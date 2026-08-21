#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._strict import AbstractAttribute, StrictModule
from .._tree_math import (
    tree_add_scaled as _tree_add_scaled,
    tree_allfinite as _tree_allfinite,
    tree_inner as _tree_inner,
    tree_negative as _tree_negative,
    tree_norm as _tree_norm,
    tree_scale as _tree_scale,
    validate_real_inexact_tree as _validate_real_inexact_tree,
)
from ..linalg import (
    DenseLinearOperator,
    eigen,
    FunctionLinearOperator,
    GMRES,
    JacobianLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    OperatorProperties,
    plan as plan_linear,
    prepare as prepare_linear,
    prepare_linearization,
    PyTreeSpace,
    refresh as refresh_linear,
    solve as solve_linear,
    TolerancePolicy,
)


ContinuationEventKind = Literal[
    "fold-candidate",
    "hopf-candidate",
    "corrector-retry",
    "corrector-failure",
    "tangent-fallback",
    "coordinate-bound",
    "stability-real-crossing",
    "stability-near-zero",
    "stability-analysis-failure",
    "user",
]
EventBracketKind = Literal["fold-candidate", "hopf-candidate"]


class ContinuationStatus(IntEnum):
    """Portable terminal status for a continuation run."""

    SUCCESS = 0
    ITERATING = 1
    INITIAL_CORRECTOR_FAILED = 2
    CORRECTOR_FAILED = 3
    TANGENT_FAILED = 4
    COORDINATE_BOUND_REACHED = 5
    NONFINITE = 6


def _real_scalar(value: Any, /, *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be one real floating-point scalar array.")
    return array


class ContinuationCurveProblem(StrictModule):
    """A residual equation restricted to one declared scalar curve.

    The solved equation is ``F(x, gamma(s)) = 0`` for one real scalar curve
    coordinate ``s``. This contract traces the declared one-dimensional path
    ``gamma``; it does not represent an unconstrained multi-parameter solution
    manifold.
    """

    coordinate_lower: AbstractAttribute[float]
    __strict_abstract__ = True
    coordinate_upper: AbstractAttribute[float]
    problem_id: AbstractAttribute[str]

    @abc.abstractmethod
    def residual(
        self,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(
        self,
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        """Evaluate the physical parameters on the declared curve."""
        raise NotImplementedError

    def parameters_jvp(
        self,
        coordinate: Any,
        coordinate_tangent: Any,
        args: Any = None,
        /,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Evaluate ``gamma(s)`` and its JVP along a scalar coordinate tangent."""
        coordinate_ = _real_scalar(coordinate, name="continuation coordinate")
        tangent_ = _real_scalar(
            coordinate_tangent,
            name="continuation coordinate tangent",
        )
        return jax.jvp(
            lambda value: self.parameters(value, args),
            (coordinate_,),
            (tangent_,),
        )

    def tangent_parameters(
        self,
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        """Return the physical-parameter tangent for increasing curve coordinate."""
        coordinate_ = _real_scalar(coordinate, name="continuation coordinate")
        return self.parameters_jvp(
            coordinate_,
            jnp.ones_like(coordinate_),
            args,
        )[1]

    def contains_coordinate(self, coordinate: Any, /) -> Array:
        value = jnp.asarray(coordinate)
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            return jnp.asarray(False)
        return (
            jnp.isfinite(value)
            & (value >= self.coordinate_lower)
            & (value <= self.coordinate_upper)
        )


def _problem_identity(
    *,
    coordinate_lower: Any,
    coordinate_upper: Any,
    problem_id: str,
) -> tuple[float, float, str]:
    lower = float(coordinate_lower)
    upper = float(coordinate_upper)
    if lower != lower or upper != upper or lower > upper:
        raise ValueError(
            "Curve-coordinate bounds must be ordered real values or infinities."
        )
    identifier = str(problem_id)
    if not identifier:
        raise ValueError("problem_id must be non-empty.")
    return lower, upper, identifier


def _validate_curve_residual(
    state: PyTree[Any],
    residual: PyTree[Any],
    /,
) -> PyTree[Array]:
    state_ = _validate_real_inexact_tree(state, name="continuation state")
    residual_ = _validate_real_inexact_tree(
        residual,
        name="continuation residual",
    )
    if jax.tree.structure(residual_) != jax.tree.structure(state_):
        raise ValueError(
            "Continuation residual and state must have the same PyTree structure."
        )
    if any(
        value.shape != template.shape
        for value, template in zip(
            jax.tree.leaves(residual_),
            jax.tree.leaves(state_),
            strict=True,
        )
    ):
        raise ValueError(
            "Continuation residual and state leaves must have matching shapes."
        )
    return jax.tree.map(
        lambda value, template: jnp.asarray(value, dtype=template.dtype),
        residual_,
        state_,
    )


class ParameterContinuationProblem(ContinuationCurveProblem):
    """Scalar physical-parameter continuation with ``gamma(s) = s``."""

    residual_function: Callable[[PyTree[Any], Array, Any], PyTree[Any]]
    coordinate_lower: float = eqx.field(static=True)
    coordinate_upper: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: Callable[[PyTree[Any], Array, Any], PyTree[Any]],
        /,
        *,
        parameter_lower: float = -jnp.inf,
        parameter_upper: float = jnp.inf,
        problem_id: str = "parameter-continuation",
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        lower, upper, identifier = _problem_identity(
            coordinate_lower=parameter_lower,
            coordinate_upper=parameter_upper,
            problem_id=problem_id,
        )
        self.residual_function = residual
        self.coordinate_lower = lower
        self.coordinate_upper = upper
        self.problem_id = identifier

    def residual(
        self,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        state_ = _validate_real_inexact_tree(state, name="continuation state")
        coordinate_ = _real_scalar(coordinate, name="continuation coordinate")
        return _validate_curve_residual(
            state_,
            self.residual_function(state_, coordinate_, args),
        )

    def parameters(
        self,
        coordinate: Any,
        args: Any = None,
        /,
    ) -> Array:
        del args
        coordinate_ = _real_scalar(coordinate, name="continuation coordinate")
        return eqx.error_if(
            coordinate_,
            ~jnp.isfinite(coordinate_),
            "Continuation coordinate must be finite.",
        )


class ParameterPathContinuationProblem(ContinuationCurveProblem):
    """Physical residual restricted to a declared PyTree-valued parameter path."""

    residual_function: Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Any]]
    path_function: Callable[[Array, Any], PyTree[Any]]
    parameter_template: PyTree[Array]
    coordinate_lower: float = eqx.field(static=True)
    coordinate_upper: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Any]],
        path: Callable[[Array, Any], PyTree[Any]],
        parameter_template: PyTree[Any],
        /,
        *,
        coordinate_lower: float = -jnp.inf,
        coordinate_upper: float = jnp.inf,
        problem_id: str = "parameter-path-continuation",
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if not callable(path):
            raise TypeError("path must be callable.")
        template = _validate_real_inexact_tree(
            parameter_template,
            name="physical parameter template",
        )
        if not all(
            bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(template)
        ):
            raise ValueError("Physical parameter template must be finite.")
        lower, upper, identifier = _problem_identity(
            coordinate_lower=coordinate_lower,
            coordinate_upper=coordinate_upper,
            problem_id=problem_id,
        )
        self.residual_function = residual
        self.path_function = path
        self.parameter_template = template
        self.coordinate_lower = lower
        self.coordinate_upper = upper
        self.problem_id = identifier

    def parameters(
        self,
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        coordinate_ = _real_scalar(coordinate, name="continuation coordinate")
        coordinate_ = eqx.error_if(
            coordinate_,
            ~jnp.isfinite(coordinate_),
            "Continuation coordinate must be finite.",
        )
        parameters = _validate_real_inexact_tree(
            self.path_function(coordinate_, args),
            name="physical parameter path value",
        )
        if jax.tree.structure(parameters) != jax.tree.structure(self.parameter_template):
            raise ValueError(
                "Physical parameter path and template must have the same PyTree "
                "structure."
            )
        for value, template in zip(
            jax.tree.leaves(parameters),
            jax.tree.leaves(self.parameter_template),
            strict=True,
        ):
            if value.shape != template.shape:
                raise ValueError(
                    "Physical parameter path and template leaf shapes must match."
                )
            if value.dtype != template.dtype:
                raise TypeError(
                    "Physical parameter path and template leaf dtypes must match."
                )
        leaves, treedef = jax.tree.flatten(parameters)
        leaves[0] = eqx.error_if(
            leaves[0],
            ~jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in leaves))),
            "Physical parameter path values must be finite.",
        )
        return jax.tree.unflatten(treedef, leaves)

    def residual(
        self,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        state_ = _validate_real_inexact_tree(state, name="continuation state")
        parameters = self.parameters(coordinate, args)
        return _validate_curve_residual(
            state_,
            self.residual_function(state_, parameters, args),
        )


class StabilityAnalysisStatus(IntEnum):
    """Validity of one continuation-point spectral analysis."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    NONFINITE = 2


class StabilityEvidence(StrictModule):
    """Eigenvalue evidence and continuous-time stability diagnostics at one point."""

    eigenvalues: Array
    mode_mask: Array
    leading_eigenvalue: Array
    leading_real_part: Array
    leading_complex_eigenvalue: Array
    leading_complex_real_part: Array
    unstable_count: Array
    marginal_count: Array
    near_zero_count: Array
    conjugate_pair_count: Array
    unpaired_complex_count: Array
    stability: Array
    status: Array
    source_status: Array
    analyzer_id: str = eqx.field(static=True)
    full_spectrum: bool = eqx.field(static=True)
    zero_tolerance: float = eqx.field(static=True)
    pair_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        eigenvalues: Any,
        mode_mask: Any,
        leading_eigenvalue: Any,
        leading_real_part: Any,
        leading_complex_eigenvalue: Any,
        leading_complex_real_part: Any,
        unstable_count: Any,
        marginal_count: Any,
        near_zero_count: Any,
        conjugate_pair_count: Any,
        unpaired_complex_count: Any,
        stability: Any,
        status: Any,
        source_status: Any,
        analyzer_id: str,
        full_spectrum: bool,
        zero_tolerance: float,
        pair_tolerance: float,
    ):
        values = jnp.asarray(eigenvalues)
        mask = jnp.asarray(mode_mask, dtype=bool)
        if values.ndim != 1 or mask.shape != values.shape or not values.size:
            raise ValueError(
                "Stability eigenvalues and mode_mask must be non-empty rank-one arrays."
            )
        identifier = str(analyzer_id)
        if not identifier:
            raise ValueError("analyzer_id must be non-empty.")
        self.eigenvalues = values
        self.mode_mask = mask
        self.leading_eigenvalue = jnp.asarray(leading_eigenvalue)
        self.leading_real_part = jnp.asarray(leading_real_part)
        self.leading_complex_eigenvalue = jnp.asarray(leading_complex_eigenvalue)
        self.leading_complex_real_part = jnp.asarray(leading_complex_real_part)
        self.unstable_count = jnp.asarray(unstable_count, dtype=jnp.int32)
        self.marginal_count = jnp.asarray(marginal_count, dtype=jnp.int32)
        self.near_zero_count = jnp.asarray(near_zero_count, dtype=jnp.int32)
        self.conjugate_pair_count = jnp.asarray(
            conjugate_pair_count,
            dtype=jnp.int32,
        )
        self.unpaired_complex_count = jnp.asarray(
            unpaired_complex_count,
            dtype=jnp.int32,
        )
        self.stability = jnp.asarray(stability, dtype=jnp.int32)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.source_status = jnp.asarray(source_status, dtype=jnp.int32)
        self.analyzer_id = identifier
        self.full_spectrum = bool(full_spectrum)
        self.zero_tolerance = float(zero_tolerance)
        self.pair_tolerance = float(pair_tolerance)

    @property
    def successful(self) -> Array:
        return self.status == int(StabilityAnalysisStatus.SUCCESS)


class AbstractStabilityAnalyzer(StrictModule):
    """Explicit spectral analysis policy for continuation equilibria."""

    analyzer_id: AbstractAttribute[str]
    zero_tolerance: AbstractAttribute[float]
    pair_tolerance: AbstractAttribute[float]

    @abc.abstractmethod
    def analyze(
        self,
        problem: ContinuationCurveProblem,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> StabilityEvidence:
        raise NotImplementedError


class DenseSchurStabilityAnalyzer(AbstractStabilityAnalyzer):
    """Full dense complex-Schur analysis for modest, possibly nonnormal systems."""

    policy: eigen.SchurSolvePolicy
    maximum_dimension: int = eqx.field(static=True)
    zero_tolerance: float = eqx.field(static=True)
    pair_tolerance: float = eqx.field(static=True)
    analyzer_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        policy: eigen.SchurSolvePolicy | None = None,
        maximum_dimension: int = 256,
        zero_tolerance: float = 1e-7,
        pair_tolerance: float = 1e-6,
        analyzer_id: str = "dense-schur-stability",
    ):
        policy_ = eigen.SchurSolvePolicy() if policy is None else policy
        dimension = int(maximum_dimension)
        zero = float(zero_tolerance)
        pair = float(pair_tolerance)
        identifier = str(analyzer_id)
        if not isinstance(policy_, eigen.SchurSolvePolicy):
            raise TypeError("policy must be a SchurSolvePolicy or None.")
        if dimension < 1:
            raise ValueError("maximum_dimension must be positive.")
        if not isfinite(zero) or not isfinite(pair) or zero < 0.0 or pair < 0.0:
            raise ValueError("Stability tolerances must be finite and non-negative.")
        if not identifier:
            raise ValueError("analyzer_id must be non-empty.")
        self.policy = policy_
        self.maximum_dimension = dimension
        self.zero_tolerance = zero
        self.pair_tolerance = pair
        self.analyzer_id = identifier

    def analyze(
        self,
        problem: ContinuationCurveProblem,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> StabilityEvidence:
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        state_ = _validate_real_inexact_tree(state, name="stability state")
        flat_state, unravel = ravel_pytree(state_)
        if flat_state.size > self.maximum_dimension:
            raise ValueError(
                "Dense stability analysis exceeds maximum_dimension; "
                "use SelfAdjointKrylovStabilityAnalyzer for a self-adjoint Jacobian."
            )

        def flat_residual(flat_parameters):
            residual = problem.residual(unravel(flat_parameters), coordinate, args)
            flat_value, _ = ravel_pytree(residual)
            return flat_value

        jacobian = jax.jacrev(flat_residual)(flat_state)
        if jacobian.shape != (flat_state.size, flat_state.size):
            raise ValueError("Dense stability analysis requires a square state Jacobian.")
        operator = DenseLinearOperator(
            jacobian,
            operator_id=f"{problem.problem_id}/stability-jacobian",
        )
        spectral_problem = eigen.SchurEigenproblem(
            operator,
            problem_id=f"{problem.problem_id}/stability-schur",
        )
        result = eigen.schur_eigensolve(spectral_problem, policy=self.policy)
        return _build_stability_evidence(
            result.eigenvalues,
            jnp.ones(result.eigenvalues.shape, dtype=bool),
            source_success=result.status == int(eigen.SchurSolveStatus.SUCCESS),
            source_status=result.status,
            analyzer_id=self.analyzer_id,
            full_spectrum=True,
            zero_tolerance=self.zero_tolerance,
            pair_tolerance=self.pair_tolerance,
        )


class SelfAdjointKrylovStabilityAnalyzer(AbstractStabilityAnalyzer):
    """Matrix-free leading-mode analysis for a user-declared self-adjoint Jacobian."""

    policy: eigen.EigenSolvePolicy
    zero_tolerance: float = eqx.field(static=True)
    pair_tolerance: float = eqx.field(static=True)
    analyzer_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        policy: eigen.EigenSolvePolicy | None = None,
        mode_count: int = 2,
        zero_tolerance: float = 1e-7,
        pair_tolerance: float = 1e-6,
        analyzer_id: str = "self-adjoint-krylov-stability",
    ):
        count = int(mode_count)
        policy_ = (
            eigen.EigenSolvePolicy(
                eigen.RestartedLanczos(),
                count=count,
                which="largest-algebraic",
            )
            if policy is None
            else policy
        )
        zero = float(zero_tolerance)
        pair = float(pair_tolerance)
        identifier = str(analyzer_id)
        if count < 1:
            raise ValueError("mode_count must be positive.")
        if not isinstance(policy_, eigen.EigenSolvePolicy):
            raise TypeError("policy must be an EigenSolvePolicy or None.")
        if policy_.which != "largest-algebraic":
            raise ValueError("Self-adjoint stability requires which='largest-algebraic'.")
        if not isfinite(zero) or not isfinite(pair) or zero < 0.0 or pair < 0.0:
            raise ValueError("Stability tolerances must be finite and non-negative.")
        if not identifier:
            raise ValueError("analyzer_id must be non-empty.")
        self.policy = policy_
        self.zero_tolerance = zero
        self.pair_tolerance = pair
        self.analyzer_id = identifier

    def analyze(
        self,
        problem: ContinuationCurveProblem,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> StabilityEvidence:
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        state_ = _validate_real_inexact_tree(state, name="stability state")
        space = PyTreeSpace(state_)
        residual, linearized = jax.linearize(
            lambda candidate: problem.residual(candidate, coordinate, args),
            state_,
        )
        space.validate(residual)
        if self.policy.count > space.size:
            raise ValueError("Requested stability mode_count exceeds state dimension.")
        operator = FunctionLinearOperator(
            linearized,
            source=space,
            target=space,
            properties=OperatorProperties(
                self_adjoint=True,
                evidence={"self_adjoint": "construction"},
            ),
            operator_id=f"{problem.problem_id}/self-adjoint-stability-jacobian",
        )
        spectral_problem = eigen.Eigenproblem(
            operator,
            problem_id=f"{problem.problem_id}/stability-krylov",
        )
        result = eigen.eigensolve(spectral_problem, policy=self.policy)
        return _build_stability_evidence(
            result.eigenvalues,
            result.mode_mask,
            source_success=result.successful,
            source_status=result.status,
            analyzer_id=self.analyzer_id,
            full_spectrum=False,
            zero_tolerance=self.zero_tolerance,
            pair_tolerance=self.pair_tolerance,
        )


class GeneralKrylovStabilityAnalyzer(AbstractStabilityAnalyzer):
    """Matrix-free nonsymmetric stability analysis with native restarted Arnoldi."""

    policy: eigen.GeneralEigenSolvePolicy
    zero_tolerance: float = eqx.field(static=True)
    pair_tolerance: float = eqx.field(static=True)
    analyzer_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        policy: eigen.GeneralEigenSolvePolicy | None = None,
        mode_count: int = 2,
        zero_tolerance: float = 1e-7,
        pair_tolerance: float = 1e-6,
        analyzer_id: str = "general-krylov-stability",
    ):
        count = int(mode_count)
        policy_ = (
            eigen.GeneralEigenSolvePolicy(
                eigen.RestartedArnoldi(),
                selection=eigen.GeneralEigenSelection(
                    "largest-real",
                    count=count,
                ),
            )
            if policy is None
            else policy
        )
        zero = float(zero_tolerance)
        pair = float(pair_tolerance)
        identifier = str(analyzer_id)
        if count < 1:
            raise ValueError("mode_count must be positive.")
        if not isinstance(policy_, eigen.GeneralEigenSolvePolicy):
            raise TypeError("policy must be a GeneralEigenSolvePolicy or None.")
        if not isinstance(policy_.method, eigen.RestartedArnoldi):
            raise ValueError(
                "General Krylov stability requires the native RestartedArnoldi method."
            )
        if policy_.selection.kind != "largest-real" or policy_.selection.count is None:
            raise ValueError(
                "General Krylov stability requires an explicit largest-real selection."
            )
        if policy_.transform.name != "standard":
            raise ValueError(
                "Largest-real stability analysis requires the standard transform."
            )
        if policy_.failure.mode != "status":
            raise ValueError(
                "General Krylov stability requires failure mode 'status' so spectral "
                "failures remain explicit."
            )
        if not isfinite(zero) or not isfinite(pair) or zero < 0.0 or pair < 0.0:
            raise ValueError("Stability tolerances must be finite and non-negative.")
        if not identifier:
            raise ValueError("analyzer_id must be non-empty.")
        self.policy = policy_
        self.zero_tolerance = zero
        self.pair_tolerance = pair
        self.analyzer_id = identifier

    def analyze(
        self,
        problem: ContinuationCurveProblem,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> StabilityEvidence:
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        state_ = _validate_real_inexact_tree(state, name="stability state")
        space = PyTreeSpace(state_)
        residual, linearized = jax.linearize(
            lambda candidate: problem.residual(candidate, coordinate, args),
            state_,
        )
        space.validate(residual)
        operator = FunctionLinearOperator(
            linearized,
            source=space,
            target=space,
            operator_id=f"{problem.problem_id}/general-stability-jacobian",
        )
        spectral_problem = eigen.GeneralEigenproblem(
            operator,
            problem_id=f"{problem.problem_id}/general-stability-krylov",
        )
        spectral_plan = eigen.plan_general_eigensolve(
            spectral_problem,
            self.policy,
        )
        prepared = eigen.prepare_general_eigensolve(
            spectral_problem,
            spectral_plan,
        )
        result = eigen.general_eigensolve(prepared)
        return _build_stability_evidence(
            result.eigenvalues,
            result.diagnostics.finite_mask,
            source_success=result.successful,
            source_status=result.status,
            analyzer_id=self.analyzer_id,
            full_spectrum=False,
            zero_tolerance=self.zero_tolerance,
            pair_tolerance=self.pair_tolerance,
        )


def _build_stability_evidence(
    eigenvalues: Any,
    mode_mask: Any,
    /,
    *,
    source_success: Any,
    source_status: Any,
    analyzer_id: str,
    full_spectrum: bool,
    zero_tolerance: float,
    pair_tolerance: float,
) -> StabilityEvidence:
    values = jnp.asarray(eigenvalues)
    complex_dtype = jnp.result_type(values.dtype, jnp.complex64)
    values = values.astype(complex_dtype)
    mask = jnp.asarray(mode_mask, dtype=bool)
    finite = jnp.any(mask) & jnp.all(jnp.where(mask, jnp.isfinite(values), True))
    real_parts = jnp.real(values)
    leading_index = jnp.argmax(jnp.where(mask, real_parts, -jnp.inf))
    leading = jnp.where(
        jnp.any(mask),
        values[leading_index],
        jnp.asarray(jnp.nan, dtype=complex_dtype),
    )
    unstable = mask & (real_parts > zero_tolerance)
    marginal = mask & (jnp.abs(real_parts) <= zero_tolerance)
    near_zero = mask & (jnp.abs(values) <= zero_tolerance)
    pair_scale = pair_tolerance * (1.0 + jnp.abs(values[:, None]))
    pair_matches = jnp.abs(values[:, None] - jnp.conj(values)[None, :]) <= pair_scale
    pair_matches = pair_matches & mask[:, None] & mask[None, :]
    pair_matches = pair_matches & ~jnp.eye(values.size, dtype=bool)
    has_pair = jnp.any(pair_matches, axis=1)
    complex_modes = mask & (jnp.abs(jnp.imag(values)) > pair_tolerance)
    paired_positive_modes = complex_modes & (jnp.imag(values) > 0.0) & has_pair
    conjugate_pairs = jnp.sum(paired_positive_modes, dtype=jnp.int32)
    complex_index = jnp.argmax(jnp.where(paired_positive_modes, real_parts, -jnp.inf))
    leading_complex = jnp.where(
        jnp.any(paired_positive_modes),
        values[complex_index],
        jnp.asarray(jnp.nan, dtype=complex_dtype),
    )
    unpaired_complex = jnp.sum(complex_modes & ~has_pair, dtype=jnp.int32)
    status = jnp.where(
        ~finite,
        int(StabilityAnalysisStatus.NONFINITE),
        jnp.where(
            source_success,
            int(StabilityAnalysisStatus.SUCCESS),
            int(StabilityAnalysisStatus.SOURCE_FAILURE),
        ),
    ).astype(jnp.int32)
    stability = jnp.where(
        status != int(StabilityAnalysisStatus.SUCCESS),
        int(eigen.SpectralStabilityStatus.UNKNOWN),
        jnp.where(
            jnp.real(leading) < -zero_tolerance,
            int(eigen.SpectralStabilityStatus.STABLE),
            jnp.where(
                jnp.real(leading) > zero_tolerance,
                int(eigen.SpectralStabilityStatus.UNSTABLE),
                int(eigen.SpectralStabilityStatus.MARGINAL),
            ),
        ),
    ).astype(jnp.int32)
    return StabilityEvidence(
        eigenvalues=values,
        mode_mask=mask,
        leading_eigenvalue=leading,
        leading_real_part=jnp.real(leading),
        leading_complex_eigenvalue=leading_complex,
        leading_complex_real_part=jnp.real(leading_complex),
        unstable_count=jnp.sum(unstable, dtype=jnp.int32),
        marginal_count=jnp.sum(marginal, dtype=jnp.int32),
        near_zero_count=jnp.sum(near_zero, dtype=jnp.int32),
        conjugate_pair_count=conjugate_pairs,
        unpaired_complex_count=unpaired_complex,
        stability=stability,
        status=status,
        source_status=source_status,
        analyzer_id=analyzer_id,
        full_spectrum=full_spectrum,
        zero_tolerance=zero_tolerance,
        pair_tolerance=pair_tolerance,
    )


class BifurcationIndicators(StrictModule):
    """Heuristic scalar indicators attached to one accepted branch point."""

    fold: Array
    hopf: Array

    def __init__(self, *, fold: Any, hopf: Any = jnp.nan):
        self.fold = _real_scalar(fold, name="fold indicator")
        self.hopf = _real_scalar(hopf, name="Hopf indicator")


class BranchPoint(StrictModule):
    """One immutable corrected point on a scalar-coordinate solution branch."""

    state: PyTree[Array]
    coordinate: Array
    parameters: PyTree[Array]
    tangent_state: PyTree[Array]
    tangent_coordinate: Array
    tangent_parameters: PyTree[Array]
    residual_norm: Array
    step_size: Array
    corrector_iterations: Array
    corrector_retries: Array
    status: Array
    tangent_status: Array
    indicators: BifurcationIndicators
    stability: StabilityEvidence | None
    fold_candidate: bool = eqx.field(static=True)
    point_id: str = eqx.field(static=True)
    parent_point_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: PyTree[Any],
        coordinate: Any,
        parameters: PyTree[Any],
        tangent_state: PyTree[Any],
        tangent_coordinate: Any,
        tangent_parameters: PyTree[Any],
        residual_norm: Any,
        step_size: Any,
        corrector_iterations: Any,
        corrector_retries: Any,
        status: Any,
        fold_candidate: bool,
        point_id: str,
        parent_point_id: str = "",
        stability: StabilityEvidence | None = None,
        tangent_status: Any = LinearSolveStatus.SUCCESS,
    ):
        state_ = _validate_real_inexact_tree(state, name="continuation state")
        tangent_ = _validate_real_inexact_tree(
            tangent_state,
            name="state tangent",
        )
        if jax.tree.structure(state_) != jax.tree.structure(tangent_):
            raise ValueError(
                "Branch state and tangent PyTrees must have the same structure."
            )
        if any(
            state_leaf.shape != tangent_leaf.shape
            for state_leaf, tangent_leaf in zip(
                jax.tree.leaves(state_), jax.tree.leaves(tangent_), strict=True
            )
        ):
            raise ValueError("Branch state and tangent leaf shapes must match.")
        parameters_ = _validate_real_inexact_tree(
            parameters,
            name="branch physical parameters",
        )
        tangent_parameters_ = _validate_real_inexact_tree(
            tangent_parameters,
            name="branch physical parameter tangent",
        )
        if jax.tree.structure(parameters_) != jax.tree.structure(tangent_parameters_):
            raise ValueError(
                "Branch physical parameters and tangents must share one PyTree structure."
            )
        if any(
            value.shape != tangent.shape or value.dtype != tangent.dtype
            for value, tangent in zip(
                jax.tree.leaves(parameters_),
                jax.tree.leaves(tangent_parameters_),
                strict=True,
            )
        ):
            raise ValueError(
                "Branch physical parameter and tangent leaf shapes/dtypes must match."
            )
        if stability is not None and not isinstance(stability, StabilityEvidence):
            raise TypeError("stability must be StabilityEvidence or None.")
        identifier = str(point_id)
        if not identifier:
            raise ValueError("point_id must be non-empty.")
        self.state = state_
        self.coordinate = _real_scalar(coordinate, name="branch coordinate")
        self.parameters = parameters_
        self.tangent_state = tangent_
        self.tangent_coordinate = _real_scalar(
            tangent_coordinate,
            name="branch coordinate tangent",
        )
        self.tangent_parameters = tangent_parameters_
        self.residual_norm = jnp.asarray(residual_norm)
        self.step_size = jnp.asarray(step_size)
        self.corrector_iterations = jnp.asarray(corrector_iterations, dtype=jnp.int32)
        self.corrector_retries = jnp.asarray(corrector_retries, dtype=jnp.int32)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.tangent_status = jnp.asarray(tangent_status, dtype=jnp.int32)
        self.indicators = BifurcationIndicators(
            fold=self.tangent_coordinate,
            hopf=(jnp.nan if stability is None else stability.leading_complex_real_part),
        )
        self.stability = stability
        self.fold_candidate = bool(fold_candidate)
        self.point_id = identifier
        self.parent_point_id = str(parent_point_id)


class EventBracket(StrictModule):
    """Two accepted points bracketing a heuristic indicator sign change."""

    left_coordinate: Array
    right_coordinate: Array
    left_indicator: Array
    right_indicator: Array
    kind: EventBracketKind = eqx.field(static=True)
    bracket_id: str = eqx.field(static=True)
    left_point_id: str = eqx.field(static=True)
    right_point_id: str = eqx.field(static=True)
    certified: bool = eqx.field(static=True)

    def __init__(
        self,
        kind: EventBracketKind,
        /,
        *,
        bracket_id: str,
        left_point_id: str,
        right_point_id: str,
        left_coordinate: Any,
        right_coordinate: Any,
        left_indicator: Any,
        right_indicator: Any,
    ):
        if kind not in ("fold-candidate", "hopf-candidate"):
            raise ValueError("Unknown event bracket kind.")
        identifiers = tuple(
            str(value) for value in (bracket_id, left_point_id, right_point_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("Bracket and endpoint point IDs must be non-empty.")
        if identifiers[1] == identifiers[2]:
            raise ValueError("Event bracket endpoints must be distinct points.")
        self.kind = kind
        self.bracket_id, self.left_point_id, self.right_point_id = identifiers
        self.left_coordinate = _real_scalar(
            left_coordinate, name="left bracket coordinate"
        )
        self.right_coordinate = _real_scalar(
            right_coordinate, name="right bracket coordinate"
        )
        self.left_indicator = _real_scalar(left_indicator, name="left bracket indicator")
        self.right_indicator = _real_scalar(
            right_indicator, name="right bracket indicator"
        )
        if not _sign_crossed(self.left_indicator, self.right_indicator):
            raise ValueError("Event bracket indicators must contain a sign change.")
        self.certified = False


class ContinuationEvent(StrictModule):
    """Typed branch event with an optional explicit bracket reference."""

    coordinate: Array
    indicator: Array
    source_status: Array
    kind: ContinuationEventKind = eqx.field(static=True)
    point_id: str = eqx.field(static=True)
    bracket_id: str = eqx.field(static=True)
    message: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ContinuationEventKind,
        coordinate: Any,
        /,
        *,
        indicator: Any = 0.0,
        source_status: Any = 0,
        point_id: str = "",
        bracket_id: str = "",
        message: str = "",
    ):
        if kind not in (
            "fold-candidate",
            "hopf-candidate",
            "corrector-retry",
            "corrector-failure",
            "tangent-fallback",
            "coordinate-bound",
            "stability-real-crossing",
            "stability-near-zero",
            "stability-analysis-failure",
            "user",
        ):
            raise ValueError("Unknown continuation event kind.")
        self.kind = kind
        self.coordinate = _real_scalar(coordinate, name="event coordinate")
        self.indicator = jnp.asarray(indicator)
        self.source_status = jnp.asarray(source_status, dtype=jnp.int32)
        self.point_id = str(point_id)
        self.bracket_id = str(bracket_id)
        self.message = str(message)


class ContinuationBranch(StrictModule):
    """Immutable ordered branch with recovery and event metadata."""

    points: tuple[BranchPoint, ...]
    events: tuple[ContinuationEvent, ...]
    brackets: tuple[EventBracket, ...]
    status: Array
    branch_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)

    def __init__(
        self,
        points: Sequence[BranchPoint],
        events: Sequence[ContinuationEvent],
        status: Any,
        /,
        *,
        brackets: Sequence[EventBracket] = (),
        branch_id: str,
        problem_id: str,
        method: str,
        termination_reason: str,
    ):
        points_ = tuple(points)
        events_ = tuple(events)
        brackets_ = tuple(brackets)
        if not points_ or any(not isinstance(point, BranchPoint) for point in points_):
            raise ValueError("A continuation branch requires BranchPoint values.")
        point_ids = tuple(point.point_id for point in points_)
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("Continuation branch point IDs must be unique.")
        if any(
            point.parent_point_id != points_[index - 1].point_id
            for index, point in enumerate(points_[1:], start=1)
        ):
            raise ValueError(
                "Each continuation point must name the preceding point as its parent."
            )
        if any(not isinstance(event, ContinuationEvent) for event in events_):
            raise TypeError("events must contain ContinuationEvent values.")
        if any(not isinstance(bracket, EventBracket) for bracket in brackets_):
            raise TypeError("brackets must contain EventBracket values.")
        bracket_ids = tuple(bracket.bracket_id for bracket in brackets_)
        if len(set(bracket_ids)) != len(bracket_ids):
            raise ValueError("Continuation event bracket IDs must be unique.")
        if any(
            event.bracket_id and event.bracket_id not in bracket_ids for event in events_
        ):
            raise ValueError("Continuation events must reference branch event brackets.")
        if any(
            bracket.left_point_id not in point_ids
            or bracket.right_point_id not in point_ids
            for bracket in brackets_
        ):
            raise ValueError("Event bracket endpoints must belong to the branch.")
        identifiers = tuple(str(value) for value in (branch_id, problem_id, method))
        if any(not value for value in identifiers):
            raise ValueError("Branch, problem, and method identifiers must be non-empty.")
        self.points = points_
        self.events = events_
        self.brackets = brackets_
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.branch_id, self.problem_id, self.method = identifiers
        self.termination_reason = str(termination_reason)

    @property
    def successful(self) -> Array:
        return (self.status == int(ContinuationStatus.SUCCESS)) | (
            self.status == int(ContinuationStatus.COORDINATE_BOUND_REACHED)
        )

    @property
    def fold_candidate_points(self) -> tuple[BranchPoint, ...]:
        return tuple(point for point in self.points if point.fold_candidate)

    @property
    def fold_brackets(self) -> tuple[EventBracket, ...]:
        return tuple(
            bracket for bracket in self.brackets if bracket.kind == "fold-candidate"
        )

    @property
    def hopf_brackets(self) -> tuple[EventBracket, ...]:
        return tuple(
            bracket for bracket in self.brackets if bracket.kind == "hopf-candidate"
        )

    @property
    def stability_points(self) -> tuple[BranchPoint, ...]:
        return tuple(point for point in self.points if point.stability is not None)

    @property
    def stability_events(self) -> tuple[ContinuationEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.kind.startswith("stability-") or event.kind == "hopf-candidate"
        )


class BranchSeed(StrictModule):
    """Explicit initial state and tangent proposed for a switched branch."""

    state: PyTree[Array]
    coordinate: Array
    tangent_state: PyTree[Array]
    tangent_coordinate: Array
    branch_id: str = eqx.field(static=True)
    source_point_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: PyTree[Any],
        coordinate: Any,
        tangent_state: PyTree[Any],
        tangent_coordinate: Any,
        branch_id: str,
        source_point_id: str,
    ):
        self.state = _validate_real_inexact_tree(state, name="branch seed state")
        self.coordinate = _real_scalar(coordinate, name="branch seed coordinate")
        self.tangent_state = _validate_real_inexact_tree(
            tangent_state,
            name="branch seed tangent",
        )
        self.tangent_coordinate = _real_scalar(
            tangent_coordinate, name="branch seed coordinate tangent"
        )
        if jax.tree.structure(self.state) != jax.tree.structure(self.tangent_state):
            raise ValueError(
                "Branch seed state and tangent PyTrees must have the same structure."
            )
        if any(
            state_leaf.shape != tangent_leaf.shape
            for state_leaf, tangent_leaf in zip(
                jax.tree.leaves(self.state),
                jax.tree.leaves(self.tangent_state),
                strict=True,
            )
        ):
            raise ValueError("Branch seed state and tangent leaf shapes must match.")
        self.branch_id = str(branch_id)
        self.source_point_id = str(source_point_id)
        if not self.branch_id or not self.source_point_id:
            raise ValueError("Branch seed branch and source point IDs must be non-empty.")


class AbstractBranchSwitchHook(StrictModule):
    """Hook that turns diagnosed branch events into explicit continuation seeds."""

    hook_id: AbstractAttribute[str]

    @abc.abstractmethod
    def propose(
        self,
        branch: ContinuationBranch,
        event: ContinuationEvent,
        args: Any = None,
        /,
    ) -> Sequence[BranchSeed]:
        raise NotImplementedError


class CallableBranchSwitchHook(AbstractBranchSwitchHook):
    """Branch-switch hook backed by one validated callable."""

    function: Callable[[ContinuationBranch, ContinuationEvent, Any], Sequence[BranchSeed]]
    hook_id: str = eqx.field(static=True)

    def __init__(self, function, /, *, hook_id: str = "callable-branch-switch"):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(hook_id)
        if not identifier:
            raise ValueError("hook_id must be non-empty.")
        self.function = function
        self.hook_id = identifier

    def propose(
        self,
        branch: ContinuationBranch,
        event: ContinuationEvent,
        args: Any = None,
        /,
    ) -> Sequence[BranchSeed]:
        seeds = tuple(self.function(branch, event, args))
        if any(not isinstance(seed, BranchSeed) for seed in seeds):
            raise TypeError("A branch-switch hook must return BranchSeed values.")
        return seeds


class AbstractBranchMonitor(StrictModule):
    """Hook observing accepted points without changing continuation state."""

    monitor_id: AbstractAttribute[str]

    @abc.abstractmethod
    def observe(
        self,
        problem: ContinuationCurveProblem,
        previous: BranchPoint | None,
        current: BranchPoint,
        args: Any = None,
        /,
    ) -> Sequence[ContinuationEvent]:
        raise NotImplementedError


class CallableBranchMonitor(AbstractBranchMonitor):
    """Accepted-point monitor backed by one deterministic callable."""

    function: Callable[
        [ContinuationCurveProblem, BranchPoint | None, BranchPoint, Any],
        Sequence[ContinuationEvent],
    ]
    monitor_id: str = eqx.field(static=True)

    def __init__(self, function, /, *, monitor_id: str = "callable-branch-monitor"):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(monitor_id)
        if not identifier:
            raise ValueError("monitor_id must be non-empty.")
        self.function = function
        self.monitor_id = identifier

    def observe(
        self,
        problem: ContinuationCurveProblem,
        previous: BranchPoint | None,
        current: BranchPoint,
        args: Any = None,
        /,
    ) -> Sequence[ContinuationEvent]:
        events = tuple(self.function(problem, previous, current, args))
        if any(not isinstance(event, ContinuationEvent) for event in events):
            raise TypeError("A branch monitor must return ContinuationEvent values.")
        return events


class AbstractContinuationMethod(StrictModule):
    """Immutable continuation method and adaptive corrector policy."""

    linear_policy: AbstractAttribute[LinearSolvePolicy]
    initial_step: AbstractAttribute[float]
    minimum_step: AbstractAttribute[float]
    maximum_step: AbstractAttribute[float]
    growth: AbstractAttribute[float]
    contraction: AbstractAttribute[float]
    target_corrector_steps: AbstractAttribute[int]
    maximum_corrector_steps: AbstractAttribute[int]
    maximum_retries: AbstractAttribute[int]
    residual_tolerance: AbstractAttribute[float]
    direction: AbstractAttribute[int]

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    def corrector_id(self) -> str:
        return f"newton/{self.linear_policy.method.name}"


def _validated_controls(
    *,
    initial_step: float,
    minimum_step: float,
    maximum_step: float,
    growth: float,
    contraction: float,
    target_corrector_steps: int,
    maximum_corrector_steps: int,
    maximum_retries: int,
    residual_tolerance: float,
    direction: int,
):
    scalar_values = tuple(
        float(value)
        for value in (
            initial_step,
            minimum_step,
            maximum_step,
            growth,
            contraction,
            residual_tolerance,
        )
    )
    if any(not isfinite(value) or value <= 0.0 for value in scalar_values):
        raise ValueError("Continuation step and tolerance values must be positive.")
    if not scalar_values[1] <= scalar_values[0] <= scalar_values[2]:
        raise ValueError("Step sizes must satisfy minimum <= initial <= maximum.")
    if scalar_values[3] <= 1.0 or scalar_values[4] >= 1.0:
        raise ValueError("growth must exceed one and contraction must be below one.")
    target = int(target_corrector_steps)
    corrector_steps = int(maximum_corrector_steps)
    retries = int(maximum_retries)
    direction_ = int(direction)
    if target < 1 or corrector_steps < 1 or retries < 0:
        raise ValueError("Corrector limits must be positive and retries non-negative.")
    if direction_ not in (-1, 1):
        raise ValueError("direction must be -1 or 1.")
    return scalar_values, target, corrector_steps, retries, direction_


def _default_corrector_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        GMRES(),
        tolerance=TolerancePolicy(relative=1e-8, absolute=1e-11),
    )


def _validated_corrector_policy(
    linear_policy: LinearSolvePolicy | None,
    /,
) -> LinearSolvePolicy:
    policy = _default_corrector_policy() if linear_policy is None else linear_policy
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
    if policy.failure.mode != "status":
        raise ValueError("Continuation correctors require failure mode 'status'.")
    return policy


class NaturalParameterContinuation(AbstractContinuationMethod):
    """Natural continuation with the scalar parameter as branch coordinate."""

    linear_policy: LinearSolvePolicy
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    growth: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    target_corrector_steps: int = eqx.field(static=True)
    maximum_corrector_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    direction: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        initial_step: float = 0.1,
        minimum_step: float = 1e-5,
        maximum_step: float = 1.0,
        growth: float = 1.5,
        contraction: float = 0.5,
        target_corrector_steps: int = 4,
        maximum_corrector_steps: int = 30,
        maximum_retries: int = 8,
        residual_tolerance: float = 1e-8,
        direction: int = 1,
    ):
        policy = _validated_corrector_policy(linear_policy)
        scalar_values, target, steps, retries, direction_ = _validated_controls(
            initial_step=initial_step,
            minimum_step=minimum_step,
            maximum_step=maximum_step,
            growth=growth,
            contraction=contraction,
            target_corrector_steps=target_corrector_steps,
            maximum_corrector_steps=maximum_corrector_steps,
            maximum_retries=maximum_retries,
            residual_tolerance=residual_tolerance,
            direction=direction,
        )
        self.linear_policy = policy
        (
            self.initial_step,
            self.minimum_step,
            self.maximum_step,
            self.growth,
            self.contraction,
            self.residual_tolerance,
        ) = scalar_values
        self.target_corrector_steps = target
        self.maximum_corrector_steps = steps
        self.maximum_retries = retries
        self.direction = direction_

    @property
    def method_id(self) -> str:
        return "natural-parameter"


class PseudoArclengthContinuation(AbstractContinuationMethod):
    """Adaptive pseudo-arclength predictor/corrector with oriented tangents."""

    linear_policy: LinearSolvePolicy
    tangent_policy: LinearSolvePolicy
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    growth: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    target_corrector_steps: int = eqx.field(static=True)
    maximum_corrector_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    direction: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        tangent_policy: LinearSolvePolicy | None = None,
        initial_step: float = 0.1,
        minimum_step: float = 1e-5,
        maximum_step: float = 1.0,
        growth: float = 1.5,
        contraction: float = 0.5,
        target_corrector_steps: int = 4,
        maximum_corrector_steps: int = 30,
        maximum_retries: int = 8,
        residual_tolerance: float = 1e-8,
        direction: int = 1,
    ):
        policy = _validated_corrector_policy(linear_policy)
        tangent_policy_ = (
            LinearSolvePolicy(
                GMRES(),
                tolerance=TolerancePolicy(relative=1e-7, absolute=1e-10),
            )
            if tangent_policy is None
            else tangent_policy
        )
        if not isinstance(tangent_policy_, LinearSolvePolicy):
            raise TypeError("tangent_policy must be a LinearSolvePolicy or None.")
        if tangent_policy_.failure.mode != "status":
            raise ValueError("Tangent solves require failure mode 'status'.")
        scalar_values, target, steps, retries, direction_ = _validated_controls(
            initial_step=initial_step,
            minimum_step=minimum_step,
            maximum_step=maximum_step,
            growth=growth,
            contraction=contraction,
            target_corrector_steps=target_corrector_steps,
            maximum_corrector_steps=maximum_corrector_steps,
            maximum_retries=maximum_retries,
            residual_tolerance=residual_tolerance,
            direction=direction,
        )
        self.linear_policy = policy
        self.tangent_policy = tangent_policy_
        (
            self.initial_step,
            self.minimum_step,
            self.maximum_step,
            self.growth,
            self.contraction,
            self.residual_tolerance,
        ) = scalar_values
        self.target_corrector_steps = target
        self.maximum_corrector_steps = steps
        self.maximum_retries = retries
        self.direction = direction_

    @property
    def method_id(self) -> str:
        return "pseudo-arclength"


class _ContinuationCorrection(StrictModule):
    state: PyTree[Array]
    residual: PyTree[Array]
    status: Array
    iterations: Array
    linear_preparations: Array
    linear_numeric_refreshes: Array
    linear_solves: Array

    def __init__(
        self,
        *,
        state: PyTree[Any],
        residual: PyTree[Any],
        status: Any,
        iterations: Any,
        linear_preparations: Any,
        linear_numeric_refreshes: Any,
        linear_solves: Any,
    ):
        self.state = state
        self.residual = residual
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.linear_preparations = jnp.asarray(
            linear_preparations,
            dtype=jnp.int32,
        )
        self.linear_numeric_refreshes = jnp.asarray(
            linear_numeric_refreshes,
            dtype=jnp.int32,
        )
        self.linear_solves = jnp.asarray(linear_solves, dtype=jnp.int32)

    @property
    def successful(self) -> Array:
        return self.status == int(LinearSolveStatus.SUCCESS)


def _newton_correct(
    residual_function: Callable[[PyTree[Any]], PyTree[Array]],
    initial_state: PyTree[Any],
    method: AbstractContinuationMethod,
    prepared_linear: Any,
    /,
    *,
    identity: str,
) -> tuple[_ContinuationCorrection, Any]:
    state = initial_state
    preparations = 0
    refreshes = 0
    linear_solves = 0
    status = jnp.asarray(
        LinearSolveStatus.MAXIMUM_STEPS_REACHED,
        dtype=jnp.int32,
    )
    residual = residual_function(state)
    for iteration in range(method.maximum_corrector_steps + 1):
        space = PyTreeSpace(state, space_id=f"{identity}/space")
        linearization = prepare_linearization(
            residual_function,
            state,
            source=space,
            target=space,
            linearization_id=f"{identity}/linearization",
        )
        residual = linearization.primal
        residual_norm = _tree_norm(residual)
        finite = (
            _tree_allfinite(state)
            & _tree_allfinite(residual)
            & jnp.isfinite(residual_norm)
        )
        if not bool(finite):
            status = jnp.asarray(
                LinearSolveStatus.NONFINITE_OUTPUT,
                dtype=jnp.int32,
            )
            return (
                _ContinuationCorrection(
                    state=state,
                    residual=residual,
                    status=status,
                    iterations=iteration,
                    linear_preparations=preparations,
                    linear_numeric_refreshes=refreshes,
                    linear_solves=linear_solves,
                ),
                prepared_linear,
            )
        if float(residual_norm) <= method.residual_tolerance:
            return (
                _ContinuationCorrection(
                    state=state,
                    residual=residual,
                    status=LinearSolveStatus.SUCCESS,
                    iterations=iteration,
                    linear_preparations=preparations,
                    linear_numeric_refreshes=refreshes,
                    linear_solves=linear_solves,
                ),
                prepared_linear,
            )
        if iteration == method.maximum_corrector_steps:
            break
        operator = JacobianLinearOperator(
            linearization,
            operator_id=f"{identity}/jacobian",
        )
        linear_problem = LinearSystem(
            operator,
            problem_id=f"{identity}/linear-system",
        )
        if prepared_linear is None:
            linear_plan = plan_linear(linear_problem, method.linear_policy)
            prepared_linear = prepare_linear(linear_problem, linear_plan)
            preparations += 1
        else:
            prepared_linear = refresh_linear(prepared_linear, linear_problem)
            refreshes += 1
        correction = solve_linear(prepared_linear, _tree_negative(residual))
        linear_solves += 1
        status = correction.status
        if not bool(correction.successful):
            return (
                _ContinuationCorrection(
                    state=state,
                    residual=residual,
                    status=status,
                    iterations=iteration + 1,
                    linear_preparations=preparations,
                    linear_numeric_refreshes=refreshes,
                    linear_solves=linear_solves,
                ),
                prepared_linear,
            )
        state = _tree_add_scaled(state, correction.value, 1.0)
    status = jnp.asarray(
        LinearSolveStatus.MAXIMUM_STEPS_REACHED,
        dtype=jnp.int32,
    )
    return (
        _ContinuationCorrection(
            state=state,
            residual=residual,
            status=status,
            iterations=method.maximum_corrector_steps,
            linear_preparations=preparations,
            linear_numeric_refreshes=refreshes,
            linear_solves=linear_solves,
        ),
        prepared_linear,
    )


def _correct_state(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    coordinate: Array,
    method: AbstractContinuationMethod,
    prepared_linear: Any,
    args: Any,
    /,
):
    return _newton_correct(
        lambda candidate: problem.residual(candidate, coordinate, args),
        state,
        method,
        prepared_linear,
        identity=f"{problem.problem_id}/fixed-coordinate-corrector",
    )


def _correct_initial_point(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    coordinate: Array,
    method: AbstractContinuationMethod,
    prepared_linear: Any,
    args: Any,
    /,
):
    return _correct_state(
        problem,
        state,
        coordinate,
        method,
        prepared_linear,
        args,
    )


def _initial_tangent(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    coordinate: Array,
    method: PseudoArclengthContinuation,
    args: Any,
    /,
):
    def residual_function(current_state):
        return problem.residual(
            current_state,
            coordinate,
            args,
        )

    residual, state_linearization = jax.linearize(residual_function, state)
    coordinate_tangent = jnp.asarray(float(method.direction), dtype=coordinate.dtype)
    coordinate_action = jax.jvp(
        lambda value: problem.residual(state, value, args),
        (coordinate,),
        (coordinate_tangent,),
    )[1]
    state_jacobian = FunctionLinearOperator(
        state_linearization,
        source=PyTreeSpace(state),
        target=PyTreeSpace(residual),
        operator_id="continuation-state-jacobian",
        closure_convert=False,
    )
    linear_result = solve_linear(
        LinearSystem(state_jacobian),
        _tree_negative(coordinate_action),
        policy=method.tangent_policy,
    )
    usable = int(linear_result.status) in (
        int(LinearSolveStatus.SUCCESS),
        int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        int(LinearSolveStatus.STAGNATION),
        int(LinearSolveStatus.CONDITION_LIMIT_REACHED),
    )
    state_tangent = linear_result.value if usable else jax.tree.map(jnp.zeros_like, state)
    norm = jnp.sqrt(_tree_inner(state_tangent, state_tangent) + coordinate_tangent**2)
    return (
        _tree_scale(1.0 / norm, state_tangent),
        coordinate_tangent / norm,
        linear_result.status,
        usable,
    )


def _normalized_secant(
    previous_state: PyTree[Any],
    previous_coordinate: Array,
    state: PyTree[Any],
    coordinate: Array,
    old_state_tangent: PyTree[Any],
    old_coordinate_tangent: Array,
    /,
):
    state_difference = jax.tree.map(
        lambda current, previous: current - previous,
        state,
        previous_state,
    )
    coordinate_difference = coordinate - previous_coordinate
    norm = jnp.sqrt(
        _tree_inner(state_difference, state_difference) + coordinate_difference**2
    )
    state_tangent = _tree_scale(1.0 / norm, state_difference)
    coordinate_tangent = coordinate_difference / norm
    orientation = (
        _tree_inner(state_tangent, old_state_tangent)
        + coordinate_tangent * old_coordinate_tangent
    )
    sign = jnp.where(orientation < 0.0, -1.0, 1.0)
    return _tree_scale(sign, state_tangent), sign * coordinate_tangent


def _sign_crossed(left: Any, right: Any, /) -> bool:
    left_ = float(left)
    right_ = float(right)
    return (
        isfinite(left_)
        and isfinite(right_)
        and ((left_ < 0.0 <= right_) or (left_ > 0.0 >= right_))
    )


def _record_stability_events(
    events: list[ContinuationEvent],
    brackets: list[EventBracket],
    previous: BranchPoint | None,
    current: BranchPoint,
    /,
) -> None:
    current_evidence = current.stability
    if current_evidence is None:
        return
    if not bool(current_evidence.successful):
        events.append(
            ContinuationEvent(
                "stability-analysis-failure",
                current.coordinate,
                indicator=current_evidence.source_status,
                source_status=current_evidence.source_status,
                point_id=current.point_id,
                message="Spectral stability analysis did not produce valid evidence.",
            )
        )
        return
    previous_evidence = None if previous is None else previous.stability
    if bool(current_evidence.near_zero_count > 0) and (
        previous_evidence is None
        or not bool(previous_evidence.successful)
        or bool(previous_evidence.near_zero_count == 0)
    ):
        events.append(
            ContinuationEvent(
                "stability-near-zero",
                current.coordinate,
                indicator=current_evidence.leading_real_part,
                point_id=current.point_id,
                message="At least one analyzed eigenvalue is near zero.",
            )
        )
    if previous is None or previous_evidence is None:
        return
    if not bool(previous_evidence.successful):
        return
    if _sign_crossed(
        previous_evidence.leading_real_part,
        current_evidence.leading_real_part,
    ):
        events.append(
            ContinuationEvent(
                "stability-real-crossing",
                current.coordinate,
                indicator=current_evidence.leading_real_part,
                point_id=current.point_id,
                message="The leading analyzed spectral real part changed sign.",
            )
        )
    if not _sign_crossed(
        previous_evidence.leading_complex_real_part,
        current_evidence.leading_complex_real_part,
    ):
        return
    bracket_id = f"{current.point_id}/hopf-bracket"
    brackets.append(
        EventBracket(
            "hopf-candidate",
            bracket_id=bracket_id,
            left_point_id=previous.point_id,
            right_point_id=current.point_id,
            left_coordinate=previous.coordinate,
            right_coordinate=current.coordinate,
            left_indicator=previous_evidence.leading_complex_real_part,
            right_indicator=current_evidence.leading_complex_real_part,
        )
    )
    events.append(
        ContinuationEvent(
            "hopf-candidate",
            current.coordinate,
            indicator=current_evidence.leading_complex_real_part,
            point_id=current.point_id,
            bracket_id=bracket_id,
            message=(
                "A monitored conjugate pair changed real-part sign; this is a "
                "heuristic Hopf bracket, not a certified bifurcation."
            ),
        )
    )


class ContinuationDiagnostics(StrictModule):
    """Deterministic numerical work and rejection evidence for one run."""

    requested_steps: Array
    attempted_steps: Array
    accepted_steps: Array
    rejected_steps: Array
    corrector_iterations: Array
    tangent_failures: Array
    spectral_evaluations: Array
    monitor_events: Array
    corrector_linear_preparations: Array
    corrector_linear_numeric_refreshes: Array
    corrector_linear_solves: Array

    def __init__(
        self,
        *,
        requested_steps: Any,
        attempted_steps: Any,
        accepted_steps: Any,
        rejected_steps: Any,
        corrector_iterations: Any,
        tangent_failures: Any,
        spectral_evaluations: Any,
        monitor_events: Any,
        corrector_linear_preparations: Any,
        corrector_linear_numeric_refreshes: Any,
        corrector_linear_solves: Any,
    ):
        values = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (
                requested_steps,
                attempted_steps,
                accepted_steps,
                rejected_steps,
                corrector_iterations,
                tangent_failures,
                spectral_evaluations,
                monitor_events,
                corrector_linear_preparations,
                corrector_linear_numeric_refreshes,
                corrector_linear_solves,
            )
        )
        (
            self.requested_steps,
            self.attempted_steps,
            self.accepted_steps,
            self.rejected_steps,
            self.corrector_iterations,
            self.tangent_failures,
            self.spectral_evaluations,
            self.monitor_events,
            self.corrector_linear_preparations,
            self.corrector_linear_numeric_refreshes,
            self.corrector_linear_solves,
        ) = values


class ContinuationProvenance(StrictModule):
    """Symbolic and numerical identities for one continuation result."""

    numeric_version: Array
    corrector_linear_numeric_version: Array
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    corrector_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    analyzer_id: str = eqx.field(static=True)
    monitor_ids: tuple[str, ...] = eqx.field(static=True)
    linear_reuse_mode: str = eqx.field(static=True)
    corrector_linear_plan_id: str = eqx.field(static=True)
    corrector_preconditioner_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        numeric_version: Any,
        problem_id: str,
        method_id: str,
        corrector_id: str,
        plan_id: str,
        prepared_id: str,
        branch_id: str,
        analyzer_id: str = "",
        monitor_ids: Sequence[str] = (),
        linear_reuse_mode: str = "none",
        corrector_linear_plan_id: str = "",
        corrector_linear_numeric_version: Any = 0,
        corrector_preconditioner_plan_id: str = "",
    ):
        identifiers = tuple(
            str(value)
            for value in (
                problem_id,
                method_id,
                corrector_id,
                plan_id,
                prepared_id,
                branch_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Continuation provenance identities must be non-empty.")
        self.numeric_version = jnp.asarray(numeric_version, dtype=jnp.int32)
        self.corrector_linear_numeric_version = jnp.asarray(
            corrector_linear_numeric_version,
            dtype=jnp.int32,
        )
        (
            self.problem_id,
            self.method_id,
            self.corrector_id,
            self.plan_id,
            self.prepared_id,
            self.branch_id,
        ) = identifiers
        self.analyzer_id = str(analyzer_id)
        reuse_mode = str(linear_reuse_mode)
        if reuse_mode not in ("none", "prepared-newton"):
            raise ValueError("linear_reuse_mode must be 'none' or 'prepared-newton'.")
        self.linear_reuse_mode = reuse_mode
        self.corrector_linear_plan_id = str(corrector_linear_plan_id)
        self.corrector_preconditioner_plan_id = str(corrector_preconditioner_plan_id)
        if reuse_mode == "prepared-newton" and not self.corrector_linear_plan_id:
            raise ValueError("Prepared Newton reuse requires a corrector linear plan ID.")
        monitors = tuple(str(value) for value in monitor_ids)
        if any(not value for value in monitors) or len(set(monitors)) != len(monitors):
            raise ValueError("Continuation monitor IDs must be non-empty and unique.")
        self.monitor_ids = monitors


class ContinuationResult(StrictModule):
    """One immutable continuation branch with terminal evidence and provenance."""

    branch: ContinuationBranch
    status: Array
    diagnostics: ContinuationDiagnostics
    provenance: ContinuationProvenance

    def __init__(
        self,
        *,
        branch: ContinuationBranch,
        status: Any,
        diagnostics: ContinuationDiagnostics,
        provenance: ContinuationProvenance,
    ):
        if not isinstance(branch, ContinuationBranch):
            raise TypeError("branch must be a ContinuationBranch.")
        if not isinstance(diagnostics, ContinuationDiagnostics):
            raise TypeError("diagnostics must be ContinuationDiagnostics.")
        if not isinstance(provenance, ContinuationProvenance):
            raise TypeError("provenance must be ContinuationProvenance.")
        self.branch = branch
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance

    @property
    def successful(self) -> Array:
        return self.branch.successful

    @property
    def termination_reason(self) -> str:
        return self.branch.termination_reason

    @property
    def points(self) -> tuple[BranchPoint, ...]:
        return self.branch.points

    @property
    def events(self) -> tuple[ContinuationEvent, ...]:
        return self.branch.events

    @property
    def brackets(self) -> tuple[EventBracket, ...]:
        return self.branch.brackets

    @property
    def fold_brackets(self) -> tuple[EventBracket, ...]:
        return self.branch.fold_brackets

    @property
    def hopf_brackets(self) -> tuple[EventBracket, ...]:
        return self.branch.hopf_brackets

    @property
    def fold_candidate_points(self) -> tuple[BranchPoint, ...]:
        return self.branch.fold_candidate_points

    @property
    def stability_points(self) -> tuple[BranchPoint, ...]:
        return self.branch.stability_points

    @property
    def stability_events(self) -> tuple[ContinuationEvent, ...]:
        return self.branch.stability_events


class ContinuationPlan(StrictModule):
    """Reusable symbolic continuation and monitoring policy."""

    method: AbstractContinuationMethod
    stability_analyzer: AbstractStabilityAnalyzer | None
    monitors: tuple[AbstractBranchMonitor, ...]
    num_steps: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: AbstractContinuationMethod,
        stability_analyzer: AbstractStabilityAnalyzer | None,
        monitors: Sequence[AbstractBranchMonitor],
        num_steps: int,
        problem_id: str,
        branch_id: str,
        plan_id: str,
    ):
        if not isinstance(method, AbstractContinuationMethod):
            raise TypeError("method must be an AbstractContinuationMethod.")
        if stability_analyzer is not None and not isinstance(
            stability_analyzer, AbstractStabilityAnalyzer
        ):
            raise TypeError(
                "stability_analyzer must be an AbstractStabilityAnalyzer or None."
            )
        monitors_ = tuple(monitors)
        if any(not isinstance(monitor, AbstractBranchMonitor) for monitor in monitors_):
            raise TypeError("monitors must contain AbstractBranchMonitor values.")
        monitor_ids = tuple(str(monitor.monitor_id) for monitor in monitors_)
        if any(not value for value in monitor_ids) or len(set(monitor_ids)) != len(
            monitor_ids
        ):
            raise ValueError("Continuation monitor IDs must be non-empty and unique.")
        steps = int(num_steps)
        if steps < 0:
            raise ValueError("num_steps must be non-negative.")
        identifiers = tuple(str(value) for value in (problem_id, branch_id, plan_id))
        if any(not value for value in identifiers):
            raise ValueError("Problem, branch, and plan IDs must be non-empty.")
        self.method = method
        self.stability_analyzer = stability_analyzer
        self.monitors = monitors_
        self.num_steps = steps
        self.problem_id, self.branch_id, self.plan_id = identifiers


class PreparedContinuation(StrictModule):
    """Numerical continuation seed bound to one reusable symbolic plan."""

    problem: ContinuationCurveProblem
    plan: ContinuationPlan
    initial_state: PyTree[Array]
    initial_coordinate: Array
    initial_tangent: tuple[PyTree[Array], Array] | None
    args: Any
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: ContinuationCurveProblem,
        plan: ContinuationPlan,
        initial_state: PyTree[Any],
        initial_coordinate: Any,
        /,
        *,
        initial_tangent: tuple[PyTree[Any], Any] | None = None,
        args: Any = None,
        numeric_version: Any = 0,
        prepared_id: str,
    ):
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        if not isinstance(plan, ContinuationPlan):
            raise TypeError("plan must be a ContinuationPlan.")
        if plan.problem_id != problem.problem_id:
            raise ValueError("Continuation plan and problem IDs must match.")
        state = _validate_real_inexact_tree(initial_state, name="initial state")
        coordinate = _real_scalar(initial_coordinate, name="initial coordinate")
        if not bool(problem.contains_coordinate(coordinate)):
            raise ValueError("initial_coordinate lies outside the continuation interval.")
        tangent = _normalize_initial_tangent(initial_tangent, state)
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version, version < 0, "numeric_version must be non-negative."
        )
        identifier = str(prepared_id)
        if not identifier:
            raise ValueError("prepared_id must be non-empty.")
        self.problem = problem
        self.plan = plan
        self.initial_state = state
        self.initial_coordinate = coordinate
        self.initial_tangent = tangent
        self.args = args
        self.numeric_version = version
        self.prepared_id = identifier


def _normalize_initial_tangent(
    tangent: tuple[PyTree[Any], Any] | None,
    state: PyTree[Any],
    /,
) -> tuple[PyTree[Array], Array] | None:
    if tangent is None:
        return None
    if not isinstance(tangent, tuple) or len(tangent) != 2:
        raise TypeError(
            "initial_tangent must be a (state_tangent, coordinate_tangent) tuple."
        )
    state_tangent = _validate_real_inexact_tree(tangent[0], name="initial state tangent")
    if jax.tree.structure(state_tangent) != jax.tree.structure(state):
        raise ValueError(
            "Initial state and tangent PyTrees must have the same structure."
        )
    if any(
        tangent_leaf.shape != state_leaf.shape
        for tangent_leaf, state_leaf in zip(
            jax.tree.leaves(state_tangent), jax.tree.leaves(state), strict=True
        )
    ):
        raise ValueError("Initial state and tangent leaf shapes must match.")
    coordinate_tangent = _real_scalar(tangent[1], name="initial coordinate tangent")
    norm = jnp.sqrt(_tree_inner(state_tangent, state_tangent) + coordinate_tangent**2)
    if not bool(jnp.isfinite(norm) & (norm > 0.0)):
        raise ValueError("initial_tangent must have finite nonzero norm.")
    return _tree_scale(1.0 / norm, state_tangent), coordinate_tangent / norm


def plan_continuation(
    problem: ContinuationCurveProblem,
    /,
    *,
    num_steps: int,
    method: AbstractContinuationMethod | None = None,
    branch_id: str = "branch-0",
    stability_analyzer: AbstractStabilityAnalyzer | None = None,
    monitors: Sequence[AbstractBranchMonitor] = (),
    plan_id: str | None = None,
) -> ContinuationPlan:
    """Create reusable symbolic continuation and monitoring policy."""
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    method_ = PseudoArclengthContinuation() if method is None else method
    if not isinstance(method_, AbstractContinuationMethod):
        raise TypeError("method must be an AbstractContinuationMethod or None.")
    identifier = (
        f"{problem.problem_id}/{method_.method_id}/continuation-plan"
        if plan_id is None
        else str(plan_id)
    )
    return ContinuationPlan(
        method=method_,
        stability_analyzer=stability_analyzer,
        monitors=monitors,
        num_steps=num_steps,
        problem_id=problem.problem_id,
        branch_id=branch_id,
        plan_id=identifier,
    )


def prepare_continuation(
    problem: ContinuationCurveProblem,
    initial_state: PyTree[Any],
    initial_coordinate: Any,
    plan: ContinuationPlan,
    /,
    *,
    initial_tangent: tuple[PyTree[Any], Any] | None = None,
    args: Any = None,
) -> PreparedContinuation:
    """Bind a numerical seed to a reusable continuation plan."""
    return PreparedContinuation(
        problem,
        plan,
        initial_state,
        initial_coordinate,
        initial_tangent=initial_tangent,
        args=args,
        numeric_version=0,
        prepared_id=f"{plan.plan_id}/prepared",
    )


def refresh_continuation(
    prepared: PreparedContinuation,
    initial_state: PyTree[Any],
    initial_coordinate: Any,
    /,
    *,
    initial_tangent: tuple[PyTree[Any], Any] | None = None,
    args: Any = None,
) -> PreparedContinuation:
    """Rebind numerical seed data while retaining symbolic plan identity."""
    if not isinstance(prepared, PreparedContinuation):
        raise TypeError("prepared must be a PreparedContinuation.")
    return PreparedContinuation(
        prepared.problem,
        prepared.plan,
        initial_state,
        initial_coordinate,
        initial_tangent=initial_tangent,
        args=args,
        numeric_version=prepared.numeric_version + 1,
        prepared_id=prepared.prepared_id,
    )


def _arclength_residual(
    variables: tuple[PyTree[Any], Array],
    payload: tuple[
        ContinuationCurveProblem,
        PyTree[Any],
        Array,
        PyTree[Any],
        Array,
        Any,
    ],
    /,
) -> tuple[PyTree[Array], Array]:
    (
        problem,
        predicted_state,
        predicted_coordinate,
        state_tangent,
        coordinate_tangent,
        args,
    ) = payload
    candidate_state, candidate_coordinate = variables
    equation = problem.residual(candidate_state, candidate_coordinate, args)
    state_displacement = jax.tree.map(
        lambda candidate, predicted: candidate - predicted,
        candidate_state,
        predicted_state,
    )
    arclength = _tree_inner(
        state_tangent,
        state_displacement,
    ) + coordinate_tangent * (candidate_coordinate - predicted_coordinate)
    return equation, arclength


def _correct_arclength(
    problem: ContinuationCurveProblem,
    predicted_state: PyTree[Any],
    predicted_coordinate: Array,
    state_tangent: PyTree[Any],
    coordinate_tangent: Array,
    method: PseudoArclengthContinuation,
    prepared_linear: Any,
    args: Any,
    /,
):
    payload = (
        problem,
        predicted_state,
        predicted_coordinate,
        state_tangent,
        coordinate_tangent,
        args,
    )
    return _newton_correct(
        lambda variables: _arclength_residual(variables, payload),
        (predicted_state, predicted_coordinate),
        method,
        prepared_linear,
        identity=f"{problem.problem_id}/arclength-corrector",
    )


def _observe_monitors(
    monitors: Sequence[AbstractBranchMonitor],
    problem: ContinuationCurveProblem,
    previous: BranchPoint | None,
    current: BranchPoint,
    args: Any,
    events: list[ContinuationEvent],
    /,
) -> int:
    count = 0
    for monitor in monitors:
        observed = tuple(monitor.observe(problem, previous, current, args))
        if any(not isinstance(event, ContinuationEvent) for event in observed):
            raise TypeError("A branch monitor must return ContinuationEvent values.")
        events.extend(observed)
        count += len(observed)
    return count


def _continuation_result(
    prepared: PreparedContinuation,
    points: Sequence[BranchPoint],
    events: Sequence[ContinuationEvent],
    brackets: Sequence[EventBracket],
    status: ContinuationStatus,
    termination_reason: str,
    /,
    *,
    attempted_steps: int,
    accepted_steps: int,
    rejected_steps: int,
    corrector_iterations: int,
    tangent_failures: int,
    spectral_evaluations: int,
    monitor_events: int,
    linear_preparations: int,
    linear_numeric_refreshes: int,
    linear_solves: int,
    corrector_prepared_linear: Any,
) -> ContinuationResult:
    plan = prepared.plan
    if corrector_prepared_linear is None:
        linear_reuse_mode = "none"
        corrector_linear_plan_id = ""
        corrector_linear_numeric_version = jnp.asarray(0, dtype=jnp.int32)
        corrector_preconditioner_plan_id = ""
    else:
        linear_reuse_mode = "prepared-newton"
        corrector_linear_plan_id = corrector_prepared_linear.plan.plan_id
        corrector_linear_numeric_version = corrector_prepared_linear.numeric_version
        preconditioner_plan = corrector_prepared_linear.plan.preconditioner_plan
        corrector_preconditioner_plan_id = (
            "" if preconditioner_plan is None else preconditioner_plan.plan_id
        )
    branch = ContinuationBranch(
        points,
        events,
        status,
        brackets=brackets,
        branch_id=plan.branch_id,
        problem_id=prepared.problem.problem_id,
        method=plan.method.method_id,
        termination_reason=termination_reason,
    )
    return ContinuationResult(
        branch=branch,
        status=status,
        diagnostics=ContinuationDiagnostics(
            requested_steps=plan.num_steps,
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            corrector_iterations=corrector_iterations,
            tangent_failures=tangent_failures,
            spectral_evaluations=spectral_evaluations,
            monitor_events=monitor_events,
            corrector_linear_preparations=linear_preparations,
            corrector_linear_numeric_refreshes=linear_numeric_refreshes,
            corrector_linear_solves=linear_solves,
        ),
        provenance=ContinuationProvenance(
            numeric_version=prepared.numeric_version,
            problem_id=prepared.problem.problem_id,
            method_id=plan.method.method_id,
            corrector_id=plan.method.corrector_id,
            plan_id=plan.plan_id,
            prepared_id=prepared.prepared_id,
            branch_id=plan.branch_id,
            analyzer_id=(
                ""
                if plan.stability_analyzer is None
                else plan.stability_analyzer.analyzer_id
            ),
            monitor_ids=tuple(monitor.monitor_id for monitor in plan.monitors),
            linear_reuse_mode=linear_reuse_mode,
            corrector_linear_plan_id=corrector_linear_plan_id,
            corrector_linear_numeric_version=corrector_linear_numeric_version,
            corrector_preconditioner_plan_id=corrector_preconditioner_plan_id,
        ),
    )


def run_continuation(prepared: PreparedContinuation, /) -> ContinuationResult:
    """Run one prepared natural or pseudo-arclength continuation."""
    if not isinstance(prepared, PreparedContinuation):
        raise TypeError("prepared must be a PreparedContinuation.")
    problem = prepared.problem
    plan = prepared.plan
    method = plan.method
    state = prepared.initial_state
    coordinate = prepared.initial_coordinate
    args = prepared.args
    events: list[ContinuationEvent] = []
    brackets: list[EventBracket] = []
    points: list[BranchPoint] = []
    attempted_steps = 0
    accepted_steps = 0
    rejected_steps = 0
    tangent_failures = 0
    spectral_evaluations = 0
    monitor_events = 0
    linear_preparations = 0
    linear_numeric_refreshes = 0
    linear_solves = 0
    corrector_prepared_linear = None

    initial_result, initial_prepared_linear = _correct_initial_point(
        problem,
        state,
        coordinate,
        method,
        None,
        args,
    )
    if isinstance(method, NaturalParameterContinuation):
        corrector_prepared_linear = initial_prepared_linear
    linear_preparations += int(initial_result.linear_preparations)
    linear_numeric_refreshes += int(initial_result.linear_numeric_refreshes)
    linear_solves += int(initial_result.linear_solves)
    state = initial_result.state
    residual_norm = _tree_norm(initial_result.residual)
    corrector_iterations = int(initial_result.iterations)
    initial_success = bool(
        initial_result.successful & (residual_norm <= method.residual_tolerance)
    )
    tangent_attempted = initial_success
    if not initial_success:
        state_tangent = jax.tree.map(jnp.zeros_like, state)
        coordinate_tangent = jnp.asarray(float(method.direction), dtype=coordinate.dtype)
        tangent_status = jnp.asarray(
            LinearSolveStatus.CAPABILITY_REJECTED, dtype=jnp.int32
        )
        tangent_usable = False
    elif prepared.initial_tangent is not None:
        state_tangent, coordinate_tangent = prepared.initial_tangent
        tangent_status = jnp.asarray(LinearSolveStatus.SUCCESS, dtype=jnp.int32)
        tangent_usable = True
    elif isinstance(method, PseudoArclengthContinuation):
        (
            state_tangent,
            coordinate_tangent,
            tangent_status,
            tangent_usable,
        ) = _initial_tangent(problem, state, coordinate, method, args)
    else:
        state_tangent = jax.tree.map(jnp.zeros_like, state)
        coordinate_tangent = jnp.asarray(float(method.direction), dtype=coordinate.dtype)
        tangent_status = jnp.asarray(LinearSolveStatus.SUCCESS, dtype=jnp.int32)
        tangent_usable = True
    if tangent_attempted and int(tangent_status) != int(LinearSolveStatus.SUCCESS):
        tangent_failures += 1
    initial_stability = (
        plan.stability_analyzer.analyze(problem, state, coordinate, args)
        if plan.stability_analyzer is not None and initial_success
        else None
    )
    if initial_stability is not None:
        spectral_evaluations += 1
    physical_parameters, physical_parameter_tangent = problem.parameters_jvp(
        coordinate,
        coordinate_tangent,
        args,
    )
    initial_point = BranchPoint(
        state=state,
        coordinate=coordinate,
        parameters=physical_parameters,
        tangent_state=state_tangent,
        tangent_coordinate=coordinate_tangent,
        tangent_parameters=physical_parameter_tangent,
        residual_norm=residual_norm,
        step_size=0.0,
        corrector_iterations=initial_result.iterations,
        corrector_retries=0,
        status=initial_result.status,
        tangent_status=tangent_status,
        fold_candidate=False,
        point_id=f"{plan.branch_id}/0",
        stability=initial_stability,
    )
    points.append(initial_point)
    _record_stability_events(events, brackets, None, initial_point)
    if tangent_attempted and not tangent_usable:
        events.append(
            ContinuationEvent(
                "tangent-fallback",
                coordinate,
                indicator=tangent_status,
                source_status=tangent_status,
                point_id=initial_point.point_id,
                message=(
                    "Initial state-Jacobian solve failed; the explicit coordinate "
                    "direction was retained."
                ),
            )
        )
    if not initial_success:
        return _continuation_result(
            prepared,
            points,
            events,
            brackets,
            ContinuationStatus.INITIAL_CORRECTOR_FAILED,
            "initial corrector failed",
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            corrector_iterations=corrector_iterations,
            tangent_failures=tangent_failures,
            spectral_evaluations=spectral_evaluations,
            monitor_events=monitor_events,
            linear_preparations=linear_preparations,
            linear_numeric_refreshes=linear_numeric_refreshes,
            linear_solves=linear_solves,
            corrector_prepared_linear=initial_prepared_linear,
        )
    monitor_events += _observe_monitors(
        plan.monitors, problem, None, initial_point, args, events
    )

    step_size = method.initial_step
    status = ContinuationStatus.ITERATING
    termination_reason = "requested points reached"
    for point_index in range(1, plan.num_steps + 1):
        accepted = False
        bound_reached = False
        retries = 0
        while retries <= method.maximum_retries and step_size >= method.minimum_step:
            if isinstance(method, NaturalParameterContinuation):
                predicted_state = state
                predicted_coordinate = coordinate + method.direction * step_size
            else:
                predicted_state = _tree_add_scaled(state, state_tangent, step_size)
                predicted_coordinate = coordinate + step_size * coordinate_tangent
            if not bool(problem.contains_coordinate(predicted_coordinate)):
                events.append(
                    ContinuationEvent(
                        "coordinate-bound",
                        coordinate,
                        indicator=predicted_coordinate,
                        point_id=points[-1].point_id,
                        message="Predictor crossed the declared coordinate interval.",
                    )
                )
                status = ContinuationStatus.COORDINATE_BOUND_REACHED
                termination_reason = "coordinate bound reached"
                bound_reached = True
                break
            attempted_steps += 1
            if isinstance(method, NaturalParameterContinuation):
                (
                    corrector_result,
                    corrector_prepared_linear,
                ) = _correct_state(
                    problem,
                    predicted_state,
                    predicted_coordinate,
                    method,
                    corrector_prepared_linear,
                    args,
                )
                candidate_state = corrector_result.state
                candidate_coordinate = predicted_coordinate
            else:
                (
                    corrector_result,
                    corrector_prepared_linear,
                ) = _correct_arclength(
                    problem,
                    predicted_state,
                    predicted_coordinate,
                    state_tangent,
                    coordinate_tangent,
                    method,
                    corrector_prepared_linear,
                    args,
                )
                candidate_state, candidate_coordinate = corrector_result.state
            linear_preparations += int(corrector_result.linear_preparations)
            linear_numeric_refreshes += int(corrector_result.linear_numeric_refreshes)
            linear_solves += int(corrector_result.linear_solves)
            corrector_iterations += int(corrector_result.iterations)
            equation_residual = problem.residual(
                candidate_state, candidate_coordinate, args
            )
            candidate_residual_norm = _tree_norm(equation_residual)
            accepted = bool(
                corrector_result.successful
                & (candidate_residual_norm <= method.residual_tolerance)
                & problem.contains_coordinate(candidate_coordinate)
            )
            if accepted:
                break
            rejected_steps += 1
            retries += 1
            step_size = max(method.minimum_step, step_size * method.contraction)
            events.append(
                ContinuationEvent(
                    "corrector-retry",
                    coordinate,
                    indicator=step_size,
                    source_status=corrector_result.status,
                    point_id=points[-1].point_id,
                    message="Corrector rejected; continuation step reduced.",
                )
            )
        if bound_reached:
            break
        if not accepted:
            status = ContinuationStatus.CORRECTOR_FAILED
            termination_reason = "corrector recovery exhausted"
            events.append(
                ContinuationEvent(
                    "corrector-failure",
                    coordinate,
                    indicator=step_size,
                    source_status=corrector_result.status,
                    point_id=points[-1].point_id,
                    message="Minimum step or retry budget reached.",
                )
            )
            break

        previous_point = points[-1]
        old_state_tangent = state_tangent
        old_coordinate_tangent = coordinate_tangent
        state_tangent, coordinate_tangent = _normalized_secant(
            state,
            coordinate,
            candidate_state,
            candidate_coordinate,
            old_state_tangent,
            old_coordinate_tangent,
        )
        state = candidate_state
        coordinate = candidate_coordinate
        fold_candidate = _sign_crossed(old_coordinate_tangent, coordinate_tangent)
        point_id = f"{plan.branch_id}/{point_index}"
        stability = (
            plan.stability_analyzer.analyze(problem, state, coordinate, args)
            if plan.stability_analyzer is not None
            else None
        )
        if stability is not None:
            spectral_evaluations += 1
        physical_parameters, physical_parameter_tangent = problem.parameters_jvp(
            coordinate,
            coordinate_tangent,
            args,
        )
        current_point = BranchPoint(
            state=state,
            coordinate=coordinate,
            parameters=physical_parameters,
            tangent_state=state_tangent,
            tangent_coordinate=coordinate_tangent,
            tangent_parameters=physical_parameter_tangent,
            residual_norm=candidate_residual_norm,
            step_size=step_size,
            corrector_iterations=corrector_result.iterations,
            corrector_retries=retries,
            status=corrector_result.status,
            tangent_status=LinearSolveStatus.SUCCESS,
            fold_candidate=fold_candidate,
            point_id=point_id,
            parent_point_id=previous_point.point_id,
            stability=stability,
        )
        if fold_candidate:
            bracket_id = f"{point_id}/fold-bracket"
            brackets.append(
                EventBracket(
                    "fold-candidate",
                    bracket_id=bracket_id,
                    left_point_id=previous_point.point_id,
                    right_point_id=point_id,
                    left_coordinate=previous_point.coordinate,
                    right_coordinate=current_point.coordinate,
                    left_indicator=previous_point.indicators.fold,
                    right_indicator=current_point.indicators.fold,
                )
            )
            events.append(
                ContinuationEvent(
                    "fold-candidate",
                    coordinate,
                    indicator=coordinate_tangent,
                    point_id=point_id,
                    bracket_id=bracket_id,
                    message=(
                        "The oriented coordinate tangent changed sign; this is a "
                        "heuristic fold bracket, not a certified bifurcation."
                    ),
                )
            )
        _record_stability_events(events, brackets, previous_point, current_point)
        monitor_events += _observe_monitors(
            plan.monitors, problem, previous_point, current_point, args, events
        )
        points.append(current_point)
        accepted_steps += 1
        iterations = int(corrector_result.iterations)
        if retries == 0 and iterations <= method.target_corrector_steps:
            step_size = min(method.maximum_step, step_size * method.growth)
        elif iterations > 2 * method.target_corrector_steps:
            step_size = max(method.minimum_step, step_size * method.contraction)
    else:
        status = ContinuationStatus.SUCCESS

    return _continuation_result(
        prepared,
        points,
        events,
        brackets,
        status,
        termination_reason,
        attempted_steps=attempted_steps,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        corrector_iterations=corrector_iterations,
        tangent_failures=tangent_failures,
        spectral_evaluations=spectral_evaluations,
        monitor_events=monitor_events,
        linear_preparations=linear_preparations,
        linear_numeric_refreshes=linear_numeric_refreshes,
        linear_solves=linear_solves,
        corrector_prepared_linear=(
            initial_prepared_linear
            if corrector_prepared_linear is None
            else corrector_prepared_linear
        ),
    )


def continue_branch(
    problem: ContinuationCurveProblem,
    initial_state: PyTree[Any],
    initial_coordinate: Any,
    /,
    *,
    num_steps: int,
    method: AbstractContinuationMethod | None = None,
    initial_tangent: tuple[PyTree[Any], Any] | None = None,
    branch_id: str = "branch-0",
    args: Any = None,
    stability_analyzer: AbstractStabilityAnalyzer | None = None,
    monitors: Sequence[AbstractBranchMonitor] = (),
    plan_id: str | None = None,
) -> ContinuationResult:
    """Plan, prepare, and run one immutable continuation result."""
    plan = plan_continuation(
        problem,
        num_steps=num_steps,
        method=method,
        branch_id=branch_id,
        stability_analyzer=stability_analyzer,
        monitors=monitors,
        plan_id=plan_id,
    )
    prepared = prepare_continuation(
        problem,
        initial_state,
        initial_coordinate,
        plan,
        initial_tangent=initial_tangent,
        args=args,
    )
    return run_continuation(prepared)


def propose_branch_seeds(
    branch: ContinuationBranch,
    hooks: Sequence[AbstractBranchSwitchHook],
    /,
    *,
    args: Any = None,
) -> tuple[BranchSeed, ...]:
    """Apply branch-switch hooks to every diagnosed event."""

    if not isinstance(branch, ContinuationBranch):
        raise TypeError("branch must be a ContinuationBranch.")
    hooks_ = tuple(hooks)
    if any(not isinstance(hook, AbstractBranchSwitchHook) for hook in hooks_):
        raise TypeError("hooks must contain AbstractBranchSwitchHook values.")
    seeds = []
    for event in branch.events:
        for hook in hooks_:
            proposed = tuple(hook.propose(branch, event, args))
            if any(not isinstance(seed, BranchSeed) for seed in proposed):
                raise TypeError("A branch-switch hook must return BranchSeed values.")
            seeds.extend(proposed)
    return tuple(seeds)


__all__ = [
    "AbstractBranchMonitor",
    "AbstractBranchSwitchHook",
    "AbstractContinuationMethod",
    "AbstractStabilityAnalyzer",
    "ContinuationCurveProblem",
    "BifurcationIndicators",
    "BranchPoint",
    "BranchSeed",
    "CallableBranchMonitor",
    "CallableBranchSwitchHook",
    "ContinuationBranch",
    "ContinuationDiagnostics",
    "ContinuationEvent",
    "ContinuationPlan",
    "ContinuationProvenance",
    "ContinuationResult",
    "ContinuationStatus",
    "DenseSchurStabilityAnalyzer",
    "EventBracket",
    "GeneralKrylovStabilityAnalyzer",
    "NaturalParameterContinuation",
    "ParameterContinuationProblem",
    "ParameterPathContinuationProblem",
    "PreparedContinuation",
    "PseudoArclengthContinuation",
    "SelfAdjointKrylovStabilityAnalyzer",
    "StabilityAnalysisStatus",
    "StabilityEvidence",
    "continue_branch",
    "plan_continuation",
    "prepare_continuation",
    "propose_branch_seeds",
    "refresh_continuation",
    "run_continuation",
]
