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
from jaxtyping import Array, PyTree

from .._strict import AbstractAttribute, StrictModule
from .._tree_math import (
    tree_add_scaled as _tree_add_scaled,
    tree_allfinite as _tree_allfinite,
    validate_inexact_tree as _validate_inexact_tree,
    validate_real_inexact_tree as _validate_real_inexact_tree,
)
from ..linalg import (
    AbstractVectorSpace,
    ArraySpace,
    BlockSpace,
    DenseLinearOperator,
    eigen,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    OperatorProperties,
    solve as solve_linear,
    TolerancePolicy,
)
from ..nonlinear import (
    AbstractNonlinearMethod,
    ImplicitRootDerivativePolicy,
    NewtonKrylov,
    NewtonTrustRegion,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
)
from ..nonlinear._prepared import _solve_prepared_nonlinear_stateful
from ._checkpoint import (
    continuation_checkpoint,
    ContinuationCheckpoint,
    ContinuationReplayEvidence,
)
from ._geometry import ContinuationGeometry, ContinuationRepresentationPolicy
from ._state_machine import (
    AbstractContinuationAdapter,
    CallableContinuationAdapter,
    continuation_step_decision_id,
    ContinuationAcceptedState,
    ContinuationCandidate,
    ContinuationStepResult,
    ParameterRealization,
    ParameterTransferEvidence,
)


ContinuationEventKind = Literal[
    "fold-candidate",
    "hopf-candidate",
    "corrector-retry",
    "corrector-failure",
    "target-corrector-retry",
    "coordinate-target",
    "predictor-fallback",
    "tangent-retry",
    "curvature-retry",
    "tangent-fallback",
    "application-retry",
    "application-failure",
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
    TARGET_NOT_REACHED = 7
    TARGET_CORRECTOR_FAILED = 8
    CURVATURE_LIMIT_REACHED = 9
    APPLICATION_REJECTED = 10


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

    @abc.abstractmethod
    def declared_spaces(
        self,
        /,
    ) -> tuple[AbstractVectorSpace | None, AbstractVectorSpace | None]:
        """Return optional public state and residual spaces."""
        return None, None

    @abc.abstractmethod
    def representation_policy(self, /) -> ContinuationRepresentationPolicy:
        """Return the public-to-real execution representation."""
        return ContinuationRepresentationPolicy()

    @abc.abstractmethod
    def state_jacobian_action(
        self,
        state: PyTree[Any],
        coordinate: Any,
        tangent: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        """Apply the state Jacobian to one public-state tangent."""
        state_ = _validate_inexact_tree(state, name="continuation state")
        tangent_ = _validate_inexact_tree(tangent, name="continuation state tangent")
        return jax.jvp(
            lambda value: self.residual(value, coordinate, args),
            (state_,),
            (tangent_,),
        )[1]

    @abc.abstractmethod
    def coordinate_derivative(
        self,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        """Differentiate the public residual along increasing curve coordinate."""
        coordinate_ = _real_scalar(coordinate, name="continuation coordinate")
        return jax.jvp(
            lambda value: self.residual(state, value, args),
            (coordinate_,),
            (jnp.ones_like(coordinate_),),
        )[1]

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
    *,
    state_space: AbstractVectorSpace | None = None,
    residual_space: AbstractVectorSpace | None = None,
) -> PyTree[Array]:
    if state_space is None:
        _validate_inexact_tree(state, name="continuation state")
    else:
        state_space.validate(state)
    if residual_space is None:
        return _validate_inexact_tree(
            residual,
            name="continuation residual",
        )
    return residual_space.validate(residual)


class ParameterContinuationProblem(ContinuationCurveProblem):
    """Scalar physical-parameter continuation with ``gamma(s) = s``."""

    residual_function: Callable[[PyTree[Any], Array, Any], PyTree[Any]]
    state_jacobian_function: Callable[..., PyTree[Any]] | None
    coordinate_derivative_function: Callable[..., PyTree[Any]] | None
    state_space: AbstractVectorSpace | None
    residual_space: AbstractVectorSpace | None
    representation: ContinuationRepresentationPolicy
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
        state_space: AbstractVectorSpace | None = None,
        residual_space: AbstractVectorSpace | None = None,
        representation: ContinuationRepresentationPolicy | None = None,
        state_jacobian_action: Callable[..., PyTree[Any]] | None = None,
        coordinate_derivative: Callable[..., PyTree[Any]] | None = None,
        problem_id: str = "parameter-continuation",
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        for value, name in (
            (state_space, "state_space"),
            (residual_space, "residual_space"),
        ):
            if value is not None and not isinstance(value, AbstractVectorSpace):
                raise TypeError(f"{name} must be an AbstractVectorSpace or None.")
        for value, name in (
            (state_jacobian_action, "state_jacobian_action"),
            (coordinate_derivative, "coordinate_derivative"),
        ):
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None.")
        representation_ = (
            ContinuationRepresentationPolicy()
            if representation is None
            else representation
        )
        if not isinstance(representation_, ContinuationRepresentationPolicy):
            raise TypeError(
                "representation must be ContinuationRepresentationPolicy or None."
            )
        lower, upper, identifier = _problem_identity(
            coordinate_lower=parameter_lower,
            coordinate_upper=parameter_upper,
            problem_id=problem_id,
        )
        self.residual_function = residual
        self.state_jacobian_function = state_jacobian_action
        self.coordinate_derivative_function = coordinate_derivative
        self.state_space = state_space
        self.residual_space = residual_space
        self.representation = representation_
        self.coordinate_lower = lower
        self.coordinate_upper = upper
        self.problem_id = identifier

    def declared_spaces(
        self,
        /,
    ) -> tuple[AbstractVectorSpace | None, AbstractVectorSpace | None]:
        return self.state_space, self.residual_space

    def representation_policy(self, /) -> ContinuationRepresentationPolicy:
        return self.representation

    def residual(
        self,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        state_ = (
            _validate_inexact_tree(state, name="continuation state")
            if self.state_space is None
            else self.state_space.validate(state)
        )
        coordinate_ = _real_scalar(coordinate, name="continuation coordinate")
        return _validate_curve_residual(
            state_,
            self.residual_function(state_, coordinate_, args),
            state_space=self.state_space,
            residual_space=self.residual_space,
        )

    def state_jacobian_action(
        self,
        state: PyTree[Any],
        coordinate: Any,
        tangent: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        if self.state_jacobian_function is None:
            return super().state_jacobian_action(
                state,
                coordinate,
                tangent,
                args,
            )
        return _validate_curve_residual(
            state,
            self.state_jacobian_function(state, coordinate, tangent, args),
            state_space=self.state_space,
            residual_space=self.residual_space,
        )

    def coordinate_derivative(
        self,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        if self.coordinate_derivative_function is None:
            return super().coordinate_derivative(state, coordinate, args)
        return _validate_curve_residual(
            state,
            self.coordinate_derivative_function(state, coordinate, args),
            state_space=self.state_space,
            residual_space=self.residual_space,
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
    state_jacobian_function: Callable[..., PyTree[Any]] | None
    coordinate_derivative_function: Callable[..., PyTree[Any]] | None
    parameter_template: PyTree[Array]
    state_space: AbstractVectorSpace | None
    residual_space: AbstractVectorSpace | None
    representation: ContinuationRepresentationPolicy
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
        state_space: AbstractVectorSpace | None = None,
        residual_space: AbstractVectorSpace | None = None,
        representation: ContinuationRepresentationPolicy | None = None,
        state_jacobian_action: Callable[..., PyTree[Any]] | None = None,
        coordinate_derivative: Callable[..., PyTree[Any]] | None = None,
        problem_id: str = "parameter-path-continuation",
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if not callable(path):
            raise TypeError("path must be callable.")
        for value, name in (
            (state_space, "state_space"),
            (residual_space, "residual_space"),
        ):
            if value is not None and not isinstance(value, AbstractVectorSpace):
                raise TypeError(f"{name} must be an AbstractVectorSpace or None.")
        for value, name in (
            (state_jacobian_action, "state_jacobian_action"),
            (coordinate_derivative, "coordinate_derivative"),
        ):
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None.")
        representation_ = (
            ContinuationRepresentationPolicy()
            if representation is None
            else representation
        )
        if not isinstance(representation_, ContinuationRepresentationPolicy):
            raise TypeError(
                "representation must be ContinuationRepresentationPolicy or None."
            )
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
        self.state_jacobian_function = state_jacobian_action
        self.coordinate_derivative_function = coordinate_derivative
        self.parameter_template = template
        self.state_space = state_space
        self.residual_space = residual_space
        self.representation = representation_
        self.coordinate_lower = lower
        self.coordinate_upper = upper
        self.problem_id = identifier

    def declared_spaces(
        self,
        /,
    ) -> tuple[AbstractVectorSpace | None, AbstractVectorSpace | None]:
        return self.state_space, self.residual_space

    def representation_policy(self, /) -> ContinuationRepresentationPolicy:
        return self.representation

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
        state_ = (
            _validate_inexact_tree(state, name="continuation state")
            if self.state_space is None
            else self.state_space.validate(state)
        )
        parameters = self.parameters(coordinate, args)
        return _validate_curve_residual(
            state_,
            self.residual_function(state_, parameters, args),
            state_space=self.state_space,
            residual_space=self.residual_space,
        )

    def state_jacobian_action(
        self,
        state: PyTree[Any],
        coordinate: Any,
        tangent: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        if self.state_jacobian_function is None:
            return super().state_jacobian_action(
                state,
                coordinate,
                tangent,
                args,
            )
        return _validate_curve_residual(
            state,
            self.state_jacobian_function(state, coordinate, tangent, args),
            state_space=self.state_space,
            residual_space=self.residual_space,
        )

    def coordinate_derivative(
        self,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        if self.coordinate_derivative_function is None:
            return super().coordinate_derivative(state, coordinate, args)
        return _validate_curve_residual(
            state,
            self.coordinate_derivative_function(state, coordinate, args),
            state_space=self.state_space,
            residual_space=self.residual_space,
        )


class StabilityAnalysisStatus(IntEnum):
    """Validity of one continuation-point spectral analysis."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    NONFINITE = 2
    CAPABILITY_REJECTED = 3


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
        *,
        geometry: ContinuationGeometry | None = None,
    ) -> StabilityEvidence:
        raise NotImplementedError


def _resolve_stability_geometry(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    coordinate: Any,
    args: Any,
    geometry: ContinuationGeometry | None,
    /,
) -> ContinuationGeometry:
    if geometry is not None:
        if not isinstance(geometry, ContinuationGeometry):
            raise TypeError("geometry must be a ContinuationGeometry or None.")
        return geometry
    residual = problem.residual(state, coordinate, args)
    state_space, residual_space = problem.declared_spaces()
    return ContinuationGeometry.resolve(
        state,
        residual,
        state_space=state_space,
        residual_space=residual_space,
        representation=problem.representation_policy(),
    )


def _capability_stability_evidence(
    analyzer: AbstractStabilityAnalyzer,
    /,
    *,
    full_spectrum: bool,
) -> StabilityEvidence:
    value = jnp.asarray([jnp.nan + 0.0j])
    return StabilityEvidence(
        eigenvalues=value,
        mode_mask=jnp.asarray([False]),
        leading_eigenvalue=value[0],
        leading_real_part=jnp.nan,
        leading_complex_eigenvalue=value[0],
        leading_complex_real_part=jnp.nan,
        unstable_count=0,
        marginal_count=0,
        near_zero_count=0,
        conjugate_pair_count=0,
        unpaired_complex_count=0,
        stability=eigen.SpectralStabilityStatus.UNKNOWN,
        status=StabilityAnalysisStatus.CAPABILITY_REJECTED,
        source_status=-1,
        analyzer_id=analyzer.analyzer_id,
        full_spectrum=full_spectrum,
        zero_tolerance=analyzer.zero_tolerance,
        pair_tolerance=analyzer.pair_tolerance,
    )


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
        *,
        geometry: ContinuationGeometry | None = None,
    ) -> StabilityEvidence:
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        geometry_ = _resolve_stability_geometry(
            problem,
            state,
            coordinate,
            args,
            geometry,
        )
        if not geometry_.execution_state_space.compatible(
            geometry_.execution_residual_space
        ):
            return _capability_stability_evidence(self, full_spectrum=True)
        state_ = geometry_.state_to_execution(state)
        flat_state = geometry_.execution_state_space.flatten(state_)
        if flat_state.size > self.maximum_dimension:
            raise ValueError(
                "Dense stability analysis exceeds maximum_dimension; "
                "use SelfAdjointKrylovStabilityAnalyzer for a self-adjoint Jacobian."
            )

        def flat_residual(flat_parameters):
            residual = _execution_residual(
                problem,
                geometry_,
                geometry_.execution_state_space.unflatten(flat_parameters),
                coordinate,
                args,
            )
            return geometry_.execution_residual_space.flatten(residual)

        jacobian = jax.jacrev(flat_residual)(flat_state)
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
        *,
        geometry: ContinuationGeometry | None = None,
    ) -> StabilityEvidence:
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        geometry_ = _resolve_stability_geometry(
            problem,
            state,
            coordinate,
            args,
            geometry,
        )
        if not geometry_.execution_state_space.compatible(
            geometry_.execution_residual_space
        ):
            return _capability_stability_evidence(self, full_spectrum=False)
        state_ = geometry_.state_to_execution(state)
        residual, linearized = jax.linearize(
            lambda candidate: _execution_residual(
                problem,
                geometry_,
                candidate,
                coordinate,
                args,
            ),
            state_,
        )
        geometry_.execution_residual_space.validate(residual)
        if self.policy.count > geometry_.execution_state_space.size:
            raise ValueError("Requested stability mode_count exceeds state dimension.")
        operator = FunctionLinearOperator(
            linearized,
            source=geometry_.execution_state_space,
            target=geometry_.execution_residual_space,
            properties=OperatorProperties(
                self_adjoint=True,
                evidence={"self_adjoint": "user-declared-stability-policy"},
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
        *,
        geometry: ContinuationGeometry | None = None,
    ) -> StabilityEvidence:
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        geometry_ = _resolve_stability_geometry(
            problem,
            state,
            coordinate,
            args,
            geometry,
        )
        if not geometry_.execution_state_space.compatible(
            geometry_.execution_residual_space
        ):
            return _capability_stability_evidence(self, full_spectrum=False)
        state_ = geometry_.state_to_execution(state)
        residual, linearized = jax.linearize(
            lambda candidate: _execution_residual(
                problem,
                geometry_,
                candidate,
                coordinate,
                args,
            ),
            state_,
        )
        geometry_.execution_residual_space.validate(residual)
        operator = FunctionLinearOperator(
            linearized,
            source=geometry_.execution_state_space,
            target=geometry_.execution_residual_space,
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
    tangent_residual_norm: Array
    tangent_alignment: Array
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
        tangent_residual_norm: Any = jnp.nan,
        tangent_alignment: Any = jnp.nan,
        corrector_iterations: Any,
        corrector_retries: Any,
        status: Any,
        fold_candidate: bool,
        point_id: str,
        parent_point_id: str = "",
        stability: StabilityEvidence | None = None,
        tangent_status: Any = LinearSolveStatus.SUCCESS,
    ):
        state_ = _validate_inexact_tree(state, name="continuation state")
        tangent_ = _validate_inexact_tree(
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
        self.tangent_residual_norm = jnp.asarray(tangent_residual_norm)
        self.tangent_alignment = jnp.asarray(tangent_alignment)
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
            "target-corrector-retry",
            "coordinate-target",
            "predictor-fallback",
            "tangent-retry",
            "curvature-retry",
            "tangent-fallback",
            "application-retry",
            "application-failure",
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
    geometry: ContinuationGeometry
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
        geometry: ContinuationGeometry,
        branch_id: str,
        problem_id: str,
        method: str,
        termination_reason: str,
    ):
        points_ = tuple(points)
        events_ = tuple(events)
        brackets_ = tuple(brackets)
        if not isinstance(geometry, ContinuationGeometry):
            raise TypeError("geometry must be a ContinuationGeometry.")
        if not points_ or any(not isinstance(point, BranchPoint) for point in points_):
            raise ValueError("A continuation branch requires BranchPoint values.")
        for point in points_:
            geometry.public_state_space.validate(point.state)
            geometry.public_state_space.validate(point.tangent_state)
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
        self.geometry = geometry
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
        self.state = _validate_inexact_tree(state, name="branch seed state")
        self.coordinate = _real_scalar(coordinate, name="branch seed coordinate")
        self.tangent_state = _validate_inexact_tree(
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

    corrector: AbstractAttribute[AbstractNonlinearMethod]
    termination: AbstractAttribute[NonlinearTermination]
    derivative_policy: AbstractAttribute[ImplicitRootDerivativePolicy]
    initial_step: AbstractAttribute[float]
    minimum_step: AbstractAttribute[float]
    maximum_step: AbstractAttribute[float]
    growth: AbstractAttribute[float]
    contraction: AbstractAttribute[float]
    coordinate_scale: AbstractAttribute[float]
    target_corrector_steps: AbstractAttribute[int]
    maximum_retries: AbstractAttribute[int]
    direction: AbstractAttribute[int]

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    def corrector_id(self) -> str:
        return self.corrector.method_id


def _validated_controls(
    *,
    initial_step: float,
    minimum_step: float,
    maximum_step: float,
    growth: float,
    contraction: float,
    coordinate_scale: float,
    target_corrector_steps: int,
    maximum_retries: int,
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
        )
    )
    if any(not isfinite(value) or value <= 0.0 for value in scalar_values):
        raise ValueError("Continuation step controls must be finite and positive.")
    if not scalar_values[1] <= scalar_values[0] <= scalar_values[2]:
        raise ValueError("Step sizes must satisfy minimum <= initial <= maximum.")
    if scalar_values[3] <= 1.0 or scalar_values[4] >= 1.0:
        raise ValueError("growth must exceed one and contraction must be below one.")
    scale = float(coordinate_scale)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale must be finite and positive.")
    target = int(target_corrector_steps)
    retries = int(maximum_retries)
    direction_ = int(direction)
    if target < 1 or retries < 0:
        raise ValueError("Corrector target must be positive and retries non-negative.")
    if direction_ not in (-1, 1):
        raise ValueError("direction must be -1 or 1.")
    return scalar_values, scale, target, retries, direction_


def _default_corrector() -> NewtonKrylov:
    return NewtonKrylov(
        linear_policy=LinearSolvePolicy(
            GMRES(),
            tolerance=TolerancePolicy(relative=1e-8, absolute=1e-11),
        )
    )


def _default_corrector_termination() -> NonlinearTermination:
    return NonlinearTermination(
        absolute_residual=1e-8,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=30,
    )


def _validated_corrector(
    corrector: AbstractNonlinearMethod | None,
    termination: NonlinearTermination | None,
    derivative_policy: ImplicitRootDerivativePolicy | None,
    /,
) -> tuple[
    AbstractNonlinearMethod,
    NonlinearTermination,
    ImplicitRootDerivativePolicy,
]:
    corrector_ = _default_corrector() if corrector is None else corrector
    termination_ = (
        _default_corrector_termination() if termination is None else termination
    )
    derivative_ = (
        ImplicitRootDerivativePolicy() if derivative_policy is None else derivative_policy
    )
    if not isinstance(corrector_, AbstractNonlinearMethod):
        raise TypeError("corrector must be an AbstractNonlinearMethod or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be a NonlinearTermination or None.")
    if not isinstance(derivative_, ImplicitRootDerivativePolicy):
        raise TypeError("derivative_policy must be ImplicitRootDerivativePolicy or None.")
    return corrector_, termination_, derivative_


class NaturalParameterContinuation(AbstractContinuationMethod):
    """Natural continuation with the scalar parameter as branch coordinate."""

    corrector: AbstractNonlinearMethod
    termination: NonlinearTermination
    derivative_policy: ImplicitRootDerivativePolicy
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    growth: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    coordinate_scale: float = eqx.field(static=True)
    target_corrector_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    direction: int = eqx.field(static=True)
    predictor: Literal["constant", "tangent"] = eqx.field(static=True)
    predictor_failure: Literal["terminate", "constant"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        corrector: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        derivative_policy: ImplicitRootDerivativePolicy | None = None,
        initial_step: float = 0.1,
        minimum_step: float = 1e-5,
        maximum_step: float = 1.0,
        growth: float = 1.5,
        contraction: float = 0.5,
        coordinate_scale: float = 1.0,
        target_corrector_steps: int = 4,
        maximum_retries: int = 8,
        direction: int = 1,
        predictor: Literal["constant", "tangent"] = "constant",
        predictor_failure: Literal["terminate", "constant"] = "terminate",
    ):
        corrector_, termination_, derivative_ = _validated_corrector(
            corrector,
            termination,
            derivative_policy,
        )
        scalar_values, scale, target, retries, direction_ = _validated_controls(
            initial_step=initial_step,
            minimum_step=minimum_step,
            maximum_step=maximum_step,
            growth=growth,
            contraction=contraction,
            coordinate_scale=coordinate_scale,
            target_corrector_steps=target_corrector_steps,
            maximum_retries=maximum_retries,
            direction=direction,
        )
        if target > termination_.maximum_steps:
            raise ValueError(
                "target_corrector_steps cannot exceed termination.maximum_steps."
            )
        if predictor not in ("constant", "tangent"):
            raise ValueError("predictor must be 'constant' or 'tangent'.")
        if predictor_failure not in ("terminate", "constant"):
            raise ValueError("predictor_failure must be 'terminate' or 'constant'.")
        self.corrector = corrector_
        self.termination = termination_
        self.derivative_policy = derivative_
        (
            self.initial_step,
            self.minimum_step,
            self.maximum_step,
            self.growth,
            self.contraction,
        ) = scalar_values
        self.coordinate_scale = scale
        self.target_corrector_steps = target
        self.maximum_retries = retries
        self.direction = direction_
        self.predictor = predictor
        self.predictor_failure = predictor_failure

    @property
    def method_id(self) -> str:
        return "natural-parameter"


class PseudoArclengthContinuation(AbstractContinuationMethod):
    """Adaptive pseudo-arclength predictor/corrector with oriented tangents."""

    corrector: AbstractNonlinearMethod
    termination: NonlinearTermination
    derivative_policy: ImplicitRootDerivativePolicy
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    growth: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    coordinate_scale: float = eqx.field(static=True)
    target_corrector_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    direction: int = eqx.field(static=True)
    tangent_update: Literal["secant", "bordered"] = eqx.field(static=True)
    minimum_tangent_alignment: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        corrector: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        derivative_policy: ImplicitRootDerivativePolicy | None = None,
        initial_step: float = 0.1,
        minimum_step: float = 1e-5,
        maximum_step: float = 1.0,
        growth: float = 1.5,
        contraction: float = 0.5,
        coordinate_scale: float = 1.0,
        target_corrector_steps: int = 4,
        maximum_retries: int = 8,
        direction: int = 1,
        tangent_update: Literal["secant", "bordered"] = "secant",
        minimum_tangent_alignment: float | None = None,
    ):
        corrector_, termination_, derivative_ = _validated_corrector(
            corrector,
            termination,
            derivative_policy,
        )
        scalar_values, scale, target, retries, direction_ = _validated_controls(
            initial_step=initial_step,
            minimum_step=minimum_step,
            maximum_step=maximum_step,
            growth=growth,
            contraction=contraction,
            coordinate_scale=coordinate_scale,
            target_corrector_steps=target_corrector_steps,
            maximum_retries=maximum_retries,
            direction=direction,
        )
        if target > termination_.maximum_steps:
            raise ValueError(
                "target_corrector_steps cannot exceed termination.maximum_steps."
            )
        if tangent_update not in ("secant", "bordered"):
            raise ValueError("tangent_update must be 'secant' or 'bordered'.")
        alignment = (
            None
            if minimum_tangent_alignment is None
            else float(minimum_tangent_alignment)
        )
        if alignment is not None and (
            not isfinite(alignment) or not 0.0 <= alignment <= 1.0
        ):
            raise ValueError("minimum_tangent_alignment must lie in [0, 1] or be None.")
        self.corrector = corrector_
        self.termination = termination_
        self.derivative_policy = derivative_
        (
            self.initial_step,
            self.minimum_step,
            self.maximum_step,
            self.growth,
            self.contraction,
        ) = scalar_values
        self.coordinate_scale = scale
        self.target_corrector_steps = target
        self.maximum_retries = retries
        self.direction = direction_
        self.tangent_update = tangent_update
        self.minimum_tangent_alignment = alignment

    @property
    def method_id(self) -> str:
        return "pseudo-arclength"


def _supports_prepared_corrector(method: AbstractNonlinearMethod, /) -> bool:
    return isinstance(method, (NewtonKrylov, NewtonTrustRegion))


def _run_nonlinear_corrector(
    residual_function: Callable[[PyTree[Any]], PyTree[Array]],
    initial_state: PyTree[Any],
    corrector: AbstractNonlinearMethod,
    termination: NonlinearTermination,
    prepared_corrector: PreparedNonlinearSolve | None,
    /,
    *,
    identity: str,
) -> tuple[NonlinearResult, PreparedNonlinearSolve | None]:
    nonlinear_problem = NonlinearSystemProblem(
        lambda state, _: residual_function(state),
        problem_id=identity,
    )
    if _supports_prepared_corrector(corrector):
        if prepared_corrector is None:
            prepared = prepare_nonlinear(
                nonlinear_problem,
                initial_state,
                method=corrector,
                termination=termination,
            )
        else:
            prepared = refresh_nonlinear(
                prepared_corrector,
                nonlinear_problem,
                initial_state,
            )
        return _solve_prepared_nonlinear_stateful(
            prepared,
            termination=termination,
        )
    result = corrector.solve(
        nonlinear_problem,
        initial_state,
        termination=termination,
    )
    return result, None


def _run_corrector(
    residual_function: Callable[[PyTree[Any]], PyTree[Array]],
    initial_state: PyTree[Any],
    method: AbstractContinuationMethod,
    prepared_corrector: PreparedNonlinearSolve | None,
    /,
    *,
    identity: str,
) -> tuple[NonlinearResult, PreparedNonlinearSolve | None]:
    return _run_nonlinear_corrector(
        residual_function,
        initial_state,
        method.corrector,
        method.termination,
        prepared_corrector,
        identity=identity,
    )


def _corrector_success(
    result: NonlinearResult,
    residual_norm: Any,
    method: AbstractContinuationMethod,
    /,
) -> bool:
    threshold = method.termination.residual_threshold(
        result.diagnostics.initial_residual_norm
    )
    return bool(
        result.successful
        & jnp.isfinite(residual_norm)
        & (jnp.asarray(residual_norm) <= threshold)
    )


def _corrector_work(
    result: NonlinearResult,
    /,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    diagnostics = result.diagnostics
    return (
        int(diagnostics.iterations),
        int(diagnostics.residual_evaluations),
        int(diagnostics.jvp_evaluations),
        int(diagnostics.vjp_evaluations),
        int(diagnostics.jacobian_preparations),
        int(diagnostics.linear_solves),
        int(diagnostics.linear_iterations),
        int(diagnostics.setup_refreshes),
        int(diagnostics.numeric_refreshes),
    )


def _execution_residual(
    problem: ContinuationCurveProblem,
    geometry: ContinuationGeometry,
    state: PyTree[Any],
    coordinate: Any,
    args: Any,
    /,
) -> PyTree[Array]:
    public_state = geometry.state_from_execution(state)
    public_residual = problem.residual(public_state, coordinate, args)
    return geometry.residual_to_execution(public_residual)


def _correct_state(
    problem: ContinuationCurveProblem,
    geometry: ContinuationGeometry,
    state: PyTree[Any],
    coordinate: Array,
    method: AbstractContinuationMethod,
    prepared_corrector: PreparedNonlinearSolve | None,
    args: Any,
    /,
):
    return _run_corrector(
        lambda candidate: _execution_residual(
            problem,
            geometry,
            candidate,
            coordinate,
            args,
        ),
        state,
        method,
        prepared_corrector,
        identity=f"{problem.problem_id}/fixed-coordinate-corrector",
    )


def _state_derivative_operator(
    problem: ContinuationCurveProblem,
    geometry: ContinuationGeometry,
    state: PyTree[Any],
    coordinate: Array,
    args: Any,
    /,
) -> tuple[FunctionLinearOperator, PyTree[Array]]:
    public_state = geometry.state_from_execution(state)

    def state_action(tangent):
        public_tangent = geometry.state_tangent_from_execution(state, tangent)
        public_action = problem.state_jacobian_action(
            public_state,
            coordinate,
            public_tangent,
            args,
        )
        return geometry.residual_to_execution(public_action)

    public_coordinate_action = problem.coordinate_derivative(
        public_state,
        coordinate,
        args,
    )
    coordinate_action = geometry.residual_to_execution(public_coordinate_action)
    return (
        FunctionLinearOperator(
            state_action,
            source=geometry.execution_state_space,
            target=geometry.execution_residual_space,
            operator_id=f"{problem.problem_id}/continuation-state-jacobian",
            closure_convert=False,
        ),
        coordinate_action,
    )


def _state_parameter_tangent(
    problem: ContinuationCurveProblem,
    geometry: ContinuationGeometry,
    state: PyTree[Any],
    coordinate: Array,
    method: AbstractContinuationMethod,
    args: Any,
    /,
) -> tuple[PyTree[Array], Array, Array, bool]:
    state_jacobian, coordinate_action = _state_derivative_operator(
        problem,
        geometry,
        state,
        coordinate,
        args,
    )
    coordinate_space = ArraySpace(
        (geometry.execution_state_space.size,),
        dtype=geometry.coordinate_dtype,
        space_id=f"{problem.problem_id}/state-tangent-coordinates",
    )
    coordinate_operator = FunctionLinearOperator(
        lambda value: geometry.execution_residual_space.flatten(
            state_jacobian(geometry.execution_state_space.unflatten(value))
        ),
        source=coordinate_space,
        target=coordinate_space,
        operator_id=f"{problem.problem_id}/state-tangent-coordinate-operator",
        closure_convert=False,
    )
    tangent_policy, _ = method.derivative_policy.resolve(method.corrector)
    linear_result = solve_linear(
        LinearSystem(coordinate_operator),
        -geometry.execution_residual_space.flatten(coordinate_action),
        policy=tangent_policy,
    )
    tangent = geometry.execution_state_space.unflatten(linear_result.value)
    residual = jax.tree.map(
        lambda state_value, coordinate_value: state_value + coordinate_value,
        state_jacobian(tangent),
        coordinate_action,
    )
    residual_norm = geometry.residual_norm(residual)
    usable = bool(
        linear_result.successful & _tree_allfinite(tangent) & jnp.isfinite(residual_norm)
    )
    return tangent, linear_result.status, residual_norm, usable


def _tangent_residual_norm(
    problem: ContinuationCurveProblem,
    geometry: ContinuationGeometry,
    state: PyTree[Any],
    coordinate: Array,
    state_tangent: PyTree[Any],
    coordinate_tangent: Array,
    args: Any,
    /,
) -> Array:
    state_jacobian, coordinate_action = _state_derivative_operator(
        problem,
        geometry,
        state,
        coordinate,
        args,
    )
    residual = jax.tree.map(
        lambda state_value, coordinate_value: (
            state_value + coordinate_tangent * coordinate_value
        ),
        state_jacobian(state_tangent),
        coordinate_action,
    )
    return geometry.residual_norm(residual)


def _bordered_tangent(
    problem: ContinuationCurveProblem,
    geometry: ContinuationGeometry,
    state: PyTree[Any],
    coordinate: Array,
    previous_state_tangent: PyTree[Any],
    previous_coordinate_tangent: Array,
    method: PseudoArclengthContinuation,
    args: Any,
    /,
) -> tuple[PyTree[Array], Array, Array, bool, Array, Array]:
    state_jacobian, coordinate_action = _state_derivative_operator(
        problem,
        geometry,
        state,
        coordinate,
        args,
    )
    scalar_space = ArraySpace((), dtype=geometry.coordinate_dtype)
    source = BlockSpace(
        (geometry.execution_state_space, scalar_space),
        names=("state", "coordinate"),
    )
    target = BlockSpace(
        (geometry.execution_residual_space, scalar_space),
        names=("residual", "normalization"),
    )

    def tangent_action(tangent):
        state_value, coordinate_value = tangent
        equation = jax.tree.map(
            lambda state_action, coordinate_action_: (
                state_action + coordinate_value * coordinate_action_
            ),
            state_jacobian(state_value),
            coordinate_action,
        )
        normalization = geometry.augmented_inner(
            previous_state_tangent,
            previous_coordinate_tangent,
            state_value,
            coordinate_value,
        )
        return equation, normalization

    operator = FunctionLinearOperator(
        tangent_action,
        source=source,
        target=target,
        operator_id=f"{problem.problem_id}/bordered-tangent",
        closure_convert=False,
    )
    coordinate_space = ArraySpace(
        (source.size,),
        dtype=geometry.coordinate_dtype,
        space_id=f"{problem.problem_id}/bordered-tangent-coordinates",
    )
    coordinate_operator = FunctionLinearOperator(
        lambda value: target.flatten(operator(source.unflatten(value))),
        source=coordinate_space,
        target=coordinate_space,
        operator_id=f"{problem.problem_id}/bordered-tangent-coordinate-operator",
        closure_convert=False,
    )
    right_hand_side = target.flatten(
        (
            geometry.execution_residual_space.zeros(),
            jnp.ones((), dtype=geometry.coordinate_dtype),
        )
    )
    tangent_policy, _ = method.derivative_policy.resolve(method.corrector)
    linear_result = solve_linear(
        LinearSystem(coordinate_operator),
        right_hand_side,
        policy=tangent_policy,
    )
    state_tangent, coordinate_tangent = source.unflatten(linear_result.value)
    norm = geometry.augmented_norm(state_tangent, coordinate_tangent)
    state_tangent = jax.tree.map(lambda value: value / norm, state_tangent)
    coordinate_tangent = coordinate_tangent / norm
    alignment = geometry.augmented_inner(
        previous_state_tangent,
        previous_coordinate_tangent,
        state_tangent,
        coordinate_tangent,
    )
    residual_norm = _tangent_residual_norm(
        problem,
        geometry,
        state,
        coordinate,
        state_tangent,
        coordinate_tangent,
        args,
    )
    usable = bool(
        linear_result.successful
        & _tree_allfinite(state_tangent)
        & jnp.isfinite(coordinate_tangent)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(alignment)
    )
    return (
        state_tangent,
        coordinate_tangent,
        linear_result.status,
        usable,
        residual_norm,
        alignment,
    )


def _correct_initial_point(
    problem: ContinuationCurveProblem,
    geometry: ContinuationGeometry,
    state: PyTree[Any],
    coordinate: Array,
    method: AbstractContinuationMethod,
    prepared_corrector: PreparedNonlinearSolve | None,
    args: Any,
    /,
):
    return _correct_state(
        problem,
        geometry,
        state,
        coordinate,
        method,
        prepared_corrector,
        args,
    )


def _initial_tangent(
    problem: ContinuationCurveProblem,
    geometry: ContinuationGeometry,
    state: PyTree[Any],
    coordinate: Array,
    method: PseudoArclengthContinuation,
    args: Any,
    /,
):
    (
        state_parameter_tangent,
        tangent_status,
        tangent_residual_norm,
        usable,
    ) = _state_parameter_tangent(
        problem,
        geometry,
        state,
        coordinate,
        method,
        args,
    )
    direction = jnp.asarray(float(method.direction), dtype=coordinate.dtype)
    state_tangent = jax.tree.map(
        lambda value: direction * value,
        state_parameter_tangent,
    )
    if not usable:
        state_tangent = jax.tree.map(jnp.zeros_like, state)
    norm = geometry.augmented_norm(state_tangent, direction)
    state_tangent = jax.tree.map(lambda value: value / norm, state_tangent)
    coordinate_tangent = direction / norm
    return (
        state_tangent,
        coordinate_tangent,
        tangent_status,
        usable,
        tangent_residual_norm,
        jnp.asarray(1.0, dtype=coordinate.dtype),
    )


def _normalized_secant(
    previous_state: PyTree[Any],
    previous_coordinate: Array,
    state: PyTree[Any],
    coordinate: Array,
    old_state_tangent: PyTree[Any],
    old_coordinate_tangent: Array,
    geometry: ContinuationGeometry,
    /,
):
    state_difference = jax.tree.map(
        lambda current, previous: current - previous,
        state,
        previous_state,
    )
    coordinate_difference = coordinate - previous_coordinate
    norm = geometry.augmented_norm(state_difference, coordinate_difference)
    state_tangent = jax.tree.map(lambda value: value / norm, state_difference)
    coordinate_tangent = coordinate_difference / norm
    orientation = geometry.augmented_inner(
        state_tangent,
        coordinate_tangent,
        old_state_tangent,
        old_coordinate_tangent,
    )
    sign = jnp.where(orientation < 0.0, -1.0, 1.0)
    return jax.tree.map(lambda value: sign * value, state_tangent), (
        sign * coordinate_tangent
    )


def _sign_crossed(left: Any, right: Any, /) -> bool:
    left_ = float(left)
    right_ = float(right)
    return (
        isfinite(left_)
        and isfinite(right_)
        and ((left_ < 0.0 <= right_) or (left_ > 0.0 >= right_))
    )


def _coordinate_target_crossed(
    left: Any,
    right: Any,
    target: float,
    /,
) -> bool:
    left_difference = float(left) - target
    right_difference = float(right) - target
    return (
        isfinite(left_difference)
        and isfinite(right_difference)
        and left_difference != 0.0
        and left_difference * right_difference <= 0.0
    )


def _branch_step_size(
    previous_state: PyTree[Any],
    previous_coordinate: Any,
    state: PyTree[Any],
    coordinate: Any,
    geometry: ContinuationGeometry,
    /,
) -> Array:
    state_difference = jax.tree.map(
        lambda current, previous: current - previous,
        state,
        previous_state,
    )
    coordinate_difference = jnp.asarray(coordinate) - jnp.asarray(previous_coordinate)
    return geometry.augmented_norm(state_difference, coordinate_difference)


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
    corrector_residual_evaluations: Array
    corrector_jvp_evaluations: Array
    corrector_vjp_evaluations: Array
    corrector_jacobian_preparations: Array
    corrector_linear_solves: Array
    corrector_linear_iterations: Array
    corrector_setup_refreshes: Array
    corrector_numeric_refreshes: Array
    tangent_failures: Array
    spectral_evaluations: Array
    monitor_events: Array
    target_corrections: Array
    curvature_rejections: Array

    def __init__(
        self,
        *,
        requested_steps: Any,
        attempted_steps: Any,
        accepted_steps: Any,
        rejected_steps: Any,
        corrector_iterations: Any,
        corrector_residual_evaluations: Any,
        corrector_jvp_evaluations: Any,
        corrector_vjp_evaluations: Any,
        corrector_jacobian_preparations: Any,
        corrector_linear_solves: Any,
        corrector_linear_iterations: Any,
        corrector_setup_refreshes: Any,
        corrector_numeric_refreshes: Any,
        tangent_failures: Any,
        spectral_evaluations: Any,
        monitor_events: Any,
        target_corrections: Any = 0,
        curvature_rejections: Any = 0,
    ):
        values = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (
                requested_steps,
                attempted_steps,
                accepted_steps,
                rejected_steps,
                corrector_iterations,
                corrector_residual_evaluations,
                corrector_jvp_evaluations,
                corrector_vjp_evaluations,
                corrector_jacobian_preparations,
                corrector_linear_solves,
                corrector_linear_iterations,
                corrector_setup_refreshes,
                corrector_numeric_refreshes,
                tangent_failures,
                spectral_evaluations,
                monitor_events,
                target_corrections,
                curvature_rejections,
            )
        )
        (
            self.requested_steps,
            self.attempted_steps,
            self.accepted_steps,
            self.rejected_steps,
            self.corrector_iterations,
            self.corrector_residual_evaluations,
            self.corrector_jvp_evaluations,
            self.corrector_vjp_evaluations,
            self.corrector_jacobian_preparations,
            self.corrector_linear_solves,
            self.corrector_linear_iterations,
            self.corrector_setup_refreshes,
            self.corrector_numeric_refreshes,
            self.tangent_failures,
            self.spectral_evaluations,
            self.monitor_events,
            self.target_corrections,
            self.curvature_rejections,
        ) = values


class ContinuationProvenance(StrictModule):
    """Symbolic and numerical identities for one continuation result."""

    numeric_version: Array
    corrector_linear_numeric_version: Array
    corrector_provenance: NonlinearProvenance
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    corrector_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    public_state_space_id: str = eqx.field(static=True)
    public_residual_space_id: str = eqx.field(static=True)
    execution_state_space_id: str = eqx.field(static=True)
    execution_residual_space_id: str = eqx.field(static=True)
    analyzer_id: str = eqx.field(static=True)
    monitor_ids: tuple[str, ...] = eqx.field(static=True)
    linear_reuse_mode: str = eqx.field(static=True)
    corrector_linear_plan_id: str = eqx.field(static=True)
    corrector_preconditioner_plan_id: str = eqx.field(static=True)
    terminal_coordinate: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        numeric_version: Any,
        corrector_provenance: NonlinearProvenance,
        problem_id: str,
        method_id: str,
        corrector_id: str,
        plan_id: str,
        prepared_id: str,
        branch_id: str,
        geometry_id: str,
        representation_id: str,
        public_state_space_id: str,
        public_residual_space_id: str,
        execution_state_space_id: str,
        execution_residual_space_id: str,
        analyzer_id: str = "",
        monitor_ids: Sequence[str] = (),
        linear_reuse_mode: str = "none",
        corrector_linear_plan_id: str = "",
        corrector_linear_numeric_version: Any = 0,
        corrector_preconditioner_plan_id: str = "",
        terminal_coordinate: float | None = None,
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
        if not isinstance(corrector_provenance, NonlinearProvenance):
            raise TypeError("corrector_provenance must be a NonlinearProvenance.")
        self.corrector_provenance = corrector_provenance
        geometry_identifiers = tuple(
            str(value)
            for value in (
                geometry_id,
                representation_id,
                public_state_space_id,
                public_residual_space_id,
                execution_state_space_id,
                execution_residual_space_id,
            )
        )
        if any(not value for value in geometry_identifiers):
            raise ValueError("Continuation geometry identities must be non-empty.")
        (
            self.geometry_id,
            self.representation_id,
            self.public_state_space_id,
            self.public_residual_space_id,
            self.execution_state_space_id,
            self.execution_residual_space_id,
        ) = geometry_identifiers
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
        self.terminal_coordinate = (
            None if terminal_coordinate is None else float(terminal_coordinate)
        )
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
    """Immutable branch output with every attempt and final committed checkpoint."""

    branch: ContinuationBranch
    status: Array
    diagnostics: ContinuationDiagnostics
    provenance: ContinuationProvenance
    steps: tuple[ContinuationStepResult, ...]
    accepted_state: ContinuationAcceptedState | None
    checkpoint: ContinuationCheckpoint | None

    def __init__(
        self,
        *,
        branch: ContinuationBranch,
        status: Any,
        diagnostics: ContinuationDiagnostics,
        provenance: ContinuationProvenance,
        steps: Sequence[ContinuationStepResult],
        accepted_state: ContinuationAcceptedState | None,
        checkpoint: ContinuationCheckpoint | None,
    ):
        if not isinstance(branch, ContinuationBranch):
            raise TypeError("branch must be a ContinuationBranch.")
        if not isinstance(diagnostics, ContinuationDiagnostics):
            raise TypeError("diagnostics must be ContinuationDiagnostics.")
        if not isinstance(provenance, ContinuationProvenance):
            raise TypeError("provenance must be ContinuationProvenance.")
        steps_ = tuple(steps)
        if any(not isinstance(step, ContinuationStepResult) for step in steps_):
            raise TypeError("steps must contain ContinuationStepResult values.")
        if accepted_state is not None and not isinstance(
            accepted_state, ContinuationAcceptedState
        ):
            raise TypeError("accepted_state must be ContinuationAcceptedState or None.")
        if checkpoint is not None and not isinstance(checkpoint, ContinuationCheckpoint):
            raise TypeError("checkpoint must be a ContinuationCheckpoint or None.")
        if (accepted_state is None) != (checkpoint is None):
            raise ValueError("A final accepted state and checkpoint must exist together.")
        if accepted_state is not None and (
            checkpoint.application_state_id != accepted_state.application_state_id
            or checkpoint.candidate.candidate_id != accepted_state.candidate.candidate_id
        ):
            raise ValueError(
                "Continuation checkpoint does not represent the accepted state."
            )
        self.branch = branch
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.steps = steps_
        self.accepted_state = accepted_state
        self.checkpoint = checkpoint

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


def _resolve_continuation_adapter(
    value: ContinuationCurveProblem | AbstractContinuationAdapter,
    /,
) -> tuple[ContinuationCurveProblem, AbstractContinuationAdapter]:
    if isinstance(value, AbstractContinuationAdapter):
        problem = value.continuation_problem
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError(
                "Continuation adapter continuation_problem must be a "
                "ContinuationCurveProblem."
            )
        return problem, value
    if isinstance(value, ContinuationCurveProblem):
        return value, CallableContinuationAdapter(value)
    raise TypeError("Expected a ContinuationCurveProblem or AbstractContinuationAdapter.")


class ContinuationPlan(StrictModule):
    """Reusable symbolic continuation and monitoring policy."""

    method: AbstractContinuationMethod
    stability_analyzer: AbstractStabilityAnalyzer | None
    monitors: tuple[AbstractBranchMonitor, ...]
    num_steps: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    terminal_coordinate: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: AbstractContinuationMethod,
        stability_analyzer: AbstractStabilityAnalyzer | None,
        monitors: Sequence[AbstractBranchMonitor],
        num_steps: int,
        problem_id: str,
        adapter_id: str,
        branch_id: str,
        plan_id: str,
        terminal_coordinate: float | None = None,
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
        identifiers = tuple(
            str(value) for value in (problem_id, adapter_id, branch_id, plan_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("Problem, adapter, branch, and plan IDs must be non-empty.")
        target = None if terminal_coordinate is None else float(terminal_coordinate)
        if target is not None and not isfinite(target):
            raise ValueError("terminal_coordinate must be finite or None.")
        self.method = method
        self.stability_analyzer = stability_analyzer
        self.monitors = monitors_
        self.num_steps = steps
        self.terminal_coordinate = target
        self.problem_id, self.adapter_id, self.branch_id, self.plan_id = identifiers


class PreparedContinuation(StrictModule):
    """Numerical continuation seed bound to one reusable symbolic plan."""

    problem: ContinuationCurveProblem
    adapter: AbstractContinuationAdapter
    plan: ContinuationPlan
    geometry: ContinuationGeometry
    initial_state: PyTree[Array]
    initial_coordinate: Array
    initial_tangent: tuple[PyTree[Array], Array] | None
    args: Any
    application_state: Any
    replay_evidence: ContinuationReplayEvidence | None
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)
    decision_history: tuple[str, ...] = eqx.field(static=True)
    attempt_history: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        problem: ContinuationCurveProblem | AbstractContinuationAdapter,
        plan: ContinuationPlan,
        initial_state: PyTree[Any],
        initial_coordinate: Any,
        /,
        *,
        initial_tangent: tuple[PyTree[Any], Any] | None = None,
        args: Any = None,
        application_state: Any = None,
        replay_evidence: ContinuationReplayEvidence | None = None,
        decision_history: Sequence[str] = (),
        attempt_history: Sequence[str] = (),
        numeric_version: Any = 0,
        prepared_id: str,
    ):
        problem_, adapter = _resolve_continuation_adapter(problem)
        if not isinstance(plan, ContinuationPlan):
            raise TypeError("plan must be a ContinuationPlan.")
        if plan.problem_id != problem_.problem_id:
            raise ValueError("Continuation plan and problem IDs must match.")
        if plan.adapter_id != adapter.adapter_id:
            raise ValueError("Continuation plan and adapter IDs must match.")
        public_state = _validate_inexact_tree(initial_state, name="initial state")
        coordinate = _real_scalar(initial_coordinate, name="initial coordinate")
        if not bool(problem_.contains_coordinate(coordinate)):
            raise ValueError("initial_coordinate lies outside the continuation interval.")
        public_residual = problem_.residual(public_state, coordinate, args)
        declared_state_space, declared_residual_space = problem_.declared_spaces()
        geometry = ContinuationGeometry.resolve(
            public_state,
            public_residual,
            state_space=declared_state_space,
            residual_space=declared_residual_space,
            representation=problem_.representation_policy(),
            coordinate_scale=plan.method.coordinate_scale,
        )
        state = geometry.state_to_execution(public_state)
        coordinate = jnp.asarray(coordinate, dtype=geometry.coordinate_dtype)
        if (
            plan.terminal_coordinate is not None
            and isinstance(plan.method, NaturalParameterContinuation)
            and plan.method.direction * (plan.terminal_coordinate - float(coordinate))
            < 0.0
        ):
            raise ValueError(
                "terminal_coordinate lies opposite the natural continuation direction."
            )
        tangent = _normalize_initial_tangent(
            initial_tangent,
            public_state,
            geometry,
        )
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version, version < 0, "numeric_version must be non-negative."
        )
        identifier = str(prepared_id)
        if not identifier:
            raise ValueError("prepared_id must be non-empty.")
        if replay_evidence is not None and not isinstance(
            replay_evidence, ContinuationReplayEvidence
        ):
            raise TypeError(
                "replay_evidence must be a ContinuationReplayEvidence or None."
            )
        history = tuple(str(value) for value in decision_history)
        attempts = tuple(str(value) for value in attempt_history)
        if any(not value for value in history + attempts):
            raise ValueError("Continuation decision identities must be non-empty.")
        if any(value not in attempts for value in history):
            raise ValueError("Accepted decisions must belong to the attempt history.")
        if isinstance(adapter, CallableContinuationAdapter):
            if application_state is not None and not adapter.supports_opaque_state:
                raise ValueError(
                    "Opaque application state requires a complete continuation "
                    "transaction callback bundle."
                )
        application_id = adapter.application_state_identity(application_state, args)
        if not application_id:
            raise ValueError("Prepared application state identity must be non-empty.")
        if replay_evidence is not None and not bool(replay_evidence.matches):
            raise ValueError("Continuation checkpoint replay evidence did not match.")
        self.problem = problem_
        self.adapter = adapter
        self.plan = plan
        self.geometry = geometry
        self.initial_state = state
        self.initial_coordinate = coordinate
        self.initial_tangent = tangent
        self.args = args
        self.application_state = application_state
        self.replay_evidence = replay_evidence
        self.numeric_version = version
        self.prepared_id = identifier
        self.decision_history = history
        self.attempt_history = attempts


def _normalize_initial_tangent(
    tangent: tuple[PyTree[Any], Any] | None,
    public_state: PyTree[Any],
    geometry: ContinuationGeometry,
    /,
) -> tuple[PyTree[Array], Array] | None:
    if tangent is None:
        return None
    if not isinstance(tangent, tuple) or len(tangent) != 2:
        raise TypeError(
            "initial_tangent must be a (state_tangent, coordinate_tangent) tuple."
        )
    state_tangent = geometry.state_tangent_to_execution(
        public_state,
        tangent[0],
    )
    coordinate_tangent = _real_scalar(
        tangent[1],
        name="initial coordinate tangent",
    ).astype(geometry.coordinate_dtype)
    norm = geometry.augmented_norm(state_tangent, coordinate_tangent)
    if not bool(jnp.isfinite(norm) & (norm > 0.0)):
        raise ValueError("initial_tangent must have finite nonzero norm.")
    return (
        jax.tree.map(lambda value: value / norm, state_tangent),
        coordinate_tangent / norm,
    )


def plan_continuation(
    problem: ContinuationCurveProblem | AbstractContinuationAdapter,
    /,
    *,
    num_steps: int,
    method: AbstractContinuationMethod | None = None,
    branch_id: str = "branch-0",
    stability_analyzer: AbstractStabilityAnalyzer | None = None,
    monitors: Sequence[AbstractBranchMonitor] = (),
    terminal_coordinate: float | None = None,
    plan_id: str | None = None,
) -> ContinuationPlan:
    """Create reusable symbolic continuation and monitoring policy."""
    problem_, adapter = _resolve_continuation_adapter(problem)
    method_ = PseudoArclengthContinuation() if method is None else method
    if not isinstance(method_, AbstractContinuationMethod):
        raise TypeError("method must be an AbstractContinuationMethod or None.")
    target = None if terminal_coordinate is None else float(terminal_coordinate)
    if target is not None:
        if not isfinite(target):
            raise ValueError("terminal_coordinate must be finite or None.")
        if not problem_.coordinate_lower <= target <= problem_.coordinate_upper:
            raise ValueError(
                "terminal_coordinate lies outside the continuation interval."
            )
    identifier = (
        f"{problem_.problem_id}/{method_.method_id}/continuation-plan"
        if plan_id is None
        else str(plan_id)
    )
    return ContinuationPlan(
        method=method_,
        stability_analyzer=stability_analyzer,
        monitors=monitors,
        num_steps=num_steps,
        problem_id=problem_.problem_id,
        adapter_id=adapter.adapter_id,
        branch_id=branch_id,
        terminal_coordinate=target,
        plan_id=identifier,
    )


def prepare_continuation(
    problem: ContinuationCurveProblem | AbstractContinuationAdapter,
    initial_state: PyTree[Any],
    initial_coordinate: Any,
    plan: ContinuationPlan,
    /,
    *,
    initial_tangent: tuple[PyTree[Any], Any] | None = None,
    args: Any = None,
    application_state: Any = None,
) -> PreparedContinuation:
    """Bind a numerical seed to a reusable continuation plan."""
    return PreparedContinuation(
        problem,
        plan,
        initial_state,
        initial_coordinate,
        initial_tangent=initial_tangent,
        args=args,
        application_state=application_state,
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
    refreshed = PreparedContinuation(
        prepared.adapter,
        prepared.plan,
        initial_state,
        initial_coordinate,
        initial_tangent=initial_tangent,
        args=args,
        application_state=prepared.application_state,
        replay_evidence=prepared.replay_evidence,
        decision_history=prepared.decision_history,
        attempt_history=prepared.attempt_history,
        numeric_version=prepared.numeric_version + 1,
        prepared_id=prepared.prepared_id,
    )
    if refreshed.geometry.geometry_id != prepared.geometry.geometry_id:
        raise ValueError("Continuation refresh changed the bound geometry.")
    return refreshed


def _arclength_residual(
    variables: tuple[PyTree[Any], Array],
    payload: tuple[
        ContinuationCurveProblem,
        ContinuationGeometry,
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
        geometry,
        predicted_state,
        predicted_coordinate,
        state_tangent,
        coordinate_tangent,
        args,
    ) = payload
    candidate_state, candidate_coordinate = variables
    equation = _execution_residual(
        problem,
        geometry,
        candidate_state,
        candidate_coordinate,
        args,
    )
    state_displacement = jax.tree.map(
        lambda candidate, predicted: candidate - predicted,
        candidate_state,
        predicted_state,
    )
    arclength = geometry.augmented_inner(
        state_tangent,
        coordinate_tangent,
        state_displacement,
        candidate_coordinate - predicted_coordinate,
    )
    return equation, arclength


def _correct_arclength(
    problem: ContinuationCurveProblem,
    geometry: ContinuationGeometry,
    predicted_state: PyTree[Any],
    predicted_coordinate: Array,
    state_tangent: PyTree[Any],
    coordinate_tangent: Array,
    method: PseudoArclengthContinuation,
    prepared_corrector: PreparedNonlinearSolve | None,
    args: Any,
    /,
):
    payload = (
        problem,
        geometry,
        predicted_state,
        predicted_coordinate,
        state_tangent,
        coordinate_tangent,
        args,
    )
    return _run_corrector(
        lambda variables: _arclength_residual(variables, payload),
        (predicted_state, predicted_coordinate),
        method,
        prepared_corrector,
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


def _continuation_candidate(
    prepared: PreparedContinuation,
    state: PyTree[Any],
    coordinate: Any,
    tangent_state: PyTree[Any],
    tangent_coordinate: Any,
    /,
    *,
    residual_norm: Any,
    step_size: Any,
    tangent_residual_norm: Any,
    tangent_alignment: Any,
    corrector_iterations: Any,
    corrector_status: Any,
    tangent_status: Any,
    numerical_accepted: Any,
    point_id: str,
    parent_point_id: str,
    attempt_index: int,
    retry_index: int,
) -> ContinuationCandidate:
    geometry = prepared.geometry
    public_state = geometry.state_from_execution(state)
    public_tangent = geometry.state_tangent_from_execution(state, tangent_state)
    parameters, tangent_parameters = prepared.problem.parameters_jvp(
        coordinate,
        tangent_coordinate,
        prepared.args,
    )
    realization = ParameterRealization(
        parameters,
        coordinate,
        problem_id=prepared.problem.problem_id,
    )
    return ContinuationCandidate(
        state=public_state,
        coordinate=coordinate,
        tangent_state=public_tangent,
        tangent_coordinate=tangent_coordinate,
        tangent_parameters=tangent_parameters,
        residual_norm=residual_norm,
        step_size=step_size,
        tangent_residual_norm=tangent_residual_norm,
        tangent_alignment=tangent_alignment,
        corrector_iterations=corrector_iterations,
        corrector_status=corrector_status,
        tangent_status=tangent_status,
        realization=realization,
        numerical_accepted=numerical_accepted,
        point_id=point_id,
        parent_point_id=parent_point_id,
        attempt_index=attempt_index,
        retry_index=retry_index,
    )


def _decide_continuation_candidate(
    prepared: PreparedContinuation,
    source: ContinuationAcceptedState | None,
    application_state: Any,
    transaction: Any,
    source_application_state_id: str,
    candidate: ContinuationCandidate,
    /,
    *,
    message: str,
) -> tuple[ContinuationStepResult, ContinuationAcceptedState | None, Any]:
    adapter = prepared.adapter
    args = prepared.args
    source_id = str(source_application_state_id)
    if source is not None and (
        source.application_state_id != source_id
        or source.application_state is not application_state
    ):
        raise ValueError(
            "Continuation application state is not the last committed accepted state."
        )
    transaction_id = adapter.application_state_identity(transaction, args)
    if adapter.application_state_identity(application_state, args) != source_id:
        raise ValueError("Frozen candidate source identity changed before evaluation.")
    if bool(candidate.numerical_accepted):
        transfer = adapter.evaluate_candidate(transaction, source, candidate, args)
    else:
        transfer = ParameterTransferEvidence.not_evaluated(
            source,
            candidate,
            message=message,
        )
    expected_source_realization = (
        "" if source is None else source.realization.realization_id
    )
    if (
        transfer.source_realization_id != expected_source_realization
        or transfer.target_realization_id != candidate.realization.realization_id
        or transfer.parameter_paths != candidate.realization.parameter_paths
    ):
        raise ValueError(
            "Candidate transfer evidence does not match the committed realization route."
        )
    if adapter.application_state_identity(application_state, args) != source_id:
        raise ValueError("Candidate evaluation mutated the committed application state.")
    accepted = bool(candidate.numerical_accepted) and bool(transfer.accepted)
    if accepted:
        restored_state = adapter.commit_candidate(
            transaction,
            source,
            candidate,
            transfer,
            args,
        )
    else:
        restored_state = adapter.rollback_candidate(
            transaction,
            source,
            candidate,
            transfer,
            args,
        )
    if adapter.application_state_identity(application_state, args) != source_id:
        raise ValueError(
            "A continuation commit or rollback mutated the prior accepted state."
        )
    restored_id = adapter.application_state_identity(restored_state, args)
    decision_id = continuation_step_decision_id(
        candidate,
        transfer,
        transaction_id=transaction_id,
        source_application_state_id=source_id,
        restored_application_state_id=restored_id,
        numerical_accepted=bool(candidate.numerical_accepted),
        accepted=accepted,
        committed=accepted,
        rolled_back=not accepted,
        message=message,
    )
    accepted_state = (
        ContinuationAcceptedState(
            candidate,
            restored_state,
            application_state_id=restored_id,
            decision_id=decision_id,
            accepted_index=(
                len(prepared.decision_history)
                if source is None
                else source.accepted_index + 1
            ),
        )
        if accepted
        else None
    )
    step = ContinuationStepResult(
        candidate,
        transfer,
        accepted_state=accepted_state,
        numerical_accepted=candidate.numerical_accepted,
        accepted=accepted,
        committed=accepted,
        rolled_back=not accepted,
        transaction_id=transaction_id,
        source_application_state_id=source_id,
        restored_application_state_id=restored_id,
        message=message,
    )
    return step, accepted_state, restored_state


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
    corrector_residual_evaluations: int,
    corrector_jvp_evaluations: int,
    corrector_vjp_evaluations: int,
    corrector_jacobian_preparations: int,
    corrector_linear_solves: int,
    corrector_linear_iterations: int,
    corrector_setup_refreshes: int,
    corrector_numeric_refreshes: int,
    tangent_failures: int,
    spectral_evaluations: int,
    monitor_events: int,
    target_corrections: int,
    curvature_rejections: int,
    corrector_prepared_linear: PreparedNonlinearSolve | None,
    steps: Sequence[ContinuationStepResult],
    accepted_state: ContinuationAcceptedState | None,
    corrector_provenance: NonlinearProvenance,
) -> ContinuationResult:
    plan = prepared.plan
    if corrector_prepared_linear is None:
        linear_reuse_mode = "none"
        corrector_linear_plan_id = ""
        corrector_linear_numeric_version = jnp.asarray(0, dtype=jnp.int32)
        corrector_preconditioner_plan_id = ""
    else:
        linear_reuse_mode = "prepared-newton"
        corrector_linear_plan_id = corrector_prepared_linear.linear_plan_id
        corrector_linear_numeric_version = (
            corrector_prepared_linear.linear_refresh_state.numeric_version
        )
        preconditioner_plan = corrector_prepared_linear.linear_plan.preconditioner_plan
        corrector_preconditioner_plan_id = (
            "" if preconditioner_plan is None else preconditioner_plan.plan_id
        )
    checkpoint: ContinuationCheckpoint | None = None
    if accepted_state is not None:
        application_id = prepared.adapter.application_state_identity(
            accepted_state.application_state,
            prepared.args,
        )
        if application_id != accepted_state.application_state_id:
            raise ValueError("Final accepted application state identity changed.")
        application_data = prepared.adapter.checkpoint_application_state(
            accepted_state.application_state,
            prepared.args,
        )
        if (
            prepared.adapter.application_state_identity(
                accepted_state.application_state,
                prepared.args,
            )
            != application_id
        ):
            raise ValueError("Checkpointing mutated the accepted application state.")
        checkpoint = continuation_checkpoint(
            accepted_state,
            steps,
            application_data,
            problem_id=prepared.problem.problem_id,
            adapter_id=prepared.adapter.adapter_id,
            plan_id=plan.plan_id,
            prepared_id=prepared.prepared_id,
            branch_id=plan.branch_id,
            prior_accepted_decision_ids=prepared.decision_history,
            prior_attempt_decision_ids=prepared.attempt_history,
        )
    branch = ContinuationBranch(
        points,
        events,
        status,
        brackets=brackets,
        geometry=prepared.geometry,
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
            corrector_residual_evaluations=corrector_residual_evaluations,
            corrector_jvp_evaluations=corrector_jvp_evaluations,
            corrector_vjp_evaluations=corrector_vjp_evaluations,
            corrector_jacobian_preparations=corrector_jacobian_preparations,
            corrector_linear_solves=corrector_linear_solves,
            corrector_linear_iterations=corrector_linear_iterations,
            corrector_setup_refreshes=corrector_setup_refreshes,
            corrector_numeric_refreshes=corrector_numeric_refreshes,
            tangent_failures=tangent_failures,
            spectral_evaluations=spectral_evaluations,
            monitor_events=monitor_events,
            target_corrections=target_corrections,
            curvature_rejections=curvature_rejections,
        ),
        provenance=ContinuationProvenance(
            numeric_version=prepared.numeric_version,
            problem_id=prepared.problem.problem_id,
            method_id=plan.method.method_id,
            corrector_id=plan.method.corrector_id,
            plan_id=plan.plan_id,
            prepared_id=prepared.prepared_id,
            branch_id=plan.branch_id,
            corrector_provenance=corrector_provenance,
            geometry_id=prepared.geometry.geometry_id,
            representation_id=prepared.geometry.representation.policy_id,
            public_state_space_id=prepared.geometry.public_state_space.space_id,
            public_residual_space_id=prepared.geometry.public_residual_space.space_id,
            execution_state_space_id=prepared.geometry.execution_state_space.space_id,
            execution_residual_space_id=(
                prepared.geometry.execution_residual_space.space_id
            ),
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
            terminal_coordinate=plan.terminal_coordinate,
        ),
        steps=steps,
        accepted_state=accepted_state,
        checkpoint=checkpoint,
    )


def run_continuation(prepared: PreparedContinuation, /) -> ContinuationResult:
    """Run one prepared natural or pseudo-arclength continuation."""
    if not isinstance(prepared, PreparedContinuation):
        raise TypeError("prepared must be a PreparedContinuation.")
    problem = prepared.problem
    plan = prepared.plan
    method = plan.method
    geometry = prepared.geometry
    state = prepared.initial_state
    coordinate = prepared.initial_coordinate
    args = prepared.args
    events: list[ContinuationEvent] = []
    brackets: list[EventBracket] = []
    points: list[BranchPoint] = []
    steps: list[ContinuationStepResult] = []
    application_state = prepared.application_state
    accepted_application_state: ContinuationAcceptedState | None = None
    attempted_steps = 0
    accepted_steps = 0
    rejected_steps = 0
    tangent_failures = 0
    spectral_evaluations = 0
    monitor_events = 0
    corrector_residual_evaluations = 0
    corrector_jvp_evaluations = 0
    corrector_vjp_evaluations = 0
    corrector_jacobian_preparations = 0
    corrector_linear_solves = 0
    corrector_linear_iterations = 0
    corrector_setup_refreshes = 0
    corrector_numeric_refreshes = 0
    target_corrections = 0
    curvature_rejections = 0
    corrector_prepared_linear: PreparedNonlinearSolve | None = None

    initial_source_id = prepared.adapter.application_state_identity(
        application_state,
        args,
    )
    initial_transaction = prepared.adapter.freeze_application_state(
        application_state,
        args,
    )
    if (
        prepared.adapter.application_state_identity(application_state, args)
        != initial_source_id
    ):
        raise ValueError("Freezing the initial candidate mutated application state.")
    initial_result, initial_prepared_linear = _correct_initial_point(
        problem,
        geometry,
        state,
        coordinate,
        method,
        None,
        args,
    )
    if isinstance(method, NaturalParameterContinuation):
        corrector_prepared_linear = initial_prepared_linear
    (
        corrector_iterations,
        initial_residual_evaluations,
        initial_jvp_evaluations,
        initial_vjp_evaluations,
        initial_jacobian_preparations,
        initial_linear_solves,
        initial_linear_iterations,
        initial_setup_refreshes,
        initial_numeric_refreshes,
    ) = _corrector_work(initial_result)
    last_corrector_provenance = initial_result.provenance
    corrector_residual_evaluations += initial_residual_evaluations
    corrector_jvp_evaluations += initial_jvp_evaluations
    corrector_vjp_evaluations += initial_vjp_evaluations
    corrector_jacobian_preparations += initial_jacobian_preparations
    corrector_linear_solves += initial_linear_solves
    corrector_linear_iterations += initial_linear_iterations
    corrector_setup_refreshes += initial_setup_refreshes
    corrector_numeric_refreshes += initial_numeric_refreshes
    state = initial_result.state
    residual_norm = geometry.residual_norm(initial_result.residual)
    initial_success = _corrector_success(initial_result, residual_norm, method)
    tangent_attempted = initial_success
    if not initial_success:
        state_tangent = jax.tree.map(jnp.zeros_like, state)
        coordinate_tangent = jnp.asarray(float(method.direction), dtype=coordinate.dtype)
        tangent_status = jnp.asarray(
            LinearSolveStatus.CAPABILITY_REJECTED, dtype=jnp.int32
        )
        tangent_usable = False
        tangent_residual_norm = jnp.asarray(jnp.inf, dtype=coordinate.dtype)
        tangent_alignment = jnp.asarray(jnp.nan, dtype=coordinate.dtype)
    elif prepared.initial_tangent is not None:
        state_tangent, coordinate_tangent = prepared.initial_tangent
        tangent_status = jnp.asarray(LinearSolveStatus.SUCCESS, dtype=jnp.int32)
        tangent_usable = True
        tangent_residual_norm = _tangent_residual_norm(
            problem,
            geometry,
            state,
            coordinate,
            state_tangent,
            coordinate_tangent,
            args,
        )
        tangent_alignment = jnp.asarray(1.0, dtype=coordinate.dtype)
    elif isinstance(method, PseudoArclengthContinuation):
        (
            state_tangent,
            coordinate_tangent,
            tangent_status,
            tangent_usable,
            tangent_residual_norm,
            tangent_alignment,
        ) = _initial_tangent(problem, geometry, state, coordinate, method, args)
    else:
        state_tangent = jax.tree.map(jnp.zeros_like, state)
        coordinate_tangent = jnp.asarray(float(method.direction), dtype=coordinate.dtype)
        tangent_status = jnp.asarray(LinearSolveStatus.SUCCESS, dtype=jnp.int32)
        tangent_usable = True
        tangent_residual_norm = jnp.asarray(jnp.nan, dtype=coordinate.dtype)
        tangent_alignment = jnp.asarray(1.0, dtype=coordinate.dtype)
    if tangent_attempted and int(tangent_status) != int(LinearSolveStatus.SUCCESS):
        tangent_failures += 1
    public_state = geometry.state_from_execution(state)
    public_state_tangent = geometry.state_tangent_from_execution(
        state,
        state_tangent,
    )
    initial_stability = (
        plan.stability_analyzer.analyze(
            problem,
            public_state,
            coordinate,
            args,
            geometry=geometry,
        )
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
        state=public_state,
        coordinate=coordinate,
        parameters=physical_parameters,
        tangent_state=public_state_tangent,
        tangent_coordinate=coordinate_tangent,
        tangent_parameters=physical_parameter_tangent,
        residual_norm=residual_norm,
        step_size=0.0,
        tangent_residual_norm=tangent_residual_norm,
        tangent_alignment=tangent_alignment,
        corrector_iterations=initial_result.diagnostics.iterations,
        corrector_retries=0,
        status=initial_result.status,
        tangent_status=tangent_status,
        fold_candidate=False,
        point_id=f"{plan.branch_id}/0",
        stability=initial_stability,
    )
    initial_numerical_accepted = initial_success and tangent_usable
    initial_candidate = _continuation_candidate(
        prepared,
        state,
        coordinate,
        state_tangent,
        coordinate_tangent,
        residual_norm=residual_norm,
        step_size=0.0,
        tangent_residual_norm=tangent_residual_norm,
        tangent_alignment=tangent_alignment,
        corrector_iterations=initial_result.diagnostics.iterations,
        corrector_status=initial_result.status,
        tangent_status=tangent_status,
        numerical_accepted=initial_numerical_accepted,
        point_id=initial_point.point_id,
        parent_point_id="",
        attempt_index=len(prepared.attempt_history),
        retry_index=0,
    )
    initial_message = (
        "initial candidate accepted"
        if initial_numerical_accepted
        else (
            "initial tangent rejected"
            if initial_success
            else "initial corrector rejected"
        )
    )
    initial_step, initial_accepted_state, application_state = (
        _decide_continuation_candidate(
            prepared,
            None,
            application_state,
            initial_transaction,
            initial_source_id,
            initial_candidate,
            message=initial_message,
        )
    )
    steps.append(initial_step)
    if initial_accepted_state is not None:
        accepted_application_state = initial_accepted_state
    points.append(initial_point)
    _record_stability_events(events, brackets, None, initial_point)
    if tangent_attempted and not tangent_usable:
        events.append(
            ContinuationEvent(
                "tangent-retry",
                coordinate,
                indicator=tangent_residual_norm,
                source_status=tangent_status,
                point_id=initial_point.point_id,
                message="Initial tangent solve failed its residual contract.",
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
            corrector_residual_evaluations=corrector_residual_evaluations,
            corrector_jvp_evaluations=corrector_jvp_evaluations,
            corrector_vjp_evaluations=corrector_vjp_evaluations,
            corrector_jacobian_preparations=corrector_jacobian_preparations,
            corrector_linear_solves=corrector_linear_solves,
            corrector_linear_iterations=corrector_linear_iterations,
            corrector_setup_refreshes=corrector_setup_refreshes,
            corrector_numeric_refreshes=corrector_numeric_refreshes,
            tangent_failures=tangent_failures,
            spectral_evaluations=spectral_evaluations,
            monitor_events=monitor_events,
            target_corrections=target_corrections,
            curvature_rejections=curvature_rejections,
            corrector_provenance=last_corrector_provenance,
            corrector_prepared_linear=initial_prepared_linear,
            steps=steps,
            accepted_state=accepted_application_state,
        )
    if tangent_attempted and not tangent_usable:
        return _continuation_result(
            prepared,
            points,
            events,
            brackets,
            ContinuationStatus.TANGENT_FAILED,
            "initial tangent solve failed",
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            corrector_iterations=corrector_iterations,
            corrector_residual_evaluations=corrector_residual_evaluations,
            corrector_jvp_evaluations=corrector_jvp_evaluations,
            corrector_vjp_evaluations=corrector_vjp_evaluations,
            corrector_jacobian_preparations=corrector_jacobian_preparations,
            corrector_linear_solves=corrector_linear_solves,
            corrector_linear_iterations=corrector_linear_iterations,
            corrector_setup_refreshes=corrector_setup_refreshes,
            corrector_numeric_refreshes=corrector_numeric_refreshes,
            tangent_failures=tangent_failures,
            spectral_evaluations=spectral_evaluations,
            monitor_events=monitor_events,
            target_corrections=target_corrections,
            curvature_rejections=curvature_rejections,
            corrector_provenance=last_corrector_provenance,
            corrector_prepared_linear=initial_prepared_linear,
            steps=steps,
            accepted_state=accepted_application_state,
        )
    if not bool(initial_step.accepted):
        return _continuation_result(
            prepared,
            points,
            events,
            brackets,
            ContinuationStatus.APPLICATION_REJECTED,
            "initial application candidate rejected",
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            corrector_iterations=corrector_iterations,
            corrector_residual_evaluations=corrector_residual_evaluations,
            corrector_jvp_evaluations=corrector_jvp_evaluations,
            corrector_vjp_evaluations=corrector_vjp_evaluations,
            corrector_jacobian_preparations=corrector_jacobian_preparations,
            corrector_linear_solves=corrector_linear_solves,
            corrector_linear_iterations=corrector_linear_iterations,
            corrector_setup_refreshes=corrector_setup_refreshes,
            corrector_numeric_refreshes=corrector_numeric_refreshes,
            tangent_failures=tangent_failures,
            spectral_evaluations=spectral_evaluations,
            monitor_events=monitor_events,
            target_corrections=target_corrections,
            curvature_rejections=curvature_rejections,
            corrector_provenance=last_corrector_provenance,
            corrector_prepared_linear=initial_prepared_linear,
            steps=steps,
            accepted_state=accepted_application_state,
        )
    monitor_events += _observe_monitors(
        plan.monitors, problem, None, initial_point, args, events
    )

    step_size = method.initial_step
    status = ContinuationStatus.ITERATING
    termination_reason = "requested points reached"
    terminal_reached = bool(
        plan.terminal_coordinate is not None
        and coordinate == jnp.asarray(plan.terminal_coordinate, dtype=coordinate.dtype)
    )
    if terminal_reached:
        events.append(
            ContinuationEvent(
                "coordinate-target",
                coordinate,
                indicator=coordinate,
                point_id=initial_point.point_id,
                message="The corrected initial point is the terminal coordinate.",
            )
        )
    for point_index in () if terminal_reached else range(1, plan.num_steps + 1):
        accepted = False
        bound_reached = False
        target_failure_seen = False
        target_reached_this_step = False
        tangent_failure_seen = False
        curvature_failure_seen = False
        application_failure_seen = False
        retries = 0
        last_corrector_status = jnp.asarray(-1, dtype=jnp.int32)
        accepted_step_size = jnp.asarray(step_size, dtype=coordinate.dtype)
        while retries <= method.maximum_retries and step_size >= method.minimum_step:
            attempt_decided = False
            direct_target_correction = False
            if isinstance(method, NaturalParameterContinuation):
                predicted_state = state
                proposed_coordinate = coordinate + method.direction * step_size
                if (
                    plan.terminal_coordinate is not None
                    and method.direction
                    * (float(proposed_coordinate) - plan.terminal_coordinate)
                    >= 0.0
                ):
                    predicted_coordinate = jnp.asarray(
                        plan.terminal_coordinate,
                        dtype=coordinate.dtype,
                    )
                    direct_target_correction = True
                else:
                    predicted_coordinate = proposed_coordinate
                if method.predictor == "tangent":
                    (
                        state_parameter_tangent,
                        predictor_status,
                        predictor_residual_norm,
                        predictor_usable,
                    ) = _state_parameter_tangent(
                        problem,
                        geometry,
                        state,
                        coordinate,
                        method,
                        args,
                    )
                    if predictor_usable:
                        predicted_state = _tree_add_scaled(
                            state,
                            state_parameter_tangent,
                            predicted_coordinate - coordinate,
                        )
                    elif method.predictor_failure == "constant":
                        tangent_failures += 1
                        events.append(
                            ContinuationEvent(
                                "predictor-fallback",
                                coordinate,
                                indicator=predictor_residual_norm,
                                source_status=predictor_status,
                                point_id=points[-1].point_id,
                                message=(
                                    "Tangent predictor failed; constant predictor used."
                                ),
                            )
                        )
                    else:
                        tangent_failures += 1
                        status = ContinuationStatus.TANGENT_FAILED
                        termination_reason = "natural tangent predictor failed"
                        events.append(
                            ContinuationEvent(
                                "tangent-retry",
                                coordinate,
                                indicator=predictor_residual_norm,
                                source_status=predictor_status,
                                point_id=points[-1].point_id,
                                message="Natural tangent predictor failed.",
                            )
                        )
                        bound_reached = True
                        break
            else:
                predicted_state = _tree_add_scaled(state, state_tangent, step_size)
                predicted_coordinate = coordinate + step_size * coordinate_tangent
                if (
                    plan.terminal_coordinate is not None
                    and not bool(problem.contains_coordinate(predicted_coordinate))
                    and _coordinate_target_crossed(
                        coordinate,
                        predicted_coordinate,
                        plan.terminal_coordinate,
                    )
                ):
                    denominator = float(predicted_coordinate - coordinate)
                    fraction = (
                        plan.terminal_coordinate - float(coordinate)
                    ) / denominator
                    predicted_state = jax.tree.map(
                        lambda left, right, fraction=fraction: (
                            left + fraction * (right - left)
                        ),
                        state,
                        predicted_state,
                    )
                    predicted_coordinate = jnp.asarray(
                        plan.terminal_coordinate,
                        dtype=coordinate.dtype,
                    )
                    direct_target_correction = True
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
                if plan.terminal_coordinate is None:
                    status = ContinuationStatus.COORDINATE_BOUND_REACHED
                    termination_reason = "coordinate bound reached"
                else:
                    status = ContinuationStatus.TARGET_NOT_REACHED
                    termination_reason = "coordinate bound reached before target"
                bound_reached = True
                break
            attempt_source_id = prepared.adapter.application_state_identity(
                application_state,
                args,
            )
            attempt_transaction = prepared.adapter.freeze_application_state(
                application_state,
                args,
            )
            if (
                prepared.adapter.application_state_identity(application_state, args)
                != attempt_source_id
            ):
                raise ValueError(
                    "Freezing an attempt mutated accepted application state."
                )
            attempted_steps += 1
            if (
                isinstance(method, NaturalParameterContinuation)
                or direct_target_correction
            ):
                target_prepared = (
                    corrector_prepared_linear
                    if isinstance(method, NaturalParameterContinuation)
                    else None
                )
                corrector_result, refreshed_target = _correct_state(
                    problem,
                    geometry,
                    predicted_state,
                    predicted_coordinate,
                    method,
                    target_prepared,
                    args,
                )
                if isinstance(method, NaturalParameterContinuation):
                    corrector_prepared_linear = refreshed_target
                candidate_state = corrector_result.state
                candidate_coordinate = predicted_coordinate
            else:
                (
                    corrector_result,
                    corrector_prepared_linear,
                ) = _correct_arclength(
                    problem,
                    geometry,
                    predicted_state,
                    predicted_coordinate,
                    state_tangent,
                    coordinate_tangent,
                    method,
                    corrector_prepared_linear,
                    args,
                )
                candidate_state, candidate_coordinate = corrector_result.state
            (
                candidate_iterations,
                candidate_residual_evaluations,
                candidate_jvp_evaluations,
                candidate_vjp_evaluations,
                candidate_jacobian_preparations,
                candidate_linear_solves,
                candidate_linear_iterations,
                candidate_setup_refreshes,
                candidate_numeric_refreshes,
            ) = _corrector_work(corrector_result)
            corrector_iterations += candidate_iterations
            corrector_residual_evaluations += candidate_residual_evaluations
            corrector_jvp_evaluations += candidate_jvp_evaluations
            corrector_vjp_evaluations += candidate_vjp_evaluations
            corrector_jacobian_preparations += candidate_jacobian_preparations
            corrector_linear_solves += candidate_linear_solves
            corrector_linear_iterations += candidate_linear_iterations
            corrector_setup_refreshes += candidate_setup_refreshes
            corrector_numeric_refreshes += candidate_numeric_refreshes
            last_corrector_provenance = corrector_result.provenance
            if direct_target_correction:
                target_corrections += 1
            last_corrector_status = corrector_result.status
            equation_residual = _execution_residual(
                problem,
                geometry,
                candidate_state,
                candidate_coordinate,
                args,
            )
            candidate_residual_norm = geometry.residual_norm(equation_residual)
            accepted = bool(
                _corrector_success(
                    corrector_result,
                    candidate_residual_norm,
                    method,
                )
                and problem.contains_coordinate(candidate_coordinate)
            )
            if (
                direct_target_correction
                and plan.terminal_coordinate is not None
                and not accepted
            ):
                target_failure_seen = True
            target_reached_this_step = bool(
                accepted
                and plan.terminal_coordinate is not None
                and candidate_coordinate
                == jnp.asarray(plan.terminal_coordinate, dtype=coordinate.dtype)
            )
            if (
                accepted
                and not target_reached_this_step
                and plan.terminal_coordinate is not None
                and _coordinate_target_crossed(
                    coordinate,
                    candidate_coordinate,
                    plan.terminal_coordinate,
                )
            ):
                denominator = float(candidate_coordinate - coordinate)
                fraction = (plan.terminal_coordinate - float(coordinate)) / denominator
                target_seed = jax.tree.map(
                    lambda left, right, fraction=fraction: (
                        left + fraction * (right - left)
                    ),
                    state,
                    candidate_state,
                )
                target_coordinate = jnp.asarray(
                    plan.terminal_coordinate,
                    dtype=coordinate.dtype,
                )
                target_result, _ = _correct_state(
                    problem,
                    geometry,
                    target_seed,
                    target_coordinate,
                    method,
                    None,
                    args,
                )
                (
                    target_iterations,
                    target_residual_evaluations,
                    target_jvp_evaluations,
                    target_vjp_evaluations,
                    target_jacobian_preparations,
                    target_linear_solves,
                    target_linear_iterations,
                    target_setup_refreshes,
                    target_numeric_refreshes,
                ) = _corrector_work(target_result)
                corrector_iterations += target_iterations
                corrector_residual_evaluations += target_residual_evaluations
                corrector_jvp_evaluations += target_jvp_evaluations
                corrector_vjp_evaluations += target_vjp_evaluations
                corrector_jacobian_preparations += target_jacobian_preparations
                corrector_linear_solves += target_linear_solves
                corrector_linear_iterations += target_linear_iterations
                corrector_setup_refreshes += target_setup_refreshes
                corrector_numeric_refreshes += target_numeric_refreshes
                target_corrections += 1
                last_corrector_provenance = target_result.provenance
                target_residual = _execution_residual(
                    problem,
                    geometry,
                    target_result.state,
                    target_coordinate,
                    args,
                )
                target_residual_norm = geometry.residual_norm(target_residual)
                target_success = _corrector_success(
                    target_result,
                    target_residual_norm,
                    method,
                )
                if target_success:
                    candidate_state = target_result.state
                    candidate_coordinate = target_coordinate
                    candidate_residual_norm = target_residual_norm
                    corrector_result = target_result
                    last_corrector_status = target_result.status
                    target_reached_this_step = True
                else:
                    candidate_state = target_result.state
                    candidate_coordinate = target_coordinate
                    candidate_residual_norm = target_residual_norm
                    corrector_result = target_result
                    accepted = False
                    target_failure_seen = True
                    last_corrector_status = target_result.status
            if accepted:
                old_state_tangent = state_tangent
                old_coordinate_tangent = coordinate_tangent
                if (
                    isinstance(method, PseudoArclengthContinuation)
                    and method.tangent_update == "bordered"
                ):
                    (
                        candidate_state_tangent,
                        candidate_coordinate_tangent,
                        candidate_tangent_status,
                        candidate_tangent_usable,
                        candidate_tangent_residual_norm,
                        candidate_tangent_alignment,
                    ) = _bordered_tangent(
                        problem,
                        geometry,
                        candidate_state,
                        candidate_coordinate,
                        old_state_tangent,
                        old_coordinate_tangent,
                        method,
                        args,
                    )
                else:
                    (
                        candidate_state_tangent,
                        candidate_coordinate_tangent,
                    ) = _normalized_secant(
                        state,
                        coordinate,
                        candidate_state,
                        candidate_coordinate,
                        old_state_tangent,
                        old_coordinate_tangent,
                        geometry,
                    )
                    candidate_tangent_status = jnp.asarray(
                        LinearSolveStatus.SUCCESS,
                        dtype=jnp.int32,
                    )
                    candidate_tangent_residual_norm = _tangent_residual_norm(
                        problem,
                        geometry,
                        candidate_state,
                        candidate_coordinate,
                        candidate_state_tangent,
                        candidate_coordinate_tangent,
                        args,
                    )
                    candidate_tangent_alignment = geometry.augmented_inner(
                        old_state_tangent,
                        old_coordinate_tangent,
                        candidate_state_tangent,
                        candidate_coordinate_tangent,
                    )
                    candidate_tangent_usable = bool(
                        _tree_allfinite(candidate_state_tangent)
                        & jnp.isfinite(candidate_coordinate_tangent)
                        & jnp.isfinite(candidate_tangent_residual_norm)
                        & jnp.isfinite(candidate_tangent_alignment)
                    )
                if not candidate_tangent_usable:
                    rejected_candidate = _continuation_candidate(
                        prepared,
                        candidate_state,
                        candidate_coordinate,
                        candidate_state_tangent,
                        candidate_coordinate_tangent,
                        residual_norm=candidate_residual_norm,
                        step_size=step_size,
                        tangent_residual_norm=candidate_tangent_residual_norm,
                        tangent_alignment=candidate_tangent_alignment,
                        corrector_iterations=corrector_result.diagnostics.iterations,
                        corrector_status=corrector_result.status,
                        tangent_status=candidate_tangent_status,
                        numerical_accepted=False,
                        point_id=f"{plan.branch_id}/{point_index}",
                        parent_point_id=points[-1].point_id,
                        attempt_index=len(prepared.attempt_history) + len(steps),
                        retry_index=retries,
                    )
                    rejected_decision, _, application_state = (
                        _decide_continuation_candidate(
                            prepared,
                            accepted_application_state,
                            application_state,
                            attempt_transaction,
                            attempt_source_id,
                            rejected_candidate,
                            message="candidate tangent rejected",
                        )
                    )
                    steps.append(rejected_decision)
                    attempt_decided = True
                    tangent_failure_seen = True
                    tangent_failures += 1
                    rejected_steps += 1
                    retries += 1
                    step_size = max(
                        method.minimum_step,
                        step_size * method.contraction,
                    )
                    events.append(
                        ContinuationEvent(
                            "tangent-retry",
                            coordinate,
                            indicator=candidate_tangent_residual_norm,
                            source_status=candidate_tangent_status,
                            point_id=points[-1].point_id,
                            message="Candidate tangent failed; continuation step reduced.",
                        )
                    )
                    accepted = False
                    continue
                if (
                    isinstance(method, PseudoArclengthContinuation)
                    and method.minimum_tangent_alignment is not None
                    and float(candidate_tangent_alignment)
                    < method.minimum_tangent_alignment
                ):
                    rejected_candidate = _continuation_candidate(
                        prepared,
                        candidate_state,
                        candidate_coordinate,
                        candidate_state_tangent,
                        candidate_coordinate_tangent,
                        residual_norm=candidate_residual_norm,
                        step_size=step_size,
                        tangent_residual_norm=candidate_tangent_residual_norm,
                        tangent_alignment=candidate_tangent_alignment,
                        corrector_iterations=corrector_result.diagnostics.iterations,
                        corrector_status=corrector_result.status,
                        tangent_status=candidate_tangent_status,
                        numerical_accepted=False,
                        point_id=f"{plan.branch_id}/{point_index}",
                        parent_point_id=points[-1].point_id,
                        attempt_index=len(prepared.attempt_history) + len(steps),
                        retry_index=retries,
                    )
                    rejected_decision, _, application_state = (
                        _decide_continuation_candidate(
                            prepared,
                            accepted_application_state,
                            application_state,
                            attempt_transaction,
                            attempt_source_id,
                            rejected_candidate,
                            message="candidate curvature rejected",
                        )
                    )
                    steps.append(rejected_decision)
                    attempt_decided = True
                    curvature_rejections += 1
                    rejected_steps += 1
                    retries += 1
                    step_size = max(
                        method.minimum_step,
                        step_size * method.contraction,
                    )
                    events.append(
                        ContinuationEvent(
                            "curvature-retry",
                            coordinate,
                            indicator=candidate_tangent_alignment,
                            source_status=candidate_tangent_status,
                            point_id=points[-1].point_id,
                            message=(
                                "Candidate tangent alignment was below the configured "
                                "minimum; continuation step reduced."
                            ),
                        )
                    )
                    accepted = False
                    continue
                accepted_step_size = _branch_step_size(
                    state,
                    coordinate,
                    candidate_state,
                    candidate_coordinate,
                    geometry,
                )
                accepted_candidate = _continuation_candidate(
                    prepared,
                    candidate_state,
                    candidate_coordinate,
                    candidate_state_tangent,
                    candidate_coordinate_tangent,
                    residual_norm=candidate_residual_norm,
                    step_size=accepted_step_size,
                    tangent_residual_norm=candidate_tangent_residual_norm,
                    tangent_alignment=candidate_tangent_alignment,
                    corrector_iterations=corrector_result.diagnostics.iterations,
                    corrector_status=corrector_result.status,
                    tangent_status=candidate_tangent_status,
                    numerical_accepted=True,
                    point_id=f"{plan.branch_id}/{point_index}",
                    parent_point_id=points[-1].point_id,
                    attempt_index=len(prepared.attempt_history) + len(steps),
                    retry_index=retries,
                )
                accepted_decision, committed_state, application_state = (
                    _decide_continuation_candidate(
                        prepared,
                        accepted_application_state,
                        application_state,
                        attempt_transaction,
                        attempt_source_id,
                        accepted_candidate,
                        message="application candidate evaluated",
                    )
                )
                steps.append(accepted_decision)
                attempt_decided = True
                if committed_state is not None:
                    accepted_application_state = committed_state
                    accepted = True
                    break
                application_failure_seen = True
                accepted = False
            if not attempt_decided:
                rejected_candidate = _continuation_candidate(
                    prepared,
                    candidate_state,
                    candidate_coordinate,
                    state_tangent,
                    coordinate_tangent,
                    residual_norm=candidate_residual_norm,
                    step_size=step_size,
                    tangent_residual_norm=tangent_residual_norm,
                    tangent_alignment=tangent_alignment,
                    corrector_iterations=corrector_result.diagnostics.iterations,
                    corrector_status=corrector_result.status,
                    tangent_status=tangent_status,
                    numerical_accepted=False,
                    point_id=f"{plan.branch_id}/{point_index}",
                    parent_point_id=points[-1].point_id,
                    attempt_index=len(prepared.attempt_history) + len(steps),
                    retry_index=retries,
                )
                rejected_decision, _, application_state = _decide_continuation_candidate(
                    prepared,
                    accepted_application_state,
                    application_state,
                    attempt_transaction,
                    attempt_source_id,
                    rejected_candidate,
                    message=(
                        "target corrector rejected"
                        if target_failure_seen
                        else "corrector rejected"
                    ),
                )
                steps.append(rejected_decision)
            rejected_steps += 1
            retries += 1
            step_size = max(method.minimum_step, step_size * method.contraction)
            events.append(
                ContinuationEvent(
                    (
                        "application-retry"
                        if application_failure_seen
                        else (
                            "target-corrector-retry"
                            if target_failure_seen
                            else "corrector-retry"
                        )
                    ),
                    coordinate,
                    indicator=step_size,
                    source_status=last_corrector_status,
                    point_id=points[-1].point_id,
                    message=(
                        "Application candidate rejected; continuation step reduced."
                        if application_failure_seen
                        else (
                            "Target corrector rejected; continuation step reduced."
                            if target_failure_seen
                            else "Corrector rejected; continuation step reduced."
                        )
                    ),
                )
            )
        if bound_reached:
            break
        if not accepted:
            if application_failure_seen:
                status = ContinuationStatus.APPLICATION_REJECTED
                termination_reason = "application candidate recovery exhausted"
            elif curvature_failure_seen:
                status = ContinuationStatus.CURVATURE_LIMIT_REACHED
                termination_reason = "curvature recovery exhausted"
            elif tangent_failure_seen:
                status = ContinuationStatus.TANGENT_FAILED
                termination_reason = "tangent recovery exhausted"
            elif target_failure_seen:
                status = ContinuationStatus.TARGET_CORRECTOR_FAILED
                termination_reason = "target corrector recovery exhausted"
            else:
                status = ContinuationStatus.CORRECTOR_FAILED
                termination_reason = "corrector recovery exhausted"
            events.append(
                ContinuationEvent(
                    (
                        "application-failure"
                        if application_failure_seen
                        else "corrector-failure"
                    ),
                    coordinate,
                    indicator=step_size,
                    source_status=last_corrector_status,
                    point_id=points[-1].point_id,
                    message=(
                        "Application acceptance retry budget reached."
                        if application_failure_seen
                        else "Minimum step or retry budget reached."
                    ),
                )
            )
            break

        previous_point = points[-1]
        old_coordinate_tangent = coordinate_tangent
        state_tangent = candidate_state_tangent
        coordinate_tangent = candidate_coordinate_tangent
        tangent_status = candidate_tangent_status
        tangent_residual_norm = candidate_tangent_residual_norm
        tangent_alignment = candidate_tangent_alignment
        state = candidate_state
        coordinate = candidate_coordinate
        fold_candidate = _sign_crossed(old_coordinate_tangent, coordinate_tangent)
        point_id = f"{plan.branch_id}/{point_index}"
        public_state = geometry.state_from_execution(state)
        public_state_tangent = geometry.state_tangent_from_execution(
            state,
            state_tangent,
        )
        stability = (
            plan.stability_analyzer.analyze(
                problem,
                public_state,
                coordinate,
                args,
                geometry=geometry,
            )
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
            state=public_state,
            coordinate=coordinate,
            parameters=physical_parameters,
            tangent_state=public_state_tangent,
            tangent_coordinate=coordinate_tangent,
            tangent_parameters=physical_parameter_tangent,
            residual_norm=candidate_residual_norm,
            step_size=accepted_step_size,
            tangent_residual_norm=tangent_residual_norm,
            tangent_alignment=tangent_alignment,
            corrector_iterations=corrector_result.diagnostics.iterations,
            corrector_retries=retries,
            status=corrector_result.status,
            tangent_status=tangent_status,
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
            plan.monitors,
            problem,
            previous_point,
            current_point,
            args,
            events,
        )
        points.append(current_point)
        accepted_steps += 1
        if target_reached_this_step:
            events.append(
                ContinuationEvent(
                    "coordinate-target",
                    coordinate,
                    indicator=coordinate,
                    point_id=current_point.point_id,
                    message="The corrected branch reached the terminal coordinate.",
                )
            )
            terminal_reached = True
            status = ContinuationStatus.SUCCESS
            termination_reason = "terminal coordinate reached"
            break
        iterations = int(corrector_result.diagnostics.iterations)
        if retries == 0 and iterations <= method.target_corrector_steps:
            step_size = min(method.maximum_step, step_size * method.growth)
        elif iterations > 2 * method.target_corrector_steps:
            step_size = max(method.minimum_step, step_size * method.contraction)
    else:
        if plan.terminal_coordinate is None or terminal_reached:
            status = ContinuationStatus.SUCCESS
        else:
            status = ContinuationStatus.TARGET_NOT_REACHED
            termination_reason = "requested steps exhausted before target"

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
        steps=steps,
        accepted_state=accepted_application_state,
        corrector_residual_evaluations=corrector_residual_evaluations,
        corrector_jvp_evaluations=corrector_jvp_evaluations,
        corrector_vjp_evaluations=corrector_vjp_evaluations,
        corrector_jacobian_preparations=corrector_jacobian_preparations,
        corrector_linear_solves=corrector_linear_solves,
        corrector_linear_iterations=corrector_linear_iterations,
        corrector_setup_refreshes=corrector_setup_refreshes,
        corrector_numeric_refreshes=corrector_numeric_refreshes,
        tangent_failures=tangent_failures,
        spectral_evaluations=spectral_evaluations,
        monitor_events=monitor_events,
        target_corrections=target_corrections,
        curvature_rejections=curvature_rejections,
        corrector_prepared_linear=(
            initial_prepared_linear
            if corrector_prepared_linear is None
            else corrector_prepared_linear
        ),
        corrector_provenance=last_corrector_provenance,
    )


def continue_branch(
    problem: ContinuationCurveProblem | AbstractContinuationAdapter,
    initial_state: PyTree[Any],
    initial_coordinate: Any,
    /,
    *,
    num_steps: int,
    method: AbstractContinuationMethod | None = None,
    initial_tangent: tuple[PyTree[Any], Any] | None = None,
    branch_id: str = "branch-0",
    args: Any = None,
    application_state: Any = None,
    stability_analyzer: AbstractStabilityAnalyzer | None = None,
    monitors: Sequence[AbstractBranchMonitor] = (),
    terminal_coordinate: float | None = None,
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
        terminal_coordinate=terminal_coordinate,
        plan_id=plan_id,
    )
    prepared = prepare_continuation(
        problem,
        initial_state,
        initial_coordinate,
        plan,
        initial_tangent=initial_tangent,
        args=args,
        application_state=application_state,
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
