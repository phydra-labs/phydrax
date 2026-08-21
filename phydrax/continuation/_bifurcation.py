#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import AbstractAttribute, StrictModule
from .._tree_math import tree_add_scaled, tree_norm
from ..linalg import AbstractVectorSpace
from ..nonlinear import (
    AbstractNonlinearMethod,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._core import ContinuationCurveProblem


BifurcationKind = Literal["fold", "hopf", "branch-point", "pitchfork"]


class ExtendedSystemStatus(IntEnum):
    """Terminal status of a fold or Hopf extended-system correction."""

    CONVERGED_CANDIDATE = 0
    NONLINEAR_SOLVE_FAILED = 1
    BLOCK_RESIDUAL_TOO_LARGE = 2
    PARAMETER_OUT_OF_RANGE = 3
    INVALID_FREQUENCY = 4
    NONFINITE = 5


class BifurcationStatus(IntEnum):
    """Evidence-based status of a mathematical bifurcation claim."""

    CERTIFIED = 0
    CANDIDATE_ONLY = 1
    EXTENDED_SYSTEM_NOT_CONVERGED = 2
    ASSUMPTIONS_UNVERIFIED = 3
    SPECTRAL_SOURCE_FAILED = 4
    INSUFFICIENT_SPECTRAL_EVIDENCE = 5
    NULLSPACE_NOT_SIMPLE = 6
    NULLSPACE_RESIDUAL_TOO_LARGE = 7
    TRANSVERSALITY_FAILED = 8
    NONDEGENERACY_FAILED = 9
    SYMMETRY_EVIDENCE_FAILED = 10
    ILL_CONDITIONED = 11
    NONFINITE_EVIDENCE = 12


class BifurcationTolerances(StrictModule):
    """Numerical thresholds used by explicit bifurcation certificates."""

    residual: float = eqx.field(static=True)
    null_residual: float = eqx.field(static=True)
    normalization: float = eqx.field(static=True)
    spectral_zero: float = eqx.field(static=True)
    spectral_gap: float = eqx.field(static=True)
    transversality: float = eqx.field(static=True)
    branch_projection: float = eqx.field(static=True)
    nondegeneracy: float = eqx.field(static=True)
    symmetry: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual: float = 1e-8,
        null_residual: float = 1e-7,
        normalization: float = 1e-6,
        spectral_zero: float = 1e-7,
        spectral_gap: float = 1e-5,
        transversality: float = 1e-6,
        branch_projection: float = 1e-7,
        nondegeneracy: float = 1e-6,
        symmetry: float = 1e-7,
        maximum_condition: float = 1e8,
    ):
        values = tuple(
            float(value)
            for value in (
                residual,
                null_residual,
                normalization,
                spectral_zero,
                spectral_gap,
                transversality,
                branch_projection,
                nondegeneracy,
                symmetry,
                maximum_condition,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in values[:-1]):
            raise ValueError("Bifurcation tolerances must be finite and non-negative.")
        if not isfinite(values[-1]) or values[-1] < 1.0:
            raise ValueError("maximum_condition must be finite and at least one.")
        (
            self.residual,
            self.null_residual,
            self.normalization,
            self.spectral_zero,
            self.spectral_gap,
            self.transversality,
            self.branch_projection,
            self.nondegeneracy,
            self.symmetry,
            self.maximum_condition,
        ) = values


class FoldAssumptions(StrictModule):
    """Analytical assumptions required for a fold theorem."""

    smoothness_order: int = eqx.field(static=True)
    scalar_parameter_verified: bool = eqx.field(static=True)
    local_fredholm_index_zero_verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        smoothness_order: int,
        scalar_parameter_verified: bool,
        local_fredholm_index_zero_verified: bool,
    ):
        order = int(smoothness_order)
        if order < 0:
            raise ValueError("smoothness_order must be non-negative.")
        self.smoothness_order = order
        self.scalar_parameter_verified = bool(scalar_parameter_verified)
        self.local_fredholm_index_zero_verified = bool(local_fredholm_index_zero_verified)

    @property
    def verified(self) -> bool:
        return (
            self.smoothness_order >= 2
            and self.scalar_parameter_verified
            and self.local_fredholm_index_zero_verified
        )


class BranchPointAssumptions(StrictModule):
    """Crandall--Rabinowitz assumptions not inferable from local numerics."""

    smoothness_order: int = eqx.field(static=True)
    scalar_parameter_verified: bool = eqx.field(static=True)
    reference_branch_verified: bool = eqx.field(static=True)
    local_fredholm_index_zero_verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        smoothness_order: int,
        scalar_parameter_verified: bool,
        reference_branch_verified: bool,
        local_fredholm_index_zero_verified: bool,
    ):
        order = int(smoothness_order)
        if order < 0:
            raise ValueError("smoothness_order must be non-negative.")
        self.smoothness_order = order
        self.scalar_parameter_verified = bool(scalar_parameter_verified)
        self.reference_branch_verified = bool(reference_branch_verified)
        self.local_fredholm_index_zero_verified = bool(local_fredholm_index_zero_verified)

    @property
    def verified(self) -> bool:
        return (
            self.smoothness_order >= 2
            and self.scalar_parameter_verified
            and self.reference_branch_verified
            and self.local_fredholm_index_zero_verified
        )


class HopfAssumptions(StrictModule):
    """Analytical assumptions required for a local Hopf theorem."""

    smoothness_order: int = eqx.field(static=True)
    autonomous_flow_verified: bool = eqx.field(static=True)
    scalar_parameter_verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        smoothness_order: int,
        autonomous_flow_verified: bool,
        scalar_parameter_verified: bool,
    ):
        order = int(smoothness_order)
        if order < 0:
            raise ValueError("smoothness_order must be non-negative.")
        self.smoothness_order = order
        self.autonomous_flow_verified = bool(autonomous_flow_verified)
        self.scalar_parameter_verified = bool(scalar_parameter_verified)

    @property
    def verified(self) -> bool:
        return (
            self.smoothness_order >= 3
            and self.autonomous_flow_verified
            and self.scalar_parameter_verified
        )


class PitchforkAssumptions(StrictModule):
    """Declared symmetry assumptions required for pitchfork certification."""

    smoothness_order: int = eqx.field(static=True)
    symmetry_is_linear: bool = eqx.field(static=True)
    symmetry_is_involutive: bool = eqx.field(static=True)
    equation_equivariance_verified: bool = eqx.field(static=True)
    reference_branch_symmetric: bool = eqx.field(static=True)
    critical_mode_is_odd: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        smoothness_order: int,
        symmetry_is_linear: bool,
        symmetry_is_involutive: bool,
        equation_equivariance_verified: bool,
        reference_branch_symmetric: bool,
        critical_mode_is_odd: bool,
    ):
        order = int(smoothness_order)
        if order < 0:
            raise ValueError("smoothness_order must be non-negative.")
        self.smoothness_order = order
        self.symmetry_is_linear = bool(symmetry_is_linear)
        self.symmetry_is_involutive = bool(symmetry_is_involutive)
        self.equation_equivariance_verified = bool(equation_equivariance_verified)
        self.reference_branch_symmetric = bool(reference_branch_symmetric)
        self.critical_mode_is_odd = bool(critical_mode_is_odd)

    @property
    def verified(self) -> bool:
        return (
            self.smoothness_order >= 3
            and self.symmetry_is_linear
            and self.symmetry_is_involutive
            and self.equation_equivariance_verified
            and self.reference_branch_symmetric
            and self.critical_mode_is_odd
        )


class FoldState(StrictModule):
    """State of the minimally augmented fold system."""

    physical_state: PyTree[Array]
    parameter: Array
    nullvector: PyTree[Array]

    def __init__(
        self,
        physical_state: PyTree[Any],
        parameter: Any,
        nullvector: PyTree[Any],
        /,
    ):
        self.physical_state = physical_state
        self.parameter = jnp.asarray(parameter)
        self.nullvector = nullvector


class FoldResidualBlocks(StrictModule):
    """Physical, kernel, and normalization blocks of a fold system."""

    equilibrium: PyTree[Array]
    kernel: PyTree[Array]
    normalization: Array

    def __init__(
        self,
        *,
        equilibrium: PyTree[Any],
        kernel: PyTree[Any],
        normalization: Any,
    ):
        self.equilibrium = equilibrium
        self.kernel = kernel
        self.normalization = jnp.asarray(normalization)


class HopfState(StrictModule):
    """Real representation of a complex Hopf mode and its frequency."""

    physical_state: PyTree[Array]
    parameter: Array
    mode_real: PyTree[Array]
    mode_imaginary: PyTree[Array]
    frequency: Array

    def __init__(
        self,
        physical_state: PyTree[Any],
        parameter: Any,
        mode_real: PyTree[Any],
        mode_imaginary: PyTree[Any],
        frequency: Any,
        /,
    ):
        self.physical_state = physical_state
        self.parameter = jnp.asarray(parameter)
        self.mode_real = mode_real
        self.mode_imaginary = mode_imaginary
        self.frequency = jnp.asarray(frequency)


class HopfResidualBlocks(StrictModule):
    """Physical, eigenmode, phase, and normalization blocks of a Hopf system."""

    equilibrium: PyTree[Array]
    eigen_real: PyTree[Array]
    eigen_imaginary: PyTree[Array]
    normalization: Array
    phase: Array

    def __init__(
        self,
        *,
        equilibrium: PyTree[Any],
        eigen_real: PyTree[Any],
        eigen_imaginary: PyTree[Any],
        normalization: Any,
        phase: Any,
    ):
        self.equilibrium = equilibrium
        self.eigen_real = eigen_real
        self.eigen_imaginary = eigen_imaginary
        self.normalization = jnp.asarray(normalization)
        self.phase = jnp.asarray(phase)


class ExtendedSystemCertificate(StrictModule):
    """Residual evidence for a converged extended system, not a bifurcation claim."""

    block_norms: Array
    total_residual_norm: Array
    residual_tolerance: Array
    nonlinear_success: Array
    parameter_valid: Array
    frequency_valid: Array
    finite: Array
    status: Array

    def __init__(
        self,
        *,
        block_norms: Any,
        total_residual_norm: Any,
        residual_tolerance: Any,
        nonlinear_success: Any,
        parameter_valid: Any,
        frequency_valid: Any = True,
        finite: Any,
        status: Any,
    ):
        norms = jnp.asarray(block_norms)
        if norms.ndim != 1 or not norms.size:
            raise ValueError("block_norms must be a nonempty rank-one array.")
        self.block_norms = norms
        self.total_residual_norm = jnp.asarray(total_residual_norm)
        self.residual_tolerance = jnp.asarray(residual_tolerance)
        self.nonlinear_success = jnp.asarray(nonlinear_success, dtype=bool)
        self.parameter_valid = jnp.asarray(parameter_valid, dtype=bool)
        self.frequency_valid = jnp.asarray(frequency_valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)

    @property
    def converged(self) -> Array:
        return self.status == int(ExtendedSystemStatus.CONVERGED_CANDIDATE)


class ExtendedSystemProvenance(StrictModule):
    """Problem, extended method, root solver, and derivative identities."""

    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    nonlinear_method_id: str = eqx.field(static=True)
    derivative_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem_id: str,
        method_id: str,
        nonlinear_method_id: str,
        derivative_id: str = "jax-jvp",
    ):
        values = tuple(
            str(value)
            for value in (
                problem_id,
                method_id,
                nonlinear_method_id,
                derivative_id,
            )
        )
        if any(not value for value in values):
            raise ValueError("Extended-system provenance identifiers must be non-empty.")
        (
            self.problem_id,
            self.method_id,
            self.nonlinear_method_id,
            self.derivative_id,
        ) = values


class FoldResult(StrictModule):
    """Fold extended-system candidate and its complete nonlinear evidence."""

    state: FoldState
    residual_blocks: FoldResidualBlocks
    convergence: ExtendedSystemCertificate
    nonlinear_result: NonlinearResult
    provenance: ExtendedSystemProvenance

    def __init__(
        self,
        *,
        state: FoldState,
        residual_blocks: FoldResidualBlocks,
        convergence: ExtendedSystemCertificate,
        nonlinear_result: NonlinearResult,
        provenance: ExtendedSystemProvenance,
    ):
        if not isinstance(state, FoldState):
            raise TypeError("state must be a FoldState.")
        if not isinstance(residual_blocks, FoldResidualBlocks):
            raise TypeError("residual_blocks must be FoldResidualBlocks.")
        if not isinstance(convergence, ExtendedSystemCertificate):
            raise TypeError("convergence must be an ExtendedSystemCertificate.")
        if not isinstance(nonlinear_result, NonlinearResult):
            raise TypeError("nonlinear_result must be a NonlinearResult.")
        if not isinstance(provenance, ExtendedSystemProvenance):
            raise TypeError("provenance must be ExtendedSystemProvenance.")
        self.state = state
        self.residual_blocks = residual_blocks
        self.convergence = convergence
        self.nonlinear_result = nonlinear_result
        self.provenance = provenance

    @property
    def candidate_converged(self) -> Array:
        return self.convergence.converged


class HopfResult(StrictModule):
    """Hopf extended-system candidate and its complete nonlinear evidence."""

    state: HopfState
    residual_blocks: HopfResidualBlocks
    convergence: ExtendedSystemCertificate
    nonlinear_result: NonlinearResult
    provenance: ExtendedSystemProvenance

    def __init__(
        self,
        *,
        state: HopfState,
        residual_blocks: HopfResidualBlocks,
        convergence: ExtendedSystemCertificate,
        nonlinear_result: NonlinearResult,
        provenance: ExtendedSystemProvenance,
    ):
        if not isinstance(state, HopfState):
            raise TypeError("state must be a HopfState.")
        if not isinstance(residual_blocks, HopfResidualBlocks):
            raise TypeError("residual_blocks must be HopfResidualBlocks.")
        if not isinstance(convergence, ExtendedSystemCertificate):
            raise TypeError("convergence must be an ExtendedSystemCertificate.")
        if not isinstance(nonlinear_result, NonlinearResult):
            raise TypeError("nonlinear_result must be a NonlinearResult.")
        if not isinstance(provenance, ExtendedSystemProvenance):
            raise TypeError("provenance must be ExtendedSystemProvenance.")
        self.state = state
        self.residual_blocks = residual_blocks
        self.convergence = convergence
        self.nonlinear_result = nonlinear_result
        self.provenance = provenance

    @property
    def candidate_converged(self) -> Array:
        return self.convergence.converged


def _validate_scalar(value: Any, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != () or not jnp.issubdtype(scalar.dtype, jnp.floating):
        raise TypeError(f"{name} must be one real floating-point scalar array.")
    return scalar


def _tree_linear_combination(
    left_scale: Any,
    left: PyTree[Any],
    right_scale: Any,
    right: PyTree[Any],
    /,
) -> PyTree[Array]:
    return jax.tree.map(
        lambda x, y: left_scale * x + right_scale * y,
        left,
        right,
    )


class FoldProblem(StrictModule):
    """Minimally augmented fold system built over a continuation problem."""

    problem: ContinuationCurveProblem
    state_space: AbstractVectorSpace
    reference_nullvector: PyTree[Array]
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: ContinuationCurveProblem,
        state_space: AbstractVectorSpace,
        reference_nullvector: PyTree[Any],
        /,
        *,
        problem_id: str | None = None,
    ):
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        if not isinstance(state_space, AbstractVectorSpace):
            raise TypeError("state_space must be an AbstractVectorSpace.")
        reference = state_space.validate(reference_nullvector)
        reference_norm = float(
            jnp.sqrt(jnp.real(state_space.inner(reference, reference)))
        )
        if not isfinite(reference_norm) or reference_norm == 0.0:
            raise ValueError("reference_nullvector must have finite nonzero norm.")
        identifier = (
            f"{problem.problem_id}/fold" if problem_id is None else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.problem = problem
        self.state_space = state_space
        self.reference_nullvector = reference
        self.problem_id = identifier

    def residual_blocks(
        self,
        extended_state: FoldState,
        args: Any = None,
        /,
    ) -> FoldResidualBlocks:
        if not isinstance(extended_state, FoldState):
            raise TypeError("extended_state must be a FoldState.")
        state = self.state_space.validate(extended_state.physical_state)
        nullvector = self.state_space.validate(extended_state.nullvector)
        parameter = _validate_scalar(extended_state.parameter, "fold parameter")
        equilibrium = self.state_space.validate(
            self.problem.residual(state, parameter, args)
        )
        kernel = jax.jvp(
            lambda value: self.problem.residual(value, parameter, args),
            (state,),
            (nullvector,),
        )[1]
        kernel = self.state_space.validate(kernel)
        normalization = (
            self.state_space.inner(
                self.reference_nullvector,
                nullvector,
            )
            - 1.0
        )
        return FoldResidualBlocks(
            equilibrium=equilibrium,
            kernel=kernel,
            normalization=normalization,
        )

    def as_nonlinear_problem(self, /) -> NonlinearSystemProblem:
        return NonlinearSystemProblem(
            lambda extended_state, args: self.residual_blocks(extended_state, args),
            problem_id=self.problem_id,
        )


class HopfProblem(StrictModule):
    """Real extended system for a nonzero imaginary eigenvalue pair."""

    problem: ContinuationCurveProblem
    state_space: AbstractVectorSpace
    reference_mode_real: PyTree[Array]
    reference_mode_imaginary: PyTree[Array]
    minimum_frequency: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: ContinuationCurveProblem,
        state_space: AbstractVectorSpace,
        reference_mode_real: PyTree[Any],
        reference_mode_imaginary: PyTree[Any],
        /,
        *,
        minimum_frequency: float = 1e-7,
        problem_id: str | None = None,
    ):
        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        if not isinstance(state_space, AbstractVectorSpace):
            raise TypeError("state_space must be an AbstractVectorSpace.")
        real_mode = state_space.validate(reference_mode_real)
        imaginary_mode = state_space.validate(reference_mode_imaginary)
        norm_squared = jnp.real(
            state_space.inner(real_mode, real_mode)
            + state_space.inner(imaginary_mode, imaginary_mode)
        )
        mode_norm = float(jnp.sqrt(jnp.maximum(norm_squared, 0.0)))
        frequency = float(minimum_frequency)
        if not isfinite(mode_norm) or mode_norm == 0.0:
            raise ValueError("The reference Hopf mode must have finite nonzero norm.")
        if not isfinite(frequency) or frequency <= 0.0:
            raise ValueError("minimum_frequency must be finite and positive.")
        identifier = (
            f"{problem.problem_id}/hopf" if problem_id is None else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.problem = problem
        self.state_space = state_space
        self.reference_mode_real = real_mode
        self.reference_mode_imaginary = imaginary_mode
        self.minimum_frequency = frequency
        self.problem_id = identifier

    def residual_blocks(
        self,
        extended_state: HopfState,
        args: Any = None,
        /,
    ) -> HopfResidualBlocks:
        if not isinstance(extended_state, HopfState):
            raise TypeError("extended_state must be a HopfState.")
        state = self.state_space.validate(extended_state.physical_state)
        mode_real = self.state_space.validate(extended_state.mode_real)
        mode_imaginary = self.state_space.validate(extended_state.mode_imaginary)
        parameter = _validate_scalar(extended_state.parameter, "Hopf parameter")
        frequency = _validate_scalar(extended_state.frequency, "Hopf frequency")
        residual_function = lambda value: self.problem.residual(value, parameter, args)
        equilibrium = self.state_space.validate(residual_function(state))
        action_real = jax.jvp(residual_function, (state,), (mode_real,))[1]
        action_imaginary = jax.jvp(residual_function, (state,), (mode_imaginary,))[1]
        eigen_real = _tree_linear_combination(
            1.0,
            action_real,
            frequency,
            mode_imaginary,
        )
        eigen_imaginary = _tree_linear_combination(
            1.0,
            action_imaginary,
            -frequency,
            mode_real,
        )
        normalization = (
            self.state_space.inner(self.reference_mode_real, mode_real)
            + self.state_space.inner(self.reference_mode_imaginary, mode_imaginary)
            - 1.0
        )
        phase = self.state_space.inner(
            self.reference_mode_real, mode_imaginary
        ) - self.state_space.inner(self.reference_mode_imaginary, mode_real)
        return HopfResidualBlocks(
            equilibrium=equilibrium,
            eigen_real=self.state_space.validate(eigen_real),
            eigen_imaginary=self.state_space.validate(eigen_imaginary),
            normalization=normalization,
            phase=phase,
        )

    def as_nonlinear_problem(self, /) -> NonlinearSystemProblem:
        return NonlinearSystemProblem(
            lambda extended_state, args: self.residual_blocks(extended_state, args),
            problem_id=self.problem_id,
        )


def _fold_block_norms(blocks: FoldResidualBlocks, /) -> Array:
    return jnp.stack(
        (
            tree_norm(blocks.equilibrium),
            tree_norm(blocks.kernel),
            jnp.abs(blocks.normalization),
        )
    )


def _hopf_block_norms(blocks: HopfResidualBlocks, /) -> Array:
    return jnp.stack(
        (
            tree_norm(blocks.equilibrium),
            tree_norm(blocks.eigen_real),
            tree_norm(blocks.eigen_imaginary),
            jnp.abs(blocks.normalization),
            jnp.abs(blocks.phase),
        )
    )


def _extended_certificate(
    block_norms: Array,
    nonlinear_result: NonlinearResult,
    parameter_valid: Any,
    residual_tolerance: float,
    /,
    *,
    frequency_valid: Any = True,
) -> ExtendedSystemCertificate:
    total = jnp.linalg.vector_norm(block_norms)
    finite = jnp.all(jnp.isfinite(block_norms)) & jnp.isfinite(total)
    within_tolerance = jnp.all(block_norms <= residual_tolerance)
    nonlinear_success = nonlinear_result.successful
    parameter_valid_ = jnp.asarray(parameter_valid, dtype=bool)
    frequency_valid_ = jnp.asarray(frequency_valid, dtype=bool)
    status = jnp.where(
        ~finite,
        int(ExtendedSystemStatus.NONFINITE),
        jnp.where(
            ~parameter_valid_,
            int(ExtendedSystemStatus.PARAMETER_OUT_OF_RANGE),
            jnp.where(
                ~frequency_valid_,
                int(ExtendedSystemStatus.INVALID_FREQUENCY),
                jnp.where(
                    ~nonlinear_success,
                    int(ExtendedSystemStatus.NONLINEAR_SOLVE_FAILED),
                    jnp.where(
                        within_tolerance,
                        int(ExtendedSystemStatus.CONVERGED_CANDIDATE),
                        int(ExtendedSystemStatus.BLOCK_RESIDUAL_TOO_LARGE),
                    ),
                ),
            ),
        ),
    )
    return ExtendedSystemCertificate(
        block_norms=block_norms,
        total_residual_norm=total,
        residual_tolerance=residual_tolerance,
        nonlinear_success=nonlinear_success,
        parameter_valid=parameter_valid_,
        frequency_valid=frequency_valid_,
        finite=finite,
        status=status,
    )


class FoldMethod(StrictModule):
    """Fold corrector delegating the augmented root solve to a nonlinear method."""

    root_method: AbstractNonlinearMethod
    termination: NonlinearTermination
    residual_tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_method: AbstractNonlinearMethod,
        /,
        *,
        termination: NonlinearTermination | None = None,
        residual_tolerance: float = 1e-7,
        method_id: str = "fold-extended-system",
    ):
        if not isinstance(root_method, AbstractNonlinearMethod):
            raise TypeError("root_method must be an AbstractNonlinearMethod.")
        tolerance = float(residual_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("residual_tolerance must be finite and non-negative.")
        termination_ = (
            NonlinearTermination(
                absolute_residual=tolerance,
                relative_residual=0.0,
            )
            if termination is None
            else termination
        )
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination or None.")
        identifier = str(method_id)
        if not identifier:
            raise ValueError("method_id must be non-empty.")
        self.root_method = root_method
        self.termination = termination_
        self.residual_tolerance = tolerance
        self.method_id = identifier

    def solve(
        self,
        problem: FoldProblem,
        initial_state: FoldState,
        /,
        *,
        args: Any = None,
    ) -> FoldResult:
        if not isinstance(problem, FoldProblem):
            raise TypeError("problem must be a FoldProblem.")
        if not isinstance(initial_state, FoldState):
            raise TypeError("initial_state must be a FoldState.")
        nonlinear_result = self.root_method.solve(
            problem.as_nonlinear_problem(),
            initial_state,
            termination=self.termination,
            args=args,
        )
        state = nonlinear_result.state
        if not isinstance(state, FoldState):
            raise TypeError("The nonlinear method did not preserve FoldState structure.")
        blocks = problem.residual_blocks(state, args)
        convergence = _extended_certificate(
            _fold_block_norms(blocks),
            nonlinear_result,
            problem.problem.contains_coordinate(state.parameter),
            self.residual_tolerance,
        )
        return FoldResult(
            state=state,
            residual_blocks=blocks,
            convergence=convergence,
            nonlinear_result=nonlinear_result,
            provenance=ExtendedSystemProvenance(
                problem_id=problem.problem_id,
                method_id=self.method_id,
                nonlinear_method_id=self.root_method.method_id,
            ),
        )


class HopfMethod(StrictModule):
    """Hopf corrector delegating the augmented root solve to a nonlinear method."""

    root_method: AbstractNonlinearMethod
    termination: NonlinearTermination
    residual_tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_method: AbstractNonlinearMethod,
        /,
        *,
        termination: NonlinearTermination | None = None,
        residual_tolerance: float = 1e-7,
        method_id: str = "hopf-extended-system",
    ):
        if not isinstance(root_method, AbstractNonlinearMethod):
            raise TypeError("root_method must be an AbstractNonlinearMethod.")
        tolerance = float(residual_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("residual_tolerance must be finite and non-negative.")
        termination_ = (
            NonlinearTermination(
                absolute_residual=tolerance,
                relative_residual=0.0,
            )
            if termination is None
            else termination
        )
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination or None.")
        identifier = str(method_id)
        if not identifier:
            raise ValueError("method_id must be non-empty.")
        self.root_method = root_method
        self.termination = termination_
        self.residual_tolerance = tolerance
        self.method_id = identifier

    def solve(
        self,
        problem: HopfProblem,
        initial_state: HopfState,
        /,
        *,
        args: Any = None,
    ) -> HopfResult:
        if not isinstance(problem, HopfProblem):
            raise TypeError("problem must be a HopfProblem.")
        if not isinstance(initial_state, HopfState):
            raise TypeError("initial_state must be a HopfState.")
        nonlinear_result = self.root_method.solve(
            problem.as_nonlinear_problem(),
            initial_state,
            termination=self.termination,
            args=args,
        )
        state = nonlinear_result.state
        if not isinstance(state, HopfState):
            raise TypeError("The nonlinear method did not preserve HopfState structure.")
        blocks = problem.residual_blocks(state, args)
        convergence = _extended_certificate(
            _hopf_block_norms(blocks),
            nonlinear_result,
            problem.problem.contains_coordinate(state.parameter),
            self.residual_tolerance,
            frequency_valid=state.frequency >= problem.minimum_frequency,
        )
        return HopfResult(
            state=state,
            residual_blocks=blocks,
            convergence=convergence,
            nonlinear_result=nonlinear_result,
            provenance=ExtendedSystemProvenance(
                problem_id=problem.problem_id,
                method_id=self.method_id,
                nonlinear_method_id=self.root_method.method_id,
            ),
        )


class NullspaceEvidence(StrictModule):
    """Right/left zero-mode evidence supplied by a general spectral analyzer."""

    right_nullvector: PyTree[Array]
    left_nullvector: PyTree[Array]
    singular_values: Array
    right_residual_norm: Array
    left_residual_norm: Array
    right_norm: Array
    left_norm: Array
    left_right_pairing: Array
    eigenvalue_condition: Array
    source_status: Array
    source_success: Array
    analyzer_id: str = eqx.field(static=True)
    full_spectrum: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        right_nullvector: PyTree[Any],
        left_nullvector: PyTree[Any],
        singular_values: Any,
        right_residual_norm: Any,
        left_residual_norm: Any,
        right_norm: Any,
        left_norm: Any,
        left_right_pairing: Any,
        eigenvalue_condition: Any,
        source_status: Any,
        source_success: Any,
        analyzer_id: str,
        full_spectrum: bool,
    ):
        values = jnp.asarray(singular_values)
        if values.ndim != 1 or not values.size:
            raise ValueError("singular_values must be a nonempty rank-one array.")
        identifier = str(analyzer_id)
        if not identifier:
            raise ValueError("analyzer_id must be non-empty.")
        self.right_nullvector = right_nullvector
        self.left_nullvector = left_nullvector
        self.singular_values = values
        self.right_residual_norm = jnp.asarray(right_residual_norm)
        self.left_residual_norm = jnp.asarray(left_residual_norm)
        self.right_norm = jnp.asarray(right_norm)
        self.left_norm = jnp.asarray(left_norm)
        self.left_right_pairing = jnp.asarray(left_right_pairing)
        self.eigenvalue_condition = jnp.asarray(eigenvalue_condition)
        self.source_status = jnp.asarray(source_status, dtype=jnp.int32)
        self.source_success = jnp.asarray(source_success, dtype=bool)
        self.analyzer_id = identifier
        self.full_spectrum = bool(full_spectrum)


class AbstractNullspaceAnalyzer(StrictModule):
    """Hook from a general eigensolver/SVD implementation to nullspace evidence."""

    analyzer_id: AbstractAttribute[str]

    @abc.abstractmethod
    def analyze(
        self,
        problem: ContinuationCurveProblem,
        state: PyTree[Any],
        parameter: Any,
        state_space: AbstractVectorSpace,
        args: Any = None,
        /,
    ) -> NullspaceEvidence:
        raise NotImplementedError


class CallableNullspaceAnalyzer(AbstractNullspaceAnalyzer):
    """Adapter for externally solved right/left nullspaces."""

    function: Callable[
        [ContinuationCurveProblem, PyTree[Any], Any, AbstractVectorSpace, Any],
        NullspaceEvidence,
    ]
    analyzer_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[
            [ContinuationCurveProblem, PyTree[Any], Any, AbstractVectorSpace, Any],
            NullspaceEvidence,
        ],
        /,
        *,
        analyzer_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(analyzer_id)
        if not identifier:
            raise ValueError("analyzer_id must be non-empty.")
        self.function = function
        self.analyzer_id = identifier

    def analyze(
        self,
        problem: ContinuationCurveProblem,
        state: PyTree[Any],
        parameter: Any,
        state_space: AbstractVectorSpace,
        args: Any = None,
        /,
    ) -> NullspaceEvidence:
        evidence = self.function(problem, state, parameter, state_space, args)
        if not isinstance(evidence, NullspaceEvidence):
            raise TypeError("A nullspace analyzer must return NullspaceEvidence.")
        return evidence


def evaluate_nullspace(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Any,
    state_space: AbstractVectorSpace,
    right_nullvector: PyTree[Any],
    left_nullvector: PyTree[Any],
    singular_values: Any,
    /,
    *,
    source_success: Any,
    source_status: Any = 0,
    full_spectrum: bool,
    analyzer_id: str,
    args: Any = None,
) -> NullspaceEvidence:
    """Evaluate residuals for nullvectors produced by an external spectral solve."""
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    if not isinstance(state_space, AbstractVectorSpace):
        raise TypeError("state_space must be an AbstractVectorSpace.")
    state_ = state_space.validate(state)
    right = state_space.validate(right_nullvector)
    left = state_space.validate(left_nullvector)
    parameter_ = _validate_scalar(parameter, "nullspace parameter")
    residual_function = lambda value: problem.residual(value, parameter_, args)
    right_action = jax.jvp(residual_function, (state_,), (right,))[1]
    _, pullback = jax.vjp(residual_function, state_)
    left_covector = state_space.riesz(left)
    adjoint_covector = pullback(left_covector)[0]
    left_action = state_space.inverse_riesz(adjoint_covector)
    right_norm = jnp.sqrt(jnp.maximum(jnp.real(state_space.inner(right, right)), 0.0))
    left_norm = jnp.sqrt(jnp.maximum(jnp.real(state_space.inner(left, left)), 0.0))
    pairing = state_space.inner(left, right)
    overlap = jnp.abs(pairing)
    condition = jnp.where(
        overlap > 0.0,
        left_norm * right_norm / overlap,
        jnp.inf,
    )
    return NullspaceEvidence(
        right_nullvector=right,
        left_nullvector=left,
        singular_values=singular_values,
        right_residual_norm=tree_norm(right_action),
        left_residual_norm=tree_norm(left_action),
        right_norm=right_norm,
        left_norm=left_norm,
        left_right_pairing=pairing,
        eigenvalue_condition=condition,
        source_status=source_status,
        source_success=source_success,
        analyzer_id=analyzer_id,
        full_spectrum=full_spectrum,
    )


class HopfEigenEvidence(StrictModule):
    """General-eigensolver evidence needed beyond the Hopf extended system."""

    eigenvalues: Array
    critical_pair_residual: Array
    adjoint_pair_residual: Array
    crossing_speed: Array
    pair_condition: Array
    source_status: Array
    source_success: Array
    analyzer_id: str = eqx.field(static=True)
    full_spectrum: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        eigenvalues: Any,
        critical_pair_residual: Any,
        adjoint_pair_residual: Any,
        crossing_speed: Any,
        pair_condition: Any,
        source_status: Any,
        source_success: Any,
        analyzer_id: str,
        full_spectrum: bool,
    ):
        values = jnp.asarray(eigenvalues)
        if values.ndim != 1 or not values.size:
            raise ValueError("eigenvalues must be a nonempty rank-one array.")
        identifier = str(analyzer_id)
        if not identifier:
            raise ValueError("analyzer_id must be non-empty.")
        self.eigenvalues = values
        self.critical_pair_residual = jnp.asarray(critical_pair_residual)
        self.adjoint_pair_residual = jnp.asarray(adjoint_pair_residual)
        self.crossing_speed = jnp.asarray(crossing_speed)
        self.pair_condition = jnp.asarray(pair_condition)
        self.source_status = jnp.asarray(source_status, dtype=jnp.int32)
        self.source_success = jnp.asarray(source_success, dtype=bool)
        self.analyzer_id = identifier
        self.full_spectrum = bool(full_spectrum)


class AbstractHopfAnalyzer(StrictModule):
    """Hook from a general eigensolver to a complete critical-pair analysis."""

    analyzer_id: AbstractAttribute[str]

    @abc.abstractmethod
    def analyze(
        self,
        problem: ContinuationCurveProblem,
        candidate: HopfState,
        state_space: AbstractVectorSpace,
        args: Any = None,
        /,
    ) -> HopfEigenEvidence:
        raise NotImplementedError


class CallableHopfAnalyzer(AbstractHopfAnalyzer):
    """Adapter for externally computed Hopf spectral evidence."""

    function: Callable[
        [ContinuationCurveProblem, HopfState, AbstractVectorSpace, Any],
        HopfEigenEvidence,
    ]
    analyzer_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[
            [ContinuationCurveProblem, HopfState, AbstractVectorSpace, Any],
            HopfEigenEvidence,
        ],
        /,
        *,
        analyzer_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(analyzer_id)
        if not identifier:
            raise ValueError("analyzer_id must be non-empty.")
        self.function = function
        self.analyzer_id = identifier

    def analyze(
        self,
        problem: ContinuationCurveProblem,
        candidate: HopfState,
        state_space: AbstractVectorSpace,
        args: Any = None,
        /,
    ) -> HopfEigenEvidence:
        evidence = self.function(problem, candidate, state_space, args)
        if not isinstance(evidence, HopfEigenEvidence):
            raise TypeError("A Hopf analyzer must return HopfEigenEvidence.")
        return evidence


class FoldEvidence(StrictModule):
    equilibrium_residual: Array
    nullspace: NullspaceEvidence
    parameter_transversality: Array
    quadratic_coefficient: Array


class BranchPointEvidence(StrictModule):
    equilibrium_residual: Array
    nullspace: NullspaceEvidence
    parameter_range_projection: Array
    mixed_transversality: Array


class HopfEvidence(StrictModule):
    extended_system: ExtendedSystemCertificate
    spectrum: HopfEigenEvidence
    critical_pair_count: Array
    pair_frequency_error: Array
    isolation_gap: Array


class PitchforkEvidence(StrictModule):
    branch_point: BranchPointEvidence
    fixed_state_defect: Array
    odd_mode_defect: Array
    involution_state_defect: Array
    local_equivariance_defect: Array
    quadratic_coefficient: Array
    cubic_coefficient: Array
    normal_form_solve_residual: Array
    normal_form_condition: Array
    normal_form_success: Array


class BifurcationCertificate(StrictModule):
    """Candidate state plus sufficient evidence for one named local theorem."""

    state: PyTree[Array]
    parameter: Array
    right_nullvector: PyTree[Array] | None
    left_nullvector: PyTree[Array] | None
    evidence: Any
    status: Array
    kind: BifurcationKind = eqx.field(static=True)
    assumptions_verified: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: PyTree[Any],
        parameter: Any,
        right_nullvector: PyTree[Any] | None,
        left_nullvector: PyTree[Any] | None,
        evidence: Any,
        status: Any,
        kind: BifurcationKind,
        assumptions_verified: bool,
        certificate_id: str,
    ):
        if kind not in ("fold", "hopf", "branch-point", "pitchfork"):
            raise ValueError("Unsupported bifurcation kind.")
        identifier = str(certificate_id)
        if not identifier:
            raise ValueError("certificate_id must be non-empty.")
        self.state = state
        self.parameter = jnp.asarray(parameter)
        self.right_nullvector = right_nullvector
        self.left_nullvector = left_nullvector
        self.evidence = evidence
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.kind = kind
        self.assumptions_verified = bool(assumptions_verified)
        self.certificate_id = identifier

    @property
    def certified(self) -> Array:
        return self.status == int(BifurcationStatus.CERTIFIED)


def _nullspace_conditions(
    evidence: NullspaceEvidence,
    tolerances: BifurcationTolerances,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    singular = jnp.sort(jnp.abs(evidence.singular_values))
    nullity = jnp.sum(singular <= tolerances.spectral_zero)
    gap = jnp.where(singular.size > 1, singular[1], jnp.inf)
    residual_valid = (evidence.right_residual_norm <= tolerances.null_residual) & (
        evidence.left_residual_norm <= tolerances.null_residual
    )
    normalization_valid = jnp.abs(evidence.left_right_pairing) > tolerances.normalization
    simple = (nullity == 1) & (gap >= tolerances.spectral_gap)
    conditioned = evidence.eigenvalue_condition <= tolerances.maximum_condition
    finite = (
        jnp.all(jnp.isfinite(evidence.singular_values))
        & jnp.isfinite(evidence.right_residual_norm)
        & jnp.isfinite(evidence.left_residual_norm)
        & jnp.isfinite(evidence.eigenvalue_condition)
    )
    return residual_valid, normalization_valid, simple, conditioned, finite


def _parameter_derivative(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Array,
    args: Any,
    /,
) -> PyTree[Array]:
    return jax.jvp(
        lambda value: problem.residual(state, value, args),
        (parameter,),
        (jnp.ones_like(parameter),),
    )[1]


def _second_state_derivative(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Array,
    direction: PyTree[Any],
    args: Any,
    /,
) -> PyTree[Array]:
    def first(value):
        return jax.jvp(
            lambda inner: problem.residual(inner, parameter, args),
            (value,),
            (direction,),
        )[1]

    return jax.jvp(first, (state,), (direction,))[1]


def _mixed_derivative(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Array,
    direction: PyTree[Any],
    args: Any,
    /,
) -> PyTree[Array]:
    def state_action(parameter_value):
        return jax.jvp(
            lambda value: problem.residual(value, parameter_value, args),
            (state,),
            (direction,),
        )[1]

    return jax.jvp(
        state_action,
        (parameter,),
        (jnp.ones_like(parameter),),
    )[1]


def _spectral_status(
    evidence: NullspaceEvidence,
    tolerances: BifurcationTolerances,
    /,
) -> Array:
    residual_valid, normalization_valid, simple, conditioned, finite = (
        _nullspace_conditions(evidence, tolerances)
    )
    return jnp.where(
        ~finite,
        int(BifurcationStatus.NONFINITE_EVIDENCE),
        jnp.where(
            ~evidence.source_success,
            int(BifurcationStatus.SPECTRAL_SOURCE_FAILED),
            jnp.where(
                not evidence.full_spectrum,
                int(BifurcationStatus.INSUFFICIENT_SPECTRAL_EVIDENCE),
                jnp.where(
                    ~simple,
                    int(BifurcationStatus.NULLSPACE_NOT_SIMPLE),
                    jnp.where(
                        ~(residual_valid & normalization_valid),
                        int(BifurcationStatus.NULLSPACE_RESIDUAL_TOO_LARGE),
                        jnp.where(
                            ~conditioned,
                            int(BifurcationStatus.ILL_CONDITIONED),
                            int(BifurcationStatus.CERTIFIED),
                        ),
                    ),
                ),
            ),
        ),
    )


def certify_fold(
    problem: FoldProblem,
    result: FoldResult,
    analyzer: AbstractNullspaceAnalyzer,
    assumptions: FoldAssumptions,
    /,
    *,
    tolerances: BifurcationTolerances | None = None,
    args: Any = None,
) -> BifurcationCertificate:
    """Certify a nondegenerate fold; mere extended convergence is insufficient."""
    if not isinstance(problem, FoldProblem):
        raise TypeError("problem must be a FoldProblem.")
    if not isinstance(result, FoldResult):
        raise TypeError("result must be a FoldResult.")
    if not isinstance(analyzer, AbstractNullspaceAnalyzer):
        raise TypeError("analyzer must be an AbstractNullspaceAnalyzer.")
    if not isinstance(assumptions, FoldAssumptions):
        raise TypeError("assumptions must be FoldAssumptions.")
    policy = BifurcationTolerances() if tolerances is None else tolerances
    if not isinstance(policy, BifurcationTolerances):
        raise TypeError("tolerances must be BifurcationTolerances or None.")
    candidate = result.state
    nullspace = analyzer.analyze(
        problem.problem,
        candidate.physical_state,
        candidate.parameter,
        problem.state_space,
        args,
    )
    parameter_derivative = _parameter_derivative(
        problem.problem,
        candidate.physical_state,
        candidate.parameter,
        args,
    )
    curvature = _second_state_derivative(
        problem.problem,
        candidate.physical_state,
        candidate.parameter,
        nullspace.right_nullvector,
        args,
    )
    transversality = jnp.abs(
        problem.state_space.inner(nullspace.left_nullvector, parameter_derivative)
    )
    quadratic = 0.5 * problem.state_space.inner(
        nullspace.left_nullvector,
        curvature,
    )
    evidence = FoldEvidence(
        equilibrium_residual=tree_norm(result.residual_blocks.equilibrium),
        nullspace=nullspace,
        parameter_transversality=transversality,
        quadratic_coefficient=quadratic,
    )
    status = _spectral_status(nullspace, policy)
    status = jnp.where(
        ~result.candidate_converged,
        int(BifurcationStatus.EXTENDED_SYSTEM_NOT_CONVERGED),
        status,
    )
    if not assumptions.verified:
        status = jnp.asarray(int(BifurcationStatus.ASSUMPTIONS_UNVERIFIED))
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED))
        & (transversality < policy.transversality),
        int(BifurcationStatus.TRANSVERSALITY_FAILED),
        status,
    )
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED))
        & (jnp.abs(quadratic) < policy.nondegeneracy),
        int(BifurcationStatus.NONDEGENERACY_FAILED),
        status,
    )
    status = jnp.where(
        jnp.isfinite(transversality) & jnp.isfinite(quadratic),
        status,
        int(BifurcationStatus.NONFINITE_EVIDENCE),
    )
    return BifurcationCertificate(
        state=candidate.physical_state,
        parameter=candidate.parameter,
        right_nullvector=nullspace.right_nullvector,
        left_nullvector=nullspace.left_nullvector,
        evidence=evidence,
        status=status,
        kind="fold",
        assumptions_verified=assumptions.verified,
        certificate_id=f"{problem.problem_id}/{analyzer.analyzer_id}/certificate",
    )


def certify_branch_point(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Any,
    state_space: AbstractVectorSpace,
    analyzer: AbstractNullspaceAnalyzer,
    assumptions: BranchPointAssumptions,
    /,
    *,
    tolerances: BifurcationTolerances | None = None,
    args: Any = None,
) -> BifurcationCertificate:
    """Certify a simple branch point with explicit range and mixed tests."""
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    if not isinstance(state_space, AbstractVectorSpace):
        raise TypeError("state_space must be an AbstractVectorSpace.")
    if not isinstance(analyzer, AbstractNullspaceAnalyzer):
        raise TypeError("analyzer must be an AbstractNullspaceAnalyzer.")
    if not isinstance(assumptions, BranchPointAssumptions):
        raise TypeError("assumptions must be BranchPointAssumptions.")
    policy = BifurcationTolerances() if tolerances is None else tolerances
    if not isinstance(policy, BifurcationTolerances):
        raise TypeError("tolerances must be BifurcationTolerances or None.")
    state_ = state_space.validate(state)
    parameter_ = _validate_scalar(parameter, "branch-point parameter")
    nullspace = analyzer.analyze(problem, state_, parameter_, state_space, args)
    equilibrium_residual = tree_norm(problem.residual(state_, parameter_, args))
    parameter_derivative = _parameter_derivative(problem, state_, parameter_, args)
    mixed = _mixed_derivative(
        problem,
        state_,
        parameter_,
        nullspace.right_nullvector,
        args,
    )
    parameter_projection = jnp.abs(
        state_space.inner(nullspace.left_nullvector, parameter_derivative)
    )
    mixed_transversality = jnp.abs(state_space.inner(nullspace.left_nullvector, mixed))
    evidence = BranchPointEvidence(
        equilibrium_residual=equilibrium_residual,
        nullspace=nullspace,
        parameter_range_projection=parameter_projection,
        mixed_transversality=mixed_transversality,
    )
    status = _spectral_status(nullspace, policy)
    if not assumptions.verified:
        status = jnp.asarray(int(BifurcationStatus.ASSUMPTIONS_UNVERIFIED))
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED))
        & (
            (equilibrium_residual > policy.residual)
            | (parameter_projection > policy.branch_projection)
            | (mixed_transversality < policy.transversality)
        ),
        int(BifurcationStatus.TRANSVERSALITY_FAILED),
        status,
    )
    status = jnp.where(
        jnp.isfinite(equilibrium_residual)
        & jnp.isfinite(parameter_projection)
        & jnp.isfinite(mixed_transversality),
        status,
        int(BifurcationStatus.NONFINITE_EVIDENCE),
    )
    return BifurcationCertificate(
        state=state_,
        parameter=parameter_,
        right_nullvector=nullspace.right_nullvector,
        left_nullvector=nullspace.left_nullvector,
        evidence=evidence,
        status=status,
        kind="branch-point",
        assumptions_verified=assumptions.verified,
        certificate_id=f"{problem.problem_id}/{analyzer.analyzer_id}/branch-certificate",
    )


def certify_hopf(
    problem: HopfProblem,
    result: HopfResult,
    analyzer: AbstractHopfAnalyzer,
    assumptions: HopfAssumptions,
    /,
    *,
    tolerances: BifurcationTolerances | None = None,
    args: Any = None,
) -> BifurcationCertificate:
    """Certify a simple isolated Hopf pair with an explicit crossing speed."""
    if not isinstance(problem, HopfProblem):
        raise TypeError("problem must be a HopfProblem.")
    if not isinstance(result, HopfResult):
        raise TypeError("result must be a HopfResult.")
    if not isinstance(analyzer, AbstractHopfAnalyzer):
        raise TypeError("analyzer must be an AbstractHopfAnalyzer.")
    if not isinstance(assumptions, HopfAssumptions):
        raise TypeError("assumptions must be HopfAssumptions.")
    policy = BifurcationTolerances() if tolerances is None else tolerances
    if not isinstance(policy, BifurcationTolerances):
        raise TypeError("tolerances must be BifurcationTolerances or None.")
    spectrum = analyzer.analyze(problem.problem, result.state, problem.state_space, args)
    values = spectrum.eigenvalues
    target_distance = jnp.minimum(
        jnp.abs(values - 1j * result.state.frequency),
        jnp.abs(values + 1j * result.state.frequency),
    )
    critical = target_distance <= policy.spectral_zero
    pair_count = jnp.sum(critical)
    pair_error = jnp.max(jnp.where(critical, target_distance, 0.0))
    isolation_gap = jnp.min(jnp.where(critical, jnp.inf, target_distance))
    evidence = HopfEvidence(
        extended_system=result.convergence,
        spectrum=spectrum,
        critical_pair_count=pair_count,
        pair_frequency_error=pair_error,
        isolation_gap=isolation_gap,
    )
    finite = (
        jnp.all(jnp.isfinite(values))
        & jnp.isfinite(spectrum.critical_pair_residual)
        & jnp.isfinite(spectrum.adjoint_pair_residual)
        & jnp.isfinite(spectrum.crossing_speed)
        & jnp.isfinite(spectrum.pair_condition)
    )
    status = jnp.asarray(int(BifurcationStatus.CERTIFIED))
    status = jnp.where(
        ~result.candidate_converged,
        int(BifurcationStatus.EXTENDED_SYSTEM_NOT_CONVERGED),
        status,
    )
    if not assumptions.verified:
        status = jnp.asarray(int(BifurcationStatus.ASSUMPTIONS_UNVERIFIED))
    status = jnp.where(
        ~finite,
        int(BifurcationStatus.NONFINITE_EVIDENCE),
        status,
    )
    status = jnp.where(
        ~spectrum.source_success,
        int(BifurcationStatus.SPECTRAL_SOURCE_FAILED),
        status,
    )
    if not spectrum.full_spectrum:
        status = jnp.asarray(int(BifurcationStatus.INSUFFICIENT_SPECTRAL_EVIDENCE))
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED))
        & ((pair_count != 2) | (isolation_gap < policy.spectral_gap)),
        int(BifurcationStatus.INSUFFICIENT_SPECTRAL_EVIDENCE),
        status,
    )
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED))
        & (
            (spectrum.critical_pair_residual > policy.null_residual)
            | (spectrum.adjoint_pair_residual > policy.null_residual)
        ),
        int(BifurcationStatus.NULLSPACE_RESIDUAL_TOO_LARGE),
        status,
    )
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED))
        & (jnp.abs(spectrum.crossing_speed) < policy.transversality),
        int(BifurcationStatus.TRANSVERSALITY_FAILED),
        status,
    )
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED))
        & (spectrum.pair_condition > policy.maximum_condition),
        int(BifurcationStatus.ILL_CONDITIONED),
        status,
    )
    return BifurcationCertificate(
        state=result.state.physical_state,
        parameter=result.state.parameter,
        right_nullvector=None,
        left_nullvector=None,
        evidence=evidence,
        status=status,
        kind="hopf",
        assumptions_verified=assumptions.verified,
        certificate_id=f"{problem.problem_id}/{analyzer.analyzer_id}/certificate",
    )


def certify_pitchfork(
    branch_certificate: BifurcationCertificate,
    problem: ContinuationCurveProblem,
    state_space: AbstractVectorSpace,
    symmetry: Callable[[PyTree[Any]], PyTree[Any]],
    assumptions: PitchforkAssumptions,
    /,
    *,
    quadratic_coefficient: Any,
    cubic_coefficient: Any,
    normal_form_solve_residual: Any,
    normal_form_condition: Any,
    normal_form_success: Any,
    probe_scale: float = 1e-4,
    tolerances: BifurcationTolerances | None = None,
    args: Any = None,
) -> BifurcationCertificate:
    """Certify a symmetry-breaking pitchfork from a certified branch point."""
    if not isinstance(branch_certificate, BifurcationCertificate):
        raise TypeError("branch_certificate must be a BifurcationCertificate.")
    if branch_certificate.kind != "branch-point":
        raise ValueError("Pitchfork certification requires a branch-point certificate.")
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    if not isinstance(state_space, AbstractVectorSpace):
        raise TypeError("state_space must be an AbstractVectorSpace.")
    if not callable(symmetry):
        raise TypeError("symmetry must be callable.")
    if not isinstance(assumptions, PitchforkAssumptions):
        raise TypeError("assumptions must be PitchforkAssumptions.")
    scale = float(probe_scale)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("probe_scale must be finite and positive.")
    policy = BifurcationTolerances() if tolerances is None else tolerances
    if not isinstance(policy, BifurcationTolerances):
        raise TypeError("tolerances must be BifurcationTolerances or None.")
    if branch_certificate.right_nullvector is None:
        raise ValueError("The branch-point certificate has no right nullvector.")
    state = state_space.validate(branch_certificate.state)
    mode = state_space.validate(branch_certificate.right_nullvector)
    symmetric_state = state_space.validate(symmetry(state))
    symmetric_mode = state_space.validate(symmetry(mode))
    twice_symmetric_state = state_space.validate(symmetry(symmetric_state))
    fixed_defect = tree_norm(jax.tree.map(lambda x, y: x - y, symmetric_state, state))
    odd_defect = tree_norm(jax.tree.map(lambda x, y: x + y, symmetric_mode, mode))
    involution_defect = tree_norm(
        jax.tree.map(lambda x, y: x - y, twice_symmetric_state, state)
    )
    probe = tree_add_scaled(state, mode, scale)
    symmetric_probe = state_space.validate(symmetry(probe))
    residual_at_symmetric = problem.residual(
        symmetric_probe,
        branch_certificate.parameter,
        args,
    )
    symmetric_residual = state_space.validate(
        symmetry(problem.residual(probe, branch_certificate.parameter, args))
    )
    equivariance_defect = tree_norm(
        jax.tree.map(
            lambda x, y: x - y,
            residual_at_symmetric,
            symmetric_residual,
        )
    )
    quadratic = jnp.asarray(quadratic_coefficient)
    cubic = jnp.asarray(cubic_coefficient)
    solve_residual = jnp.asarray(normal_form_solve_residual)
    condition = jnp.asarray(normal_form_condition)
    normal_success = jnp.asarray(normal_form_success, dtype=bool)
    branch_evidence = branch_certificate.evidence
    if not isinstance(branch_evidence, BranchPointEvidence):
        raise TypeError("branch certificate evidence must be BranchPointEvidence.")
    evidence = PitchforkEvidence(
        branch_point=branch_evidence,
        fixed_state_defect=fixed_defect,
        odd_mode_defect=odd_defect,
        involution_state_defect=involution_defect,
        local_equivariance_defect=equivariance_defect,
        quadratic_coefficient=quadratic,
        cubic_coefficient=cubic,
        normal_form_solve_residual=solve_residual,
        normal_form_condition=condition,
        normal_form_success=normal_success,
    )
    status = jnp.where(
        branch_certificate.certified,
        int(BifurcationStatus.CERTIFIED),
        int(BifurcationStatus.CANDIDATE_ONLY),
    )
    if not assumptions.verified:
        status = jnp.asarray(int(BifurcationStatus.ASSUMPTIONS_UNVERIFIED))
    symmetry_valid = (
        (fixed_defect <= policy.symmetry)
        & (odd_defect <= policy.symmetry)
        & (involution_defect <= policy.symmetry)
        & (equivariance_defect <= policy.symmetry)
    )
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED)) & ~symmetry_valid,
        int(BifurcationStatus.SYMMETRY_EVIDENCE_FAILED),
        status,
    )
    normal_finite = (
        jnp.isfinite(quadratic)
        & jnp.isfinite(cubic)
        & jnp.isfinite(solve_residual)
        & jnp.isfinite(condition)
    )
    status = jnp.where(
        ~normal_finite,
        int(BifurcationStatus.NONFINITE_EVIDENCE),
        status,
    )
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED))
        & (
            ~normal_success
            | (solve_residual > policy.null_residual)
            | (condition > policy.maximum_condition)
        ),
        int(BifurcationStatus.ILL_CONDITIONED),
        status,
    )
    status = jnp.where(
        (status == int(BifurcationStatus.CERTIFIED))
        & (
            (jnp.abs(quadratic) > policy.symmetry)
            | (jnp.abs(cubic) < policy.nondegeneracy)
        ),
        int(BifurcationStatus.NONDEGENERACY_FAILED),
        status,
    )
    return BifurcationCertificate(
        state=state,
        parameter=branch_certificate.parameter,
        right_nullvector=branch_certificate.right_nullvector,
        left_nullvector=branch_certificate.left_nullvector,
        evidence=evidence,
        status=status,
        kind="pitchfork",
        assumptions_verified=assumptions.verified,
        certificate_id=f"{branch_certificate.certificate_id}/pitchfork",
    )


def switch_branches_from_nullspace(
    certificate: BifurcationCertificate,
    /,
    *,
    amplitude: float,
) -> tuple[tuple[PyTree[Array], Array], tuple[PyTree[Array], Array]]:
    """Construct the two local branch seeds from a certified critical nullspace."""
    if not isinstance(certificate, BifurcationCertificate):
        raise TypeError("certificate must be a BifurcationCertificate.")
    if not bool(certificate.certified):
        raise ValueError("Automatic branch switching requires a certified nullspace.")
    if certificate.kind not in ("branch-point", "pitchfork"):
        raise ValueError("Automatic switching is defined for certified branch points.")
    if certificate.right_nullvector is None:
        raise ValueError("The certificate does not contain a right nullvector.")
    amplitude_ = float(amplitude)
    if not isfinite(amplitude_) or amplitude_ <= 0.0:
        raise ValueError("amplitude must be finite and positive.")
    mode_norm = tree_norm(certificate.right_nullvector)
    if not bool(jnp.isfinite(mode_norm) & (mode_norm > 0.0)):
        raise ValueError("The certified right nullvector has invalid Euclidean norm.")
    unit_mode = jax.tree.map(
        lambda value: value / mode_norm,
        certificate.right_nullvector,
    )
    plus = tree_add_scaled(certificate.state, unit_mode, amplitude_)
    minus = tree_add_scaled(certificate.state, unit_mode, -amplitude_)
    return (
        (plus, certificate.parameter),
        (minus, certificate.parameter),
    )


__all__ = [
    "AbstractHopfAnalyzer",
    "AbstractNullspaceAnalyzer",
    "BifurcationCertificate",
    "BifurcationKind",
    "BifurcationStatus",
    "BifurcationTolerances",
    "BranchPointAssumptions",
    "BranchPointEvidence",
    "CallableHopfAnalyzer",
    "CallableNullspaceAnalyzer",
    "ExtendedSystemCertificate",
    "ExtendedSystemProvenance",
    "ExtendedSystemStatus",
    "FoldAssumptions",
    "FoldEvidence",
    "FoldMethod",
    "FoldProblem",
    "FoldResidualBlocks",
    "FoldResult",
    "FoldState",
    "HopfAssumptions",
    "HopfEigenEvidence",
    "HopfEvidence",
    "HopfMethod",
    "HopfProblem",
    "HopfResidualBlocks",
    "HopfResult",
    "HopfState",
    "NullspaceEvidence",
    "PitchforkAssumptions",
    "PitchforkEvidence",
    "certify_branch_point",
    "certify_fold",
    "certify_hopf",
    "certify_pitchfork",
    "evaluate_nullspace",
    "switch_branches_from_nullspace",
]
