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

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._tree_math import tree_allfinite, validate_inexact_tree
from ..linalg import AbstractLinearOperator, AbstractVectorSpace, PyTreeSpace


class NonlinearStatus(IntEnum):
    """Portable terminal status for nonlinear algebraic solves."""

    SUCCESS = 0
    ITERATING = 1
    MAXIMUM_STEPS_REACHED = 2
    MAXIMUM_EVALUATIONS_REACHED = 3
    RESIDUAL_STAGNATION = 4
    LINE_SEARCH_FAILED = 5
    TRUST_REGION_FAILED = 6
    LINEAR_SOLVE_FAILED = 7
    SINGULAR_JACOBIAN = 8
    NONFINITE_INPUT = 9
    NONFINITE_EVALUATION = 10
    RECOVERABLE_DOMAIN_FAILURE = 11
    UNRECOVERABLE_DOMAIN_FAILURE = 12
    DIVERGENCE = 13
    CAPABILITY_REJECTED = 14
    BACKEND_FAILED = 15
    TRANSFORMATION_CERTIFICATION_FAILED = 16
    MAXIMUM_LINEAR_ITERATIONS_REACHED = 17


_STATUS_MESSAGES = {
    NonlinearStatus.SUCCESS: "success",
    NonlinearStatus.ITERATING: "iteration remains active",
    NonlinearStatus.MAXIMUM_STEPS_REACHED: "maximum nonlinear steps reached",
    NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED: "maximum residual evaluations reached",
    NonlinearStatus.RESIDUAL_STAGNATION: (
        "step stagnated before the residual tolerance was satisfied"
    ),
    NonlinearStatus.LINE_SEARCH_FAILED: "line search failed to accept a trial state",
    NonlinearStatus.TRUST_REGION_FAILED: (
        "trust-region model failed to accept a trial state"
    ),
    NonlinearStatus.LINEAR_SOLVE_FAILED: "Jacobian system solve failed",
    NonlinearStatus.SINGULAR_JACOBIAN: "Jacobian is singular or numerically unresolved",
    NonlinearStatus.NONFINITE_INPUT: "initial state contains non-finite values",
    NonlinearStatus.NONFINITE_EVALUATION: "nonlinear evaluation was non-finite",
    NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE: "trial state was outside the residual domain",
    NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE: (
        "the accepted state is outside the residual domain"
    ),
    NonlinearStatus.DIVERGENCE: "nonlinear residual diverged",
    NonlinearStatus.CAPABILITY_REJECTED: "requested nonlinear capability is unavailable",
    NonlinearStatus.BACKEND_FAILED: "nonlinear backend failed",
    NonlinearStatus.TRANSFORMATION_CERTIFICATION_FAILED: (
        "transformed root failed physical residual certification"
    ),
    NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED: (
        "maximum inner linear iterations reached"
    ),
}


def nonlinear_status_message(status: int | NonlinearStatus, /) -> str:
    return _STATUS_MESSAGES[NonlinearStatus(int(status))]


class NonlinearTermination(StrictModule):
    """Scale-aware residual and work limits for one nonlinear solve."""

    absolute_residual: float = eqx.field(static=True)
    relative_residual: float = eqx.field(static=True)
    absolute_step: float = eqx.field(static=True)
    relative_step: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_evaluations: int | None = eqx.field(static=True)
    maximum_linear_iterations: int | None = eqx.field(static=True)
    divergence_factor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_residual: float = 1e-8,
        relative_residual: float = 1e-8,
        absolute_step: float = 1e-12,
        relative_step: float = 1e-10,
        maximum_steps: int = 100,
        maximum_evaluations: int | None = None,
        maximum_linear_iterations: int | None = None,
        divergence_factor: float = 1e8,
    ):
        tolerances = tuple(
            float(value)
            for value in (
                absolute_residual,
                relative_residual,
                absolute_step,
                relative_step,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Nonlinear tolerances must be finite and non-negative.")
        steps = int(maximum_steps)
        evaluations = None if maximum_evaluations is None else int(maximum_evaluations)
        linear_iterations = (
            None if maximum_linear_iterations is None else int(maximum_linear_iterations)
        )
        divergence = float(divergence_factor)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        if evaluations is not None and evaluations < 1:
            raise ValueError("maximum_evaluations must be positive or None.")
        if linear_iterations is not None and linear_iterations < 1:
            raise ValueError("maximum_linear_iterations must be positive or None.")
        if not isfinite(divergence) or divergence <= 1.0:
            raise ValueError("divergence_factor must be finite and exceed one.")
        (
            self.absolute_residual,
            self.relative_residual,
            self.absolute_step,
            self.relative_step,
        ) = tolerances
        self.maximum_steps = steps
        self.maximum_evaluations = evaluations
        self.maximum_linear_iterations = linear_iterations
        self.divergence_factor = divergence

    def residual_threshold(self, initial_residual: Any, /) -> Array:
        return self.absolute_residual + self.relative_residual * jnp.asarray(
            initial_residual
        )

    def step_threshold(self, state_norm: Any, /) -> Array:
        return self.absolute_step + self.relative_step * jnp.asarray(state_norm)


class NonlinearCapabilities(StrictModule):
    """Static transformation and algorithm capabilities of one method."""

    matrix_free: bool = eqx.field(static=True)
    prepared_refresh: bool = eqx.field(static=True)
    jit: bool = eqx.field(static=True)
    implicit_differentiation: bool = eqx.field(static=True)
    fixed_point: bool = eqx.field(static=True)
    nonlinear_preconditioning: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        matrix_free: bool,
        prepared_refresh: bool,
        jit: bool,
        implicit_differentiation: bool,
        fixed_point: bool = False,
        nonlinear_preconditioning: bool = False,
    ):
        self.matrix_free = bool(matrix_free)
        self.prepared_refresh = bool(prepared_refresh)
        self.jit = bool(jit)
        self.implicit_differentiation = bool(implicit_differentiation)
        self.fixed_point = bool(fixed_point)
        self.nonlinear_preconditioning = bool(nonlinear_preconditioning)


class NonlinearSystemProblem(StrictModule):
    """Residual equation ``F(state, args) = 0`` with explicit space semantics."""

    residual_function: Callable[[PyTree[Any], Any], Any]
    validity_function: Callable[[PyTree[Any], PyTree[Any], Any, Any], Any] | None
    linear_setup_function: Callable[[PyTree[Any], Any], AbstractLinearOperator] | None
    state_space: AbstractVectorSpace | None
    residual_space: AbstractVectorSpace | None
    has_aux: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: Callable[[PyTree[Any], Any], Any],
        /,
        *,
        state_space: AbstractVectorSpace | None = None,
        residual_space: AbstractVectorSpace | None = None,
        has_aux: bool = False,
        validity: Callable[[PyTree[Any], PyTree[Any], Any, Any], Any] | None = None,
        linear_setup: Callable[[PyTree[Any], Any], AbstractLinearOperator] | None = None,
        problem_id: str = "nonlinear-system",
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if state_space is not None and not isinstance(state_space, AbstractVectorSpace):
            raise TypeError("state_space must be an AbstractVectorSpace or None.")
        if residual_space is not None and not isinstance(
            residual_space, AbstractVectorSpace
        ):
            raise TypeError("residual_space must be an AbstractVectorSpace or None.")
        if validity is not None and not callable(validity):
            raise TypeError("validity must be callable or None.")
        if linear_setup is not None and not callable(linear_setup):
            raise TypeError("linear_setup must be callable or None.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.residual_function = residual
        self.validity_function = validity
        self.linear_setup_function = linear_setup
        self.state_space = state_space
        self.residual_space = residual_space
        self.has_aux = bool(has_aux)
        self.problem_id = identifier

    def validate_state(self, state: PyTree[Any], /) -> PyTree[Array]:
        if self.state_space is None:
            return validate_inexact_tree(state, name="nonlinear state")
        return self.state_space.validate(state)

    def validate_residual(self, residual: PyTree[Any], /) -> PyTree[Array]:
        if self.residual_space is None:
            return validate_inexact_tree(residual, name="nonlinear residual")
        return self.residual_space.validate(residual)

    def evaluate(
        self,
        state: PyTree[Any],
        args: Any = None,
        /,
    ) -> tuple[PyTree[Array], Any]:
        state_ = self.validate_state(state)
        output = self.residual_function(state_, args)
        if self.has_aux:
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError(
                    "A has_aux nonlinear residual must return (residual, auxiliary)."
                )
            residual, auxiliary = output
        else:
            residual, auxiliary = output, None
        return self.validate_residual(residual), auxiliary

    def residual(self, state: PyTree[Any], args: Any = None, /) -> PyTree[Array]:
        return self.evaluate(state, args)[0]

    def valid(
        self,
        state: PyTree[Any],
        residual: PyTree[Any],
        auxiliary: Any,
        args: Any = None,
        /,
    ) -> Array:
        state_ = self.validate_state(state)
        residual_ = self.validate_residual(residual)
        finite = tree_allfinite(state_) & tree_allfinite(residual_)
        if self.validity_function is None:
            return finite
        return finite & jnp.asarray(
            self.validity_function(state_, residual_, auxiliary, args), dtype=bool
        )

    def linear_setup(
        self,
        state: PyTree[Any],
        args: Any = None,
        /,
    ) -> AbstractLinearOperator | None:
        if self.linear_setup_function is None:
            return None
        operator = self.linear_setup_function(self.validate_state(state), args)
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("linear_setup must return an AbstractLinearOperator.")
        return operator

    def bind_spaces(
        self,
        initial_state: PyTree[Any],
        residual: PyTree[Any] | None = None,
        /,
        *,
        args: Any = None,
    ) -> NonlinearSystemProblem:
        """Return this problem with any missing vector spaces inferred and bound."""
        state = self.validate_state(initial_state)
        if residual is None:
            residual_, _ = self.evaluate(state, args)
        else:
            residual_ = self.validate_residual(residual)
        if self.state_space is not None and self.residual_space is not None:
            return self
        return NonlinearSystemProblem(
            self.residual_function,
            state_space=(
                PyTreeSpace(state) if self.state_space is None else self.state_space
            ),
            residual_space=(
                PyTreeSpace(residual_)
                if self.residual_space is None
                else self.residual_space
            ),
            has_aux=self.has_aux,
            validity=self.validity_function,
            linear_setup=self.linear_setup_function,
            problem_id=self.problem_id,
        )


class FixedPointProblem(StrictModule):
    """Fixed-point equation ``state = mapping(state, args)``."""

    mapping_function: Callable[[PyTree[Any], Any], PyTree[Any]]
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        mapping: Callable[[PyTree[Any], Any], PyTree[Any]],
        /,
        *,
        problem_id: str = "fixed-point",
    ):
        if not callable(mapping):
            raise TypeError("mapping must be callable.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.mapping_function = mapping
        self.problem_id = identifier

    def mapping(self, state: PyTree[Any], args: Any = None, /) -> PyTree[Array]:
        state_ = validate_inexact_tree(state, name="fixed-point state")
        mapped = validate_inexact_tree(
            self.mapping_function(state_, args), name="fixed-point mapping"
        )
        if jax.tree.structure(mapped) != jax.tree.structure(state_):
            raise ValueError(
                "A fixed-point mapping must preserve the state PyTree structure."
            )
        return jax.tree.map(
            lambda value, template: jnp.asarray(value, dtype=template.dtype),
            mapped,
            state_,
        )


class NonlinearDiagnostics(StrictModule):
    """JAX-compatible numerical evidence from one nonlinear solve."""

    initial_residual_norm: Array
    final_residual_norm: Array
    final_step_norm: Array
    iterations: Array
    residual_evaluations: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array
    accepted_steps: Array
    rejected_steps: Array
    domain_failures: Array
    nonfinite_trials: Array
    acceleration_restarts: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    final_forcing: Array
    final_trust_radius: Array
    final_linear_status: Array
    final_linear_rank: Array
    final_linear_condition_estimate: Array
    final_linear_residual_norm: Array
    final_linear_converged: Array
    counts_complete: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_residual_norm: Any,
        final_residual_norm: Any,
        final_step_norm: Any = 0.0,
        iterations: Any = 0,
        residual_evaluations: Any = 0,
        jvp_evaluations: Any = 0,
        vjp_evaluations: Any = 0,
        jacobian_preparations: Any = 0,
        linear_solves: Any = 0,
        linear_iterations: Any = 0,
        accepted_steps: Any = 0,
        rejected_steps: Any = 0,
        domain_failures: Any = 0,
        nonfinite_trials: Any = 0,
        acceleration_restarts: Any = 0,
        setup_refreshes: Any = 0,
        numeric_refreshes: Any = 0,
        final_forcing: Any = jnp.nan,
        final_trust_radius: Any = jnp.nan,
        final_linear_status: Any = -1,
        final_linear_rank: Any = -1,
        final_linear_condition_estimate: Any = jnp.nan,
        final_linear_residual_norm: Any = jnp.nan,
        final_linear_converged: Any = False,
        counts_complete: bool = True,
    ):
        self.initial_residual_norm = jnp.asarray(initial_residual_norm)
        self.final_residual_norm = jnp.asarray(final_residual_norm)
        self.final_step_norm = jnp.asarray(final_step_norm)
        integer_values = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (
                iterations,
                residual_evaluations,
                jvp_evaluations,
                vjp_evaluations,
                jacobian_preparations,
                linear_solves,
                linear_iterations,
                accepted_steps,
                rejected_steps,
                domain_failures,
                nonfinite_trials,
                acceleration_restarts,
                setup_refreshes,
                numeric_refreshes,
            )
        )
        (
            self.iterations,
            self.residual_evaluations,
            self.jvp_evaluations,
            self.vjp_evaluations,
            self.jacobian_preparations,
            self.linear_solves,
            self.linear_iterations,
            self.accepted_steps,
            self.rejected_steps,
            self.domain_failures,
            self.nonfinite_trials,
            self.acceleration_restarts,
            self.setup_refreshes,
            self.numeric_refreshes,
        ) = integer_values
        self.final_forcing = jnp.asarray(final_forcing)
        self.final_trust_radius = jnp.asarray(final_trust_radius)
        self.final_linear_status = jnp.asarray(final_linear_status, dtype=jnp.int32)
        self.final_linear_rank = jnp.asarray(final_linear_rank, dtype=jnp.int32)
        self.final_linear_condition_estimate = jnp.asarray(
            final_linear_condition_estimate
        )
        self.final_linear_residual_norm = jnp.asarray(final_linear_residual_norm)
        self.final_linear_converged = jnp.asarray(final_linear_converged, dtype=bool)
        self.counts_complete = bool(counts_complete)


class NonlinearProvenance(StrictModule):
    """Static method, derivative, globalization, and backend identities."""

    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    derivative_id: str = eqx.field(static=True)
    globalization_id: str = eqx.field(static=True)
    linear_plan_id: str = eqx.field(static=True)
    precision_policy_id: str | None = eqx.field(static=True)
    notes: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem_id: str,
        method_id: str,
        derivative_id: str,
        globalization_id: str,
        linear_plan_id: str = "",
        precision_policy_id: str | None = None,
        notes: str = "",
    ):
        identifiers = tuple(
            str(value)
            for value in (problem_id, method_id, derivative_id, globalization_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("Nonlinear provenance identifiers must be non-empty.")
        (
            self.problem_id,
            self.method_id,
            self.derivative_id,
            self.globalization_id,
        ) = identifiers
        self.linear_plan_id = str(linear_plan_id)
        self.precision_policy_id = (
            None if precision_policy_id is None else str(precision_policy_id)
        )
        self.notes = str(notes)


class NonlinearTransformationEvidence(StrictModule):
    """Solver-coordinate evidence retained after physical reconstruction."""

    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any

    def __init__(
        self,
        *,
        state: PyTree[Any],
        residual: PyTree[Any],
        auxiliary: Any,
    ):
        self.state = validate_inexact_tree(state, name="transformed nonlinear state")
        self.residual = validate_inexact_tree(
            residual, name="transformed nonlinear residual"
        )
        self.auxiliary = auxiliary


class NonlinearResult(StrictModule):
    """Accepted nonlinear state, physical residual, and complete evidence."""

    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    status: Array
    diagnostics: NonlinearDiagnostics
    provenance: NonlinearProvenance
    transformation_evidence: NonlinearTransformationEvidence | None
    precision_evidence: PrecisionEvidenceEnvelope | None = eqx.field(static=True)
    attempts: tuple[Any, ...]

    def __init__(
        self,
        *,
        state: PyTree[Any],
        residual: PyTree[Any],
        auxiliary: Any,
        status: Any,
        diagnostics: NonlinearDiagnostics,
        provenance: NonlinearProvenance,
        transformation_evidence: NonlinearTransformationEvidence | None = None,
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
        attempts: tuple[Any, ...] = (),
    ):
        if not isinstance(diagnostics, NonlinearDiagnostics):
            raise TypeError("diagnostics must be NonlinearDiagnostics.")
        if not isinstance(provenance, NonlinearProvenance):
            raise TypeError("provenance must be NonlinearProvenance.")
        if transformation_evidence is not None and not isinstance(
            transformation_evidence, NonlinearTransformationEvidence
        ):
            raise TypeError(
                "transformation_evidence must be NonlinearTransformationEvidence or None."
            )
        if precision_evidence is not None and not isinstance(
            precision_evidence,
            PrecisionEvidenceEnvelope,
        ):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        self.state = state
        self.residual = residual
        self.auxiliary = auxiliary
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.transformation_evidence = transformation_evidence
        self.precision_evidence = precision_evidence
        self.attempts = tuple(attempts)

    @property
    def successful(self) -> Array:
        return self.status == int(NonlinearStatus.SUCCESS)


class AbstractNonlinearMethod(StrictModule):
    """Method implementing one complete nonlinear-system solve contract."""

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> NonlinearCapabilities:
        raise NotImplementedError

    @abc.abstractmethod
    def solve(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
    ) -> NonlinearResult:
        raise NotImplementedError


__all__ = [
    "AbstractNonlinearMethod",
    "FixedPointProblem",
    "NonlinearCapabilities",
    "NonlinearDiagnostics",
    "NonlinearProvenance",
    "NonlinearResult",
    "NonlinearStatus",
    "NonlinearSystemProblem",
    "NonlinearTermination",
    "NonlinearTransformationEvidence",
    "nonlinear_status_message",
]
