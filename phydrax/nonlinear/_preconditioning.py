#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ..linalg import AbstractVectorSpace
from ._types import (
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    NonlinearTransformationEvidence,
)


def _space_norm(space: AbstractVectorSpace, vector: PyTree[Any], /) -> Array:
    squared = jnp.real(space.inner(vector, vector))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


class AbstractLeftNonlinearPreconditioner(StrictModule):
    """State-dependent map from a physical residual to a solver residual."""

    state_space: AbstractVectorSpace
    source: AbstractVectorSpace
    target: AbstractVectorSpace
    preconditioner_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def _apply(
        self,
        state: PyTree[Any],
        residual: PyTree[Any],
        args: Any,
        /,
    ) -> PyTree[Array]:
        raise NotImplementedError

    def apply(
        self,
        state: PyTree[Any],
        residual: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        state_ = self.state_space.validate(state)
        residual_ = self.source.validate(residual)
        return self.target.validate(self._apply(state_, residual_, args))


class FunctionLeftNonlinearPreconditioner(AbstractLeftNonlinearPreconditioner):
    """Callable left nonlinear preconditioner over declared vector spaces."""

    function: Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Array]]

    def __init__(
        self,
        function: Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Array]],
        /,
        *,
        state_space: AbstractVectorSpace,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
        preconditioner_id: str = "function-left-nonlinear-preconditioner",
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        if not all(
            isinstance(space, AbstractVectorSpace)
            for space in (state_space, source, target)
        ):
            raise TypeError(
                "state_space, source, and target must be AbstractVectorSpace values."
            )
        identifier = str(preconditioner_id)
        if not identifier:
            raise ValueError("preconditioner_id must be non-empty.")
        self.function = function
        self.state_space = state_space
        self.source = source
        self.target = target
        self.preconditioner_id = identifier

    def _apply(
        self,
        state: PyTree[Any],
        residual: PyTree[Any],
        args: Any,
        /,
    ) -> PyTree[Array]:
        return self.function(state, residual, args)


class AbstractRightNonlinearPreconditioner(StrictModule):
    """Nonlinear map from latent solver coordinates to physical states."""

    source: AbstractVectorSpace
    target: AbstractVectorSpace
    preconditioner_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def _reconstruct(self, latent: PyTree[Any], args: Any, /) -> PyTree[Array]:
        raise NotImplementedError

    def reconstruct(
        self,
        latent: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        latent_ = self.source.validate(latent)
        return self.target.validate(self._reconstruct(latent_, args))


class FunctionRightNonlinearPreconditioner(AbstractRightNonlinearPreconditioner):
    """Callable right nonlinear reconstruction over declared vector spaces."""

    function: Callable[[PyTree[Any], Any], PyTree[Array]]

    def __init__(
        self,
        function: Callable[[PyTree[Any], Any], PyTree[Array]],
        /,
        *,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
        preconditioner_id: str = "function-right-nonlinear-preconditioner",
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        if not isinstance(source, AbstractVectorSpace) or not isinstance(
            target, AbstractVectorSpace
        ):
            raise TypeError("source and target must be AbstractVectorSpace values.")
        identifier = str(preconditioner_id)
        if not identifier:
            raise ValueError("preconditioner_id must be non-empty.")
        self.function = function
        self.source = source
        self.target = target
        self.preconditioner_id = identifier

    def _reconstruct(self, latent: PyTree[Any], args: Any, /) -> PyTree[Array]:
        return self.function(latent, args)


class _TransformationEvaluation(StrictModule):
    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any

    def __init__(
        self,
        state: PyTree[Array],
        residual: PyTree[Array],
        auxiliary: Any,
        /,
    ):
        self.state = state
        self.residual = residual
        self.auxiliary = auxiliary


def _require_compatible_space(
    declared: AbstractVectorSpace | None,
    transformation: AbstractVectorSpace,
    name: str,
    /,
) -> None:
    if declared is not None and not declared.compatible(transformation):
        raise ValueError(
            f"The problem {name} is incompatible with the nonlinear transformation."
        )


class AbstractNonlinearSystemTransformation(StrictModule):
    """Explicit solver-coordinate transformation of one physical root problem."""

    original: NonlinearSystemProblem
    preconditioner: (
        AbstractLeftNonlinearPreconditioner | AbstractRightNonlinearPreconditioner
    )
    problem: NonlinearSystemProblem
    transformation_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def reconstruct(
        self,
        transformed_state: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        raise NotImplementedError

    def finalize_result(
        self,
        result: NonlinearResult,
        initial_state: PyTree[Any],
        termination: NonlinearTermination,
        /,
        *,
        args: Any = None,
    ) -> NonlinearResult:
        """Reconstruct and independently certify one transformed solver result."""
        if not isinstance(result, NonlinearResult):
            raise TypeError("result must be a NonlinearResult.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination.")
        if not isinstance(result.auxiliary, _TransformationEvaluation):
            raise TypeError(
                "The transformed solver result has invalid auxiliary evidence."
            )

        initial_physical_state = self.reconstruct(initial_state, args)
        initial_physical_residual, _ = self.original.evaluate(
            initial_physical_state, args
        )
        physical_problem = self.original.bind_spaces(
            initial_physical_state, initial_physical_residual
        )
        residual_space = physical_problem.residual_space
        if residual_space is None:
            raise ValueError("A bound physical problem must have a residual space.")
        physical_state = self.reconstruct(result.state, args)
        physical_residual = physical_problem.validate_residual(result.auxiliary.residual)
        physical_auxiliary = result.auxiliary.auxiliary
        initial_physical_norm = _space_norm(
            residual_space,
            initial_physical_residual,
        )
        physical_norm = _space_norm(residual_space, physical_residual)
        physical_finite = tree_allfinite(physical_state) & tree_allfinite(
            physical_residual
        )
        physical_valid = physical_problem.valid(
            physical_state, physical_residual, physical_auxiliary, args
        )
        certified = (
            physical_finite
            & physical_valid
            & (physical_norm <= termination.residual_threshold(initial_physical_norm))
        )
        status = jnp.where(
            result.successful & ~certified,
            int(NonlinearStatus.TRANSFORMATION_CERTIFICATION_FAILED),
            result.status,
        ).astype(jnp.int32)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=initial_physical_norm,
            final_residual_norm=physical_norm,
            final_step_norm=result.diagnostics.final_step_norm,
            iterations=result.diagnostics.iterations,
            residual_evaluations=result.diagnostics.residual_evaluations + 1,
            jvp_evaluations=result.diagnostics.jvp_evaluations,
            vjp_evaluations=result.diagnostics.vjp_evaluations,
            jacobian_preparations=result.diagnostics.jacobian_preparations,
            linear_solves=result.diagnostics.linear_solves,
            linear_iterations=result.diagnostics.linear_iterations,
            accepted_steps=result.diagnostics.accepted_steps,
            rejected_steps=result.diagnostics.rejected_steps,
            domain_failures=result.diagnostics.domain_failures,
            nonfinite_trials=result.diagnostics.nonfinite_trials,
            acceleration_restarts=result.diagnostics.acceleration_restarts,
            setup_refreshes=result.diagnostics.setup_refreshes,
            numeric_refreshes=result.diagnostics.numeric_refreshes,
            final_forcing=result.diagnostics.final_forcing,
            final_trust_radius=result.diagnostics.final_trust_radius,
            final_linear_status=result.diagnostics.final_linear_status,
            final_linear_rank=result.diagnostics.final_linear_rank,
            final_linear_condition_estimate=(
                result.diagnostics.final_linear_condition_estimate
            ),
            final_linear_residual_norm=result.diagnostics.final_linear_residual_norm,
            final_linear_converged=result.diagnostics.final_linear_converged,
            counts_complete=result.diagnostics.counts_complete,
        )
        note = f"nonlinear-transformation={self.transformation_id}"
        notes = result.provenance.notes
        provenance = NonlinearProvenance(
            problem_id=self.original.problem_id,
            method_id=result.provenance.method_id,
            derivative_id=result.provenance.derivative_id,
            globalization_id=result.provenance.globalization_id,
            linear_plan_id=result.provenance.linear_plan_id,
            notes=f"{notes};{note}" if notes else note,
        )
        transformed_auxiliary = result.auxiliary.auxiliary
        return NonlinearResult(
            state=physical_state,
            residual=physical_residual,
            auxiliary=physical_auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=provenance,
            transformation_evidence=NonlinearTransformationEvidence(
                state=result.state,
                residual=result.residual,
                auxiliary=transformed_auxiliary,
            ),
            attempts=result.attempts,
        )


class LeftPreconditionedSystem(AbstractNonlinearSystemTransformation):
    """Physical-state system with a left-transformed solver residual."""

    def __init__(
        self,
        problem: NonlinearSystemProblem,
        preconditioner: AbstractLeftNonlinearPreconditioner,
        /,
    ):
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be a NonlinearSystemProblem.")
        if not isinstance(preconditioner, AbstractLeftNonlinearPreconditioner):
            raise TypeError(
                "preconditioner must be an AbstractLeftNonlinearPreconditioner."
            )
        _require_compatible_space(
            problem.state_space, preconditioner.state_space, "state_space"
        )
        _require_compatible_space(
            problem.residual_space, preconditioner.source, "residual_space"
        )

        def residual(state, args):
            physical, auxiliary = problem.evaluate(state, args)
            transformed = preconditioner.apply(state, physical, args)
            return transformed, _TransformationEvaluation(state, physical, auxiliary)

        def valid(_, __, payload, args):
            return problem.valid(payload.state, payload.residual, payload.auxiliary, args)

        self.original = problem
        self.preconditioner = preconditioner
        self.transformation_id = f"left/{preconditioner.preconditioner_id}"
        self.problem = NonlinearSystemProblem(
            residual,
            state_space=preconditioner.state_space,
            residual_space=preconditioner.target,
            validity=valid,
            has_aux=True,
            problem_id=f"{problem.problem_id}/{self.transformation_id}",
        )

    def reconstruct(self, state: PyTree[Any], args: Any = None, /) -> PyTree[Array]:
        del args
        preconditioner = self.preconditioner
        if not isinstance(preconditioner, AbstractLeftNonlinearPreconditioner):
            raise TypeError("Left transformation has an invalid preconditioner.")
        return preconditioner.state_space.validate(state)


class RightPreconditionedSystem(AbstractNonlinearSystemTransformation):
    """Latent-coordinate system with explicit physical reconstruction."""

    def __init__(
        self,
        problem: NonlinearSystemProblem,
        preconditioner: AbstractRightNonlinearPreconditioner,
        /,
    ):
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be a NonlinearSystemProblem.")
        if not isinstance(preconditioner, AbstractRightNonlinearPreconditioner):
            raise TypeError(
                "preconditioner must be an AbstractRightNonlinearPreconditioner."
            )
        _require_compatible_space(
            problem.state_space, preconditioner.target, "state_space"
        )

        def residual(latent, args):
            state = preconditioner.reconstruct(latent, args)
            physical, auxiliary = problem.evaluate(state, args)
            return physical, _TransformationEvaluation(state, physical, auxiliary)

        def valid(_, __, payload, args):
            return problem.valid(payload.state, payload.residual, payload.auxiliary, args)

        self.original = problem
        self.preconditioner = preconditioner
        self.transformation_id = f"right/{preconditioner.preconditioner_id}"
        self.problem = NonlinearSystemProblem(
            residual,
            state_space=preconditioner.source,
            residual_space=problem.residual_space,
            validity=valid,
            has_aux=True,
            problem_id=f"{problem.problem_id}/{self.transformation_id}",
        )

    def reconstruct(self, latent: PyTree[Any], args: Any = None, /) -> PyTree[Array]:
        preconditioner = self.preconditioner
        if not isinstance(preconditioner, AbstractRightNonlinearPreconditioner):
            raise TypeError("Right transformation has an invalid preconditioner.")
        return preconditioner.reconstruct(latent, args)


def left_precondition(
    problem: NonlinearSystemProblem,
    preconditioner: AbstractLeftNonlinearPreconditioner,
    /,
) -> LeftPreconditionedSystem:
    """Build an explicit left nonlinear system transformation."""
    return LeftPreconditionedSystem(problem, preconditioner)


def right_precondition(
    problem: NonlinearSystemProblem,
    preconditioner: AbstractRightNonlinearPreconditioner,
    /,
) -> RightPreconditionedSystem:
    """Build an explicit right nonlinear system transformation."""
    return RightPreconditionedSystem(problem, preconditioner)


__all__ = [
    "AbstractLeftNonlinearPreconditioner",
    "AbstractNonlinearSystemTransformation",
    "AbstractRightNonlinearPreconditioner",
    "FunctionLeftNonlinearPreconditioner",
    "FunctionRightNonlinearPreconditioner",
    "LeftPreconditionedSystem",
    "RightPreconditionedSystem",
    "left_precondition",
    "right_precondition",
]
