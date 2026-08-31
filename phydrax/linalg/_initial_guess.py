#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._operators import AbstractLinearOperator
from ._spaces import AbstractVectorSpace


LinearInitialGuessStrategy = Literal[
    "zero",
    "last-solution",
    "projection",
    "rolling-qr",
    "stabilized-extrapolation",
]


class LinearSolveHistoryPolicy(StrictModule, NonTrainableState):
    strategy: LinearInitialGuessStrategy = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    extrapolation_degree: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    reorthogonalizations: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        strategy: LinearInitialGuessStrategy = "projection",
        /,
        *,
        capacity: int = 6,
        extrapolation_degree: int = 2,
        rank_tolerance: float = 1.0e-10,
        reorthogonalizations: int = 2,
    ):
        strategy_ = str(strategy)
        capacity_ = int(capacity)
        degree = int(extrapolation_degree)
        tolerance = float(rank_tolerance)
        reorthogonalizations_ = int(reorthogonalizations)
        if strategy_ not in (
            "zero",
            "last-solution",
            "projection",
            "rolling-qr",
            "stabilized-extrapolation",
        ):
            raise ValueError("Unknown linear initial-guess strategy.")
        if (
            capacity_ < 1
            or degree < 0
            or (strategy_ == "stabilized-extrapolation" and degree >= capacity_)
        ):
            raise ValueError("History capacity/degree are incompatible.")
        if tolerance <= 0.0 or reorthogonalizations_ < 1:
            raise ValueError("History rank controls are invalid.")
        self.strategy = strategy_
        self.capacity = capacity_
        self.extrapolation_degree = degree
        self.rank_tolerance = tolerance
        self.reorthogonalizations = reorthogonalizations_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "linear-solve-history-policy",
                "strategy": strategy_,
                "capacity": capacity_,
                "extrapolation_degree": degree,
                "rank_tolerance": tolerance,
                "reorthogonalizations": reorthogonalizations_,
            }
        )


class LinearInitialGuessDiagnostics(StrictModule):
    effective_dimension: Array
    rank: Array
    projection_residual_norm: Array
    reused: Array
    strategy: str = eqx.field(static=True)


class LinearSolveHistory(StrictModule):
    source: AbstractVectorSpace
    target: AbstractVectorSpace
    solution_basis: Array
    rhs_image_basis: Array
    times: Array
    effective_dimension: Array
    update_count: Array
    operator_family_id: str = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)
    nullspace_policy_id: str = eqx.field(static=True)
    policy: LinearSolveHistoryPolicy
    history_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
        policy: LinearSolveHistoryPolicy,
        operator_family_id: str,
        /,
        *,
        constraint_id: str = "unconstrained",
        nullspace_policy_id: str = "none",
        solution_basis: ArrayLike | None = None,
        rhs_image_basis: ArrayLike | None = None,
        times: ArrayLike | None = None,
        effective_dimension: ArrayLike = 0,
        update_count: ArrayLike = 0,
    ):
        if not isinstance(source, AbstractVectorSpace) or not isinstance(
            target, AbstractVectorSpace
        ):
            raise TypeError("History spaces must be AbstractVectorSpace values.")
        if not isinstance(policy, LinearSolveHistoryPolicy):
            raise TypeError("policy must be LinearSolveHistoryPolicy.")
        identifiers = (
            str(operator_family_id),
            str(constraint_id),
            str(nullspace_policy_id),
        )
        if any(not value for value in identifiers):
            raise ValueError("History compatibility identities must be non-empty.")
        source_dtype = source.flatten(source.zeros()).dtype
        target_dtype = target.flatten(target.zeros()).dtype
        solutions = (
            jnp.zeros((source.size, policy.capacity), dtype=source_dtype)
            if solution_basis is None
            else jnp.asarray(solution_basis)
        )
        images = (
            jnp.zeros((target.size, policy.capacity), dtype=target_dtype)
            if rhs_image_basis is None
            else jnp.asarray(rhs_image_basis)
        )
        times_ = (
            jnp.full((policy.capacity,), jnp.nan, dtype=jnp.float64)
            if times is None
            else jnp.asarray(times)
        )
        effective = jnp.asarray(effective_dimension, dtype=jnp.int32)
        updates = jnp.asarray(update_count, dtype=jnp.int32)
        if solutions.shape != (source.size, policy.capacity):
            raise ValueError("Solution history basis has an invalid shape.")
        if images.shape != (target.size, policy.capacity):
            raise ValueError("RHS-image history basis has an invalid shape.")
        if (
            times_.shape != (policy.capacity,)
            or effective.shape != ()
            or updates.shape != ()
        ):
            raise ValueError("History time/counter layouts are invalid.")
        effective = eqx.error_if(
            effective,
            (effective < 0) | (effective > policy.capacity),
            "History effective dimension is out of bounds.",
        )
        self.source = source
        self.target = target
        self.solution_basis = solutions
        self.rhs_image_basis = images
        self.times = times_
        self.effective_dimension = effective
        self.update_count = updates
        self.operator_family_id, self.constraint_id, self.nullspace_policy_id = (
            identifiers
        )
        self.policy = policy
        self.history_id = canonical_fingerprint(
            {
                "kind": "linear-solve-history",
                "source": source.space_id,
                "target": target.space_id,
                "policy": policy.policy_id,
                "operator_family": identifiers[0],
                "constraint": identifiers[1],
                "nullspace": identifiers[2],
            }
        )

    @classmethod
    def empty(
        cls,
        operator: AbstractLinearOperator,
        policy: LinearSolveHistoryPolicy,
        operator_family_id: str,
        /,
        *,
        constraint_id: str = "unconstrained",
        nullspace_policy_id: str = "none",
    ) -> LinearSolveHistory:
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be AbstractLinearOperator.")
        return cls(
            operator.source,
            operator.target,
            policy,
            operator_family_id,
            constraint_id=constraint_id,
            nullspace_policy_id=nullspace_policy_id,
        )

    def compatible(
        self,
        operator: AbstractLinearOperator,
        operator_family_id: str,
        /,
        *,
        constraint_id: str = "unconstrained",
        nullspace_policy_id: str = "none",
    ) -> bool:
        return (
            isinstance(operator, AbstractLinearOperator)
            and self.source.compatible(operator.source)
            and self.target.compatible(operator.target)
            and self.operator_family_id == str(operator_family_id)
            and self.constraint_id == str(constraint_id)
            and self.nullspace_policy_id == str(nullspace_policy_id)
        )

    def _active_mask(self) -> Array:
        return jnp.arange(self.policy.capacity) < self.effective_dimension

    def initial_guess(
        self,
        rhs: PyTree[Any],
        /,
        *,
        time: ArrayLike | None = None,
    ) -> tuple[PyTree[Array], LinearInitialGuessDiagnostics]:
        rhs_ = self.target.validate(rhs)
        rhs_coordinates = self.target.flatten(rhs_)
        mask = self._active_mask()
        effective = self.effective_dimension
        zero = self.source.zeros()
        if self.policy.strategy == "zero":
            return zero, LinearInitialGuessDiagnostics(
                effective_dimension=effective,
                rank=jnp.asarray(0, dtype=jnp.int32),
                projection_residual_norm=self.target.norm(rhs_),
                reused=jnp.asarray(False),
                strategy=self.policy.strategy,
            )
        if self.policy.strategy == "last-solution":
            index = jnp.maximum(effective - 1, 0)
            coordinates = jnp.where(
                effective > 0,
                self.solution_basis[:, index],
                jnp.zeros((self.source.size,), dtype=self.solution_basis.dtype),
            )
            return self.source.unflatten(
                jax.lax.stop_gradient(coordinates)
            ), LinearInitialGuessDiagnostics(
                effective_dimension=effective,
                rank=jnp.minimum(effective, 1),
                projection_residual_norm=jnp.asarray(jnp.nan),
                reused=effective > 0,
                strategy=self.policy.strategy,
            )
        if self.policy.strategy == "stabilized-extrapolation":
            if time is None:
                raise ValueError("Extrapolation initial guesses require a target time.")
            degree_count = min(self.policy.extrapolation_degree + 1, self.policy.capacity)
            used = jnp.minimum(effective, degree_count)
            start = jnp.maximum(effective - degree_count, 0)
            recent_solutions = jnp.roll(self.solution_basis, -start, axis=1)[
                :, :degree_count
            ]
            recent_times = jnp.roll(self.times, -start)[:degree_count]
            valid = jnp.arange(degree_count) < used
            scale = jnp.maximum(
                jnp.max(jnp.where(valid, jnp.abs(recent_times), 0.0)), 1.0
            )
            nodes = recent_times / scale
            target = jnp.asarray(time) / scale
            powers = jnp.arange(degree_count)
            vandermonde = nodes[:, None] ** powers[None, :]
            system = jnp.where(valid[:, None], vandermonde, 0.0)
            system = system + jnp.diag((~valid).astype(system.dtype))
            target_values = target**powers
            coefficients = jnp.linalg.solve(system.T, target_values)
            coordinates = recent_solutions @ coefficients
            coordinates = jnp.where(used > 0, coordinates, 0.0)
            return self.source.unflatten(
                jax.lax.stop_gradient(coordinates)
            ), LinearInitialGuessDiagnostics(
                effective_dimension=effective,
                rank=used,
                projection_residual_norm=jnp.asarray(jnp.nan),
                reused=used > 0,
                strategy=self.policy.strategy,
            )
        active_images = jnp.where(mask[None, :], self.rhs_image_basis, 0.0)
        if self.policy.strategy == "rolling-qr":
            q_basis, upper = jnp.linalg.qr(active_images, mode="reduced")
            coefficients = jnp.linalg.pinv(upper, rtol=self.policy.rank_tolerance) @ (
                jnp.conj(q_basis.T) @ rhs_coordinates
            )
            coefficients = jnp.where(mask, coefficients, 0.0)
            coordinates = self.solution_basis @ coefficients
            projected = active_images @ coefficients
            residual = rhs_coordinates - projected
            singular_values = jnp.linalg.svd(upper, compute_uv=False)
            largest = jnp.maximum(jnp.max(singular_values), 1.0)
            rank = jnp.sum(
                singular_values > self.policy.rank_tolerance * largest,
                dtype=jnp.int32,
            )
            return self.source.unflatten(
                jax.lax.stop_gradient(coordinates)
            ), LinearInitialGuessDiagnostics(
                effective_dimension=effective,
                rank=rank,
                projection_residual_norm=jnp.sqrt(jnp.real(jnp.vdot(residual, residual))),
                reused=effective > 0,
                strategy=self.policy.strategy,
            )
        gram = jnp.conj(active_images.T) @ active_images
        rhs_projection = jnp.conj(active_images.T) @ rhs_coordinates
        inactive = (~mask).astype(gram.dtype)
        gram = gram + jnp.diag(inactive)
        eigenvalues = jnp.linalg.eigvalsh(0.5 * (gram + jnp.conj(gram.T)))
        largest = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
        threshold = self.policy.rank_tolerance * largest
        rank = jnp.sum((eigenvalues > threshold) & mask, dtype=jnp.int32)
        coefficients = (
            jnp.linalg.pinv(gram, rtol=self.policy.rank_tolerance) @ rhs_projection
        )
        coefficients = jnp.where(mask, coefficients, 0.0)
        coordinates = self.solution_basis @ coefficients
        projected = active_images @ coefficients
        residual = rhs_coordinates - projected
        return self.source.unflatten(
            jax.lax.stop_gradient(coordinates)
        ), LinearInitialGuessDiagnostics(
            effective_dimension=effective,
            rank=rank,
            projection_residual_norm=jnp.sqrt(jnp.real(jnp.vdot(residual, residual))),
            reused=effective > 0,
            strategy=self.policy.strategy,
        )

    def update(
        self,
        operator: AbstractLinearOperator,
        solution: PyTree[Any],
        /,
        *,
        rhs: PyTree[Any] | None = None,
        time: ArrayLike | None = None,
        accepted: ArrayLike = True,
    ) -> LinearSolveHistory:
        if (
            not isinstance(operator, AbstractLinearOperator)
            or not self.source.compatible(operator.source)
            or not self.target.compatible(operator.target)
        ):
            raise ValueError("History update operator is incompatible.")
        solution_ = self.source.validate(solution)
        image = operator.mv(solution_) if rhs is None else self.target.validate(rhs)
        solution_coordinates = jax.lax.stop_gradient(self.source.flatten(solution_))
        image_coordinates = jax.lax.stop_gradient(self.target.flatten(image))
        time_ = jnp.asarray(jnp.nan if time is None else time, dtype=self.times.dtype)
        accepted_ = jnp.asarray(accepted, dtype=bool)
        if accepted_.shape != () or time_.shape != ():
            raise ValueError("History acceptance/time must be scalar.")

        def perform(history):
            effective = history.effective_dimension
            full = effective >= history.policy.capacity
            solutions = jax.lax.cond(
                full,
                lambda value: jnp.concatenate(
                    (value[:, 1:], solution_coordinates[:, None]), axis=1
                ),
                lambda value: value.at[:, effective].set(solution_coordinates),
                history.solution_basis,
            )
            images = jax.lax.cond(
                full,
                lambda value: jnp.concatenate(
                    (value[:, 1:], image_coordinates[:, None]), axis=1
                ),
                lambda value: value.at[:, effective].set(image_coordinates),
                history.rhs_image_basis,
            )
            times = jax.lax.cond(
                full,
                lambda value: jnp.concatenate((value[1:], time_[None])),
                lambda value: value.at[effective].set(time_),
                history.times,
            )
            return LinearSolveHistory(
                history.source,
                history.target,
                history.policy,
                history.operator_family_id,
                constraint_id=history.constraint_id,
                nullspace_policy_id=history.nullspace_policy_id,
                solution_basis=solutions,
                rhs_image_basis=images,
                times=times,
                effective_dimension=jnp.minimum(effective + 1, history.policy.capacity),
                update_count=history.update_count + 1,
            )

        return jax.lax.cond(accepted_, perform, lambda history: history, self)


class HistoryLinearSolveResult(StrictModule):
    result: object
    history: LinearSolveHistory
    initial_guess_diagnostics: LinearInitialGuessDiagnostics

    def __init__(
        self,
        result: object,
        history: LinearSolveHistory,
        initial_guess_diagnostics: LinearInitialGuessDiagnostics,
        /,
    ):
        if not isinstance(history, LinearSolveHistory):
            raise TypeError("history must be LinearSolveHistory.")
        if not isinstance(initial_guess_diagnostics, LinearInitialGuessDiagnostics):
            raise TypeError(
                "initial_guess_diagnostics must be LinearInitialGuessDiagnostics."
            )
        self.result = result
        self.history = history
        self.initial_guess_diagnostics = initial_guess_diagnostics


def solve_with_history(
    problem_or_prepared: object,
    rhs: PyTree[Any],
    history: LinearSolveHistory,
    /,
    *,
    operator_family_id: str,
    constraint_id: str = "unconstrained",
    nullspace_policy_id: str = "none",
    time: ArrayLike | None = None,
    accepted: ArrayLike = True,
    policy: object = None,
    rhs_layout: object = None,
    control: object = None,
) -> HistoryLinearSolveResult:
    from ._plans import PreparedLinearSolve
    from ._problems import AbstractLinearProblem
    from ._runtime import solve

    if isinstance(problem_or_prepared, PreparedLinearSolve):
        operator = problem_or_prepared.problem.operator
    elif isinstance(problem_or_prepared, AbstractLinearProblem):
        operator = problem_or_prepared.operator
    else:
        raise TypeError("Expected an AbstractLinearProblem or PreparedLinearSolve.")
    if not isinstance(history, LinearSolveHistory) or not history.compatible(
        operator,
        operator_family_id,
        constraint_id=constraint_id,
        nullspace_policy_id=nullspace_policy_id,
    ):
        raise ValueError("Linear solve history is incompatible with this problem.")
    guess, diagnostics = history.initial_guess(rhs, time=time)
    result = solve(
        problem_or_prepared,
        rhs,
        policy=policy,
        rhs_layout=rhs_layout,
        initial_guess=guess,
        control=control,
    )
    accepted_ = jnp.asarray(accepted, dtype=bool) & result.successful
    updated = history.update(
        operator,
        result.value,
        time=time,
        accepted=accepted_,
    )
    return HistoryLinearSolveResult(result, updated, diagnostics)


__all__ = [
    "HistoryLinearSolveResult",
    "LinearInitialGuessDiagnostics",
    "LinearInitialGuessStrategy",
    "LinearSolveHistory",
    "LinearSolveHistoryPolicy",
    "solve_with_history",
]
