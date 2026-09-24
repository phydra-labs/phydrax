#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Untrusted initial-guess providers for linear and nonlinear solves."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, ClassVar, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._differentiation import ComponentAuthority
from .._fingerprint import canonical_fingerprint
from .._model import (
    AbstractArrayModel,
    AbstractComponentSlot,
    ComponentBinding,
    ComponentContract,
)
from .._model._component import slot_component_contracts
from .._trainable import NonTrainableState
from ._dense_pseudoinverse import apply_pseudoinverse, factor_pseudoinverse
from ._operators import AbstractLinearOperator
from ._policies import RankPolicy
from ._results import InitialGuessDiagnostics
from ._spaces import AbstractVectorSpace


HistoryInitialGuessStrategy: TypeAlias = Literal[
    "zero",
    "last-solution",
    "projection",
    "rolling-qr",
    "stabilized-extrapolation",
]


class AbstractInitialGuessProvider(AbstractComponentSlot):
    """Source of untrusted initial guesses for linear and nonlinear solves.

    The base is the neutral `ACCELERATOR` slot of solves: a provider may change
    how much work a solve performs but never the equation it solves.
    `propose(data, baseline)` maps the solve data (a linear right-hand side or
    nonlinear `args`) and the native baseline guess to a proposed solution. A
    production solve evaluates the proposal against that baseline on device and
    visibly keeps the baseline unless the proposal is valid with a strictly
    smaller residual (`InitialGuessDiagnostics`); the selected guess carries no
    derivative. Training differentiates the raw `propose` output through an
    algorithmic-work objective instead.
    """

    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.ACCELERATOR
    slot_semantic_id: ClassVar[str] = "phydrax.linalg.initial-guess"

    provider_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def propose(self, data: PyTree[Any], baseline: PyTree[Any], /) -> PyTree[Array]:
        """Return a proposed solution with the baseline's structure."""
        raise NotImplementedError


def _select_proposal(
    proposal: PyTree[Array],
    baseline: PyTree[Array],
    proposal_residual_norm: Array,
    baseline_residual_norm: Array,
    proposal_valid: Array,
    /,
    *,
    provider_id: str,
    baseline_valid: Array | bool = True,
) -> tuple[PyTree[Array], InitialGuessDiagnostics]:
    """Select a valid, strictly improving proposal over the native baseline.

    The owning solve supplies both residual norms and the proposal validity.
    Selection is elementwise over leading evidence axes. The selected guess and
    the evidence are stopped: no derivative flows through the branch or the
    proposal.
    """
    proposal_norm = jax.lax.stop_gradient(jnp.asarray(proposal_residual_norm))
    baseline_norm = jax.lax.stop_gradient(jnp.asarray(baseline_residual_norm))
    valid = jnp.asarray(proposal_valid, dtype=jnp.bool_) & jnp.isfinite(proposal_norm)
    baseline_ok = jnp.asarray(baseline_valid, dtype=jnp.bool_)
    accepted = valid & (~baseline_ok | (proposal_norm < baseline_norm))
    selected = jax.tree.map(
        lambda proposed, native: jax.lax.stop_gradient(
            jnp.where(accepted, proposed, native)
        ),
        proposal,
        baseline,
    )
    return selected, InitialGuessDiagnostics(
        proposal_residual_norm=proposal_norm,
        baseline_residual_norm=baseline_norm,
        proposal_valid=valid,
        accepted=accepted,
        provider_id=provider_id,
    )


class HistoryInitialGuess(AbstractInitialGuessProvider, NonTrainableState):
    """Accepted-solution history of one linear operator family.

    Strategies are `"zero"`, `"last-solution"`, `"projection"` (paired RHS-image
    projection minimizing the represented RHS residual), `"rolling-qr"`
    (fixed-capacity QR of the RHS images), and `"stabilized-extrapolation"`
    (polynomial extrapolation in accepted times to `at_time(time)`). The history
    records accepted solutions only through `update` and carries explicit
    operator-family, constraint, and nullspace identities; it is fixed data.
    """

    source: AbstractVectorSpace
    target: AbstractVectorSpace
    solution_basis: Array
    rhs_image_basis: Array
    times: Array
    effective_dimension: Array
    update_count: Array
    target_time: Array | None
    strategy: HistoryInitialGuessStrategy = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    extrapolation_degree: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    operator_family_id: str = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)
    nullspace_policy_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        operator_family_id: str,
        /,
        *,
        strategy: HistoryInitialGuessStrategy = "projection",
        capacity: int = 6,
        extrapolation_degree: int = 2,
        rank_tolerance: float = 1.0e-10,
        constraint_id: str = "unconstrained",
        nullspace_policy_id: str = "none",
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be AbstractLinearOperator.")
        strategy_ = str(strategy)
        capacity_ = int(capacity)
        degree = int(extrapolation_degree)
        tolerance = float(rank_tolerance)
        if strategy_ not in (
            "zero",
            "last-solution",
            "projection",
            "rolling-qr",
            "stabilized-extrapolation",
        ):
            raise ValueError("Unknown history initial-guess strategy.")
        if (
            capacity_ < 1
            or degree < 0
            or (strategy_ == "stabilized-extrapolation" and degree >= capacity_)
        ):
            raise ValueError("History capacity/degree are incompatible.")
        if not tolerance > 0.0:
            raise ValueError("History rank_tolerance must be positive.")
        identifiers = (
            str(operator_family_id),
            str(constraint_id),
            str(nullspace_policy_id),
        )
        if any(not value for value in identifiers):
            raise ValueError("History compatibility identities must be non-empty.")
        source = operator.source
        target = operator.target
        self.source = source
        self.target = target
        self.solution_basis = jnp.zeros(
            (source.size, capacity_), dtype=source.flatten(source.zeros()).dtype
        )
        self.rhs_image_basis = jnp.zeros(
            (target.size, capacity_), dtype=target.flatten(target.zeros()).dtype
        )
        self.times = jnp.full((capacity_,), jnp.nan, dtype=jnp.float64)
        self.effective_dimension = jnp.asarray(0, dtype=jnp.int32)
        self.update_count = jnp.asarray(0, dtype=jnp.int32)
        self.target_time = None
        self.strategy = strategy_
        self.capacity = capacity_
        self.extrapolation_degree = degree
        self.rank_tolerance = tolerance
        self.operator_family_id, self.constraint_id, self.nullspace_policy_id = (
            identifiers
        )
        self.provider_id = canonical_fingerprint(
            {
                "kind": "history-initial-guess",
                "source": source.space_id,
                "target": target.space_id,
                "strategy": strategy_,
                "capacity": capacity_,
                "extrapolation_degree": degree,
                "rank_tolerance": tolerance,
                "operator_family": identifiers[0],
                "constraint": identifiers[1],
                "nullspace": identifiers[2],
            }
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
        """Whether `operator` and the declared identities match this history."""
        return (
            isinstance(operator, AbstractLinearOperator)
            and self.source.compatible(operator.source)
            and self.target.compatible(operator.target)
            and self.operator_family_id == str(operator_family_id)
            and self.constraint_id == str(constraint_id)
            and self.nullspace_policy_id == str(nullspace_policy_id)
        )

    def at_time(self, time: ArrayLike, /) -> HistoryInitialGuess:
        """Return this history targeting `time` for stabilized extrapolation."""
        value = jnp.asarray(time, dtype=self.times.dtype)
        if value.shape != ():
            raise ValueError("History target time must be scalar.")
        return eqx.tree_at(
            lambda history: history.target_time,
            self,
            value,
            is_leaf=lambda node: node is None,
        )

    def propose(self, data: PyTree[Any], baseline: PyTree[Any], /) -> PyTree[Array]:
        """Propose a solution for right-hand side `data` from accepted history."""
        rhs_coordinates = self.target.flatten(self.target.validate(data))
        self.source.validate(baseline)
        match self.strategy:
            case "zero":
                coordinates = jnp.zeros(
                    (self.source.size,), dtype=self.solution_basis.dtype
                )
            case "last-solution":
                coordinates = self._last_solution()
            case "stabilized-extrapolation":
                coordinates = self._extrapolated()
            case "rolling-qr":
                coordinates = self._rolling_qr(rhs_coordinates)
            case "projection":
                coordinates = self._projection(rhs_coordinates)
            case strategy:
                raise ValueError(f"Unknown history initial-guess strategy {strategy!r}.")
        return self.source.unflatten(jax.lax.stop_gradient(coordinates))

    def _active_mask(self) -> Array:
        return jnp.arange(self.capacity) < self.effective_dimension

    def _last_solution(self) -> Array:
        effective = self.effective_dimension
        index = jnp.maximum(effective - 1, 0)
        return jnp.where(
            effective > 0,
            self.solution_basis[:, index],
            jnp.zeros((self.source.size,), dtype=self.solution_basis.dtype),
        )

    def _extrapolated(self) -> Array:
        if self.target_time is None:
            raise ValueError(
                "Extrapolation initial guesses require a target time; use at_time."
            )
        effective = self.effective_dimension
        degree_count = min(self.extrapolation_degree + 1, self.capacity)
        used = jnp.minimum(effective, degree_count)
        start = jnp.maximum(effective - degree_count, 0)
        recent_solutions = jnp.roll(self.solution_basis, -start, axis=1)[:, :degree_count]
        recent_times = jnp.roll(self.times, -start)[:degree_count]
        valid = jnp.arange(degree_count) < used
        scale = jnp.maximum(jnp.max(jnp.where(valid, jnp.abs(recent_times), 0.0)), 1.0)
        nodes = recent_times / scale
        target = self.target_time / scale
        powers = jnp.arange(degree_count)
        vandermonde = nodes[:, None] ** powers[None, :]
        system = jnp.where(valid[:, None], vandermonde, 0.0)
        system = system + jnp.diag((~valid).astype(system.dtype))
        coefficients = jnp.linalg.solve(system.T, target**powers)
        coordinates = recent_solutions @ coefficients
        return jnp.where(used > 0, coordinates, 0.0)

    def _rolling_qr(self, rhs_coordinates: Array) -> Array:
        mask = self._active_mask()
        active_images = jnp.where(mask[None, :], self.rhs_image_basis, 0.0)
        q_basis, upper = jnp.linalg.qr(active_images, mode="reduced")
        factors = factor_pseudoinverse(
            upper,
            RankPolicy(relative_cutoff=self.rank_tolerance),
        )
        coefficients = apply_pseudoinverse(
            factors,
            jnp.conj(q_basis.T) @ rhs_coordinates,
        )
        return self.solution_basis @ jnp.where(mask, coefficients, 0.0)

    def _projection(self, rhs_coordinates: Array) -> Array:
        mask = self._active_mask()
        active_images = jnp.where(mask[None, :], self.rhs_image_basis, 0.0)
        gram = jnp.conj(active_images.T) @ active_images
        gram = gram + jnp.diag((~mask).astype(gram.dtype))
        factors = factor_pseudoinverse(
            gram,
            RankPolicy(relative_cutoff=self.rank_tolerance),
            hermitian=True,
        )
        coefficients = apply_pseudoinverse(
            factors, jnp.conj(active_images.T) @ rhs_coordinates
        )
        return self.solution_basis @ jnp.where(mask, coefficients, 0.0)

    def update(
        self,
        operator: AbstractLinearOperator,
        solution: PyTree[Any],
        /,
        *,
        rhs: PyTree[Any] | None = None,
        time: ArrayLike | None = None,
        accepted: ArrayLike = True,
    ) -> HistoryInitialGuess:
        """Record one accepted solution; a rejected update is bitwise inert."""
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
        accepted_ = jnp.asarray(accepted, dtype=jnp.bool_)
        if accepted_.shape != () or time_.shape != ():
            raise ValueError("History acceptance/time must be scalar.")
        effective = self.effective_dimension
        full = effective >= self.capacity

        def append(values, entry):
            return jax.lax.cond(
                full,
                lambda current: jnp.concatenate(
                    (current[..., 1:], entry[..., None]), axis=-1
                ),
                lambda current: current.at[..., effective].set(entry),
                values,
            )

        def record(history):
            return eqx.tree_at(
                lambda value: (
                    value.solution_basis,
                    value.rhs_image_basis,
                    value.times,
                    value.effective_dimension,
                    value.update_count,
                ),
                history,
                (
                    append(history.solution_basis, solution_coordinates),
                    append(history.rhs_image_basis, image_coordinates),
                    append(history.times, time_),
                    jnp.minimum(effective + 1, self.capacity).astype(jnp.int32),
                    history.update_count + 1,
                ),
            )

        return jax.lax.cond(accepted_, record, lambda history: history, self)


def _is_component(node: Any, /) -> bool:
    return isinstance(node, (AbstractArrayModel, ComponentBinding))


class LearnedInitialGuess(AbstractInitialGuessProvider):
    """Model-backed initial-guess proposal.

    `function(data, baseline)` returns a proposed solution. It is a callable
    module holding at least one model as a dynamic child, whose arrays keep
    their roles; every `AbstractArrayModel` inside it is bound to this
    `ACCELERATOR` slot, and a `ComponentBinding` inside it must carry the same
    authority and slot. A bare array model maps model inputs, not
    `(data, baseline)`, so it enters through a callable module that defines the
    proposal. The proposal is never trusted by a production solve.
    """

    function: Callable[[PyTree[Any], PyTree[Any]], PyTree[Any]]
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[PyTree[Any], PyTree[Any]], PyTree[Any]],
        /,
        *,
        provider_id: str = "learned-initial-guess",
    ):
        if _is_component(function):
            raise TypeError(
                "function must map (data, baseline) to a proposed solution; hold an "
                "array model in a callable module that defines the proposal."
            )
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(provider_id)
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        if not slot_component_contracts(type(self), function, scope="function"):
            raise ValueError(
                "LearnedInitialGuess requires at least one model component in function."
            )
        self.function = function
        self.provider_id = identifier

    def component_contracts(self) -> tuple[tuple[str, ComponentContract], ...]:
        """Return `(location, contract)` of every model bound to this slot."""
        return slot_component_contracts(type(self), self.function, scope="function")

    def propose(self, data: PyTree[Any], baseline: PyTree[Any], /) -> PyTree[Array]:
        return self.function(data, baseline)


__all__ = [
    "AbstractInitialGuessProvider",
    "HistoryInitialGuess",
    "HistoryInitialGuessStrategy",
    "LearnedInitialGuess",
]
