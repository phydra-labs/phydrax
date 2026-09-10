#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key

from .._frozendict import frozendict
from .._strict import StrictModule
from ..domain import DomainFunction
from ..nn.parameters import ParameterSubspace
from ._functional_solver import FunctionalSolver


Optimizer = optax.GradientTransformation | optax.GradientTransformationExtraArgs


def _tree_finite(tree: Any, /) -> Array:
    values = tuple(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(tree)
        if eqx.is_inexact_array(leaf)
    )
    return jnp.all(jnp.stack(values)) if values else jnp.asarray(True)


def _functional_update(
    solver: FunctionalSolver,
    optimizer: Optimizer,
    subspace: ParameterSubspace,
    parameters: Any,
    optimizer_state: Any,
    key: Key[Array, ""],
    step: Array,
    /,
):
    def loss_fn(values):
        functions = subspace.reconstruct(values)
        bound_solver = eqx.tree_at(
            lambda value: value.functions,
            solver,
            functions,
        )
        return bound_solver.loss(key=key, step=step)

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(parameters)
    updates, candidate_optimizer_state = optimizer.update(
        gradient,
        optimizer_state,
        parameters,
    )
    candidate = optax.apply_updates(parameters, updates)
    finite = jnp.isfinite(loss) & _tree_finite(candidate)
    accepted_parameters = jax.tree_util.tree_map(
        lambda new, old: jnp.where(finite, new, old) if eqx.is_array(new) else old,
        candidate,
        parameters,
    )
    accepted_state = jax.tree_util.tree_map(
        lambda new, old: jnp.where(finite, new, old) if eqx.is_array(new) else old,
        candidate_optimizer_state,
        optimizer_state,
    )
    return accepted_parameters, accepted_state, loss, finite


_compiled_functional_update = eqx.filter_jit(_functional_update)


class FunctionalUpdateState(StrictModule):
    """Canonical functions and backend state after accepted parameter updates."""

    functions: frozendict[str, DomainFunction]
    optimizer_state: Any
    step: Array

    def __init__(
        self,
        functions: Mapping[str, DomainFunction],
        optimizer_state: Any,
        step: Any = 0,
    ):
        if isinstance(step, int) and step < 0:
            raise ValueError("step must be non-negative.")
        step_ = jnp.asarray(step, dtype=jnp.int32)
        if step_.shape != ():
            raise ValueError("step must be a scalar.")
        self.functions = frozendict(functions)
        self.optimizer_state = optimizer_state
        self.step = step_


class FunctionalUpdateEvidence(StrictModule):
    loss: Array
    accepted: Array

    def __init__(self, loss: Array, accepted: Array, /):
        self.loss = jnp.asarray(loss).reshape(())
        self.accepted = jnp.asarray(accepted, dtype=bool).reshape(())


class FunctionalUpdateKernel(StrictModule):
    """Reusable exact-parameter-subspace Optax update over a FunctionalSolver."""

    solver: FunctionalSolver
    optimizer: Optimizer
    leaf_paths: tuple[str, ...] = eqx.field(static=True)
    jit: bool = eqx.field(static=True)

    def __init__(
        self,
        solver: FunctionalSolver,
        optimizer: Optimizer,
        leaf_paths: Sequence[str],
        /,
        *,
        jit: bool = True,
    ):
        if not isinstance(solver, FunctionalSolver):
            raise TypeError("solver must be a FunctionalSolver.")
        paths = tuple(str(path) for path in leaf_paths)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("leaf_paths must contain distinct parameter paths.")
        ParameterSubspace.from_leaf_paths(solver.functions, paths)
        self.solver = solver
        self.optimizer = optimizer
        self.leaf_paths = paths
        self.jit = bool(jit)

    @classmethod
    def from_subspace(
        cls,
        solver: FunctionalSolver,
        optimizer: Optimizer,
        subspace: ParameterSubspace,
        /,
        *,
        jit: bool = True,
    ) -> FunctionalUpdateKernel:
        if not isinstance(subspace, ParameterSubspace):
            raise TypeError("subspace must be a ParameterSubspace.")
        subspace.validate_root(solver.functions)
        return cls(
            solver,
            optimizer,
            subspace.leaf_paths,
            jit=jit,
        )

    def initialize(
        self,
        functions: Mapping[str, DomainFunction] | None = None,
        /,
    ) -> FunctionalUpdateState:
        bound = self.solver.functions if functions is None else frozendict(functions)
        subspace = ParameterSubspace.from_leaf_paths(bound, self.leaf_paths)
        return FunctionalUpdateState(
            bound,
            self.optimizer.init(subspace.initial),
            0,
        )

    def advance(
        self,
        state: FunctionalUpdateState,
        /,
        *,
        key: Key[Array, ""],
    ) -> tuple[FunctionalUpdateState, FunctionalUpdateEvidence]:
        if not isinstance(state, FunctionalUpdateState):
            raise TypeError("state must be a FunctionalUpdateState.")
        subspace = ParameterSubspace.from_leaf_paths(
            state.functions,
            self.leaf_paths,
        )
        selected = subspace.initial

        update_fn = _compiled_functional_update if self.jit else _functional_update
        parameters, optimizer_state, loss, accepted = update_fn(
            self.solver,
            self.optimizer,
            subspace,
            selected,
            state.optimizer_state,
            key,
            state.step,
        )
        functions = subspace.reconstruct(parameters)
        next_state = FunctionalUpdateState(
            functions,
            optimizer_state,
            state.step + accepted.astype(jnp.int32),
        )
        return next_state, FunctionalUpdateEvidence(loss, accepted)


__all__ = [
    "FunctionalUpdateEvidence",
    "FunctionalUpdateKernel",
    "FunctionalUpdateState",
]
