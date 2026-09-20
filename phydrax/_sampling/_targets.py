#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit full and incremental target protocols for one Markov lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule


def _tree_all_finite(tree: PyTree[Any], /) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves])
    )


def _cache_matches(exact: PyTree[Any], cached: PyTree[Any], tolerance: float, /) -> Array:
    exact_leaves, exact_structure = jax.tree_util.tree_flatten(exact)
    cached_leaves, cached_structure = jax.tree_util.tree_flatten(cached)
    if exact_structure != cached_structure:
        raise ValueError("Refreshed and cached target structures must agree.")
    matches = []
    for exact_leaf, cached_leaf in zip(exact_leaves, cached_leaves, strict=True):
        exact_array, cached_array = jnp.asarray(exact_leaf), jnp.asarray(cached_leaf)
        if exact_array.shape != cached_array.shape:
            raise ValueError("Refreshed and cached target leaf shapes must agree.")
        if exact_array.dtype == jnp.bool_ or cached_array.dtype == jnp.bool_:
            matches.append(jnp.all(exact_array == cached_array))
        else:
            matches.append(jnp.all(jnp.abs(exact_array - cached_array) <= tolerance))
    return jnp.all(jnp.stack(matches)) if matches else jnp.asarray(True)


class MarkovTargetState(StrictModule):
    position: PyTree[Array]
    log_target: Array
    cache: PyTree[Array]
    valid: Array


class IncrementalTargetProposal(StrictModule):
    log_ratio: Array
    proposed_cache: PyTree[Array]
    valid: Array


class FullMarkovTarget(StrictModule):
    """Complete target evaluation with a statically declared identity."""

    evaluate: Callable[[PyTree[Any]], Array]
    target_id: str = eqx.field(static=True)

    def __init__(self, evaluate: Callable[[PyTree[Any]], Array], /, *, target_id: str):
        if not callable(evaluate):
            raise TypeError("evaluate must be callable.")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be nonempty.")
        self.evaluate = (
            evaluate if isinstance(evaluate, eqx.Partial) else eqx.Partial(evaluate)
        )
        self.target_id = target_id

    def initialize(self, position: PyTree[Any], /) -> MarkovTargetState:
        value = jnp.asarray(self.evaluate(position))
        if value.shape != ():
            raise ValueError("A Markov target must return one scalar.")
        if jnp.iscomplexobj(value):
            raise TypeError("A Markov target must be real-valued.")
        return MarkovTargetState(
            position=position,
            log_target=value,
            cache=(),
            valid=jnp.isfinite(value) & _tree_all_finite(position),
        )

    def propose(
        self, state: MarkovTargetState, proposed_position: PyTree[Any], payload=(), /
    ) -> IncrementalTargetProposal:
        del payload
        proposed = jnp.asarray(self.evaluate(proposed_position))
        if proposed.shape != ():
            raise ValueError("A Markov target must return one scalar.")
        if jnp.iscomplexobj(proposed):
            raise TypeError("A Markov target must be real-valued.")
        return IncrementalTargetProposal(
            log_ratio=proposed - state.log_target,
            proposed_cache=(),
            valid=jnp.isfinite(proposed) & _tree_all_finite(proposed_position),
        )

    def commit(
        self,
        state: MarkovTargetState,
        proposed_position,
        proposal: IncrementalTargetProposal,
        accepted: Array,
        /,
    ) -> MarkovTargetState:
        position = jax.tree_util.tree_map(
            lambda proposed, current: jnp.where(accepted, proposed, current),
            proposed_position,
            state.position,
        )
        value = state.log_target + jnp.where(accepted, proposal.log_ratio, 0.0)
        return MarkovTargetState(
            position=position,
            log_target=value,
            cache=(),
            valid=(
                state.valid
                & jnp.where(accepted, proposal.valid, True)
                & jnp.isfinite(value)
                & _tree_all_finite(position)
            ),
        )


class IncrementalMarkovTarget(StrictModule):
    """Explicit local target/cache functions; no runtime method discovery."""

    initialize_fn: Callable[[PyTree[Any]], tuple[Array, PyTree[Any]]]
    propose_fn: Callable[
        [PyTree[Any], PyTree[Any], PyTree[Any], PyTree[Any]],
        tuple[Array, PyTree[Any], Array],
    ]
    select_fn: Callable[[PyTree[Any], PyTree[Any], Array], PyTree[Any]]
    refresh_fn: Callable[[PyTree[Any]], tuple[Array, PyTree[Any]]]
    refresh_validate_fn: Callable[[PyTree[Any], PyTree[Any]], Array] | None
    target_id: str = eqx.field(static=True)
    refresh_cadence: int = eqx.field(static=True)
    cache_tolerance: float = eqx.field(static=True)
    maximum_chains: int | None = eqx.field(static=True)
    cache_bytes_per_chain: int = eqx.field(static=True)
    workspace_bytes_per_chain: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        initialize: Callable,
        propose: Callable,
        select: Callable,
        refresh: Callable,
        refresh_validate: Callable | None = None,
        target_id: str,
        refresh_cadence: int,
        cache_tolerance: float = 1e-8,
        maximum_chains: int | None = None,
        cache_bytes_per_chain: int = 0,
        workspace_bytes_per_chain: int = 0,
    ):
        if not all(callable(value) for value in (initialize, propose, select, refresh)):
            raise TypeError("All incremental target functions must be callable.")
        if refresh_validate is not None and not callable(refresh_validate):
            raise TypeError("refresh_validate must be callable or None.")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be nonempty.")
        cadence, tolerance = int(refresh_cadence), float(cache_tolerance)
        if cadence <= 0 or tolerance < 0.0:
            raise ValueError(
                "refresh_cadence must be positive and cache_tolerance non-negative."
            )
        if maximum_chains is None:
            maximum_chains_ = None
        else:
            maximum_chains_ = int(maximum_chains)
            if maximum_chains_ < 1:
                raise ValueError("maximum_chains must be positive or None.")
        cache_bytes = int(cache_bytes_per_chain)
        workspace_bytes = int(workspace_bytes_per_chain)
        if cache_bytes < 0 or workspace_bytes < 0:
            raise ValueError("Incremental target byte estimates must be non-negative.")
        self.initialize_fn = (
            initialize if isinstance(initialize, eqx.Partial) else eqx.Partial(initialize)
        )
        self.propose_fn = (
            propose if isinstance(propose, eqx.Partial) else eqx.Partial(propose)
        )
        self.select_fn = (
            select if isinstance(select, eqx.Partial) else eqx.Partial(select)
        )
        self.refresh_fn = (
            refresh if isinstance(refresh, eqx.Partial) else eqx.Partial(refresh)
        )
        self.refresh_validate_fn = (
            None
            if refresh_validate is None
            else (
                refresh_validate
                if isinstance(refresh_validate, eqx.Partial)
                else eqx.Partial(refresh_validate)
            )
        )
        self.target_id = target_id
        self.refresh_cadence = cadence
        self.cache_tolerance = tolerance
        self.maximum_chains = maximum_chains_
        self.cache_bytes_per_chain = cache_bytes
        self.workspace_bytes_per_chain = workspace_bytes

    def initialize(self, position: PyTree[Any], /) -> MarkovTargetState:
        value, cache = self.initialize_fn(position)
        scalar = jnp.asarray(value)
        if scalar.shape != () or jnp.iscomplexobj(scalar):
            raise ValueError(
                "Incremental target initialization must return a real scalar."
            )
        return MarkovTargetState(
            position=position,
            log_target=scalar,
            cache=cache,
            valid=(
                jnp.isfinite(scalar)
                & _tree_all_finite(position)
                & _tree_all_finite(cache)
            ),
        )

    def propose(
        self, state: MarkovTargetState, proposed_position, payload, /
    ) -> IncrementalTargetProposal:
        ratio, cache, valid = self.propose_fn(
            state.position, state.cache, proposed_position, payload
        )
        ratio_ = jnp.asarray(ratio)
        if ratio_.shape != () or jnp.iscomplexobj(ratio_):
            raise ValueError("Incremental target ratios must be real scalars.")
        return IncrementalTargetProposal(
            log_ratio=ratio_,
            proposed_cache=cache,
            valid=(
                jnp.asarray(valid, dtype=jnp.bool_)
                & jnp.isfinite(ratio_)
                & _tree_all_finite(cache)
                & _tree_all_finite(proposed_position)
            ),
        )

    def commit(
        self,
        state: MarkovTargetState,
        proposed_position,
        proposal: IncrementalTargetProposal,
        accepted: Array,
        /,
    ) -> MarkovTargetState:
        position = self.select_fn(state.position, proposed_position, accepted)
        cache = self.select_fn(state.cache, proposal.proposed_cache, accepted)
        value = state.log_target + jnp.where(accepted, proposal.log_ratio, 0.0)
        return MarkovTargetState(
            position=position,
            log_target=value,
            cache=cache,
            valid=(
                state.valid
                & jnp.where(accepted, proposal.valid, True)
                & jnp.isfinite(value)
                & _tree_all_finite(cache)
                & _tree_all_finite(position)
            ),
        )

    def refresh(self, state: MarkovTargetState, /) -> MarkovTargetState:
        value, cache = self.refresh_fn(state.position)
        scalar = jnp.asarray(value)
        if scalar.shape != () or jnp.iscomplexobj(scalar):
            raise ValueError("Incremental target refreshes must return real scalars.")
        residual = jnp.abs(scalar - state.log_target)
        if self.refresh_validate_fn is None:
            cache_valid = _cache_matches(cache, state.cache, self.cache_tolerance)
        else:
            cache_valid = jnp.asarray(
                self.refresh_validate_fn(state.cache, cache),
                dtype=jnp.bool_,
            )
            if cache_valid.shape != ():
                raise ValueError(
                    "Incremental target refresh validation must return one scalar."
                )
        valid = (
            state.valid
            & jnp.isfinite(scalar)
            & _tree_all_finite(cache)
            & _tree_all_finite(state.position)
            & (residual <= self.cache_tolerance)
            & cache_valid
        )
        return MarkovTargetState(
            position=state.position, log_target=scalar, cache=cache, valid=valid
        )


__all__ = [
    "FullMarkovTarget",
    "IncrementalMarkovTarget",
    "IncrementalTargetProposal",
    "MarkovTargetState",
]
