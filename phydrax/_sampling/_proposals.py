#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._strict import AbstractAttribute, StrictModule


class AbstractProposal(StrictModule):
    """Normalized proposal density over a structure-preserving position PyTree."""

    proposal_id: AbstractAttribute[str]

    @abstractmethod
    def sample(self, key: Key[Array, ""], current: PyTree[Any], /) -> PyTree[Array]:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, proposed: PyTree[Any], current: PyTree[Any], /) -> Array:
        """Return ``log q(proposed | current)`` as one real scalar."""
        raise NotImplementedError


class GaussianRandomWalkProposal(AbstractProposal):
    """Isotropic Gaussian random walk over every inexact position leaf."""

    scale: float = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(self, scale: float, /, *, proposal_id: str = "gaussian-random-walk"):
        value = float(scale)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("scale must be finite and positive.")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("proposal_id must be a non-empty string.")
        self.scale = value
        self.proposal_id = proposal_id

    def sample(self, key, current, /) -> PyTree[Array]:
        leaves, structure = jax.tree_util.tree_flatten(current)
        if not leaves or any(not eqx.is_inexact_array(leaf) for leaf in leaves):
            raise TypeError("Positions must be a non-empty PyTree of inexact arrays.")
        keys = jr.split(key, len(leaves))
        return structure.unflatten(
            jnp.asarray(leaf)
            + self.scale * jr.normal(leaf_key, leaf.shape, dtype=jnp.asarray(leaf).dtype)
            for leaf, leaf_key in zip(leaves, keys, strict=True)
        )

    def log_prob(self, proposed, current, /) -> Array:
        proposed_leaves, proposed_structure = jax.tree_util.tree_flatten(proposed)
        current_leaves, current_structure = jax.tree_util.tree_flatten(current)
        if proposed_structure != current_structure:
            raise ValueError("Proposed and current position structures must agree.")
        if not current_leaves:
            raise ValueError("Positions must contain at least one array leaf.")
        dimension = sum(int(jnp.asarray(leaf).size) for leaf in current_leaves)
        squared = sum(
            (
                jnp.sum(
                    jnp.abs(
                        (jnp.asarray(proposed_leaf) - jnp.asarray(current_leaf))
                        / self.scale
                    )
                    ** 2
                )
                for proposed_leaf, current_leaf in zip(
                    proposed_leaves, current_leaves, strict=True
                )
            ),
            jnp.zeros(()),
        )
        return -0.5 * (squared + dimension * jnp.log(2.0 * jnp.pi * self.scale**2))


class CallableProposal(AbstractProposal):
    """Normalized user-defined structure-preserving proposal."""

    sample_fn: Callable[[Array, PyTree[Any]], PyTree[Any]] = eqx.field(static=True)
    log_prob_fn: Callable[[PyTree[Any], PyTree[Any]], ArrayLike] = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample: Callable[[Array, PyTree[Any]], PyTree[Any]],
        log_prob: Callable[[PyTree[Any], PyTree[Any]], ArrayLike],
        /,
        *,
        proposal_id: str,
    ):
        if not callable(sample) or not callable(log_prob):
            raise TypeError("sample and log_prob must be callable.")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("proposal_id must be a non-empty string.")
        self.sample_fn = sample
        self.log_prob_fn = log_prob
        self.proposal_id = proposal_id

    def sample(self, key, current, /) -> PyTree[Array]:
        proposed = self.sample_fn(key, current)
        if jax.tree_util.tree_structure(proposed) != jax.tree_util.tree_structure(
            current
        ):
            raise ValueError("Proposal must preserve the position PyTree structure.")
        result = jax.tree_util.tree_map(jnp.asarray, proposed)
        for proposed_leaf, current_leaf in zip(
            jax.tree_util.tree_leaves(result),
            jax.tree_util.tree_leaves(current),
            strict=True,
        ):
            current_array = jnp.asarray(current_leaf)
            if proposed_leaf.shape != current_array.shape:
                raise ValueError("Proposal must preserve every position leaf shape.")
            if proposed_leaf.dtype != current_array.dtype:
                raise TypeError("Proposal must preserve every position leaf dtype.")
        return result

    def log_prob(self, proposed, current, /) -> Array:
        value = jnp.asarray(self.log_prob_fn(proposed, current))
        if jnp.iscomplexobj(value):
            raise TypeError("Proposal log probabilities must be real-valued.")
        if value.shape != ():
            raise ValueError("Proposal log probabilities must be scalar.")
        return value.reshape(())


__all__ = [
    "AbstractProposal",
    "CallableProposal",
    "GaussianRandomWalkProposal",
]
