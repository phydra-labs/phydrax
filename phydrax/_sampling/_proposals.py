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

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule


class ProposalMove(StrictModule):
    """One normalized proposal with fixed-shape local-update payload evidence."""

    position: PyTree[Array]
    log_forward: Array
    log_reverse: Array
    payload: PyTree[Array]
    valid: Array


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

    @abstractmethod
    def payload(
        self,
        key: Key[Array, ""],
        current: PyTree[Any],
        proposed: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        """Return fixed-shape local-update evidence for the proposed position."""
        raise NotImplementedError

    def propose(self, key: Key[Array, ""], current: PyTree[Any], /) -> ProposalMove:
        """Materialize one complete Hastings move without capability probing."""
        proposed = self.sample(key, current)
        forward = jnp.asarray(self.log_prob(proposed, current))
        reverse = jnp.asarray(self.log_prob(current, proposed))
        valid = (
            jnp.isfinite(forward)
            & jnp.isfinite(reverse)
            & jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(jnp.asarray(leaf)))
                        for leaf in jax.tree_util.tree_leaves(proposed)
                    ]
                )
            )
        )
        return ProposalMove(
            position=proposed,
            log_forward=forward,
            log_reverse=reverse,
            payload=self.payload(key, current, proposed),
            valid=valid,
        )


class SingleCoordinateProposalPayload(StrictModule):
    """Selected coordinate and displacement for one local array proposal."""

    index: Array
    displacement: Array


def _real_position(value: PyTree[Any], role: str, /) -> Array:
    leaves = jax.tree_util.tree_leaves(value)
    if len(leaves) != 1 or jax.tree_util.tree_structure(value) != jax.tree.structure(
        leaves[0]
    ):
        raise TypeError(f"{role} must be one real inexact array.")
    array = jnp.asarray(leaves[0])
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{role} must be one real inexact array.")
    if array.size == 0:
        raise ValueError(f"{role} must be non-empty.")
    return array


def _single_coordinate_log_probability(
    proposed: Array,
    current: Array,
    /,
    *,
    displacement: Array,
    scalar_log_density: Array,
) -> Array:
    if proposed.shape != current.shape or proposed.dtype != current.dtype:
        raise ValueError("Proposed and current coordinate arrays must agree.")
    changed_count = jnp.sum(jnp.ravel(proposed != current), dtype=jnp.int32)
    log_coordinate = -jnp.log(jnp.asarray(current.size, dtype=current.dtype))
    no_change = changed_count == 0
    one_change = changed_count == 1
    return jnp.where(
        no_change,
        scalar_log_density,
        jnp.where(one_change, log_coordinate + scalar_log_density, -jnp.inf),
    )


class SingleCoordinateGaussianProposal(AbstractProposal):
    """Normalized Gaussian random walk on one uniformly selected array coordinate."""

    scale: float = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(self, scale: float, /, *, proposal_id: str | None = None):
        value = float(scale)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("scale must be finite and positive.")
        self.scale = value
        self.proposal_id = (
            canonical_fingerprint({"kind": "single-coordinate-gaussian", "scale": value})
            if proposal_id is None
            else str(proposal_id)
        )
        if not self.proposal_id:
            raise ValueError("proposal_id must be non-empty.")

    def sample(self, key, current, /) -> Array:
        array = _real_position(current, "current")
        index_key, displacement_key = jr.split(key)
        index = jr.randint(index_key, (), 0, array.size)
        displacement = self.scale * jr.normal(displacement_key, (), dtype=array.dtype)
        return jnp.ravel(array).at[index].add(displacement).reshape(array.shape)

    def log_prob(self, proposed, current, /) -> Array:
        proposed_array = _real_position(proposed, "proposed")
        current_array = _real_position(current, "current")
        difference = jnp.ravel(proposed_array - current_array)
        displacement = jnp.sum(difference)
        scalar = -0.5 * (
            (displacement / self.scale) ** 2 + jnp.log(2.0 * jnp.pi * self.scale**2)
        )
        return _single_coordinate_log_probability(
            proposed_array,
            current_array,
            displacement=displacement,
            scalar_log_density=scalar,
        )

    def payload(self, key, current, proposed, /) -> SingleCoordinateProposalPayload:
        current_array = _real_position(current, "current")
        proposed_array = _real_position(proposed, "proposed")
        index_key, _ = jr.split(key)
        index = jr.randint(index_key, (), 0, current_array.size)
        displacement = jnp.ravel(proposed_array - current_array)[index]
        return SingleCoordinateProposalPayload(index=index, displacement=displacement)


class SingleCoordinatePeriodicProposal(AbstractProposal):
    """Uniform local proposal on one coordinate of a flat torus."""

    period: float = eqx.field(static=True)
    half_width: float = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        period: float,
        half_width: float,
        /,
        *,
        proposal_id: str | None = None,
    ):
        period_, half_width_ = float(period), float(half_width)
        if not np.isfinite(period_) or period_ <= 0.0:
            raise ValueError("period must be finite and positive.")
        if (
            not np.isfinite(half_width_)
            or half_width_ <= 0.0
            or half_width_ > 0.5 * period_
        ):
            raise ValueError("half_width must lie in (0, period / 2].")
        self.period = period_
        self.half_width = half_width_
        self.proposal_id = (
            canonical_fingerprint(
                {
                    "kind": "single-coordinate-periodic",
                    "period": period_,
                    "half_width": half_width_,
                }
            )
            if proposal_id is None
            else str(proposal_id)
        )
        if not self.proposal_id:
            raise ValueError("proposal_id must be non-empty.")

    def _wrap(self, value: Array, /) -> Array:
        return jnp.mod(value + 0.5 * self.period, self.period) - 0.5 * self.period

    def sample(self, key, current, /) -> Array:
        array = _real_position(current, "current")
        index_key, displacement_key = jr.split(key)
        index = jr.randint(index_key, (), 0, array.size)
        displacement = jr.uniform(
            displacement_key,
            (),
            minval=-self.half_width,
            maxval=self.half_width,
            dtype=array.dtype,
        )
        flat = jnp.ravel(array).at[index].add(displacement)
        return self._wrap(flat).reshape(array.shape)

    def log_prob(self, proposed, current, /) -> Array:
        proposed_array = _real_position(proposed, "proposed")
        current_array = _real_position(current, "current")
        difference = self._wrap(proposed_array - current_array)
        displacement = jnp.sum(jnp.ravel(difference))
        scalar = -jnp.log(jnp.asarray(2.0 * self.half_width, dtype=current_array.dtype))
        support = jnp.abs(displacement) <= self.half_width
        return jnp.where(
            support,
            _single_coordinate_log_probability(
                proposed_array,
                current_array,
                displacement=displacement,
                scalar_log_density=scalar,
            ),
            -jnp.inf,
        )

    def payload(self, key, current, proposed, /) -> SingleCoordinateProposalPayload:
        current_array = _real_position(current, "current")
        proposed_array = _real_position(proposed, "proposed")
        index_key, _ = jr.split(key)
        index = jr.randint(index_key, (), 0, current_array.size)
        displacement = self._wrap(jnp.ravel(proposed_array - current_array)[index])
        return SingleCoordinateProposalPayload(index=index, displacement=displacement)


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

    def payload(self, key, current, proposed, /):
        del key, current, proposed
        return ()


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

    def payload(self, key, current, proposed, /):
        del key, current, proposed
        return ()


__all__ = [
    "AbstractProposal",
    "CallableProposal",
    "GaussianRandomWalkProposal",
    "ProposalMove",
    "SingleCoordinateGaussianProposal",
    "SingleCoordinatePeriodicProposal",
    "SingleCoordinateProposalPayload",
]
