#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._probability import AbstractProbabilityLaw
from .._strict import StrictModule
from ._distributions import LogNormal, Normal
from ._posterior import (
    AbstractBijector,
    ExpBijector,
    IdentityBijector,
    ParameterSpace,
)


class GaussianPriorWhitening(StrictModule):
    """Affine map from declared Gaussian prior coordinates to standard normals."""

    location: PyTree[Array]
    scale: PyTree[Array]

    def __init__(self, location: PyTree[Array], scale: PyTree[Array], /):
        if jax.tree_util.tree_structure(location) != jax.tree_util.tree_structure(scale):
            raise ValueError("Whitening location and scale structures must match.")
        scale_leaves = jax.tree_util.tree_leaves(scale)
        if not scale_leaves or any(
            bool(jnp.any(~jnp.isfinite(leaf))) or bool(jnp.any(leaf <= 0.0))
            for leaf in scale_leaves
        ):
            raise ValueError("Whitening scales must be finite and strictly positive.")
        self.location = location
        self.scale = scale

    @classmethod
    def from_parameter_space(cls, space: ParameterSpace, /) -> GaussianPriorWhitening:
        """Infer exact Gaussian unconstrained priors from supported prior/bijector pairs."""
        if not isinstance(space, ParameterSpace):
            raise TypeError("space must be a ParameterSpace.")
        if space.priors is None:
            raise ValueError(
                "Gaussian prior whitening requires explicit distribution priors."
            )
        initial_leaves, treedef = jax.tree_util.tree_flatten(space.initial)
        prior_leaves = jax.tree_util.tree_leaves(
            space.priors,
            is_leaf=lambda value: isinstance(value, AbstractProbabilityLaw),
        )
        bijector_leaves = jax.tree_util.tree_leaves(
            space.bijectors,
            is_leaf=lambda value: isinstance(value, AbstractBijector),
        )
        if not (len(initial_leaves) == len(prior_leaves) == len(bijector_leaves)):
            raise ValueError("Parameter prior and bijector structures are incompatible.")
        locations: list[Array] = []
        scales: list[Array] = []
        for initial, prior, bijector in zip(
            initial_leaves,
            prior_leaves,
            bijector_leaves,
            strict=True,
        ):
            if isinstance(prior, Normal) and isinstance(bijector, IdentityBijector):
                location, scale = prior.location, prior.scale
            elif isinstance(prior, LogNormal) and isinstance(bijector, ExpBijector):
                location, scale = prior.location, prior.scale
            else:
                raise ValueError(
                    "Gaussian prior whitening supports only Normal/Identity and "
                    "LogNormal/Exp prior-bijector pairs."
                )
            initial_array = jnp.asarray(initial)
            locations.append(jnp.broadcast_to(location, initial_array.shape))
            scales.append(jnp.broadcast_to(scale, initial_array.shape))
        return cls(
            jax.tree_util.tree_unflatten(treedef, locations),
            jax.tree_util.tree_unflatten(treedef, scales),
        )

    def whiten(self, position: PyTree[Array], /) -> PyTree[Array]:
        """Map unconstrained posterior coordinates to standard-prior coordinates."""
        self._check_structure(position)
        return jax.tree_util.tree_map(
            lambda value, location, scale: (value - location) / scale,
            position,
            self.location,
            self.scale,
        )

    def unwhiten(self, whitened: PyTree[Array], /) -> PyTree[Array]:
        """Map standard-prior coordinates back to unconstrained coordinates."""
        self._check_structure(whitened)
        return jax.tree_util.tree_map(
            lambda value, location, scale: location + scale * value,
            whitened,
            self.location,
            self.scale,
        )

    def whiten_vector(self, vector: PyTree[Array], /) -> PyTree[Array]:
        """Apply the inverse affine map to a tangent vector."""
        self._check_structure(vector)
        return jax.tree_util.tree_map(
            lambda value, scale: value / scale,
            vector,
            self.scale,
        )

    def unwhiten_vector(self, vector: PyTree[Array], /) -> PyTree[Array]:
        """Apply the affine map to a tangent vector."""
        self._check_structure(vector)
        return jax.tree_util.tree_map(
            lambda value, scale: scale * value,
            vector,
            self.scale,
        )

    def _check_structure(self, value: PyTree[Any]) -> None:
        if jax.tree_util.tree_structure(value) != jax.tree_util.tree_structure(
            self.location
        ):
            raise ValueError("Whitening input has incompatible PyTree structure.")


__all__ = ["GaussianPriorWhitening"]
