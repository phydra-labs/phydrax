#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from ..._strict import StrictModule
from ._transforms import AbstractParameterTransform


class TransformedParameter(StrictModule):
    """Trainable unconstrained coordinates with a physical-value transformation."""

    raw: PyTree[Array]
    transform: AbstractParameterTransform

    def __init__(self, raw: PyTree[Array], transform: AbstractParameterTransform):
        if not isinstance(transform, AbstractParameterTransform):
            raise TypeError("transform must be an AbstractParameterTransform.")
        leaves = jax.tree_util.tree_leaves(raw)
        if not leaves:
            raise ValueError("raw must contain at least one array leaf.")
        if any(not eqx.is_inexact_array(leaf) for leaf in leaves):
            raise TypeError("Every raw parameter leaf must be an inexact JAX array.")
        self.raw = raw
        self.transform = transform

    def value(self) -> Array:
        """Return the physical parameter represented by the raw coordinates."""
        return self.transform(self.raw)

    def __call__(self) -> Array:
        return self.value()


__all__ = ["TransformedParameter"]
