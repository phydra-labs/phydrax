#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax.numpy as jnp
from jaxtyping import ArrayLike

from phydrax.domain import DomainFunction

from .._strict import StrictModule
from ..operators.linalg import einsum


class _CrossEvaluator(StrictModule):
    left: DomainFunction
    right: DomainFunction
    left_positions: tuple[int, ...]
    right_positions: tuple[int, ...]

    def __init__(
        self,
        left: DomainFunction,
        right: DomainFunction,
        left_positions: tuple[int, ...],
        right_positions: tuple[int, ...],
        /,
    ):
        self.left = left
        self.right = right
        self.left_positions = left_positions
        self.right_positions = right_positions

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        left = self.left.func(
            *(args[index] for index in self.left_positions),
            key=key,
            **kwargs,
        )
        right = self.right.func(
            *(args[index] for index in self.right_positions),
            key=key,
            **kwargs,
        )
        return jnp.cross(jnp.asarray(left), jnp.asarray(right))


def dot(left: DomainFunction, right: DomainFunction, /) -> DomainFunction:
    return einsum("...i,...i->...", left, right)


def matvec(matrix: DomainFunction, vector: DomainFunction, /) -> DomainFunction:
    return einsum("...ij,...j->...i", matrix, vector)


def outer_scalar_vector(
    scalar: DomainFunction,
    vector: DomainFunction,
    /,
) -> DomainFunction:
    return einsum("...,...i->...i", scalar, vector)


def cross(left: DomainFunction, right: DomainFunction, /) -> DomainFunction:
    joined = left.domain.join(right.domain)
    promoted_left = left.promote(joined)
    promoted_right = right.promote(joined)
    deps = tuple(
        label
        for label in joined.labels
        if label in promoted_left.deps or label in promoted_right.deps
    )
    positions = {label: index for index, label in enumerate(deps)}
    left_positions = tuple(positions[label] for label in promoted_left.deps)
    right_positions = tuple(positions[label] for label in promoted_right.deps)
    metadata = left.metadata if left.metadata == right.metadata else {}
    return DomainFunction(
        domain=joined,
        deps=deps,
        func=_CrossEvaluator(
            promoted_left,
            promoted_right,
            left_positions,
            right_positions,
        ),
        metadata=metadata,
    )


def constant_field(value: ArrayLike, like: DomainFunction, /) -> DomainFunction:
    return DomainFunction(domain=like.domain, deps=(), func=value, metadata={})


__all__ = ["constant_field", "cross", "dot", "matvec", "outer_scalar_vector"]
