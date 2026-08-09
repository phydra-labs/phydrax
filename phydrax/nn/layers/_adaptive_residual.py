# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._callable import _ensure_special_kwonly_args
from ..._strict import StrictModule
from .._keys import EvalKey


class AdaptiveResidual(StrictModule):
    r"""Identity-start residual interpolation around a shape-preserving branch.

    For input ``x`` and branch ``F``, this layer evaluates
    ``x + alpha * (F(x) - x)``. A scalar ``alpha`` gates every output; a
    channelwise gate has one coefficient for the final output axis. The default
    zero initialization makes the layer an exact identity map while preserving
    nonzero gradients with respect to ``alpha`` whenever ``F(x) != x``.
    """

    branch: Callable
    alpha: Array
    channel_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        branch: Callable[..., Any],
        /,
        *,
        channel_size: int | None = None,
        initial_alpha: ArrayLike = 0.0,
    ):
        if not callable(branch):
            raise TypeError("branch must be callable.")
        if channel_size is not None and int(channel_size) <= 0:
            raise ValueError("channel_size must be positive when supplied.")

        alpha = jnp.asarray(initial_alpha)
        if channel_size is None:
            if alpha.shape != ():
                raise ValueError(
                    "initial_alpha must be scalar when channel_size is omitted."
                )
        else:
            size = int(channel_size)
            if alpha.shape == ():
                alpha = jnp.full((size,), alpha)
            elif alpha.shape != (size,):
                raise ValueError(
                    f"initial_alpha must have shape ({size},), got {alpha.shape}."
                )

        if not jnp.issubdtype(alpha.dtype, jnp.inexact):
            alpha = alpha.astype(jnp.asarray(0.0).dtype)
        self.branch = _ensure_special_kwonly_args(branch)
        self.alpha = alpha
        self.channel_size = None if channel_size is None else int(channel_size)

    def __call__(
        self,
        x: Array,
        /,
        *args: Any,
        key: EvalKey = None,
        **kwargs: Any,
    ) -> Array:
        value = jnp.asarray(x)
        branch_value = jnp.asarray(self.branch(value, *args, key=key, **kwargs))
        if branch_value.shape != value.shape:
            raise ValueError(
                "AdaptiveResidual branch output shape must equal its input shape; "
                f"got {branch_value.shape} and {value.shape}."
            )
        if self.channel_size is not None:
            if not value.shape or value.shape[-1] != self.channel_size:
                raise ValueError(
                    "AdaptiveResidual channelwise alpha requires final input axis "
                    f"size {self.channel_size}, got {value.shape}."
                )
        return value + self.alpha * (branch_value - value)


__all__ = ["AdaptiveResidual"]
