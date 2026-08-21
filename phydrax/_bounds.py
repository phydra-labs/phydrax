#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ._strict import StrictModule
from ._tree_math import tree_all, validate_real_inexact_tree


def _static_bound_metadata(
    bound: Any,
    /,
) -> tuple[tuple[tuple[int, ...], str, tuple[Any, ...]], ...] | None:
    """Capture immutable bound values before array leaves become JAX tracers."""
    metadata = []
    try:
        for leaf in jax.tree.leaves(bound):
            array = np.asarray(leaf)
            metadata.append(
                (
                    tuple(array.shape),
                    array.dtype.str,
                    tuple(array.reshape(-1).tolist()),
                )
            )
    except (TypeError, ValueError, jax.errors.TracerArrayConversionError):
        return None
    return tuple(metadata)


class Bounds(StrictModule):
    """Broadcastable lower and upper bounds over one real array PyTree."""

    lower: Any
    upper: Any
    _lower_metadata: tuple[tuple[tuple[int, ...], str, tuple[Any, ...]], ...] | None = (
        eqx.field(static=True, repr=False)
    )
    _upper_metadata: tuple[tuple[tuple[int, ...], str, tuple[Any, ...]], ...] | None = (
        eqx.field(static=True, repr=False)
    )

    def __init__(self, lower: Any = -jnp.inf, upper: Any = jnp.inf, /):
        self.lower = lower
        self.upper = upper
        self._lower_metadata = _static_bound_metadata(lower)
        self._upper_metadata = _static_bound_metadata(upper)

    def materialize(
        self,
        parameters: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        parameters_ = validate_real_inexact_tree(parameters, name="parameters")
        structure = jax.tree.structure(parameters_)

        def materialize_bound(bound: Any, *, name: str) -> PyTree[Array]:
            if jax.tree.structure(bound) == structure:
                return jax.tree.map(
                    lambda bound_leaf, parameter: jnp.broadcast_to(
                        jnp.asarray(bound_leaf, dtype=parameter.dtype),
                        parameter.shape,
                    ),
                    bound,
                    parameters_,
                )
            scalar = jnp.asarray(bound)
            if scalar.shape != ():
                raise ValueError(
                    f"{name} must be scalar or have the parameter PyTree structure."
                )
            return jax.tree.map(
                lambda parameter: jnp.broadcast_to(
                    scalar.astype(parameter.dtype),
                    parameter.shape,
                ),
                parameters_,
            )

        lower = materialize_bound(self.lower, name="lower")
        upper = materialize_bound(self.upper, name="upper")
        valid = tree_all(jax.tree.map(lambda lo, hi: jnp.all(lo <= hi), lower, upper))
        lower = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                ~valid,
                "Lower bounds must not exceed upper bounds.",
            ),
            lower,
        )
        return lower, upper

    def project(self, parameters: PyTree[Any], /) -> PyTree[Array]:
        lower, upper = self.materialize(parameters)
        return jax.tree.map(jnp.clip, parameters, lower, upper)

    def contains(self, parameters: PyTree[Any], /) -> Array:
        lower, upper = self.materialize(parameters)
        return tree_all(
            jax.tree.map(
                lambda value, lo, hi: jnp.all(
                    jnp.isfinite(value) & (value >= lo) & (value <= hi)
                ),
                parameters,
                lower,
                upper,
            )
        )

    def violation(self, parameters: PyTree[Any], /) -> Array:
        lower, upper = self.materialize(parameters)
        violations = jax.tree.map(
            lambda value, lo, hi: jnp.maximum(
                jnp.maximum(lo - value, value - hi),
                0.0,
            ),
            parameters,
            lower,
            upper,
        )
        leaves = tuple(jnp.max(value) for value in jax.tree.leaves(violations))
        maximum = jnp.asarray(0.0)
        for value in leaves:
            maximum = jnp.maximum(maximum, value)
        return maximum

    def projected_gradient(
        self,
        parameters: PyTree[Any],
        gradient: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        trial = jax.tree.map(lambda value, grad: value - grad, parameters, gradient)
        projected = self.project(trial)
        return jax.tree.map(
            lambda value, candidate: value - candidate,
            parameters,
            projected,
        )

    def active_mask(
        self,
        parameters: PyTree[Any],
        gradient: PyTree[Any],
        /,
        *,
        tolerance: float = 1e-10,
    ) -> PyTree[Array]:
        lower, upper = self.materialize(parameters)
        tol = float(tolerance)
        if not isfinite(tol) or tol < 0.0:
            raise ValueError("Active-set tolerance must be finite and non-negative.")
        return jax.tree.map(
            lambda value, grad, lo, hi: (
                ((value <= lo + tol * (1.0 + jnp.abs(lo))) & (grad > 0.0))
                | ((value >= hi - tol * (1.0 + jnp.abs(hi))) & (grad < 0.0))
                | (lo == hi)
            ),
            parameters,
            gradient,
            lower,
            upper,
        )


__all__ = ["Bounds"]
