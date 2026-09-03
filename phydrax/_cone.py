#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._strict import StrictModule


def _scaled_cone_norm(value: Array, /) -> Array:
    scale = jnp.max(jnp.abs(value), axis=-1, initial=0.0)
    safe_scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    residual = scale * jnp.linalg.norm(value / safe_scale[..., None], axis=-1)
    return jnp.where(jnp.isinf(scale), jnp.inf, residual)


class AbstractConvexCone(StrictModule):
    """Closed convex cone over one trailing canonical-coordinate axis.

    This neutral base lives above both :mod:`phydrax.nonlinear` and
    :mod:`phydrax.optim`, allowing variational-inequality declarations to depend
    on cone semantics without creating an import cycle through the optimizer
    package.
    """

    dimension: int = eqx.field(static=True)
    cone_id: str = eqx.field(static=True)

    def _validate(self, value: Any, /) -> Array:
        array = jnp.asarray(value)
        if array.ndim < 1 or int(array.shape[-1]) != self.dimension:
            raise ValueError(
                f"Cone value must end in shape ({self.dimension},); got {array.shape}."
            )
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("Cone values must be real floating-point arrays.")
        return array

    @abc.abstractmethod
    def project(self, value: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def project_dual(self, value: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def interior_margin(self, value: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        """Return distance to the nearest nonsmooth dual-projection stratum."""
        raise NotImplementedError

    def projection_smoothness_margin(self, value: Any, /) -> Array:
        """Return distance to a nonsmooth primal-projection stratum."""
        array = self._validate(value)
        return self.dual_projection_smoothness_margin(-array)

    def residual(self, value: Any, /) -> Array:
        array = self._validate(value)
        return _scaled_cone_norm(array - self.project(array))

    def dual_residual(self, value: Any, /) -> Array:
        array = self._validate(value)
        return _scaled_cone_norm(array - self.project_dual(array))

    def contains(self, value: Any, /, *, tolerance: float = 0.0) -> Array:
        return self.residual(value) <= float(tolerance)

    def contains_dual(self, value: Any, /, *, tolerance: float = 0.0) -> Array:
        return self.dual_residual(value) <= float(tolerance)

    def complementarity(self, primal: Any, dual: Any, /) -> Array:
        primal_ = self._validate(primal)
        dual_ = self._validate(dual)
        return jnp.sum(primal_ * dual_, axis=-1)


__all__ = ["AbstractConvexCone"]
