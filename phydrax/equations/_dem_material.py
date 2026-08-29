#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


class DEMMaterialTable(StrictModule):
    """Elastic material properties and explicit symmetric interface coefficients."""

    young_modulus: Array
    poisson_ratio: Array
    restitution: Array
    friction: Array
    rolling_friction: Array
    material_count: int = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        young_modulus: ArrayLike,
        poisson_ratio: ArrayLike,
        restitution: ArrayLike,
        friction: ArrayLike,
        /,
        *,
        rolling_friction: ArrayLike | None = None,
        material_id: str | None = None,
    ):
        young = np.asarray(young_modulus)
        poisson = np.asarray(poisson_ratio)
        restitution_ = np.asarray(restitution)
        friction_ = np.asarray(friction)
        rolling_ = (
            np.zeros_like(friction_)
            if rolling_friction is None
            else np.asarray(rolling_friction)
        )
        if young.ndim != 1 or young.size == 0:
            raise ValueError("young_modulus must be a nonempty rank-1 array.")
        count = int(young.size)
        if poisson.shape != (count,):
            raise ValueError("poisson_ratio must have the young_modulus shape.")
        pair_shape = (count, count)
        if (
            restitution_.shape != pair_shape
            or friction_.shape != pair_shape
            or rolling_.shape != pair_shape
        ):
            raise ValueError(
                "Restitution, friction, and rolling friction must be square pair tables."
            )
        if np.any(~np.isfinite(young)) or np.any(young <= 0.0):
            raise ValueError("Young moduli must be finite and positive.")
        if (
            np.any(~np.isfinite(poisson))
            or np.any(poisson <= -1.0)
            or np.any(poisson >= 0.5)
        ):
            raise ValueError("Poisson ratios must be finite and lie in (-1, 0.5).")
        if (
            np.any(~np.isfinite(restitution_))
            or np.any(restitution_ <= 0.0)
            or np.any(restitution_ > 1.0)
        ):
            raise ValueError("Restitution coefficients must lie in (0, 1].")
        if np.any(~np.isfinite(friction_)) or np.any(friction_ < 0.0):
            raise ValueError("Friction coefficients must be finite and nonnegative.")
        if np.any(~np.isfinite(rolling_)) or np.any(rolling_ < 0.0):
            raise ValueError(
                "Rolling-friction coefficients must be finite and nonnegative."
            )
        if not np.array_equal(restitution_, restitution_.T):
            raise ValueError("Restitution table must be symmetric.")
        if not np.array_equal(friction_, friction_.T):
            raise ValueError("Friction table must be symmetric.")
        if not np.array_equal(rolling_, rolling_.T):
            raise ValueError("Rolling-friction table must be symmetric.")
        generated = canonical_fingerprint(
            {
                "kind": "dem-material-table",
                "values": array_tree_fingerprint(
                    {
                        "young_modulus": young,
                        "poisson_ratio": poisson,
                        "restitution": restitution_,
                        "friction": friction_,
                        "rolling_friction": rolling_,
                    }
                ),
            }
        )
        identifier = generated if material_id is None else str(material_id)
        if not identifier:
            raise ValueError("material_id must be nonempty.")
        dtype = jnp.result_type(
            young.dtype,
            poisson.dtype,
            restitution_.dtype,
            friction_.dtype,
            rolling_.dtype,
        )
        if not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.dtype(jnp.float32)
        self.young_modulus = jnp.asarray(young, dtype=dtype)
        self.poisson_ratio = jnp.asarray(poisson, dtype=dtype)
        self.restitution = jnp.asarray(restitution_, dtype=dtype)
        self.friction = jnp.asarray(friction_, dtype=dtype)
        self.rolling_friction = jnp.asarray(rolling_, dtype=dtype)
        self.material_count = count
        self.material_id = identifier

    def ids_admissible(self, material_ids: ArrayLike, /) -> Array:
        ids = jnp.asarray(material_ids)
        return jnp.all((ids >= 0) & (ids < self.material_count))

    def pair_restitution(self, left_ids: ArrayLike, right_ids: ArrayLike, /) -> Array:
        left, right = self._pair_ids(left_ids, right_ids)
        return self.restitution[left, right]

    def pair_friction(self, left_ids: ArrayLike, right_ids: ArrayLike, /) -> Array:
        left, right = self._pair_ids(left_ids, right_ids)
        return self.friction[left, right]

    def pair_rolling_friction(
        self, left_ids: ArrayLike, right_ids: ArrayLike, /
    ) -> Array:
        left, right = self._pair_ids(left_ids, right_ids)
        return self.rolling_friction[left, right]

    def effective_young_modulus(
        self, left_ids: ArrayLike, right_ids: ArrayLike, /
    ) -> Array:
        left, right = self._pair_ids(left_ids, right_ids)
        inverse = (1.0 - self.poisson_ratio[left] ** 2) / self.young_modulus[left] + (
            1.0 - self.poisson_ratio[right] ** 2
        ) / self.young_modulus[right]
        return 1.0 / inverse

    def effective_shear_modulus(
        self, left_ids: ArrayLike, right_ids: ArrayLike, /
    ) -> Array:
        left, right = self._pair_ids(left_ids, right_ids)
        shear = self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
        inverse = (2.0 - self.poisson_ratio[left]) / shear[left] + (
            2.0 - self.poisson_ratio[right]
        ) / shear[right]
        return 1.0 / inverse

    def admissible(self) -> Array:
        young = self.young_modulus
        poisson = self.poisson_ratio
        restitution = self.restitution
        friction = self.friction
        rolling = self.rolling_friction
        return (
            jnp.all(jnp.isfinite(young) & (young > 0.0))
            & jnp.all(jnp.isfinite(poisson) & (poisson > -1.0) & (poisson < 0.5))
            & jnp.all(
                jnp.isfinite(restitution) & (restitution > 0.0) & (restitution <= 1.0)
            )
            & jnp.all(jnp.isfinite(friction) & (friction >= 0.0))
            & jnp.all(jnp.isfinite(rolling) & (rolling >= 0.0))
            & jnp.all(restitution == restitution.T)
            & jnp.all(friction == friction.T)
            & jnp.all(rolling == rolling.T)
        )

    def _pair_ids(
        self, left_ids: ArrayLike, right_ids: ArrayLike, /
    ) -> tuple[Array, Array]:
        left = jnp.asarray(left_ids)
        right = jnp.asarray(right_ids)
        if left.shape != right.shape:
            raise ValueError("Material pair ID arrays must have equal shapes.")
        if not jnp.issubdtype(left.dtype, jnp.integer) or not jnp.issubdtype(
            right.dtype, jnp.integer
        ):
            raise TypeError("Material pair IDs must be integers.")
        left = eqx.error_if(
            left,
            jnp.any((left < 0) | (left >= self.material_count)),
            "Left DEM material ID is out of range.",
        )
        right = eqx.error_if(
            right,
            jnp.any((right < 0) | (right >= self.material_count)),
            "Right DEM material ID is out of range.",
        )
        return left.astype(jnp.int32), right.astype(jnp.int32)


__all__ = ["DEMMaterialTable"]
