#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._probability import AbstractProbabilityLaw
from ..domain._measure import MeasureKind


class AutoregressiveLaw(AbstractProbabilityLaw):
    """Normalized ordered event law built from explicit scalar conditional laws."""

    conditional: Any
    length: int = eqx.field(static=True)
    dtype: Any = eqx.field(static=True)
    order_id: str = eqx.field(static=True)
    _measure_kind: MeasureKind = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        conditional,
        length: int,
        /,
        *,
        dtype=float,
        order_id: str,
        law_id: str | None = None,
    ):
        if not callable(conditional):
            raise TypeError("conditional must be callable.")
        size = int(length)
        if size <= 0 or not order_id:
            raise ValueError("length must be positive and order_id non-empty.")
        resolved_dtype = jnp.dtype(dtype)
        probe = conditional(jnp.empty((0,), dtype=resolved_dtype), 0)
        if not isinstance(probe, AbstractProbabilityLaw) or probe.event_shape != ():
            raise ValueError("Autoregressive conditionals must be scalar probability laws.")
        self.conditional = conditional
        self.length = size
        self.dtype = resolved_dtype
        self.order_id = order_id
        self._measure_kind = probe.density_measure_kind
        self.law_id = law_id or canonical_fingerprint(
            {
                "kind": "autoregressive-law",
                "length": size,
                "dtype": str(self.dtype),
                "order_id": order_id,
            }
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self.length,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def density_measure_kind(self) -> MeasureKind:
        return self._measure_kind

    def _conditional_law(self, prefix: Array, index: int, /) -> AbstractProbabilityLaw:
        law = self.conditional(prefix, index)
        if not isinstance(law, AbstractProbabilityLaw) or law.event_shape != ():
            raise ValueError("Autoregressive conditional must return a scalar law.")
        if law.density_measure_kind != self.density_measure_kind:
            raise ValueError("Every autoregressive conditional must use one reference measure.")
        return law

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        count = 1
        for size in samples:
            count *= size
        keys = jr.split(key, count * self.length).reshape(
            (count, self.length) + tuple(key.shape)
        )
        rows = []
        for row_index in range(count):
            row = jnp.empty((self.length,), dtype=self.dtype)
            for index in range(self.length):
                law = self._conditional_law(row[:index], index)
                row = row.at[index].set(law.sample(keys[row_index, index]))
            rows.append(row)
        stacked = jnp.stack(rows)
        return stacked.reshape(samples + self.event_shape) if samples else stacked[0]

    def _rows(self, value: ArrayLike, /) -> tuple[Array, tuple[int, ...]]:
        array = jnp.asarray(value, dtype=self.dtype)
        if array.ndim < 1 or int(array.shape[-1]) != self.length:
            raise ValueError("Autoregressive values must end in the declared length.")
        return array.reshape((-1, self.length)), tuple(array.shape[:-1])

    def log_prob(self, value: ArrayLike, /) -> Array:
        rows, leading = self._rows(value)
        outputs = []
        for row in rows:
            total = jnp.asarray(0.0)
            for index in range(self.length):
                law = self._conditional_law(row[:index], index)
                total = total + law.log_prob(row[index])
            outputs.append(total)
        return jnp.stack(outputs).reshape(leading)

    def contains(self, value: ArrayLike, /) -> Array:
        rows, leading = self._rows(value)
        outputs = []
        for row in rows:
            valid = jnp.asarray(True)
            for index in range(self.length):
                law = self._conditional_law(row[:index], index)
                valid = valid & law.contains(row[index])
            outputs.append(valid)
        return jnp.stack(outputs).reshape(leading)


__all__ = ["AutoregressiveLaw"]
