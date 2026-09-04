#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Private fixed-shape helpers shared by tissue reaction models.

This module contains layout and exact first-order gate mechanics only.  It does
not define a biological model, a registry, or a public electrophysiology API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


@dataclass(frozen=True, slots=True)
class _FixedReactionLayout:
    """Pinned final-axis names, units, and source symbols for one model array."""

    names: tuple[str, ...]
    units: tuple[str, ...]
    source_symbols: tuple[str, ...]
    label: str
    _indices: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = tuple(self.names)
        units = tuple(self.units)
        symbols = tuple(self.source_symbols)
        label = str(self.label)
        if not label:
            raise ValueError("Reaction layout label must be non-empty.")
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{label} names must be non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError(f"{label} names must be unique.")
        if len(units) != len(names) or any(
            not isinstance(unit, str) or not unit for unit in units
        ):
            raise ValueError(f"{label} units must have one non-empty entry per name.")
        if len(symbols) != len(names) or any(
            not isinstance(symbol, str) or "/" not in symbol for symbol in symbols
        ):
            raise ValueError(
                f"{label} source symbols must map every entry as component/name."
            )
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"{label} source symbols must be unique.")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "source_symbols", symbols)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "_indices", {name: i for i, name in enumerate(names)})

    @property
    def count(self) -> int:
        return len(self.names)

    def index(self, name: str, /) -> int:
        if name not in self._indices:
            raise KeyError(f"Unknown {self.label} entry {name!r}.")
        return self._indices[name]

    def source_symbol(self, name: str, /) -> str:
        return self.source_symbols[self.index(name)]

    def require(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[-1] != self.count:
            raise ValueError(
                f"{self.label} must have final axis {self.count}; got {array.shape}."
            )
        return array

    def pack(self, values: Mapping[str, ArrayLike], /) -> Array:
        missing = tuple(name for name in self.names if name not in values)
        extra = tuple(name for name in values if name not in self._indices)
        if missing or extra:
            raise ValueError(
                f"{self.label} entries do not match layout; "
                f"missing={missing}, extra={extra}."
            )
        arrays = tuple(jnp.asarray(values[name]) for name in self.names)
        return jnp.stack(jnp.broadcast_arrays(*arrays), axis=-1)

    def unpack(self, values: ArrayLike, /) -> dict[str, Array]:
        array = self.require(values)
        return {name: array[..., index] for index, name in enumerate(self.names)}


class _ExactFirstOrderGates(StrictModule, NonTrainableState):
    """Rush--Larsen update for a pinned set of independent first-order gates."""

    indices: tuple[int, ...] = eqx.field(static=True)
    substrate_id: str = eqx.field(static=True)

    def __init__(self, indices: tuple[int, ...], /, *, substrate_id: str):
        resolved = tuple(int(index) for index in indices)
        if not resolved or any(index < 0 for index in resolved):
            raise ValueError("Exact gate indices must be non-empty and nonnegative.")
        if len(set(resolved)) != len(resolved):
            raise ValueError("Exact gate indices must be unique.")
        identifier = str(substrate_id)
        if not identifier:
            raise ValueError("substrate_id must be non-empty.")
        self.indices = resolved
        self.substrate_id = identifier

    def update(
        self,
        state: ArrayLike,
        steady_state: ArrayLike,
        time_constant_ms: ArrayLike,
        dt_ms: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(state)
        steady = jnp.asarray(steady_state, dtype=values.dtype)
        time_constant = jnp.asarray(time_constant_ms, dtype=values.dtype)
        step = jnp.asarray(dt_ms, dtype=values.dtype)
        expected = values.shape[:-1] + (len(self.indices),)
        if steady.shape != expected or time_constant.shape != expected:
            raise ValueError(
                "Exact gate steady states and time constants must have shape "
                f"{expected}."
            )
        if step.shape not in ((), values.shape[:-1]):
            raise ValueError("dt_ms must be scalar or match the state batch axes.")
        fraction = -jnp.expm1(-step[..., None] / time_constant)
        current = values[..., jnp.asarray(self.indices)]
        return values.at[..., jnp.asarray(self.indices)].set(
            current + fraction * (steady - current)
        )


__all__: list[str] = []
