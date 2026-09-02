#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import coordax as cx
import equinox as eqx
import jax.numpy as jnp

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ResidualBlockLayout(StrictModule, NonTrainableState):
    """Named partition of one residual event axis.

    ``event_axis`` indexes only unnamed ``coordax.Field`` event axes. ``sizes``
    partitions that axis in order. When omitted, every named block has size one.
    The layout is metadata: it never changes the authored quadratic residual loss.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    sizes: tuple[int, ...] = eqx.field(static=True)
    event_axis: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        names: Sequence[str],
        /,
        *,
        sizes: Sequence[int] | None = None,
        event_axis: int = 0,
        layout_id: str | None = None,
    ):
        names_ = tuple(str(name) for name in names)
        if not names_ or any(not name for name in names_):
            raise ValueError("Residual block names must be non-empty.")
        if len(set(names_)) != len(names_):
            raise ValueError("Residual block names must be unique.")
        sizes_ = (
            tuple(1 for _ in names_)
            if sizes is None
            else tuple(int(size) for size in sizes)
        )
        if len(sizes_) != len(names_) or any(size <= 0 for size in sizes_):
            raise ValueError(
                "Residual block sizes must be positive and align with block names."
            )
        axis = int(event_axis)
        if axis < 0:
            raise ValueError("event_axis must be non-negative among unnamed event axes.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "residual-block-layout",
                    "names": names_,
                    "sizes": sizes_,
                    "event_axis": axis,
                }
            )
            if layout_id is None
            else str(layout_id)
        )
        if not identifier:
            raise ValueError("layout_id must be non-empty.")
        self.names = names_
        self.sizes = sizes_
        self.event_axis = axis
        self.layout_id = identifier

    @property
    def block_count(self) -> int:
        return len(self.names)

    @property
    def event_size(self) -> int:
        return sum(self.sizes)

    def block_index(self, name: str, /) -> int:
        name_ = str(name)
        if name_ not in self.names:
            raise KeyError(f"Unknown residual block {name!r}.")
        return self.names.index(name_)

    def split(self, field: cx.Field, /) -> tuple[cx.Field, ...]:
        """Split a residual field while preserving its event-axis dimension."""
        if not isinstance(field, cx.Field):
            raise TypeError("ResidualBlockLayout.split requires a coordax.Field.")
        event_positions = tuple(
            index for index, dimension in enumerate(field.dims) if dimension is None
        )
        if self.event_axis >= len(event_positions):
            raise ValueError(
                f"Residual field has {len(event_positions)} event axes; "
                f"layout requests event axis {self.event_axis}."
            )
        axis = event_positions[self.event_axis]
        data = jnp.asarray(field.data)
        if int(data.shape[axis]) != self.event_size:
            raise ValueError(
                f"Residual block layout requires event size {self.event_size}, "
                f"got {data.shape[axis]}."
            )
        blocks: list[cx.Field] = []
        start = 0
        for size in self.sizes:
            indices = jnp.arange(start, start + size, dtype=jnp.int32)
            blocks.append(cx.Field(jnp.take(data, indices, axis=axis), dims=field.dims))
            start += size
        return tuple(blocks)


class ResidualBlockRef(StrictModule, NonTrainableState):
    """Stable reference to one complete residual term or named residual block."""

    term_index: int = eqx.field(static=True)
    block_name: str | None = eqx.field(static=True)

    def __init__(self, term_index: int, block_name: str | None = None, /):
        index = int(term_index)
        if index < 0:
            raise ValueError("term_index must be non-negative.")
        name = None if block_name is None else str(block_name)
        if name == "":
            raise ValueError("block_name must be non-empty when supplied.")
        self.term_index = index
        self.block_name = name


__all__ = ["ResidualBlockLayout", "ResidualBlockRef"]
