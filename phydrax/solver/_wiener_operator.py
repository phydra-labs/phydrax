#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


class WienerNoiseBlock(StrictModule):
    """One named contiguous block in flattened Wiener coordinates."""

    name: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)
    start: int = eqx.field(static=True)
    stop: int = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        start: int,
        /,
        *,
        basis_id: str | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("Wiener noise block name must be non-empty.")
        resolved_shape = tuple(int(size) for size in shape)
        if any(size <= 0 for size in resolved_shape):
            raise ValueError("Wiener noise block dimensions must be positive.")
        offset = int(start)
        if offset < 0:
            raise ValueError("Wiener noise block start must be nonnegative.")
        size = prod(resolved_shape)
        if basis_id is not None and (not isinstance(basis_id, str) or not basis_id):
            raise ValueError("basis_id must be non-empty or None.")
        self.name = name
        self.shape = resolved_shape
        self.size = size
        self.start = offset
        self.stop = offset + size
        self.basis_id = basis_id


class WienerNoiseLayout(StrictModule):
    """Ordered named Wiener blocks with one deterministic flattened coordinate layout."""

    blocks: tuple[WienerNoiseBlock, ...]
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    total_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[tuple[str, Sequence[int], str | None]],
        /,
        *,
        layout_id: str | None = None,
    ):
        values = []
        offset = 0
        for name, shape, basis_id in blocks:
            block = WienerNoiseBlock(name, shape, offset, basis_id=basis_id)
            values.append(block)
            offset = block.stop
        if not values:
            raise ValueError("WienerNoiseLayout requires at least one block.")
        names = tuple(block.name for block in values)
        if len(set(names)) != len(names):
            raise ValueError("Wiener noise block names must be unique.")
        resolved = layout_id or canonical_fingerprint(
            {
                "kind": "wiener-noise-layout",
                "blocks": [
                    {
                        "name": block.name,
                        "shape": list(block.shape),
                        "basis_id": block.basis_id,
                    }
                    for block in values
                ],
            }
        )
        if not isinstance(resolved, str) or not resolved:
            raise ValueError("layout_id must be non-empty or None.")
        self.blocks = tuple(values)
        self.noise_shape = (offset,)
        self.total_size = offset
        self.layout_id = resolved

    def block(self, name: str, /) -> WienerNoiseBlock:
        matches = tuple(block for block in self.blocks if block.name == name)
        if not matches:
            raise KeyError(name)
        return matches[0]


__all__ = ["WienerNoiseBlock", "WienerNoiseLayout"]
