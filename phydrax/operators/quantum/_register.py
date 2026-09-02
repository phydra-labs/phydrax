#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


def _target_wire_ids(value: Sequence[str], /) -> tuple[str, ...]:
    targets = tuple(str(wire_id) for wire_id in value)
    if not targets or any(not wire_id for wire_id in targets):
        raise ValueError("Local quantum targets must be unique and non-empty.")
    if len(set(targets)) != len(targets):
        raise ValueError("Local quantum targets must be unique and non-empty.")
    return targets


class HilbertRegisterLayout(StrictModule):
    """Ordered finite-dimensional Hilbert-space factorization."""

    wire_ids: tuple[str, ...] = eqx.field(static=True)
    local_dimensions: tuple[int, ...] = eqx.field(static=True)
    wire_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        wire_ids: Sequence[str],
        local_dimensions: Sequence[int],
        /,
    ):
        ids = tuple(str(wire_id) for wire_id in wire_ids)
        dimensions = tuple(int(dimension) for dimension in local_dimensions)
        if not ids:
            raise ValueError("Hilbert register must contain at least one wire.")
        if len(ids) != len(dimensions):
            raise ValueError("wire_ids and local_dimensions must have equal lengths.")
        if any(not wire_id for wire_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Hilbert-register wire IDs must be unique and non-empty.")
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("Hilbert-register dimensions must be positive integers.")
        total = prod(dimensions)
        self.wire_ids = ids
        self.local_dimensions = dimensions
        self.wire_count = len(ids)
        self.dimension = total
        self.layout_id = canonical_fingerprint(
            {
                "kind": "hilbert-register-layout",
                "wire_ids": ids,
                "local_dimensions": dimensions,
            }
        )

    def wire_index(self, wire_id: str, /) -> int:
        identifier = str(wire_id)
        if identifier not in self.wire_ids:
            raise KeyError(f"Unknown Hilbert-register wire {identifier!r}.")
        return self.wire_ids.index(identifier)

    def target_indices(self, wire_ids: Sequence[str], /) -> tuple[int, ...]:
        identifiers = _target_wire_ids(wire_ids)
        return tuple(self.wire_index(wire_id) for wire_id in identifiers)

    def target_dimension(self, wire_ids: Sequence[str], /) -> int:
        return prod(
            self.local_dimensions[index] for index in self.target_indices(wire_ids)
        )


__all__ = ["HilbertRegisterLayout"]
