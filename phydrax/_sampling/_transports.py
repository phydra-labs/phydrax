#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Protocol

from jaxtyping import Array


class UnitCubeTransport(Protocol):
    """Exact sampling map from unit-cube coordinates to one target factor."""

    @property
    def reference_dimension(self) -> int: ...

    def map(self, unit: Array, /) -> Any: ...


__all__ = ["UnitCubeTransport"]
