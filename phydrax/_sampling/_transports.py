#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Protocol

from jaxtyping import Array


class ReferenceTransport(Protocol):
    """Exact map from unit-cube coordinates to one target-measure factor."""

    @property
    def reference_dimension(self) -> int: ...

    def map(self, unit: Array, /) -> Any: ...


__all__ = ["ReferenceTransport"]
