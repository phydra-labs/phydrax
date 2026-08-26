#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FiniteRealAlgebraProvider(Protocol):
    @property
    def coordinate_dimension(self) -> int: ...

    @property
    def basis_ids(self) -> tuple[str, ...]: ...

    @property
    def algebra_id(self) -> str: ...

    def conjugate(self, value: Any, /) -> Any: ...

    def prepare_product(self, **kwargs) -> Any: ...


__all__ = ["FiniteRealAlgebraProvider"]
