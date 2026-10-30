#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from types import TracebackType
from typing import Self


class AbstractMeshingSession(abc.ABC):
    """Ephemeral host-side provider session excluded from JAX state."""

    @property
    @abc.abstractmethod
    def closed(self) -> bool: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("Cannot enter a closed meshing session.")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False


__all__ = ["AbstractMeshingSession"]
