#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

from .._trainable import NonTrainableState
from ._array import AbstractArrayModel


class FrozenModel(AbstractArrayModel, NonTrainableState):
    """Callable wrapper that keeps an entire model outside solver partitions."""

    model: AbstractArrayModel
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(self, model: AbstractArrayModel, /):
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("FrozenModel requires an AbstractArrayModel.")
        self.model = model
        self.in_size = model.in_size
        self.out_size = model.out_size

    def __call__(self, x: Any, /, *, key: Any = None):
        return self.model(x, key=key)

    def as_trainable(self, /) -> AbstractArrayModel:
        """Return the wrapped model without copying its array leaves."""
        return self.model


__all__ = ["FrozenModel"]
