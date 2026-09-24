#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

from .._trainable import ExplicitFreeze
from ._array import AbstractArrayModel


class FrozenModel(AbstractArrayModel, ExplicitFreeze):
    """Callable wrapper that intentionally freezes an entire trained model.

    Every array below the wrapper is FIXED; as an `ExplicitFreeze` holder it may
    legally sit inside other fixed state.
    """

    model: AbstractArrayModel
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    _input_binding = staticmethod(lambda wrapper: wrapper.model.input_binding())
    _delegated_methods = frozenset(
        {"decision_function", "predict", "predict_log_proba", "predict_proba"}
    )

    def __init__(self, model: AbstractArrayModel, /):
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("FrozenModel requires an AbstractArrayModel.")
        self.model = model
        self.in_size = model.in_size
        self.out_size = model.out_size

    def __call__(self, x: Any, /, *, key: Any = None):
        return self.model(x, key=key)

    def __getattr__(self, name: str, /):
        if name not in self._delegated_methods:
            raise AttributeError(name)
        model = object.__getattribute__(self, "model")
        return getattr(model, name)

    def as_trainable(self, /) -> AbstractArrayModel:
        """Return the wrapped model without copying its array leaves."""
        return self.model


__all__ = ["FrozenModel"]
