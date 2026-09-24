#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax

from .._trainable import ExplicitFreeze, NonTrainableState
from ._array import AbstractArrayModel
from ._component import ModelExecutionContract


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
        {
            "decision_function",
            "model_ports",
            "predict",
            "predict_log_proba",
            "predict_proba",
        }
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

    def model_execution_contract(self) -> ModelExecutionContract:
        """Return the wrapped model's contract; freezing changes no evaluation claim."""
        return self.model.model_execution_contract()

    def as_trainable(self, /) -> AbstractArrayModel:
        """Return the wrapped model without copying its array leaves."""
        return self.model


def trainable_provider(provider: Any, /) -> Any:
    """Return the trainable form of a frozen provider, refusing one without any.

    Frozen artifacts use it for their explicit trainable counterpart: a
    `FrozenModel` is unwrapped (without copying array leaves); any other
    terminal (`NonTrainableState`) provider cannot become trainable. The
    provider must be callable and hold a visible inexact array leaf, which
    becomes a PARAMETER in the trainable holder. Raises `ValueError` otherwise.
    """
    unwrapped = provider.as_trainable() if isinstance(provider, FrozenModel) else provider
    if not callable(unwrapped):
        raise TypeError("A trainable provider must be callable.")
    if isinstance(unwrapped, NonTrainableState):
        raise ValueError(
            f"{type(unwrapped).__name__} is a fixed provider and cannot become trainable."
        )
    if not jax.tree_util.tree_leaves(eqx.filter(unwrapped, eqx.is_inexact_array)):
        raise ValueError(
            f"{type(unwrapped).__name__} holds no visible inexact array to train; a "
            "trainable binding needs a provider whose arrays are PyTree leaves."
        )
    return unwrapped


__all__ = ["FrozenModel", "trainable_provider"]
