#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, Literal

from jaxtyping import Array

from .._strict import AbstractAttribute, StrictModule
from ._binding import ModelBinding
from ._protocols import ModelEvaluator


class AbstractArrayModel(StrictModule, ModelEvaluator):
    """Callable array model with explicit input/output sizes and domain binding."""

    # JAX hashes callable objects when a module itself is passed to ``jax.jit``.
    # Equinox's structural hash cannot hash array leaves; identity is stable because
    # Phydrax models are immutable and their leaves remain explicit PyTree state.
    __hash__ = object.__hash__

    in_size: AbstractAttribute[int | tuple[int, ...] | Literal["scalar"]]
    out_size: AbstractAttribute[int | tuple[int, ...] | Literal["scalar"]]

    _input_binding: ClassVar[ModelBinding] = ModelBinding.pointwise()

    @abstractmethod
    def __call__(
        self,
        x: Any,
        /,
        *,
        key: Any = None,
    ) -> Array:
        raise NotImplementedError

    def input_binding(self) -> ModelBinding:
        """Return the model's domain input packing and batch execution contract."""
        return self._input_binding


__all__ = ["AbstractArrayModel"]
