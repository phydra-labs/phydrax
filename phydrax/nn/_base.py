#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import jax.numpy as jnp
from jaxtyping import Array

from .._doc import DOC_KEY0
from .._model import ModelBinding, ModelEvaluator, ModelObjectiveProvider
from .._strict import AbstractAttribute, StrictModule
from ._keys import EvalKey


class _AbstractBaseModel(StrictModule, ModelEvaluator, ModelObjectiveProvider):
    """Abstract base class for callable models with defined input and output sizes."""

    in_size: AbstractAttribute[int | tuple[int, ...] | Literal["scalar"]]
    out_size: AbstractAttribute[int | tuple[int, ...] | Literal["scalar"]]

    @abstractmethod
    def __call__(
        self,
        x: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    _input_binding: ClassVar[ModelBinding] = ModelBinding.pointwise()

    def input_binding(self) -> ModelBinding:
        """Return the model's domain input packing and batch execution contract."""
        return self._input_binding

    def __loss__(
        self,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
    ) -> Array:
        del key, iter_
        return jnp.array(0.0, dtype=float)

    def local_model_objective_labels(self) -> tuple[str, ...]:
        if type(self).__loss__ is _AbstractBaseModel.__loss__:
            return ()
        return (f"{type(self).__name__}.__loss__",)

    def local_model_objective_values(
        self,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
    ) -> tuple[Array, ...]:
        if type(self).__loss__ is _AbstractBaseModel.__loss__:
            return ()
        return (
            jnp.asarray(self.__loss__(key=key, iter_=iter_), dtype=float).reshape(()),
        )

    def add_model_loss(
        self,
        penalty: Callable[..., Any],
        /,
        *,
        weight: Any = 1.0,
        label: str | None = None,
    ) -> Any:
        """Return a model wrapper that contributes an extra scalar objective term."""
        from ._loss import add_model_loss

        return add_model_loss(self, penalty, weight=weight, label=label)


class _AbstractStructuredInputModel(_AbstractBaseModel):
    """Abstract base for models whose concrete structured-input schema is model-specific."""

    _input_binding: ClassVar[ModelBinding] = ModelBinding.pointwise("structured")

    @abstractmethod
    def __call__(
        self,
        x: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError
