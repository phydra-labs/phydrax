#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Literal, TypeAlias

import jax.numpy as jnp
from jaxtyping import Array

from ...._doc import DOC_KEY0
from ...._strict import AbstractAttribute, StrictModule
from ._keys import EvalKey


DomainInputMode: TypeAlias = Literal["flat", "structured"]


class _AbstractBaseModel(StrictModule):
    """Abstract base class for callable models with defined input and output sizes."""

    in_size: AbstractAttribute[int | tuple[int, ...] | Literal["scalar"]]
    out_size: AbstractAttribute[int | tuple[int, ...] | Literal["scalar"]]

    @abstractmethod
    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    _supports_structured_input: bool = False
    _supports_blockwise_input: bool = False
    _supports_axis_batch_input: bool = False
    _warn_on_auto_fallback: bool = False
    _domain_input_mode: ClassVar[DomainInputMode] = "flat"

    @classmethod
    def supports_structured_input(cls) -> bool:
        return cls._supports_structured_input

    @classmethod
    def supports_blockwise_input(cls) -> bool:
        return cls._supports_blockwise_input

    def warn_on_auto_fallback(self) -> bool:
        return bool(self._warn_on_auto_fallback)

    def supports_axis_batch_input(self) -> bool:
        return bool(self._supports_axis_batch_input)

    @classmethod
    def domain_input_mode(cls) -> DomainInputMode:
        return cls._domain_input_mode

    def __loss__(
        self,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
    ) -> Array:
        del key, iter_
        return jnp.array(0.0, dtype=float)

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
    """Abstract base class for models that accept structured (tuple) inputs."""

    _supports_structured_input: bool = True
    _domain_input_mode: ClassVar[DomainInputMode] = "structured"

    @abstractmethod
    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError
