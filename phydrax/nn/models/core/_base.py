#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from typing import ClassVar, Literal, TypeAlias

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

    @classmethod
    def domain_input_mode(cls) -> DomainInputMode:
        return cls._domain_input_mode


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
