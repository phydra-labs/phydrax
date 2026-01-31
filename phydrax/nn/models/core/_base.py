#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from typing import Literal

from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._strict import AbstractAttribute, StrictModule


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
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    _supports_structured_input: bool = False

    @classmethod
    def supports_structured_input(cls) -> bool:
        return cls._supports_structured_input


class _AbstractStructuredInputModel(_AbstractBaseModel):
    """Abstract base class for models that accept structured (tuple) inputs."""

    _supports_structured_input: bool = True

    @abstractmethod
    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError
