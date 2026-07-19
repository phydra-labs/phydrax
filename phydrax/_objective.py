#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from jaxtyping import Array, Key

from ._doc import DOC_KEY0
from ._strict import AbstractAttribute, StrictModule
from .domain._function import DomainFunction


class AbstractObjectiveTerm(StrictModule):
    """A scalar term evaluated from the solver's current domain functions."""

    label: AbstractAttribute[str | None]

    @abstractmethod
    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | None = None,
        **kwargs: Any,
    ) -> Array:
        raise NotImplementedError


class AbstractSamplingObjectiveTerm(AbstractObjectiveTerm):
    """Objective term whose scalar estimate is evaluated on a sampled batch."""

    @abstractmethod
    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> Any:
        raise NotImplementedError


__all__ = [
    "AbstractObjectiveTerm",
    "AbstractSamplingObjectiveTerm",
]
