#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ._binding import ModelBinding


MODEL_CONSTRUCTION_CERTIFICATE_KEYS = frozenset({"trial_space_certificate"})


class ModelEvaluator(abc.ABC):
    """Callable model with an explicit domain invocation contract."""

    @abc.abstractmethod
    def input_binding(self) -> ModelBinding:
        """Return the model's input packing and batch execution contract."""
        raise NotImplementedError

@runtime_checkable
class ModelMetadataProvider(Protocol):
    """Model supplying immutable semantic metadata to a bound domain function."""

    def model_metadata(self) -> Mapping[str, Any]:
        """Return metadata attached by ``Domain.Model``."""
        ...


class AxisModelEvaluator(abc.ABC):
    """Model that evaluates one complete named-axis domain batch."""

    @abc.abstractmethod
    def __call_axis_batch__(
        self,
        batch: Any,
        deps: tuple[str, ...],
        /,
        *,
        key: Any = None,
        iter_: Any = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class StructuredDerivativeProvider(abc.ABC):
    """Model that can evaluate a structured partial derivative directly."""

    @abc.abstractmethod
    def try_structured_partial(
        self,
        *,
        deps: tuple[str, ...],
        var: str,
        axis: int,
        order: int,
        args: tuple[Any, ...],
        key: Any,
        kwargs: dict[str, Any],
    ) -> tuple[Any | None, str | None]:
        """Return an optimized derivative and diagnostic, or decline with `None`."""
        raise NotImplementedError

    @abc.abstractmethod
    def handle_structured_derivative_fallback(self, reason: str, /) -> None:
        """Apply this model's configured fallback policy to one diagnostic."""
        raise NotImplementedError


__all__ = [
    "AxisModelEvaluator",
    "MODEL_CONSTRUCTION_CERTIFICATE_KEYS",
    "ModelEvaluator",
    "ModelMetadataProvider",
    "StructuredDerivativeProvider",
]
