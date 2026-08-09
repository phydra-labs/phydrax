#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

from jaxtyping import Array, ArrayLike

from ._strict import StrictModule
from .domain._measure import MeasureKind


class AbstractProbabilityLaw(StrictModule):
    """Probability law with explicit sample, batch, event, and measure semantics."""

    @property
    @abstractmethod
    def event_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def batch_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def density_measure_kind(self) -> MeasureKind:
        raise NotImplementedError

    @abstractmethod
    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def contains(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError


__all__ = ["AbstractProbabilityLaw"]
