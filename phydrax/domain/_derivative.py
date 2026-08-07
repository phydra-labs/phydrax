#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from ._function import DomainFunction


DerivativeMode: TypeAlias = Literal["reverse", "forward"]
DerivativeBackend: TypeAlias = Literal["ad", "jet", "fd", "basis"]
DerivativeBasis: TypeAlias = Literal["poly", "fourier", "sine", "cosine"]


class DerivativeRule(abc.ABC):
    """Explicit strategy for differentiating a bound domain function."""

    @abc.abstractmethod
    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        """Return the requested derivative, or ``None`` to use generic lowering."""
        raise NotImplementedError


DerivativeCallback: TypeAlias = Callable[..., "DomainFunction | None"]


@dataclass(frozen=True, slots=True, eq=False)
class CallbackDerivativeRule(DerivativeRule):
    """Typed adapter for a custom derivative callback."""

    callback: DerivativeCallback

    def __post_init__(self) -> None:
        if not callable(self.callback):
            raise TypeError("CallbackDerivativeRule.callback must be callable.")

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        return self.callback(
            var=var,
            axis=axis,
            order=int(order),
            mode=mode,
            backend=backend,
            basis=basis,
            periodic=bool(periodic),
        )


__all__ = [
    "CallbackDerivativeRule",
    "DerivativeBackend",
    "DerivativeBasis",
    "DerivativeCallback",
    "DerivativeMode",
    "DerivativeRule",
]
