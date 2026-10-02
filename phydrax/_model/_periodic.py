#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Construction evidence that a model is periodic in selected input coordinates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

import equinox as eqx

from .._differentiation import AbstractConstructionCertificate, DerivativeRegularity
from .._fingerprint import canonical_fingerprint
from .._validation import nonnegative_integer
from ..typing import checked


class PeriodicInputCertificate(AbstractConstructionCertificate):
    """Construction claim that a model is exactly periodic in flat input entries.

    ``periodic_inputs`` lists ``(flat_index, period)`` pairs of the flattened model
    input: ``f(x + period e_i) = f(x)`` holds for every parameter value because
    every path from input entry ``i`` passes through features that are invariant
    under that translation. ``regularity`` is the declared regularity of the
    complete model, which bounds the derivative orders whose periodicity is
    claimed. The certificate says nothing about entries it does not list.
    """

    capability_id: ClassVar[str] = "periodic-input"
    input_size: int = eqx.field(static=True)
    periodic_inputs: tuple[tuple[int, float], ...] = eqx.field(static=True)
    regularity: DerivativeRegularity
    certificate_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        input_size: int,
        periodic_inputs: Sequence[tuple[int, float]],
        regularity: DerivativeRegularity,
    ) -> None:
        if isinstance(input_size, bool) or not isinstance(input_size, int):
            raise TypeError("input_size must be an integer.")
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        entries = tuple(
            sorted((index, float(period)) for index, period in periodic_inputs)
        )
        if not entries:
            raise ValueError("A periodic input certificate lists at least one entry.")
        indices = tuple(index for index, _ in entries)
        if len(set(indices)) != len(indices):
            raise ValueError("Periodic input entries must be unique.")
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise TypeError("Periodic input indices must be integers.")
        if any(index < 0 or index >= input_size for index in indices):
            raise ValueError("Periodic input indices must lie inside the model input.")
        if any(not math.isfinite(period) or period <= 0.0 for _, period in entries):
            raise ValueError("Periodic input periods must be finite and positive.")
        self.input_size = input_size
        self.periodic_inputs = entries
        self.regularity = regularity
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "periodic-input-certificate",
                "input_size": input_size,
                "periodic_inputs": [list(entry) for entry in entries],
                "continuity": regularity.continuity,
                "pieces": regularity.pieces,
            }
        )

    def period_of(self, index: int, /) -> float | None:
        """Certified period of one flat input entry, or ``None`` if uncertified."""
        for entry, period in self.periodic_inputs:
            if entry == index:
                return period
        return None

    def supports_order(self, order: int, /) -> bool:
        """Whether periodicity of the ``order``-th derivative follows from regularity."""
        order = nonnegative_integer(order, "derivative order")
        continuity = self.regularity.continuity
        return continuity == "smooth" or continuity >= order


__all__ = ["PeriodicInputCertificate"]
