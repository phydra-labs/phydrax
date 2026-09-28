#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact integer label-volume constraints solved by capacitated auction.

Following the auction-dynamics scheme of Jacobs, Merkurjev and Esedoglu (J. Comput.
Phys. 354, 2018), the thresholding step is replaced by the assignment that
minimizes ``sum_x psi_{l(x)}(x)`` subject to integer site counts
``lower_i <= #{x : l(x) = i} <= upper_i``. Volumes are exact only because every site
carries the same measure; routes with unequal site measures are refused. Auction
prices are numerical duals of the count constraints, not physical pressures.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import fixed_field, NonTrainableState
from .._validation import positive_integer
from ..combinatorial import CapacitatedAuctionPlan, EpsilonScale
from ..typing import as_host_array, Dim, HostInteger, Int32, parse, Scope


class _ConstraintLabelDim(Dim, minimum=2):
    """Declared labels carrying a count constraint."""


class LabelVolumeConstraint(StrictModule, NonTrainableState):
    """Integer site-count bounds per label and the auction configuration.

    ``upper_counts=None`` requests exact counts (``lower == upper``). Counts are
    dynamic integer leaves (they may change between steps); the auction schedule is
    static. Extinct labels are forced to zero count.
    """

    __strict_contract__ = True

    lower_counts: Int32[_ConstraintLabelDim] = fixed_field()
    upper_counts: Int32[_ConstraintLabelDim] = fixed_field()
    exact: bool = eqx.field(static=True)
    epsilon_schedule: tuple[float, ...] = eqx.field(static=True)
    epsilon_scale: EpsilonScale = eqx.field(static=True)
    maximum_rounds: int = eqx.field(static=True)

    def __init__(
        self,
        lower_counts: ArrayLike,
        upper_counts: ArrayLike | None = None,
        /,
        *,
        epsilon_schedule: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        epsilon_scale: EpsilonScale = "value-range",
        maximum_rounds: int = 20_000,
    ) -> None:
        scope = Scope()
        lower = as_host_array(
            lower_counts, HostInteger[_ConstraintLabelDim], "lower_counts", scope=scope
        )
        upper = (
            lower
            if upper_counts is None
            else as_host_array(
                upper_counts,
                HostInteger[_ConstraintLabelDim],
                "upper_counts",
                scope=scope,
            )
        )
        int32_maximum = np.iinfo(np.int32).max
        if np.any(lower < 0) or np.any(upper < 0):
            raise ValueError("Label counts must be nonnegative.")
        if np.any(lower > int32_maximum) or np.any(upper > int32_maximum):
            raise ValueError("Label counts must fit in signed int32 before conversion.")
        if np.any(upper < lower):
            raise ValueError("Label counts need lower_counts <= upper_counts.")
        schedule = tuple(float(value) for value in epsilon_schedule)
        rounds = positive_integer(maximum_rounds, "maximum_rounds")
        # The auction plan owns the schedule validation; construct it once here so
        # an invalid configuration fails at declaration, not at preparation.
        CapacitatedAuctionPlan(
            1,
            lower.shape[0],
            lower.shape[0],
            epsilon_schedule=schedule,
            epsilon_scale=epsilon_scale,
            maximum_rounds=rounds,
        )
        self.lower_counts = jnp.asarray(lower, dtype=jnp.int32)
        self.upper_counts = jnp.asarray(upper, dtype=jnp.int32)
        self.exact = upper_counts is None or bool(np.array_equal(lower, upper))
        self.epsilon_schedule = schedule
        self.epsilon_scale = parse(epsilon_scale, EpsilonScale, "epsilon_scale")
        self.maximum_rounds = rounds

    @property
    def label_count(self) -> int:
        return self.lower_counts.shape[0]

    def auction_plan(
        self, site_count: int, candidate_width: int, /
    ) -> CapacitatedAuctionPlan:
        return CapacitatedAuctionPlan(
            site_count,
            self.label_count,
            candidate_width,
            epsilon_schedule=self.epsilon_schedule,
            epsilon_scale=self.epsilon_scale,
            maximum_rounds=self.maximum_rounds,
        )


__all__ = ["LabelVolumeConstraint"]
