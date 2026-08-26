#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CochainDiscretization


class CochainRatePartition(StrictModule, NonTrainableState):
    categories: tuple[Array, ...]
    maximum_category: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        cochain: CochainDiscretization,
        categories: Sequence[ArrayLike],
        /,
    ):
        values = [np.asarray(value, dtype=np.int32).copy() for value in categories]
        if len(values) != len(cochain.cell_counts):
            raise ValueError("Rate categories must cover every cochain degree.")
        for degree, (value, count) in enumerate(
            zip(values, cochain.cell_counts, strict=True)
        ):
            if value.shape != (count,) or np.any(value < 0):
                raise ValueError(f"Rate categories[{degree}] are invalid.")
        changed = True
        while changed:
            changed = False
            for degree, incidence in enumerate(cochain.topology.incidences):
                valid = np.asarray(incidence.relation.valid, dtype=bool)
                lower = np.asarray(incidence.relation.source_indices)[valid]
                upper = np.asarray(incidence.relation.target_indices)[valid]
                required = np.maximum(values[degree][lower], values[degree + 1][upper])
                lower_new = np.maximum(values[degree][lower], required - 1)
                upper_new = np.maximum(values[degree + 1][upper], required - 1)
                if np.any(lower_new != values[degree][lower]) or np.any(
                    upper_new != values[degree + 1][upper]
                ):
                    values[degree][lower] = lower_new
                    values[degree + 1][upper] = upper_new
                    changed = True
        maximum = max(int(np.max(value, initial=0)) for value in values)
        self.categories = tuple(jnp.asarray(value) for value in values)
        self.maximum_category = maximum
        self.partition_id = canonical_fingerprint(
            {
                "kind": "cochain-rate-partition",
                "cochain": cochain.prepared_id,
                "categories": [array_tree_fingerprint(value) for value in values],
            }
        )

    def active(self, degree: int, tick: int, total_ticks: int, /) -> Array:
        category = self.categories[int(degree)]
        stride = 2 ** (self.maximum_category - category)
        return (int(tick) % stride) == 0


class CochainMultirateDiagnostics(StrictModule):
    energy_before: Array
    energy_after: Array
    relative_energy_change: Array
    substeps: int = eqx.field(static=True)


class CochainMultiratePlan(StrictModule):
    """Power-of-two synchronized cochain scheduler with explicit masks."""

    partition: CochainRatePartition
    update: Callable[[Any, Array, tuple[Array, ...]], Any] = eqx.field(static=True)
    energy: Callable[[Any], ArrayLike] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        partition: CochainRatePartition,
        update: Callable[[Any, Array, tuple[Array, ...]], Any],
        energy: Callable[[Any], ArrayLike],
        /,
    ):
        if not isinstance(partition, CochainRatePartition):
            raise TypeError("partition must be CochainRatePartition.")
        if not callable(update) or not callable(energy):
            raise TypeError("Multirate update/energy must be callable.")
        self.partition = partition
        self.update = update
        self.energy = energy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cochain-multirate-plan",
                "partition": partition.partition_id,
                "update": repr(update),
                "energy": repr(energy),
            }
        )

    def advance(
        self,
        state: Any,
        step_size: ArrayLike,
        /,
    ) -> tuple[Any, CochainMultirateDiagnostics]:
        dt = jnp.asarray(step_size)
        ticks = 2**self.partition.maximum_category
        substep = dt / ticks
        before = jnp.asarray(self.energy(state))
        value = state
        for tick in range(ticks):
            masks = tuple(
                self.partition.active(degree, tick, ticks)
                for degree in range(len(self.partition.categories))
            )
            value = self.update(value, substep, masks)
        after = jnp.asarray(self.energy(value))
        change = jnp.abs(after - before) / jnp.maximum(1.0, jnp.abs(before))
        return value, CochainMultirateDiagnostics(before, after, change, ticks)


__all__ = [
    "CochainMultirateDiagnostics",
    "CochainMultiratePlan",
    "CochainRatePartition",
]
