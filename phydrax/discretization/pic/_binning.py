#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cell-sorted particle binning with bounded work.

Particles are keyed by the flat index of the uniform grid cell containing
them and grouped by `phydrax.sparse.KeyGroupPlan` in canonical
``(cell, identity)`` order. The identity is a lexicographic key tuple — the
persistent global ID words ``(id_hi, id_lo)`` for populations, or any other
slot-independent payload — so the binned order, the per-cell membership, and
every reduction taken in that order are invariant to storage-slot order.
Work is bounded by static capacities: one ``O(N log N)`` sort over the
particle capacity and fixed-size group arrays; exceeding the occupied-cell or
per-cell capacity is reported as overflow, never truncated silently.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import KeyGroupPlan


class PICCellBins(StrictModule):
    """Canonical cell grouping of one particle capacity.

    ``cell[N]`` is the flat cell of each slot (``-1`` when unbinned);
    ``order[N]`` lists slots in ``(cell, identity)`` order with unbinned slots
    last; ``slot_group[N]`` is each slot's compact group (``-1`` when unbinned).
    Group ``g`` covers ``order[cell_starts[g] : cell_starts[g] + cell_counts[g]]``
    of flat cell ``cell_keys[g]`` when ``occupied[g]``. ``outside`` counts active
    particles beyond nonperiodic bounds, ``identity_ties`` counts binned
    particles sharing their cell and complete identity key, and ``overflow``
    flags an exceeded occupied-cell or per-cell capacity.
    """

    cell: Array
    order: Array
    slot_group: Array
    binned: Array
    cell_keys: Array
    cell_starts: Array
    cell_counts: Array
    occupied: Array
    occupied_cells: Array
    maximum_cell_count: Array
    outside: Array
    identity_ties: Array
    overflow: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _lexicographic_rank(keys: tuple[Array, ...], /) -> tuple[Array, Array]:
    """Unique rank of each slot in lexicographic key order, and tie flags.

    ``keys`` are most significant first. Ties are broken by slot index; the
    returned flags mark slots whose complete key equals their predecessor's.
    """
    count = keys[0].shape[0]
    order = jnp.lexsort(tuple(reversed(keys))).astype(jnp.int32)
    rank = (
        jnp.zeros((count,), dtype=jnp.int32)
        .at[order]
        .set(jnp.arange(count, dtype=jnp.int32))
    )
    equal = jnp.ones((count - 1,), dtype=jnp.bool_) if count > 1 else None
    if equal is None:
        return rank, jnp.zeros((count,), dtype=jnp.bool_)
    for value in keys:
        ordered = value[order]
        equal = equal & (ordered[1:] == ordered[:-1])
    sorted_tie = jnp.concatenate((jnp.zeros((1,), dtype=jnp.bool_), equal))
    tie = jnp.zeros((count,), dtype=jnp.bool_).at[order].set(sorted_tie)
    return rank, tie


class PICCellBinningPlan(StrictModule, NonTrainableState):
    """Uniform-cell particle binning with static capacity bounds."""

    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    maximum_particles_per_cell: int | None = eqx.field(static=True)
    maximum_occupied_cells: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        shape: Sequence[int],
        periodic: Sequence[bool],
        /,
        *,
        maximum_particles_per_cell: int | None = None,
        maximum_occupied_cells: int | None = None,
    ) -> None:
        lower_ = tuple(float(value) for value in lower)
        upper_ = tuple(float(value) for value in upper)
        shape_ = tuple(int(value) for value in shape)
        periodic_ = tuple(bool(value) for value in periodic)
        if not shape_ or not (
            len(lower_) == len(upper_) == len(shape_) == len(periodic_)
        ):
            raise ValueError("Binning bounds, shape, and periodicity must align.")
        if any(count <= 0 for count in shape_):
            raise ValueError("Binning cell counts must be positive.")
        if any(
            not (np.isfinite(low) and np.isfinite(high) and high > low)
            for low, high in zip(lower_, upper_, strict=True)
        ):
            raise ValueError("Binning bounds must be finite with upper > lower.")
        for name, value in (
            ("maximum_particles_per_cell", maximum_particles_per_cell),
            ("maximum_occupied_cells", maximum_occupied_cells),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided.")
        self.lower = lower_
        self.upper = upper_
        self.shape = shape_
        self.periodic = periodic_
        self.maximum_particles_per_cell = (
            None
            if maximum_particles_per_cell is None
            else int(maximum_particles_per_cell)
        )
        self.maximum_occupied_cells = (
            None if maximum_occupied_cells is None else int(maximum_occupied_cells)
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pic-cell-binning-plan",
                "lower": list(lower_),
                "upper": list(upper_),
                "shape": list(shape_),
                "periodic": list(periodic_),
                "maximum_particles_per_cell": self.maximum_particles_per_cell,
                "maximum_occupied_cells": self.maximum_occupied_cells,
            }
        )

    @property
    def cell_count(self) -> int:
        return int(np.prod(self.shape))

    def cell_index(self, position: ArrayLike, /) -> tuple[Array, Array]:
        """Flat containing cell of each position and its in-domain mask."""
        points = jnp.asarray(position)
        if points.ndim != 2 or points.shape[1] != len(self.shape):
            raise ValueError("Binning positions must have shape (N, dimension).")
        flat = jnp.zeros((points.shape[0],), dtype=jnp.int32)
        inside = jnp.all(jnp.isfinite(points), axis=-1)
        for axis, count in enumerate(self.shape):
            width = (self.upper[axis] - self.lower[axis]) / count
            normalized = (points[:, axis] - self.lower[axis]) / width
            raw = jnp.floor(jnp.where(jnp.isfinite(normalized), normalized, 0.0))
            if self.periodic[axis]:
                index = jnp.mod(raw, count)
            else:
                # The closed upper bound belongs to the last cell.
                inside = inside & (normalized >= 0.0) & (normalized <= count)
                index = jnp.clip(raw, 0, count - 1)
            flat = flat * count + index.astype(jnp.int32)
        return flat, inside

    def bin(
        self,
        position: ArrayLike,
        active: ArrayLike,
        /,
        *,
        identity: Sequence[ArrayLike] = (),
    ) -> PICCellBins:
        """Group active particles by cell in canonical ``(cell, identity)`` order.

        ``identity`` is a lexicographic key tuple, most significant first,
        of slot-independent integer or real per-particle arrays; empty, the
        position coordinates are the key.
        """
        points = jnp.asarray(position)
        active_ = jnp.asarray(active, dtype=jnp.bool_)
        count = points.shape[0]
        if active_.shape != (count,):
            raise ValueError("active must have particle-capacity shape.")
        keys = (
            tuple(points[:, axis] for axis in range(points.shape[1]))
            if not identity
            else tuple(jnp.asarray(value) for value in identity)
        )
        if any(value.shape != (count,) for value in keys):
            raise ValueError("Every identity key must have particle-capacity shape.")
        cell, inside = self.cell_index(points)
        binned = active_ & inside
        rank, tie = _lexicographic_rank((jnp.where(binned, cell, -1), *keys))
        occupied_capacity = min(count, self.cell_count)
        if self.maximum_occupied_cells is not None:
            occupied_capacity = min(occupied_capacity, self.maximum_occupied_cells)
        groups = KeyGroupPlan(
            count,
            max(occupied_capacity, 1),
            self.cell_count - 1,
            maximum_group_size=self.maximum_particles_per_cell,
        ).build(cell, binned, stable_ids=rank)
        evidence = groups.evidence
        overflow = evidence.group_overflow | evidence.member_overflow
        return PICCellBins(
            jnp.where(binned, cell, -1),
            groups.storage_to_logical,
            jnp.where(binned, groups.item_group_slots, -1),
            binned,
            groups.group_keys,
            groups.group_starts,
            groups.group_counts,
            groups.group_active,
            evidence.required_groups,
            evidence.maximum_group_size,
            jnp.sum(active_ & ~inside, dtype=jnp.int32),
            jnp.sum(binned & tie, dtype=jnp.int32),
            overflow,
            ~overflow,
            self.plan_id,
        )


__all__ = ["PICCellBinningPlan", "PICCellBins"]
