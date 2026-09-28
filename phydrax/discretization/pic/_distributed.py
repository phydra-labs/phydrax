#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Static block decomposition, halo accumulation, and particle migration for PIC.

A `PICDomainDecomposition` splits the leading ``k`` axes of a uniform structured
field over a ``k``-axis device mesh (``k = 1``: slabs; ``k = 2, 3``: pencils and
blocks). Device ``(p₀, …, p_{k−1})`` owns the cells ``[p_a n_a, (p_a+1) n_a)``
along every decomposed axis ``a`` (``n_a = N_a/P_a``) and, with linear index
``p = Σ p_a Π_{b>a} P_b`` (row-major over the mesh axes), the particle slots
``[p c, (p+1) c)`` of every species (``c = C/P``), so particle ownership is
aligned with the field decomposition. Deposition and gather run per device on
the block's window (its owned cells plus ``guard_cells`` cells on each side of
every decomposed axis). One plane-level `phydrax.discretization.DistributedHaloPlan`
per decomposed axis moves guard contributions axis by axis: accumulation sums
guards into their owners along axis 0, then along axis 1, …; guard filling
exchanges along each axis in turn with the previous axes' guards included, so
edge and corner guards (diagonal neighbors) are filled without extra messages.
Particles may lie up to ``particle_margin`` cells outside their block between
migrations.

`PICMigrationPlan` routes every particle whose owner changed through
fixed-capacity ``lax.ppermute`` packets, one packet per mesh offset in
``{−reach, …, reach}^k \\ {0}`` (diagonal neighbors included), for any
`PICSlotGroup` — a PIC species with its slot-aligned process state, or a
process-owned bank. A migrating particle is deactivated in its source slot
(the slot keeps its occupant's identity) and allocated in a free destination
slot with its persistent identity, lineage, and payload (the slot incarnation
advances). Any packet overflow, unreachable owner, or capacity shortfall on
any device rejects the whole migration; the caller rejects the step atomically.

`PICIdentityAllocator` allocates the particles a process creates on one device
within the device's slot block and gives them identities that do not depend
on the decomposition and need no communication: every allocation call reserves
``B·R`` identities after the population counter (``B`` identity tiles — a fixed
partition of the decomposed cells nested in every admissible device block —
and ``R`` the population's global capacity), and the created particle of an
event in tile ``t`` with rank ``r`` among the call's events in ``t`` (in the
process's canonical event order) receives ``counter + t R + r``. Event tiles are
those of the creating event's position, which is always on the creating
device's block, so every tile's identities are assigned by exactly one device.

The decomposition is static: ownership changes only by migration, and a
different mesh is admitted only when a run is restarted
(`repartition_destinations`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._distributed_field import DistributedHaloPlan
from ..particle import (
    ParticleAllocationRequest,
    ParticleAllocationResult,
    ParticlePopulationPlan,
    ParticlePopulationState,
    ParticlePopulationStatus,
    ParticleSlotReusePolicy,
)
from ._charge_state import PICChargeState, PICSpeciesState
from ._process import AbstractPICParticleAllocator
from ._types import PICParticleState


_NO_IDENTITY = np.uint64(0xFFFFFFFFFFFFFFFF)


def _plane_halo(
    parts: int, cells: int, guards: int, periodic: bool, /
) -> DistributedHaloPlan:
    """Plane halo of ``parts`` equal slabs: every plane within ``guards`` planes."""
    owner = np.arange(cells, dtype=np.int64) // (cells // parts)
    pairs = []
    for offset in range(1, guards + 1):
        left = np.arange(cells, dtype=np.int64)
        right = left + offset
        if periodic:
            right = right % cells
        keep = (right < cells) & (right != left)
        pairs.append(np.stack((left[keep], right[keep]), axis=1))
    return DistributedHaloPlan(owner, np.concatenate(pairs, axis=0), parts)


def _axis_tuple(
    values: Sequence[int] | int, count: int | None, name: str, /
) -> tuple[int, ...]:
    result = (
        (int(values),)
        if isinstance(values, int | np.integer)
        else tuple(int(value) for value in values)
    )
    if not result or (count is not None and len(result) != count):
        raise ValueError(f"{name} must give one value per decomposed axis.")
    return result


class PICGuardWindow(StrictModule, NonTrainableState):
    """Contiguous guard-padded windows of the owned blocks of a periodic grid.

    ``extend(owned, coordinates, axis_names)`` returns the device's owned cells
    with ``guards[a]`` cells exchanged from the neighbors added on both sides of
    every decomposed axis (periodic), in natural cell order.
    """

    halos: tuple[DistributedHaloPlan, ...]
    rows: tuple[Array, ...]
    guards: tuple[int, ...] = eqx.field(static=True)
    owned: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self, parts: tuple[int, ...], counts: tuple[int, ...], guards: tuple[int, ...], /
    ) -> None:
        halos, rows = [], []
        for part_count, cells, guard in zip(parts, counts, guards, strict=True):
            if guard < 0 or guard >= cells:
                raise ValueError("Guard widths must lie in [0, cell count).")
            owned = cells // part_count
            halo = _plane_halo(part_count, cells, max(guard, 1), True)
            ids = np.asarray(halo.local_global_ids)
            valid = np.asarray(halo.local_valid)
            table = np.zeros((part_count, owned + 2 * guard), dtype=np.int32)
            for part in range(part_count):
                position = {
                    int(value): index
                    for index, value in enumerate(ids[part])
                    if valid[part, index]
                }
                wanted = (part * owned - guard + np.arange(owned + 2 * guard)) % cells
                table[part] = [position[int(value)] for value in wanted]
            halos.append(halo)
            rows.append(jnp.asarray(table))
        self.halos = tuple(halos)
        self.rows = tuple(rows)
        self.guards = guards
        self.owned = tuple(
            cells // part for part, cells in zip(parts, counts, strict=True)
        )

    def extend(
        self,
        owned: Array,
        coordinates: tuple[Array, ...],
        axis_names: tuple[str, ...],
        /,
    ) -> Array:
        values = owned
        for axis, (halo, rows, name) in enumerate(
            zip(self.halos, self.rows, axis_names, strict=True)
        ):
            moved = jnp.moveaxis(values, axis, 0)
            buffer = (
                jnp.zeros((halo.local_capacity,) + moved.shape[1:], dtype=moved.dtype)
                .at[: self.owned[axis]]
                .set(moved)
            )
            exchanged = halo.exchange(buffer, coordinates[axis], axis_name=name)
            values = jnp.moveaxis(exchanged[rows[coordinates[axis]]], 0, axis)
        return values


class PICDomainDecomposition(StrictModule, NonTrainableState):
    """Equal whole-cell blocks of the leading axes of a uniform structured grid.

    ``parts[a]`` devices split the ``counts[a]`` cells of axis ``a``.
    ``halos[a]`` is axis ``a``'s plane-level halo plan (plane ``i`` owned by
    slab ``i // n_a``; every plane within ``guard_cells`` of a slab is in its
    halo) and ``window_masks[a][p, i]`` marks the owned and halo planes of slab
    ``p``; a device window is the product of its slabs' windows.

    A device's deposit is window-local up to, along each decomposed axis, a
    component uniform along that axis (the mean current of a periodic
    continuity-projected grid, which may vary along the other axes): axis by
    axis, that component is read from a plane outside the slab window, summed
    over the mesh axis, and added to every owned plane; the remainder is
    halo-accumulated.

    ``identity_tiles[a]`` splits axis ``a`` into identity tiles for
    `PICIdentityAllocator` (default: one tile per cell); ``parts[a]`` must
    divide it, so tiles never straddle device blocks.
    """

    halos: tuple[DistributedHaloPlan, ...]
    window_masks: tuple[Array, ...]
    outside_planes: tuple[Array, ...]
    has_outside: tuple[Array, ...]
    parts: tuple[int, ...] = eqx.field(static=True)
    counts: tuple[int, ...] = eqx.field(static=True)
    lower: tuple[float, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    guard_cells: int = eqx.field(static=True)
    particle_margin: float = eqx.field(static=True)
    identity_tiles: tuple[int, ...] = eqx.field(static=True)
    decomposition_id: str = eqx.field(static=True)

    def __init__(
        self,
        parts: Sequence[int] | int,
        counts: Sequence[int] | int,
        lower: Sequence[float] | float,
        spacing: Sequence[float] | float,
        /,
        *,
        periodic: Sequence[bool] | bool,
        guard_cells: int = 3,
        particle_margin: float = 1.0,
        identity_tiles: Sequence[int] | None = None,
    ) -> None:
        parts_ = _axis_tuple(parts, None, "parts")
        axes = len(parts_)
        counts_ = _axis_tuple(counts, axes, "counts")
        lower_ = (
            (float(lower),)
            if isinstance(lower, float | int)
            else tuple(float(value) for value in lower)
        )
        spacing_ = (
            (float(spacing),)
            if isinstance(spacing, float | int)
            else tuple(float(value) for value in spacing)
        )
        periodic_ = (
            (bool(periodic),)
            if isinstance(periodic, bool | np.bool_)
            else tuple(bool(value) for value in periodic)
        )
        if not (len(lower_) == len(spacing_) == len(periodic_) == axes) or axes > 3:
            raise ValueError("PIC decompositions split one to three leading axes.")
        guards, margin = int(guard_cells), float(particle_margin)
        if any(p <= 0 or n <= 0 or n % p for p, n in zip(parts_, counts_, strict=True)):
            raise ValueError("PIC blocks require part counts dividing the cell counts.")
        if not all(
            np.isfinite(origin) and np.isfinite(width) and width > 0.0
            for origin, width in zip(lower_, spacing_, strict=True)
        ):
            raise ValueError("PIC decomposition geometry must be finite and positive.")
        if guards < 1 or any(guards >= n for n in counts_):
            raise ValueError("guard_cells must lie in [1, cell count).")
        if not (np.isfinite(margin) and 0.0 <= margin < guards):
            raise ValueError("particle_margin must lie in [0, guard_cells).")
        tiles = (
            counts_
            if identity_tiles is None
            else _axis_tuple(identity_tiles, axes, "identity_tiles")
        )
        if any(
            t <= 0 or n % t or t % p
            for t, n, p in zip(tiles, counts_, parts_, strict=True)
        ):
            raise ValueError(
                "identity_tiles must divide the cell counts and be divisible by parts."
            )
        halos, masks, outside_planes, has_outside = [], [], [], []
        for part_count, cells, wrap in zip(parts_, counts_, periodic_, strict=True):
            halo = _plane_halo(part_count, cells, guards, wrap)
            ids = np.asarray(halo.local_global_ids)
            valid = np.asarray(halo.local_valid)
            window = np.zeros((part_count, cells), dtype=np.bool_)
            for part in range(part_count):
                window[part, ids[part][valid[part]]] = True
            outside = ~window
            halos.append(halo)
            masks.append(jnp.asarray(window))
            outside_planes.append(
                jnp.asarray(np.argmax(outside, axis=1), dtype=jnp.int32)
            )
            has_outside.append(jnp.asarray(np.any(outside, axis=1)))
        self.halos = tuple(halos)
        self.window_masks = tuple(masks)
        self.outside_planes = tuple(outside_planes)
        self.has_outside = tuple(has_outside)
        self.parts = parts_
        self.counts = counts_
        self.lower = lower_
        self.spacing = spacing_
        self.periodic = periodic_
        self.guard_cells = guards
        self.particle_margin = margin
        self.identity_tiles = tiles
        self.decomposition_id = canonical_fingerprint(
            {
                "kind": "pic-domain-decomposition",
                "parts": list(parts_),
                "counts": list(counts_),
                "lower": list(lower_),
                "spacing": list(spacing_),
                "periodic": list(periodic_),
                "guard_cells": guards,
                "particle_margin": margin,
                "identity_tiles": list(tiles),
                "halos": [value.plan_id for value in halos],
            }
        )

    # -- geometry ---------------------------------------------------------------------

    @property
    def axis_count(self) -> int:
        return len(self.parts)

    @property
    def part_count(self) -> int:
        return math.prod(self.parts)

    @property
    def block_cells(self) -> tuple[int, ...]:
        return tuple(n // p for n, p in zip(self.counts, self.parts, strict=True))

    @property
    def tile_count(self) -> int:
        return math.prod(self.identity_tiles)

    def linear(self, coordinates: Sequence[Array | int], /) -> Array:
        """Row-major linear part index of mesh coordinates."""
        index = jnp.asarray(0, dtype=jnp.int32)
        for coordinate, count in zip(coordinates, self.parts, strict=True):
            index = index * count + jnp.asarray(coordinate, dtype=jnp.int32)
        return index

    def coordinates(self, part: int, /) -> tuple[int, ...]:
        """Mesh coordinates of linear part ``part`` (host)."""
        return tuple(int(value) for value in np.unravel_index(part, self.parts))

    def _cells(self, position: Array, /) -> tuple[Array, ...]:
        cells = []
        for axis in range(self.axis_count):
            cell = jnp.floor(
                (position[:, axis] - self.lower[axis]) / self.spacing[axis]
            ).astype(jnp.int32)
            count = self.counts[axis]
            cells.append(
                jnp.mod(cell, count)
                if self.periodic[axis]
                else jnp.clip(cell, 0, count - 1)
            )
        return tuple(cells)

    def owner_coordinates(self, position: Array, /) -> tuple[Array, ...]:
        """Owning mesh coordinates of positions ``[N, d]`` (periodic images wrap)."""
        return tuple(
            cell // width
            for cell, width in zip(self._cells(position), self.block_cells, strict=True)
        )

    def owner(self, position: Array, /) -> Array:
        """Owning linear part of positions ``[N, d]``."""
        return self.linear(self.owner_coordinates(position))

    def in_region(self, position: Array, coordinates: tuple[Array, ...], /) -> Array:
        """Whether positions lie within ``particle_margin`` cells of the block."""
        inside = jnp.ones(position.shape[:1], dtype=jnp.bool_)
        margin = self.particle_margin
        for axis, (coordinate, width) in enumerate(
            zip(coordinates, self.block_cells, strict=True)
        ):
            offset = (position[:, axis] - self.lower[axis]) / self.spacing[
                axis
            ] - coordinate * width
            if self.periodic[axis]:
                wrapped = jnp.mod(offset, self.counts[axis])
                inside = inside & (
                    (wrapped < width + margin) | (wrapped >= self.counts[axis] - margin)
                )
            else:
                inside = inside & (offset >= -margin) & (offset < width + margin)
        return inside

    def tile(self, position: Array, /) -> Array:
        """Row-major identity tile of positions ``[N, d]``."""
        index = jnp.zeros(position.shape[:1], dtype=jnp.int32)
        for cell, count, tiles in zip(
            self._cells(position), self.counts, self.identity_tiles, strict=True
        ):
            index = index * tiles + cell // (count // tiles)
        return index

    def owns_tile(self, tile: Array, coordinates: tuple[Array, ...], /) -> Array:
        """Whether identity tiles lie in the block at ``coordinates``."""
        owned = jnp.ones(tile.shape, dtype=jnp.bool_)
        remainder = tile
        for axis in reversed(range(self.axis_count)):
            tiles = self.identity_tiles[axis]
            along = remainder % tiles
            remainder = remainder // tiles
            per_part = tiles // self.parts[axis]
            owned = owned & (along // per_part == coordinates[axis])
        return owned

    def window_indicator(self, part: int, /) -> np.ndarray:
        """Host mask of part ``part``'s window over the decomposed cells."""
        indicator = np.ones(self.counts, dtype=np.bool_)
        for axis, coordinate in enumerate(self.coordinates(part)):
            shape = [1] * self.axis_count
            shape[axis] = self.counts[axis]
            indicator = indicator & np.asarray(
                self.window_masks[axis][coordinate]
            ).reshape(shape)
        return indicator

    def guard_window(self, guards: Sequence[int], /) -> PICGuardWindow:
        """Contiguous guard-padded windows (periodic axes only)."""
        if not all(self.periodic):
            raise ValueError("Guard windows require periodic decomposed axes.")
        return PICGuardWindow(
            self.parts, self.counts, _axis_tuple(guards, self.axis_count, "guards")
        )

    # -- halo accumulation ------------------------------------------------------------

    def _axis_uniform(self, values: Array, axis: int, coordinate: Array, /) -> Array:
        """The component of ``values`` uniform along decomposed axis ``axis``.

        Read at a plane outside the slab window along ``axis`` (zero when the
        window covers the axis); size one along ``axis``.
        """
        plane = jnp.take(values, self.outside_planes[axis][coordinate], axis=axis)
        plane = jnp.expand_dims(plane, axis)
        return jnp.where(self.has_outside[axis][coordinate], plane, jnp.zeros_like(plane))

    def _outside_along(self, axis: int, coordinate: Array, ndim: int, /) -> Array:
        shape = [1] * ndim
        shape[axis] = self.counts[axis]
        return ~self.window_masks[axis][coordinate].reshape(shape)

    def outside_window(self, values: Array, coordinates: tuple[Array, ...], /) -> Array:
        """Whether an array is not window-local up to per-axis uniform components.

        Along each decomposed axis in turn, a component uniform along that axis
        (the mean current of a periodic continuity-projected grid, possibly
        varying along the other axes) is removed and the rest must vanish
        outside the window. The residual may carry the roundoff of the solver's
        own accumulation over the grid (``64 ε M max|values|``, ``M`` the
        decomposed cells); a footprint outside the window leaks ``O(max|values|)``.
        """
        local = values
        leaked = jnp.zeros((), dtype=values.dtype)
        for axis, coordinate in enumerate(coordinates):
            residual = local - self._axis_uniform(local, axis, coordinate)
            outside = self._outside_along(axis, coordinate, values.ndim)
            leaked = jnp.maximum(
                leaked, jnp.max(jnp.where(outside, jnp.abs(residual), 0.0), initial=0.0)
            )
            local = jnp.where(outside, 0.0, residual)
        scale = jnp.max(jnp.abs(values), initial=0.0)
        cells = math.prod(self.counts)
        return leaked > 64.0 * cells * jnp.finfo(values.dtype).eps * scale

    def accumulate(
        self,
        values: Array,
        coordinates: tuple[Array, ...],
        /,
        *,
        axis_names: tuple[str, ...],
    ) -> Array:
        """Owned cells of the sum over devices of per-device window-local arrays.

        Axis by axis: the component uniform along the axis is summed over that
        mesh axis and the remainder is halo-accumulated into its owners.
        """
        local = values
        for axis, (halo, coordinate, name) in enumerate(
            zip(self.halos, coordinates, axis_names, strict=True)
        ):
            uniform = self._axis_uniform(local, axis, coordinate)
            moved = jnp.moveaxis(local - uniform, axis, 0)
            valid = halo.local_valid[coordinate]
            window = jnp.where(
                valid.reshape(valid.shape + (1,) * (moved.ndim - 1)),
                moved[halo.local_global_ids[coordinate]],
                0.0,
            )
            summed = halo.accumulate_halo(window, coordinate, axis_name=name)
            local = jnp.moveaxis(
                summed[: self.block_cells[axis]], 0, axis
            ) + jax.lax.psum(uniform, name)
        return local

    def fill_window(
        self,
        owned: Array,
        coordinates: tuple[Array, ...],
        /,
        *,
        axis_names: tuple[str, ...],
    ) -> Array:
        """Global-shaped array holding the block's owned and guard cells.

        Guards are exchanged from their owners axis by axis (edge and corner
        guards included); cells outside the window are zero.
        """
        values = owned
        for axis, (halo, coordinate, name) in enumerate(
            zip(self.halos, coordinates, axis_names, strict=True)
        ):
            moved = jnp.moveaxis(values, axis, 0)
            buffer = (
                jnp.zeros((halo.local_capacity,) + moved.shape[1:], dtype=moved.dtype)
                .at[: self.block_cells[axis]]
                .set(moved)
            )
            values = jnp.moveaxis(
                halo.exchange(buffer, coordinate, axis_name=name), 0, axis
            )
        ids = tuple(
            jnp.where(
                halo.local_valid[coordinate], halo.local_global_ids[coordinate], count
            )
            for halo, coordinate, count in zip(
                self.halos, coordinates, self.counts, strict=True
            )
        )
        full = jnp.zeros(self.counts + owned.shape[self.axis_count :], dtype=owned.dtype)
        return full.at[jnp.ix_(*ids)].set(values, mode="drop")


def species_slot_leaves(state: PICSpeciesState, /) -> tuple[Array, ...]:
    """Every per-slot leaf of a species state, in a fixed order."""
    population = state.population
    charge = state.charge
    return (
        state.particles.position,
        state.particles.proper_velocity,
        population.active,
        population.mass,
        population.incarnation,
        population.ever_occupied,
        population.retired,
        population.id_hi,
        population.id_lo,
        population.parent_hi,
        population.parent_lo,
        charge.charge_number,
        charge.transition_count,
        charge.last_transition_step,
    )


def with_species_slot_leaves(
    state: PICSpeciesState, leaves: Sequence[Array], /
) -> PICSpeciesState:
    """Species state with its per-slot leaves replaced (population counters kept)."""
    population = state.population
    return assemble_species(leaves, (population.next_id_hi, population.next_id_lo))


def assemble_species(
    leaves: Sequence[Array], counters: tuple[Array, Array], /
) -> PICSpeciesState:
    """Species state from `species_slot_leaves` and the population ID counter."""
    (
        position,
        proper_velocity,
        active,
        mass,
        incarnation,
        ever_occupied,
        retired,
        id_hi,
        id_lo,
        parent_hi,
        parent_lo,
        charge_number,
        transition_count,
        last_transition_step,
    ) = leaves
    next_id_hi, next_id_lo = counters
    return PICSpeciesState(
        PICParticleState(position, proper_velocity),
        ParticlePopulationState(
            active,
            mass,
            incarnation,
            ever_occupied,
            retired,
            id_hi,
            id_lo,
            parent_hi,
            parent_lo,
            next_id_hi,
            next_id_lo,
        ),
        PICChargeState(charge_number, transition_count, last_transition_step),
    )


def population_slot_leaves(population: ParticlePopulationState, /) -> tuple[Array, ...]:
    """Every per-slot leaf of a population state, in a fixed order."""
    return (
        population.active,
        population.mass,
        population.incarnation,
        population.ever_occupied,
        population.retired,
        population.id_hi,
        population.id_lo,
        population.parent_hi,
        population.parent_lo,
    )


def with_population_slot_leaves(
    population: ParticlePopulationState, leaves: Sequence[Array], /
) -> ParticlePopulationState:
    """Population state with its per-slot leaves replaced (counters kept)."""
    return assemble_population(leaves, (population.next_id_hi, population.next_id_lo))


def assemble_population(
    leaves: Sequence[Array], counters: tuple[Array, Array], /
) -> ParticlePopulationState:
    """Population state from `population_slot_leaves` and the identity counter."""
    active, mass, incarnation, ever, retired, id_hi, id_lo, parent_hi, parent_lo = leaves
    return ParticlePopulationState(
        active,
        mass,
        incarnation,
        ever,
        retired,
        id_hi,
        id_lo,
        parent_hi,
        parent_lo,
        counters[0],
        counters[1],
    )


def repartition_destinations(
    decomposition: PICDomainDecomposition,
    position: Array,
    active: Array,
    /,
) -> tuple[Array, Array]:
    """Slot permutation placing every active particle in its owner's slot block.

    Returns ``destination[C]`` (source slot ``s`` moves to ``destination[s]``)
    and whether every owner's particles fit its ``C/P`` slots. Active particles
    keep their relative slot order within each owner; inactive slots fill the
    remaining slots in slot order, so the permutation is deterministic.
    """
    parts = decomposition.part_count
    capacity = active.shape[0]
    block = capacity // parts
    key = jnp.where(active, decomposition.owner(position), parts).astype(jnp.int32)
    order = jnp.argsort(key, stable=True)
    ordered = key[order]
    counts = jnp.bincount(key, length=parts + 1)
    starts = jnp.cumsum(counts) - counts
    index = jnp.arange(capacity, dtype=jnp.int32)
    rank = index - starts[ordered]
    placed = ordered < parts
    active_target = ordered * block + rank
    occupied = (
        jnp.zeros((capacity,), dtype=jnp.bool_)
        .at[jnp.where(placed, active_target, capacity)]
        .set(True, mode="drop")
    )
    free = jnp.nonzero(~occupied, size=capacity, fill_value=capacity)[0]
    placed_count = jnp.sum(placed, dtype=jnp.int32)
    target = jnp.where(
        placed,
        active_target,
        free[jnp.clip(index - placed_count, 0, capacity - 1)],
    ).astype(jnp.int32)
    destination = jnp.zeros((capacity,), dtype=jnp.int32).at[order].set(target)
    return destination, jnp.all(counts[:parts] <= block)


def permute_slots(values: Array, destination: Array, /) -> Array:
    """Move slot ``s`` of ``values`` to ``destination[s]``."""
    return jnp.zeros_like(values).at[destination].set(values)


class PICSlotGroup(StrictModule):
    """Slot-aligned particles migrating together: a population with positions.

    ``cleared`` leaves are zeroed in vacated slots (proper velocities);
    ``kept`` leaves keep the previous occupant's values there (charge state,
    slot-aligned process state), like the population's identity words.
    ``structural`` is the population's structural slot mask.
    """

    population: ParticlePopulationState
    position: Array
    cleared: tuple[Array, ...]
    kept: tuple[Array, ...]
    structural: Array


class PICMigrationEvidence(StrictModule):
    """Global evidence of one migration.

    ``sent[k]`` is the largest packet count any device sent along mesh offset
    ``shifts[k]``; ``unrouted`` counts particles whose owner lies beyond the
    reach; ``received`` is the largest per-device arrival count and
    ``free_slots`` the smallest per-device free-slot count before arrival.
    """

    sent: Array
    unrouted: Array
    received: Array
    free_slots: Array
    migrated: Array
    packet_overflow: Array
    capacity_overflow: Array
    successful: Array
    shifts: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    packet_capacity: int = eqx.field(static=True)


class PICMigrationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity ppermute particle migration over one decomposition.

    Packets travel along every mesh offset in ``{−reach, …, reach}^k``
    (offsets equal modulo the part counts are merged), diagonals included.
    """

    decomposition: PICDomainDecomposition
    packet_capacity: int = eqx.field(static=True)
    reach: int = eqx.field(static=True)
    shifts: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        decomposition: PICDomainDecomposition,
        /,
        *,
        packet_capacity: int,
        reach: int = 1,
    ) -> None:
        if not isinstance(decomposition, PICDomainDecomposition):
            raise TypeError("decomposition must be PICDomainDecomposition.")
        capacity, hops = int(packet_capacity), int(reach)
        if capacity <= 0 or hops <= 0:
            raise ValueError("packet_capacity and reach must be positive.")
        offsets = [
            sorted({step % parts for step in range(-hops, hops + 1)})
            for parts in decomposition.parts
        ]
        shifts = tuple(
            shift for shift in product(*offsets) if any(value != 0 for value in shift)
        )
        self.decomposition = decomposition
        self.packet_capacity = capacity
        self.reach = hops
        self.shifts = shifts
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pic-migration-plan",
                "decomposition": decomposition.decomposition_id,
                "packet_capacity": capacity,
                "reach": hops,
            }
        )

    def migrate_local(
        self,
        plans: tuple[ParticlePopulationPlan, ...],
        groups: tuple[PICSlotGroup, ...],
        coordinates: tuple[Array, ...],
        /,
        *,
        axis_names: tuple[str, ...],
    ) -> tuple[tuple[PICSlotGroup, ...], PICMigrationEvidence]:
        """Migrate one device's slot blocks; call inside ``shard_map``.

        ``coordinates`` are the device's mesh coordinates along ``axis_names``.
        Returns the committed blocks (unchanged unless every device succeeded)
        and replicated global evidence.
        """
        width = self.packet_capacity
        sent = jnp.zeros((len(self.shifts),), dtype=jnp.int32)
        unrouted = jnp.zeros((), dtype=jnp.int32)
        received = jnp.zeros((), dtype=jnp.int32)
        free_minimum = jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32)
        migrated = jnp.zeros((), dtype=jnp.int32)
        packet_overflow = jnp.asarray(False)
        capacity_overflow = jnp.asarray(False)
        candidates = []
        for plan, group in zip(plans, groups, strict=True):
            (
                candidate,
                group_sent,
                group_unrouted,
                arrivals,
                free,
                moved,
                incarnation_ok,
            ) = self._migrate_group(plan, group, coordinates, axis_names)
            candidates.append(candidate)
            sent = jnp.maximum(sent, group_sent)
            unrouted = unrouted + group_unrouted
            received = jnp.maximum(received, arrivals)
            free_minimum = jnp.minimum(free_minimum, free)
            migrated = migrated + moved
            packet_overflow = packet_overflow | jnp.any(group_sent > width)
            capacity_overflow = capacity_overflow | (arrivals > free) | ~incarnation_ok
        local_ok = ~packet_overflow & ~capacity_overflow & (unrouted == 0)
        successful = jax.lax.pmin(local_ok.astype(jnp.int32), axis_names) == 1
        committed = tuple(
            jax.tree.map(
                lambda new, old: jnp.where(successful, new, old), candidate, group
            )
            for candidate, group in zip(candidates, groups, strict=True)
        )
        evidence = PICMigrationEvidence(
            jax.lax.pmax(sent, axis_names),
            jax.lax.psum(unrouted, axis_names),
            jax.lax.pmax(received, axis_names),
            jax.lax.pmin(free_minimum, axis_names),
            jax.lax.psum(migrated, axis_names),
            jax.lax.pmax(packet_overflow.astype(jnp.int32), axis_names) == 1,
            jax.lax.pmax(capacity_overflow.astype(jnp.int32), axis_names) == 1,
            successful,
            self.shifts,
            width,
        )
        return committed, evidence

    def _migrate_group(
        self,
        plan: ParticlePopulationPlan,
        group: PICSlotGroup,
        coordinates: tuple[Array, ...],
        axis_names: tuple[str, ...],
        /,
    ) -> tuple[PICSlotGroup, Array, Array, Array, Array, Array, Array]:
        decomposition = self.decomposition
        parts = decomposition.parts
        width = self.packet_capacity
        population = group.population
        identity = population_slot_leaves(population)
        leaves = (*identity, group.position, *group.cleared, *group.kept)
        active = population.active
        owner = decomposition.owner_coordinates(group.position)
        home = jnp.ones_like(active)
        for coordinate, value in zip(coordinates, owner, strict=True):
            home = home & (value == coordinate)
        leaving = active & ~home
        sent = []
        routed = jnp.zeros_like(leaving)
        incoming: list[tuple[Array, ...]] = []
        incoming_valid = []
        for shift in self.shifts:
            selected = leaving
            for coordinate, value, offset, count in zip(
                coordinates, owner, shift, parts, strict=True
            ):
                selected = selected & (value == (coordinate + offset) % count)
            routed = routed | selected
            rank = jnp.cumsum(selected, dtype=jnp.int32) - 1
            rows = jnp.where(selected & (rank < width), rank, width)
            packet = tuple(
                jnp.zeros((width + 1,) + leaf.shape[1:], dtype=leaf.dtype)
                .at[rows]
                .set(leaf)[:width]
                for leaf in leaves
            )
            valid = jnp.zeros((width + 1,), dtype=jnp.bool_).at[rows].set(True)[:width]
            permutation = tuple(
                (
                    int(np.ravel_multi_index(source, parts)),
                    int(
                        np.ravel_multi_index(
                            tuple(
                                (value + offset) % count
                                for value, offset, count in zip(
                                    source, shift, parts, strict=True
                                )
                            ),
                            parts,
                        )
                    ),
                )
                for source in np.ndindex(*parts)
            )
            incoming.append(
                tuple(
                    jax.lax.ppermute(value, axis_names, permutation) for value in packet
                )
            )
            incoming_valid.append(jax.lax.ppermute(valid, axis_names, permutation))
            sent.append(jnp.sum(selected, dtype=jnp.int32))
        unrouted = jnp.sum(leaving & ~routed, dtype=jnp.int32)
        never_reuse = plan.reuse_policy is ParticleSlotReusePolicy.NEVER_REUSE
        remaining = population.active & ~leaving
        vacated = (
            remaining,
            jnp.where(leaving, 0.0, population.mass),
            population.incarnation,
            population.ever_occupied,
            population.retired | (leaving & never_reuse),
            population.id_hi,
            population.id_lo,
            population.parent_hi,
            population.parent_lo,
            jnp.where(leaving[:, None], 0.0, group.position),
            *(
                jnp.where(
                    leaving.reshape(leaving.shape + (1,) * (value.ndim - 1)),
                    jnp.zeros_like(value),
                    value,
                )
                for value in group.cleared
            ),
            *group.kept,
        )
        free = group.structural & ~remaining
        if never_reuse:
            free = free & ~population.ever_occupied & ~population.retired
        capacity = active.shape[0]
        arriving = (
            tuple(
                jnp.concatenate(tuple(packet[index] for packet in incoming))
                for index in range(len(leaves))
            )
            if incoming
            else tuple(value[:0] for value in leaves)
        )
        arriving_valid = (
            jnp.concatenate(incoming_valid)
            if incoming_valid
            else jnp.zeros((0,), dtype=jnp.bool_)
        )
        rows = arriving_valid.shape[0]
        arrivals = jnp.sum(arriving_valid, dtype=jnp.int32)
        free_count = jnp.sum(free, dtype=jnp.int32)
        slots = jnp.nonzero(free, size=max(rows, 1), fill_value=capacity)[0][:rows]
        rank = jnp.cumsum(arriving_valid, dtype=jnp.int32) - 1
        target = jnp.where(
            arriving_valid, slots[jnp.clip(rank, 0, max(rows - 1, 0))], capacity
        )
        placed = [
            value.at[target].set(incoming_value, mode="drop")
            for value, incoming_value in zip(vacated, arriving, strict=True)
        ]
        next_incarnation = population.incarnation.at[target].add(1, mode="drop")
        incarnation_ok = jnp.all(next_incarnation <= plan.incarnation_maximum)
        placed[2] = next_incarnation
        placed[3] = placed[3].at[target].set(True, mode="drop")
        placed[4] = placed[4].at[target].set(False, mode="drop")
        cleared_count = len(group.cleared)
        candidate = PICSlotGroup(
            with_population_slot_leaves(population, placed[:9]),
            placed[9],
            tuple(placed[10 : 10 + cleared_count]),
            tuple(placed[10 + cleared_count :]),
            group.structural,
        )
        return (
            candidate,
            jnp.stack(sent) if sent else jnp.zeros((0,), dtype=jnp.int32),
            unrouted,
            arrivals,
            free_count,
            jnp.sum(leaving, dtype=jnp.int32),
            incarnation_ok,
        )


def _words(value: Array, /) -> tuple[Array, Array]:
    return (value >> np.uint64(32)).astype(jnp.uint32), value.astype(jnp.uint32)


class PICIdentityAllocator(AbstractPICParticleAllocator):
    """Per-device allocation with decomposition-independent identities.

    Bound to one device (``coordinates``) inside ``shard_map``. Each call
    allocates within the device's slot block through the population plan,
    then assigns every created particle ``counter + t R + r`` (tile ``t`` of
    its event position, ``R`` the population's global capacity ``parts ×``
    block capacity, ``r`` its rank among the call's events in ``t`` in event
    order) and advances the counter by ``B R`` (``B`` identity tiles). Every
    device advances every counter identically, so identities need no
    communication. An event outside the device's tiles fails the allocation
    (`ParticlePopulationStatus.INVALID_REQUEST`); a reservation beyond the
    64-bit identity space fails it with ``IDENTITY_EXHAUSTED``.
    """

    decomposition: PICDomainDecomposition
    coordinates: tuple[Array, ...]

    def allocate(
        self,
        plan: ParticlePopulationPlan,
        state: ParticlePopulationState,
        request: ParticleAllocationRequest,
        positions: Array,
        /,
    ) -> ParticleAllocationResult:
        decomposition = self.decomposition
        width = request.valid.shape[0]
        if (
            positions.shape[:1] != (width,)
            or positions.shape[1] < decomposition.axis_count
        ):
            raise ValueError("Allocation positions must give one event position per row.")
        result = plan.allocate(state, request)
        allocated = result.allocated
        tile = decomposition.tile(positions)
        home = jnp.all(~allocated | decomposition.owns_tile(tile, self.coordinates))
        tiles = decomposition.tile_count
        reserve = plan.particles.capacity * decomposition.part_count
        order = jnp.argsort(
            jnp.where(
                request.valid,
                request.event_ids.astype(jnp.int64),
                jnp.iinfo(jnp.int64).max,
            )
        )
        ordered_tile = jnp.where(allocated[order], tile[order], tiles)
        grouping = jnp.argsort(ordered_tile, stable=True)
        grouped = ordered_tile[grouping]
        first = jnp.searchsorted(grouped, grouped, side="left")
        rank_grouped = jnp.arange(width, dtype=jnp.int32) - first.astype(jnp.int32)
        rank = jnp.zeros((width,), dtype=jnp.int32).at[order[grouping]].set(rank_grouped)
        counter = (state.next_id_hi.astype(jnp.uint64) << np.uint64(32)) | (
            state.next_id_lo.astype(jnp.uint64)
        )
        span = np.uint64(tiles * reserve)
        exhausted = counter > _NO_IDENTITY - span
        identity = (
            counter
            + tile.astype(jnp.uint64) * np.uint64(reserve)
            + rank.astype(jnp.uint64)
        )
        id_hi, id_lo = _words(identity)
        next_hi, next_lo = _words(counter + span)
        capacity = state.active.shape[0]
        written = jnp.where(allocated, result.slots, capacity)
        candidate = result.candidate_state
        candidate = ParticlePopulationState(
            candidate.active,
            candidate.mass,
            candidate.incarnation,
            candidate.ever_occupied,
            candidate.retired,
            candidate.id_hi.at[written].set(id_hi, mode="drop"),
            candidate.id_lo.at[written].set(id_lo, mode="drop"),
            candidate.parent_hi,
            candidate.parent_lo,
            next_hi,
            next_lo,
        )
        successful = result.successful & home & ~exhausted
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        status = jnp.where(
            result.successful & ~home,
            int(ParticlePopulationStatus.INVALID_REQUEST),
            jnp.where(
                result.successful & exhausted,
                int(ParticlePopulationStatus.IDENTITY_EXHAUSTED),
                result.status,
            ),
        ).astype(jnp.int32)
        return ParticleAllocationResult(
            candidate,
            accepted,
            result.slots,
            allocated & successful,
            result.requested_count,
            jnp.where(successful, result.allocated_count, 0).astype(jnp.int32),
            status,
            successful,
            result.plan_id,
        )


__all__ = [
    "PICDomainDecomposition",
    "PICGuardWindow",
    "PICIdentityAllocator",
    "PICMigrationEvidence",
    "PICMigrationPlan",
    "PICSlotGroup",
]
