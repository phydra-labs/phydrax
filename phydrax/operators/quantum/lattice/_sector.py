#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Direct fixed-charge coordinate bases; no ambient Hilbert-space enumeration."""

from __future__ import annotations

import abc
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from .._fermionic_fock import FermionModeOrder


class SectorBasisResourcePolicy(StrictModule):
    """Hard dimension and dynamic-programming table limits."""

    maximum_dimension: int = eqx.field(static=True)
    maximum_table_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, *, maximum_dimension: int, maximum_table_bytes: int):
        dimension = int(maximum_dimension)
        table_bytes = int(maximum_table_bytes)
        if dimension < 1 or table_bytes < 1:
            raise ValueError("Sector basis resource limits must be positive.")
        self.maximum_dimension = dimension
        self.maximum_table_bytes = table_bytes
        self.policy_id = canonical_fingerprint(
            {
                "kind": "sector-basis-resource-policy",
                "maximum_dimension": dimension,
                "maximum_table_bytes": table_bytes,
            }
        )


class AbstractSectorBasis(StrictModule):
    """Direct rank/unrank basis for one conserved integral quantum number."""

    site_ids: tuple[str, ...] = eqx.field(static=True)
    site_dimensions: tuple[int, ...] = eqx.field(static=True)
    quantum_number_label: str = eqx.field(static=True)
    quantum_number: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    resources: SectorBasisResourcePolicy = eqx.field(static=True)

    @abc.abstractmethod
    def coordinate(self, index: int | Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, coordinate: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def contains(self, coordinate: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def _coordinate_unchecked(self, rank: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def _rank_unchecked(self, coordinate: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def _contains_unchecked(self, coordinate: Array, /) -> Array:
        raise NotImplementedError


class _FixedChargeCoordinateBasis(AbstractSectorBasis):
    __strict_abstract__ = True
    state_charges: Array
    suffix_counts: Array
    maximum_local_dimension: int = eqx.field(static=True)
    charge_minimum: int = eqx.field(static=True)
    charge_maximum: int = eqx.field(static=True)

    @abc.abstractmethod
    def __init__(
        self,
        site_ids: Sequence[str],
        state_charges: Sequence[Sequence[int]],
        quantum_number_label: str,
        quantum_number: int,
        resources: SectorBasisResourcePolicy,
        /,
        *,
        kind: str,
        identity: object,
    ):
        if not isinstance(resources, SectorBasisResourcePolicy):
            raise TypeError("resources must be SectorBasisResourcePolicy.")
        sites = tuple(str(value) for value in site_ids)
        charges = tuple(tuple(values) for values in state_charges)
        charge_label = str(quantum_number_label)
        target = int(quantum_number)
        if (
            not sites
            or any(not value for value in sites)
            or len(set(sites)) != len(sites)
        ):
            raise ValueError("Sector site IDs must be unique and non-empty.")
        if len(charges) != len(sites) or any(not values for values in charges):
            raise ValueError("One nonempty local charge list is required per site.")
        if any(tuple(sorted(set(values))) != values for values in charges):
            raise ValueError("Local charges must be unique and strictly increasing.")
        if not charge_label:
            raise ValueError("quantum_number_label must be non-empty.")
        minimum = sum(values[0] for values in charges)
        maximum = sum(values[-1] for values in charges)
        if not minimum <= target <= maximum:
            raise ValueError("The requested sector is outside the local charge range.")
        table_host, offset = _suffix_count_table(charges)
        dimension = int(table_host[0, target + offset])
        if dimension < 1:
            raise ValueError("The requested fixed-charge sector is empty.")
        if dimension > resources.maximum_dimension:
            raise ValueError(
                f"Sector dimension {dimension} exceeds admitted maximum_dimension {resources.maximum_dimension}."
            )
        if dimension >= np.iinfo(np.int32).max:
            raise OverflowError(
                "Direct sector ranks require signed 32-bit route indices."
            )
        padded = np.zeros((len(sites), max(map(len, charges))), dtype=np.int32)
        for index, values in enumerate(charges):
            padded[index, : len(values)] = values
        required = table_host.nbytes + padded.nbytes
        if required > resources.maximum_table_bytes:
            raise ValueError(
                f"Direct sector tables require {required} bytes, exceeding "
                f"maximum_table_bytes {resources.maximum_table_bytes}."
            )
        self.state_charges = jnp.asarray(padded)
        self.suffix_counts = jnp.asarray(table_host, dtype=jnp.int64)
        self.maximum_local_dimension = padded.shape[1]
        self.charge_minimum = -offset
        self.charge_maximum = table_host.shape[1] - offset - 1
        self.site_ids = sites
        self.site_dimensions = tuple(len(values) for values in charges)
        self.quantum_number_label = charge_label
        self.quantum_number = target
        self.dimension = dimension
        self.resources = resources
        self.basis_id = canonical_fingerprint(
            {
                "kind": kind,
                "site_ids": sites,
                "state_charges": charges,
                "quantum_number_label": charge_label,
                "quantum_number": target,
                "resources": resources.policy_id,
                "identity": identity,
                "table": array_tree_fingerprint(table_host),
            }
        )

    def _coordinate_array(self, coordinate: ArrayLike, /) -> Array:
        values = jnp.asarray(coordinate)
        if not jnp.issubdtype(values.dtype, jnp.integer):
            raise TypeError("Sector coordinates must use an integer dtype.")
        return values.astype(jnp.int32)

    def coordinate(self, index: int | Array, /) -> Array:
        raw_rank = jnp.asarray(index)
        if not jnp.issubdtype(raw_rank.dtype, jnp.integer):
            raise TypeError("A sector basis index must use an integer dtype.")
        rank = raw_rank.astype(jnp.int64)
        if rank.shape != ():
            raise ValueError("A sector basis index must be scalar.")
        rank = eqx.error_if(
            rank,
            (rank < 0) | (rank >= self.dimension),
            "Sector basis index is out of range.",
        )
        return self._coordinate_unchecked(rank)

    def _coordinate_unchecked(self, rank: Array, /) -> Array:
        remaining = jnp.asarray(self.quantum_number, dtype=jnp.int32)
        coordinate = jnp.zeros((len(self.site_ids),), dtype=jnp.int32)
        for site, dimension in enumerate(self.site_dimensions):
            charges = self.state_charges[site]
            needed = remaining - charges
            indices = needed - self.charge_minimum
            in_range = (indices >= 0) & (indices < self.suffix_counts.shape[1])
            safe = jnp.clip(indices, 0, self.suffix_counts.shape[1] - 1)
            counts = jnp.where(
                in_range & (jnp.arange(self.maximum_local_dimension) < dimension),
                self.suffix_counts[site + 1, safe],
                0,
            )
            cumulative = jnp.cumsum(counts)
            selected = jnp.argmax(rank < cumulative).astype(jnp.int32)
            previous = jnp.where(selected == 0, 0, cumulative[selected - 1])
            rank = rank - previous
            remaining = remaining - charges[selected]
            coordinate = coordinate.at[site].set(selected)
        return coordinate

    def contains(self, coordinate: ArrayLike, /) -> Array:
        values = self._coordinate_array(coordinate)
        if values.shape != (len(self.site_ids),):
            raise ValueError("Sector coordinates must provide one local state per site.")
        return self._contains_unchecked(values)

    def _contains_unchecked(self, coordinate: Array, /) -> Array:
        dimensions = jnp.asarray(self.site_dimensions, dtype=jnp.int32)
        valid_indices = jnp.all((coordinate >= 0) & (coordinate < dimensions))
        safe = jnp.clip(coordinate, 0, dimensions - 1)
        charge = jnp.sum(self.state_charges[jnp.arange(len(self.site_ids)), safe])
        return valid_indices & (charge == self.quantum_number)

    def rank(self, coordinate: ArrayLike, /) -> Array:
        values = self._coordinate_array(coordinate)
        if values.shape != (len(self.site_ids),):
            raise ValueError("Sector coordinates must provide one local state per site.")
        values = eqx.error_if(
            values, ~self._contains_unchecked(values), "Coordinate is outside the sector."
        )
        return self._rank_unchecked(values)

    def _rank_unchecked(self, coordinate: Array, /) -> Array:
        rank = jnp.asarray(0, dtype=jnp.int64)
        remaining = jnp.asarray(self.quantum_number, dtype=jnp.int32)
        local_states = jnp.arange(self.maximum_local_dimension)
        for site, dimension in enumerate(self.site_dimensions):
            selected = coordinate[site]
            charges = self.state_charges[site]
            needed = remaining - charges
            indices = needed - self.charge_minimum
            in_range = (indices >= 0) & (indices < self.suffix_counts.shape[1])
            safe = jnp.clip(indices, 0, self.suffix_counts.shape[1] - 1)
            counts = jnp.where(
                in_range & (local_states < dimension) & (local_states < selected),
                self.suffix_counts[site + 1, safe],
                0,
            )
            rank = rank + jnp.sum(counts)
            remaining = remaining - charges[selected]
        return rank.astype(jnp.int32)


def _suffix_count_table(
    charges: tuple[tuple[int, ...], ...], /
) -> tuple[np.ndarray, int]:
    minimum = sum(min(values) for values in charges)
    maximum = sum(max(values) for values in charges)
    offset = -minimum
    table = np.zeros((len(charges) + 1, maximum - minimum + 1), dtype=np.int64)
    if not 0 <= offset < table.shape[1]:
        raise ValueError("Local charge ranges must include a zero suffix charge.")
    table[-1, offset] = 1
    for site in range(len(charges) - 1, -1, -1):
        for total in range(minimum, maximum + 1):
            count = 0
            for charge in charges[site]:
                suffix = total - charge
                if minimum <= suffix <= maximum:
                    count += int(table[site + 1, suffix + offset])
            if count > np.iinfo(np.int64).max:
                raise OverflowError("Sector dimension exceeds signed 64-bit counting.")
            table[site, total + offset] = count
    return table, offset


class FixedCardinalityFermionBasis(_FixedChargeCoordinateBasis):
    """Binary occupations of an explicit FermionModeOrder at fixed cardinality."""

    mode_order: FermionModeOrder = eqx.field(static=True)
    particle_count: int = eqx.field(static=True)

    def __init__(
        self,
        mode_order: FermionModeOrder,
        particle_count: int,
        /,
        *,
        resources: SectorBasisResourcePolicy,
    ):
        if not isinstance(mode_order, FermionModeOrder):
            raise TypeError("mode_order must be FermionModeOrder.")
        particles = int(particle_count)
        if not 0 <= particles <= mode_order.mode_count:
            raise ValueError("particle_count is outside the fermion mode range.")
        super().__init__(
            mode_order.labels,
            ((0, 1),) * mode_order.mode_count,
            "particle-number",
            particles,
            resources,
            kind="fixed-cardinality-fermion-basis",
            identity=mode_order.order_id,
        )
        self.mode_order = mode_order
        self.particle_count = particles


class FixedSpinProjectionBasis(_FixedChargeCoordinateBasis):
    """Direct product of spin-S sites at fixed twice-spin projection."""

    twice_spins: tuple[int, ...] = eqx.field(static=True)
    twice_projection: int = eqx.field(static=True)

    def __init__(
        self,
        site_ids: Sequence[str],
        twice_spins: Sequence[int],
        twice_projection: int,
        /,
        *,
        resources: SectorBasisResourcePolicy,
    ):
        spins = tuple(twice_spins)
        if not spins or any(value < 1 for value in spins):
            raise ValueError("twice_spins must contain positive integers.")
        projection = int(twice_projection)
        super().__init__(
            site_ids,
            tuple(tuple(range(-spin, spin + 1, 2)) for spin in spins),
            "twice-spin-projection",
            projection,
            resources,
            kind="fixed-spin-projection-basis",
            identity=spins,
        )
        self.twice_spins = spins
        self.twice_projection = projection


class FixedBosonNumberBasis(_FixedChargeCoordinateBasis):
    """Cutoff boson occupations at one fixed total boson number."""

    cutoffs: tuple[int, ...] = eqx.field(static=True)
    boson_number: int = eqx.field(static=True)

    def __init__(
        self,
        mode_ids: Sequence[str],
        cutoffs: Sequence[int],
        boson_number: int,
        /,
        *,
        resources: SectorBasisResourcePolicy,
    ):
        bounds = tuple(cutoffs)
        if not bounds or any(value < 2 for value in bounds):
            raise ValueError("Boson cutoffs must retain at least states zero and one.")
        number = int(boson_number)
        if number < 0:
            raise ValueError("boson_number must be non-negative.")
        super().__init__(
            mode_ids,
            tuple(tuple(range(cutoff)) for cutoff in bounds),
            "boson-number",
            number,
            resources,
            kind="fixed-boson-number-basis",
            identity=bounds,
        )
        self.cutoffs = bounds
        self.boson_number = number


class SectorChargeMap(StrictModule):
    """Declared source-to-target map for one fixed-charge operator."""

    source: AbstractSectorBasis
    target: AbstractSectorBasis
    charge_label: str = eqx.field(static=True)
    charge_delta: int = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: AbstractSectorBasis,
        target: AbstractSectorBasis,
        charge_delta: int,
        /,
    ):
        if not isinstance(source, AbstractSectorBasis) or not isinstance(
            target, AbstractSectorBasis
        ):
            raise TypeError("source and target must be direct sector bases.")
        if (
            source.site_ids != target.site_ids
            or source.site_dimensions != target.site_dimensions
            or source.quantum_number_label != target.quantum_number_label
        ):
            raise ValueError(
                "Sector charge maps require identical local coordinate spaces."
            )
        delta = int(charge_delta)
        if target.quantum_number - source.quantum_number != delta:
            raise ValueError("Sector quantum numbers contradict charge_delta.")
        self.source = source
        self.target = target
        self.charge_label = source.quantum_number_label
        self.charge_delta = delta
        self.map_id = canonical_fingerprint(
            {
                "kind": "sector-charge-map",
                "source": source.basis_id,
                "target": target.basis_id,
                "charge_label": self.charge_label,
                "charge_delta": delta,
            }
        )


__all__ = [
    "AbstractSectorBasis",
    "FixedBosonNumberBasis",
    "FixedCardinalityFermionBasis",
    "FixedSpinProjectionBasis",
    "SectorBasisResourcePolicy",
    "SectorChargeMap",
]
