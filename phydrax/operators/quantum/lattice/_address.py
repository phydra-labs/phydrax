#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact packed configuration addresses without a global Hilbert-space rank."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....typing import as_array, ConvertibleToArray, Dim, Int32, UInt32
from .._abelian_charge import AbelianGroup
from ._model import QuantumLatticeSpecification


class AddressWordDim(Dim):
    """Canonical most-significant-first packed word axis."""


class ConfigurationSiteDim(Dim):
    """Ordered local-state coordinate axis."""


@final
class QuantumAddress(StrictModule):
    __strict_contract__ = True

    key_words: UInt32[AddressWordDim]
    domain_id: str = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)


@final
class QuantumAddressCodec(StrictModule):
    """Prepared bit fields; no ranks, hashes, or physical sentinel keys."""

    local_dimensions: tuple[int, ...] = eqx.field(static=True)
    fields: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    word_count: int = eqx.field(static=True)
    total_bits: int = eqx.field(static=True)
    codec_id: str = eqx.field(static=True)

    def __init__(
        self, local_dimensions: Sequence[int], /, *, maximum_address_words: int = 4096
    ) -> None:
        dimensions = tuple(local_dimensions)
        if not dimensions or any(
            isinstance(d, bool) or not isinstance(d, int) for d in dimensions
        ):
            raise TypeError("Local dimensions must be a nonempty integer sequence.")
        if any(d < 1 or d > np.iinfo(np.int32).max for d in dimensions):
            raise ValueError("Local dimensions must fit positive int32 coordinates.")
        if maximum_address_words < 1:
            raise ValueError("Address word budget must be positive.")
        widths = tuple((d - 1).bit_length() for d in dimensions)
        bits = sum(widths)
        words = max(1, (bits + 31) // 32)
        if words > maximum_address_words:
            raise ValueError("Packed address exceeds its word budget.")
        remaining = bits
        fields = []
        for width in widths:
            remaining -= width
            fields.append((words - 1 - remaining // 32, remaining % 32, width))
        self.local_dimensions = dimensions
        self.fields = tuple(fields)
        self.word_count = words
        self.total_bits = bits
        self.codec_id = canonical_fingerprint(
            {
                "kind": "quantum-packed-address",
                "dimensions": dimensions,
                "fields": self.fields,
            }
        )

    def encode(self, coordinates: ConvertibleToArray, /) -> Array:
        raw = jnp.asarray(coordinates)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("Coordinates must have an integer dtype.")
        if raw.shape != (len(self.local_dimensions),):
            raise ValueError("Coordinates must contain one index per site.")
        dimensions = jnp.asarray(self.local_dimensions, dtype=jnp.int64)
        raw = raw.astype(jnp.int64)
        raw = eqx.error_if(
            raw, jnp.any((raw < 0) | (raw >= dimensions)), "Invalid local-state index."
        )
        return self._encode(raw.astype(jnp.int32))

    def _encode(self, coordinates: Array, /) -> Array:
        words = jnp.zeros(coordinates.shape[:-1] + (self.word_count,), dtype=jnp.uint32)
        for site, (word, shift, width) in enumerate(self.fields):
            if width == 0:
                continue
            value = coordinates[..., site].astype(jnp.uint32)
            words = words.at[..., word].set(
                words[..., word] | (value << jnp.uint32(shift))
            )
            if shift + width > 32:
                words = words.at[..., word - 1].set(
                    words[..., word - 1] | (value >> jnp.uint32(32 - shift))
                )
        return words

    def decode(self, keys: ArrayLike, /) -> Array:
        words = self._words(keys)
        values = []
        for word, shift, width in self.fields:
            if width == 0:
                values.append(jnp.zeros(words.shape[:-1], dtype=jnp.int32))
                continue
            value = words[..., word] >> jnp.uint32(shift)
            if shift + width > 32:
                value = value | (words[..., word - 1] << jnp.uint32(32 - shift))
            values.append((value & jnp.uint32((1 << width) - 1)).astype(jnp.int32))
        return jnp.stack(values, axis=-1)

    def contains_keys(self, keys: ArrayLike, /) -> Array:
        words = self._words(keys)
        return self._contains_coordinates(words, self.decode(words))

    def _contains_coordinates(self, words: Array, coordinates: Array, /) -> Array:
        dimensions = jnp.asarray(self.local_dimensions, dtype=jnp.int32)
        return jnp.all(
            coordinates < jnp.broadcast_to(dimensions, coordinates.shape), axis=-1
        ) & jnp.all(self._encode(coordinates) == words, axis=-1)

    def _words(self, keys: ArrayLike, /) -> Array:
        words = jnp.asarray(keys)
        if words.dtype != jnp.dtype(jnp.uint32):
            raise TypeError("Packed keys must use uint32 words.")
        if words.ndim < 1 or words.shape[-1] != self.word_count:
            raise ValueError("Packed keys must end in the declared word axis.")
        return words


@final
class QuantumConfigurationDomain(StrictModule):
    """Scientific product/charge domain, independent of coefficients and resources."""

    codec: QuantumAddressCodec
    charge_tables: tuple[Array, ...]
    site_ids: tuple[str, ...] = eqx.field(static=True)
    space_ids: tuple[str, ...] = eqx.field(static=True)
    species_ids: tuple[str, ...] = eqx.field(static=True)
    charge_group: AbelianGroup | None = eqx.field(static=True)
    charge_labels: tuple[str, ...] = eqx.field(static=True)
    total_charge: tuple[int, ...] | None = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        specification: QuantumLatticeSpecification,
        /,
        *,
        species_ids: Sequence[str],
        charge_group: AbelianGroup | None = None,
        charge_labels: Sequence[str] = (),
        total_charge: Sequence[int] | None = None,
        maximum_address_words: int = 4096,
    ) -> None:
        if not isinstance(specification, QuantumLatticeSpecification):
            raise TypeError("Domain requires a canonical lattice specification.")
        species = tuple(species_ids)
        if len(species) != len(specification.spaces) or any(
            not isinstance(s, str) or not s for s in species
        ):
            raise ValueError(
                "Explicit nonempty species/component IDs must align with sites."
            )
        labels = tuple(charge_labels)
        if charge_group is None:
            if labels or total_charge is not None:
                raise ValueError("Charge constraints require an explicit Abelian group.")
            target = None
            tables = ()
        else:
            if not isinstance(charge_group, AbelianGroup):
                raise TypeError("charge_group must be AbelianGroup.")
            if (
                len(labels) != len(charge_group.components)
                or len(set(labels)) != len(labels)
                or any(label not in specification.charge_labels for label in labels)
            ):
                raise ValueError("Constraint labels must identify each group component.")
            if total_charge is None:
                raise ValueError("A charge group requires an exact total charge.")
            target = charge_group.normalize(total_charge)
            host_tables = []
            lower = [0] * len(labels)
            upper = [0] * len(labels)
            for space in specification.spaces:
                original = np.asarray(space.charges)
                table = np.zeros((space.dimension, len(labels)), dtype=np.int64)
                for component, label in enumerate(labels):
                    if label in space.charge_labels:
                        table[:, component] = original[
                            :, space.charge_labels.index(label)
                        ]
                    modulus = charge_group.components[component]
                    if modulus is None:
                        lower[component] += min(0, int(table[:, component].min()))
                        upper[component] += max(0, int(table[:, component].max()))
                    else:
                        if modulus > np.iinfo(np.int64).max // 2:
                            raise ValueError(
                                "Modular charge exceeds exact accumulator bounds."
                            )
                        table[:, component] %= modulus
                host_tables.append(table)
            for component, modulus in enumerate(charge_group.components):
                if modulus is None and (
                    lower[component] < np.iinfo(np.int64).min
                    or upper[component] > np.iinfo(np.int64).max
                    or not np.iinfo(np.int64).min
                    <= target[component]
                    <= np.iinfo(np.int64).max
                ):
                    raise ValueError(
                        "Integral charge exceeds exact int64 accumulator bounds."
                    )
            tables = tuple(jnp.asarray(table, dtype=jnp.int64) for table in host_tables)
        codec = QuantumAddressCodec(
            specification.local_dimensions, maximum_address_words=maximum_address_words
        )
        physics = canonical_fingerprint(
            {
                "kind": "quantum-configuration-physics",
                "spaces": tuple(s.space_id for s in specification.spaces),
                "fermion_order": None
                if specification.fermion_mode_order is None
                else specification.fermion_mode_order.order_id,
            }
        )
        self.codec = codec
        self.charge_tables = tables
        self.site_ids = specification.site_ids
        self.space_ids = tuple(s.space_id for s in specification.spaces)
        self.species_ids = species
        self.charge_group = charge_group
        self.charge_labels = labels
        self.total_charge = target
        self.physics_id = physics
        self.domain_id = canonical_fingerprint(
            {
                "kind": "quantum-configuration-domain",
                "physics": physics,
                "species": species,
                "group": None if charge_group is None else charge_group.group_id,
                "labels": labels,
                "target": target,
                "codec": codec.codec_id,
            }
        )

    def contains_keys(self, keys: ArrayLike, /) -> Array:
        words = self.codec._words(keys)
        return self._contains_coordinates(words, self.codec.decode(words))

    def _contains_coordinates(self, words: Array, coordinate: Array, /) -> Array:
        valid = self.codec._contains_coordinates(words, coordinate)
        if self.charge_group is None:
            return valid
        totals = jnp.zeros(
            coordinate.shape[:-1] + (len(self.charge_labels),), dtype=jnp.int64
        )
        for site, table in enumerate(self.charge_tables):
            incoming = jnp.minimum(coordinate[..., site], table.shape[0] - 1)
            totals = totals + table[incoming]
            for component, modulus in enumerate(self.charge_group.components):
                if modulus is not None:
                    totals = totals.at[..., component].set(
                        totals[..., component] % jnp.int64(modulus)
                    )
        target = jnp.broadcast_to(
            jnp.asarray(self.total_charge, dtype=jnp.int64), totals.shape
        )
        return valid & jnp.all(totals == target, axis=-1)

    def address(self, coordinates: ConvertibleToArray, /) -> QuantumAddress:
        return self.from_key(self.codec.encode(coordinates))

    def from_key(self, key: ArrayLike, /) -> QuantumAddress:
        words = as_array(key, UInt32[AddressWordDim], "key", casting="no")
        if words.shape != (self.codec.word_count,):
            raise ValueError("Address must contain exactly the domain word count.")
        words = eqx.error_if(
            words,
            ~self.contains_keys(words),
            "Key is outside its canonical configuration domain.",
        )
        return QuantumAddress(
            key_words=words, domain_id=self.domain_id, codec_id=self.codec.codec_id
        )

    def decode(self, key: ArrayLike, /) -> Int32[ConfigurationSiteDim]:
        words = self.codec._words(key)
        if words.ndim != 1:
            raise ValueError("Domain decode accepts one packed configuration key.")
        coordinate = self.codec.decode(words)
        return eqx.error_if(
            coordinate,
            ~self._contains_coordinates(words, coordinate),
            "Key is outside its canonical configuration domain.",
        )

    def validate_address(self, address: QuantumAddress, /) -> None:
        if not isinstance(address, QuantumAddress):
            raise TypeError("Expected QuantumAddress.")
        if address.domain_id != self.domain_id or address.codec_id != self.codec.codec_id:
            raise ValueError("Address belongs to another physical domain or codec.")
        if address.key_words.shape != (self.codec.word_count,):
            raise ValueError("Address word shape is incompatible with its domain.")


__all__ = [
    "AddressWordDim",
    "ConfigurationSiteDim",
    "QuantumAddress",
    "QuantumAddressCodec",
    "QuantumConfigurationDomain",
]
