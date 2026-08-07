#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jaxtyping import Array


EPOCH_ORDER_ALGORITHM = "feistel32-v1"
_MAX_POPULATION = 1 << 31
_ROUNDS = 6
_UINT32_MASK = (1 << 32) - 1


def _mix32(value: int) -> int:
    value &= _UINT32_MASK
    value ^= value >> 16
    value = (value * 0x7FEB352D) & _UINT32_MASK
    value ^= value >> 15
    value = (value * 0x846CA68B) & _UINT32_MASK
    value ^= value >> 16
    return value & _UINT32_MASK


def _round_keys(seed: int, epoch: int) -> tuple[int, ...]:
    payload = f"{EPOCH_ORDER_ALGORITHM}:{seed}:{epoch}".encode("ascii")
    material = hashlib.sha256(payload).digest()
    return tuple(
        int.from_bytes(material[4 * index : 4 * index + 4], "little")
        for index in range(_ROUNDS)
    )


@dataclass(frozen=True, slots=True)
class StatelessIndexPermutation:
    """Versioned O(1)-memory permutation of integer positions in one epoch."""

    population: int
    seed: int
    epoch: int
    _half_bits: int = field(init=False, repr=False)
    _half_mask: int = field(init=False, repr=False)
    _round_keys: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        population = int(self.population)
        seed = int(self.seed)
        epoch = int(self.epoch)
        if population <= 0:
            raise ValueError("population must be positive.")
        if population > _MAX_POPULATION:
            raise ValueError(
                f"population must not exceed {_MAX_POPULATION} for {EPOCH_ORDER_ALGORITHM}."
            )
        if seed < 0:
            raise ValueError("seed must be nonnegative.")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        domain_bits = max(2, (population - 1).bit_length())
        if domain_bits % 2:
            domain_bits += 1
        half_bits = domain_bits // 2
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "_half_bits", half_bits)
        object.__setattr__(self, "_half_mask", (1 << half_bits) - 1)
        object.__setattr__(self, "_round_keys", _round_keys(seed, epoch))

    @property
    def algorithm(self) -> str:
        return EPOCH_ORDER_ALGORITHM

    def __call__(self, position: int, /) -> int:
        index = int(position)
        if index < 0 or index >= self.population:
            raise IndexError("Permutation position is out of range.")
        if self.population == 1:
            return 0
        candidate = self._apply(index)
        while candidate >= self.population:
            candidate = self._apply(candidate)
        return candidate

    def jax(self, position: Array, /) -> Array:
        """Evaluate the same positional permutation with JAX integer operations."""
        index = jnp.asarray(position, dtype=jnp.uint32)
        population = jnp.asarray(self.population, dtype=jnp.uint32)
        if self.population == 1:
            return jnp.zeros_like(index)

        def apply(value):
            left = value >> jnp.uint32(self._half_bits)
            right = value & jnp.uint32(self._half_mask)

            def round_body(round_index, pair):
                current_left, current_right = pair
                key = jnp.asarray(self._round_keys, dtype=jnp.uint32)[round_index]
                mixed = current_right ^ key
                mixed ^= mixed >> jnp.uint32(16)
                mixed *= jnp.uint32(0x7FEB352D)
                mixed ^= mixed >> jnp.uint32(15)
                mixed *= jnp.uint32(0x846CA68B)
                mixed ^= mixed >> jnp.uint32(16)
                return (
                    current_right,
                    (current_left ^ mixed) & jnp.uint32(self._half_mask),
                )

            left, right = jax.lax.fori_loop(0, _ROUNDS, round_body, (left, right))
            return (left << jnp.uint32(self._half_bits)) | right

        candidate = apply(index)
        return jax.lax.while_loop(
            lambda value: value >= population,
            apply,
            candidate,
        ).astype(jnp.int32)

    def _apply(self, value: int, /) -> int:
        left = int(value) >> self._half_bits
        right = int(value) & self._half_mask
        for key in self._round_keys:
            left, right = right, (left ^ _mix32(right ^ key)) & self._half_mask
        return (left << self._half_bits) | right


__all__ = ["EPOCH_ORDER_ALGORITHM", "StatelessIndexPermutation"]
