#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._amplitude import LogAmplitude


class FiniteSignedPermutationSymmetry(StrictModule):
    """Finite one-dimensional symmetry sector over signed site permutations."""

    permutations: Array
    signs: Array
    characters: Array
    num_sites: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    identity_index: int = eqx.field(static=True)
    symmetry_id: str = eqx.field(static=True)

    def __init__(
        self,
        permutations: ArrayLike,
        signs: ArrayLike,
        characters: ArrayLike,
        /,
        *,
        symmetry_id: str | None = None,
    ):
        permutations_host = np.asarray(permutations)
        signs_host = np.asarray(signs)
        characters_host = np.asarray(characters, dtype=complex)
        if permutations_host.ndim != 2 or permutations_host.shape[0] < 1:
            raise ValueError("permutations must have shape (group, site).")
        order, num_sites = (int(size) for size in permutations_host.shape)
        if signs_host.shape != (order, num_sites):
            raise ValueError("signs must match permutations shape.")
        if characters_host.shape != (order,):
            raise ValueError("characters must have one value per group action.")
        expected = np.arange(num_sites)
        if any(not np.array_equal(np.sort(row), expected) for row in permutations_host):
            raise ValueError("Every action row must be a site permutation.")
        if np.any((signs_host != 1) & (signs_host != -1)):
            raise ValueError("Signed permutation entries must be exactly -1 or +1.")
        if np.any(~np.isfinite(characters_host)) or not np.allclose(
            np.abs(characters_host), 1.0, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Sector characters must be finite and unit-modulus.")
        action_keys = [
            (
                tuple(int(value) for value in permutation),
                tuple(int(value) for value in sign),
            )
            for permutation, sign in zip(permutations_host, signs_host, strict=True)
        ]
        if len(set(action_keys)) != order:
            raise ValueError("Finite symmetry actions must be unique.")
        action_indices = {key: index for index, key in enumerate(action_keys)}
        identity_key = (tuple(expected), tuple(np.ones((num_sites,), dtype=int)))
        if identity_key not in action_indices:
            raise ValueError("Finite symmetry actions must contain the identity.")
        identity_index = action_indices[identity_key]
        if not np.allclose(characters_host[identity_index], 1.0):
            raise ValueError("The identity action must have character one.")
        for left in range(order):
            for right in range(order):
                permutation = permutations_host[right][permutations_host[left]]
                sign = signs_host[left] * signs_host[right][permutations_host[left]]
                key = (
                    tuple(int(value) for value in permutation),
                    tuple(int(value) for value in sign),
                )
                if key not in action_indices:
                    raise ValueError("Signed permutation actions are not closed.")
                product = action_indices[key]
                if not np.allclose(
                    characters_host[product],
                    characters_host[left] * characters_host[right],
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "Sector characters are not a one-dimensional representation."
                    )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "finite-signed-permutation-symmetry",
                    "actions": array_tree_fingerprint(
                        {
                            "permutations": permutations_host.astype(np.int32),
                            "signs": signs_host.astype(np.int8),
                            "characters": characters_host,
                        }
                    ),
                }
            )
            if symmetry_id is None
            else str(symmetry_id)
        )
        if not identifier:
            raise ValueError("symmetry_id must be non-empty.")
        self.permutations = jnp.asarray(permutations_host, dtype=jnp.int32)
        self.signs = jnp.asarray(signs_host, dtype=jnp.int8)
        self.characters = jnp.asarray(characters_host)
        self.num_sites = num_sites
        self.order = order
        self.identity_index = identity_index
        self.symmetry_id = identifier

    def act(self, configuration: ArrayLike, /) -> Array:
        """Return every signed group image with one new penultimate group axis."""
        values = jnp.asarray(configuration)
        if values.ndim < 1 or int(values.shape[-1]) != self.num_sites:
            raise ValueError(
                f"configuration must end in ({self.num_sites},); got {values.shape}."
            )
        transformed = jnp.take(values, self.permutations, axis=-1)
        return transformed * self.signs.astype(values.dtype)


class SymmetryProjectedAmplitude(StrictModule):
    """Project an amplitude model into a finite one-dimensional symmetry sector."""

    model: Any
    symmetry: FiniteSignedPermutationSymmetry

    def __init__(self, model: Any, symmetry: FiniteSignedPermutationSymmetry, /):
        if not callable(model):
            raise TypeError("model must be callable.")
        if not isinstance(symmetry, FiniteSignedPermutationSymmetry):
            raise TypeError("symmetry must be a FiniteSignedPermutationSymmetry.")
        self.model = model
        self.symmetry = symmetry

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        images = self.symmetry.act(configuration)
        amplitudes = jax.vmap(self.model)(images)
        if not isinstance(amplitudes, LogAmplitude):
            raise TypeError("The projected model must return LogAmplitude values.")
        nonzero = amplitudes.valid & amplitudes.nonzero
        maximum = jnp.max(jnp.where(nonzero, amplitudes.log_abs, -jnp.inf))
        reference = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
        safe_log_abs = jnp.where(nonzero, amplitudes.log_abs, reference)
        safe_phase = jnp.where(nonzero, amplitudes.phase, 0.0j)
        scaled = (
            jnp.exp(safe_log_abs - reference)
            * safe_phase
            * jnp.conj(self.symmetry.characters)
        )
        projected = jnp.sum(scaled) / self.symmetry.order
        magnitude = jnp.abs(projected)
        nonzero_projected = magnitude > 0.0
        safe_magnitude = jnp.where(nonzero_projected, magnitude, 1.0)
        log_abs = jnp.where(
            nonzero_projected,
            reference + jnp.log(safe_magnitude),
            -jnp.inf,
        )
        phase = jnp.where(nonzero_projected, projected / safe_magnitude, 1.0 + 0.0j)
        valid = jnp.all(amplitudes.valid) & jnp.isfinite(phase)
        return LogAmplitude(log_abs, phase, valid=valid)


__all__ = [
    "FiniteSignedPermutationSymmetry",
    "SymmetryProjectedAmplitude",
]
