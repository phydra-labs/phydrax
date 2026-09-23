#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Permutation-equivariant monopole-sphere determinant amplitude."""

from __future__ import annotations

from collections.abc import Sequence
from math import comb, pi, sqrt

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...operators.quantum import LogAmplitude
from ._complex_determinant import complex_determinant_mixture


class MonopoleAttentionAmplitude(StrictModule):
    """Finite attention ansatz with exact monopole-harmonic orbital phases."""

    input_weights: Array
    input_bias: Array
    query_weights: Array
    key_weights: Array
    value_weights: Array
    output_weights: Array
    orbital_coefficients: Array
    determinant_coefficients: Array
    component_embedding: Array
    jastrow_matrix: Array
    component_indices: tuple[int, ...] = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    particle_count: int = eqx.field(static=True)
    twice_monopole_flux: int = eqx.field(static=True)
    hidden_dimension: int = eqx.field(static=True)
    layer_count: int = eqx.field(static=True)
    determinant_count: int = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_count: int,
        twice_monopole_flux: int,
        key: Key[Array, ""],
        /,
        *,
        hidden_dimension: int = 32,
        layer_count: int = 2,
        determinant_count: int = 4,
        component_indices: Sequence[int] | None = None,
        maximum_parameter_elements: int = 20_000_000,
    ):
        particles = int(particle_count)
        flux = int(twice_monopole_flux)
        hidden = int(hidden_dimension)
        layers = int(layer_count)
        determinants = int(determinant_count)
        components = (
            (0,) * particles
            if component_indices is None
            else tuple(int(value) for value in component_indices)
        )
        component_count = max(components, default=0) + 1
        maximum = int(maximum_parameter_elements)
        if (
            particles < 2
            or flux < 1
            or particles > flux + 1
            or hidden < 2
            or layers < 1
            or determinants < 1
            or maximum < 1
            or len(components) != particles
            or any(value < 0 for value in components)
        ):
            raise ValueError("Monopole attention dimensions are invalid.")
        orbital_count = flux + 1
        parameter_elements = (
            4 * hidden
            + hidden
            + 4 * layers * hidden * hidden
            + determinants * particles * orbital_count * hidden
            + determinants
            + component_count * hidden
            + component_count * component_count
        )
        if parameter_elements > maximum:
            raise ValueError("Monopole attention parameters exceed resource admission.")
        keys = jr.split(key, 10)
        real_dtype = jnp.float64
        scale = 1.0 / sqrt(hidden)
        self.input_weights = scale * jr.normal(keys[0], (4, hidden), dtype=real_dtype)
        self.input_bias = jnp.zeros((hidden,), dtype=real_dtype)
        self.query_weights = scale * jr.normal(
            keys[1], (layers, hidden, hidden), dtype=real_dtype
        )
        self.key_weights = scale * jr.normal(
            keys[2], (layers, hidden, hidden), dtype=real_dtype
        )
        self.value_weights = scale * jr.normal(
            keys[3], (layers, hidden, hidden), dtype=real_dtype
        )
        self.output_weights = scale * jr.normal(
            keys[4], (layers, hidden, hidden), dtype=real_dtype
        )
        coefficient_shape = (determinants, particles, orbital_count, hidden)
        coefficient_scale = 1.0 / sqrt(hidden * orbital_count)
        self.orbital_coefficients = coefficient_scale * (
            jr.normal(keys[5], coefficient_shape, dtype=real_dtype)
            + 1.0j * jr.normal(keys[6], coefficient_shape, dtype=real_dtype)
        )
        self.determinant_coefficients = (
            jnp.ones((determinants,), dtype=jnp.complex128) / determinants
        )
        self.component_embedding = scale * jr.normal(
            keys[7], (component_count, hidden), dtype=real_dtype
        )
        self.jastrow_matrix = jnp.zeros(
            (component_count, component_count), dtype=real_dtype
        )
        self.component_indices = components
        self.component_count = component_count
        self.particle_count = particles
        self.twice_monopole_flux = flux
        self.hidden_dimension = hidden
        self.layer_count = layers
        self.determinant_count = determinants
        self.network_id = canonical_fingerprint(
            {
                "kind": "monopole-attention-amplitude",
                "particle_count": particles,
                "twice_monopole_flux": flux,
                "hidden_dimension": hidden,
                "layer_count": layers,
                "determinant_count": determinants,
                "component_indices": components,
            }
        )

    def _features(self, configuration: Array, /) -> tuple[Array, Array, Array]:
        theta, phi = configuration[:, 0], configuration[:, 1]
        sine = jnp.sin(theta)
        cartesian = jnp.stack(
            (sine * jnp.cos(phi), sine * jnp.sin(phi), jnp.cos(theta)),
            axis=-1,
        )
        inputs = jnp.concatenate(
            (cartesian, jnp.ones((self.particle_count, 1), dtype=cartesian.dtype)),
            axis=-1,
        )
        hidden = jnp.tanh(
            inputs @ self.input_weights
            + self.input_bias
            + self.component_embedding[jnp.asarray(self.component_indices)]
        )
        for layer in range(self.layer_count):
            query = hidden @ self.query_weights[layer]
            key = hidden @ self.key_weights[layer]
            value = hidden @ self.value_weights[layer]
            scores = query @ key.T / sqrt(self.hidden_dimension)
            attention = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
            attention = attention / jnp.sum(attention, axis=-1, keepdims=True)
            hidden = jnp.tanh(hidden + (attention @ value) @ self.output_weights[layer])
        return hidden, cartesian, theta + 1.0j * phi

    def _monopole_harmonics(self, configuration: Array, /) -> Array:
        theta, phi = configuration[:, 0], configuration[:, 1]
        u = jnp.cos(0.5 * theta) * jnp.exp(0.5j * phi)
        v = jnp.sin(0.5 * theta) * jnp.exp(-0.5j * phi)
        values = []
        for orbital in range(self.twice_monopole_flux + 1):
            normalization = sqrt(
                (self.twice_monopole_flux + 1)
                * comb(self.twice_monopole_flux, orbital)
                / (4.0 * pi)
            )
            values.append(
                normalization * u**orbital * v ** (self.twice_monopole_flux - orbital)
            )
        return jnp.stack(values, axis=-1)

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        coordinates = jnp.asarray(configuration)
        if coordinates.shape != (self.particle_count, 2):
            raise ValueError(
                f"Monopole attention coordinates must have shape ({self.particle_count}, 2)."
            )
        hidden, cartesian, _ = self._features(coordinates)
        harmonics = self._monopole_harmonics(coordinates)
        orbitals = contract(
            "djmh,ih,im->dij",
            self.orbital_coefficients,
            hidden,
            harmonics,
            backend="jax",
        )
        determinant_log_abs, determinant_phase, determinant_valid = (
            complex_determinant_mixture(orbitals, self.determinant_coefficients)
        )
        cosine = cartesian @ cartesian.T
        pair_mask = jnp.triu(
            jnp.ones((self.particle_count, self.particle_count), dtype=jnp.bool_),
            k=1,
        )
        chord_squared = jnp.where(pair_mask, 2.0 - 2.0 * cosine, 1.0)
        chord = jnp.sqrt(jnp.maximum(chord_squared, 0.0))
        component = jnp.asarray(self.component_indices, dtype=jnp.int32)
        pair_strength = self.jastrow_matrix[component[:, None], component[None, :]]
        jastrow = jnp.sum(jnp.where(pair_mask, pair_strength * chord, 0.0))
        valid_coordinates = (
            jnp.all(jnp.isfinite(coordinates))
            & jnp.all(coordinates[:, 0] >= 0.0)
            & jnp.all(coordinates[:, 0] <= jnp.pi)
        )
        return LogAmplitude(
            determinant_log_abs + jastrow,
            determinant_phase,
            valid=determinant_valid & valid_coordinates & jnp.isfinite(jastrow),
        )


__all__ = ["MonopoleAttentionAmplitude"]
