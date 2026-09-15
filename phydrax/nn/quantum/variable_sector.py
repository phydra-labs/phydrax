#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Permutation-covariant complex amplitudes on finite Fock-sector envelopes."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...operators.quantum._amplitude import LogAmplitude
from ...operators.quantum.variable_sector import (
    VariableParticleConfiguration,
    VariableSectorSpace,
)


def _configuration_arrays(
    space: VariableSectorSpace,
    configuration: VariableParticleConfiguration,
    /,
) -> tuple[Array, Array, Array, Array]:
    if not isinstance(configuration, VariableParticleConfiguration):
        raise TypeError("configuration must be VariableParticleConfiguration.")
    valid = space.valid(configuration)
    labels = jnp.clip(configuration.species, 0, space.species_count - 1)
    return configuration.coordinates, configuration.active_mask, labels, valid


def _pair_jastrow(
    coordinates: Array,
    active: Array,
    labels: Array,
    cusp: Array,
    ranges: Array,
    /,
) -> tuple[Array, Array]:
    capacity = int(coordinates.shape[0])
    pair = (
        active[:, None]
        & active[None, :]
        & jnp.triu(jnp.ones((capacity, capacity), dtype=bool), k=1)
    )
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    squared_distance = jnp.sum(difference * difference, axis=-1)
    distance = jnp.sqrt(jnp.where(pair, squared_distance, 1.0))
    cusp_value = cusp[labels[:, None], labels[None, :]]
    range_value = ranges[labels[:, None], labels[None, :]]
    factor = cusp_value * distance / (1.0 + range_value * distance)
    return jnp.sum(jnp.where(pair, factor, 0.0)), jnp.all(~pair | jnp.isfinite(factor))


def _log_amplitude(value: Array, valid: Array, /) -> LogAmplitude:
    phase = jnp.exp(1j * jnp.imag(value))
    return LogAmplitude(jnp.real(value), phase, valid=valid & jnp.isfinite(value))


class BosonicJastrowAmplitude(StrictModule):
    """Complex permutation-invariant Gaussian/Jastrow Fock amplitude.

    For species ``s`` the one-body term is
    ``-alpha_s |x-center_s|²/2 + i momentum_s·x``. Pair ``(s,t)`` contributes
    ``cusp[s,t] r / (1 + range[s,t] r)``. Its radial derivative at coincidence
    is exactly ``cusp[s,t]``, making the cusp convention observable.
    """

    space: VariableSectorSpace
    centers: Array
    precisions: Array
    momenta: Array
    pair_cusp: Array
    pair_range: Array
    sector_log_weights: Array
    amplitude_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        /,
        *,
        centers: ArrayLike | None = None,
        precisions: ArrayLike | float = 1.0,
        momenta: ArrayLike | None = None,
        pair_cusp: ArrayLike | float = 0.0,
        pair_range: ArrayLike | float = 0.0,
        sector_log_weights: ArrayLike | None = None,
        amplitude_id: str | None = None,
    ):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        center = (
            np.zeros((space.species_count, space.dimension), dtype=float)
            if centers is None
            else np.asarray(centers, dtype=float)
        )
        precision = np.broadcast_to(
            np.asarray(precisions, dtype=float),
            (space.species_count, space.dimension),
        ).copy()
        momentum = (
            np.zeros((space.species_count, space.dimension), dtype=float)
            if momenta is None
            else np.asarray(momenta, dtype=float)
        )
        cusp = np.broadcast_to(
            np.asarray(pair_cusp), (space.species_count, space.species_count)
        ).copy()
        ranges = np.broadcast_to(
            np.asarray(pair_range, dtype=float),
            (space.species_count, space.species_count),
        ).copy()
        sector = (
            np.zeros((space.species_count, space.capacity + 1), dtype=complex)
            if sector_log_weights is None
            else np.asarray(sector_log_weights)
        )
        cusp = 0.5 * (cusp + cusp.T)
        ranges = 0.5 * (ranges + ranges.T)
        if (
            center.shape != (space.species_count, space.dimension)
            or momentum.shape != center.shape
            or sector.shape != (space.species_count, space.capacity + 1)
            or np.any(~np.isfinite(center))
            or np.any(~np.isfinite(precision))
            or np.any(precision <= 0)
            or np.any(~np.isfinite(momentum))
            or np.any(~np.isfinite(cusp))
            or np.any(~np.isfinite(ranges))
            or np.any(ranges < 0)
            or np.any(~np.isfinite(sector))
        ):
            raise ValueError("Bosonic amplitude parameters are invalid.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "bosonic-variable-sector-jastrow-amplitude",
                    "space": space.space_id,
                    "centers": array_tree_fingerprint(center),
                    "precisions": array_tree_fingerprint(precision),
                    "momenta": array_tree_fingerprint(momentum),
                    "pair_cusp": array_tree_fingerprint(cusp),
                    "pair_range": array_tree_fingerprint(ranges),
                    "sector_log_weights": array_tree_fingerprint(sector),
                }
            )
            if amplitude_id is None
            else str(amplitude_id)
        )
        if not identifier:
            raise ValueError("amplitude_id must be non-empty.")
        self.space = space
        self.centers = jnp.asarray(center)
        self.precisions = jnp.asarray(precision)
        self.momenta = jnp.asarray(momentum)
        self.pair_cusp = jnp.asarray(cusp)
        self.pair_range = jnp.asarray(ranges)
        self.sector_log_weights = jnp.asarray(sector)
        self.amplitude_id = identifier

    def pair_log_factor(self, distance: ArrayLike, first: int, second: int, /) -> Array:
        first_, second_ = int(first), int(second)
        if (
            not 0 <= first_ < self.space.species_count
            or not 0 <= second_ < self.space.species_count
        ):
            raise ValueError("species indices are outside the amplitude space.")
        radius = jnp.asarray(distance)
        return (
            self.pair_cusp[first_, second_]
            * radius
            / (1.0 + self.pair_range[first_, second_] * radius)
        )

    def __call__(self, configuration: VariableParticleConfiguration, /) -> LogAmplitude:
        coordinate, active, labels, valid = _configuration_arrays(
            self.space, configuration
        )
        displacement = coordinate - self.centers[labels]
        one_body = -0.5 * jnp.sum(
            jnp.where(active[:, None], self.precisions[labels] * displacement**2, 0.0)
        )
        phase = contract(
            "nd,nd->",
            jnp.where(active[:, None], self.momenta[labels], 0.0),
            coordinate,
        )
        jastrow, pair_valid = _pair_jastrow(
            coordinate,
            active,
            labels,
            self.pair_cusp,
            self.pair_range,
        )
        counts = configuration.sector_counts(self.space.species_count)
        sector = jnp.sum(
            self.sector_log_weights[
                jnp.arange(self.space.species_count),
                jnp.clip(counts, 0, self.space.capacity),
            ]
        )
        return _log_amplitude(
            one_body + jastrow + 1j * phase + sector, valid & pair_valid
        )


class FermionicDeterminantJastrowAmplitude(StrictModule):
    """Species-resolved Slater determinants times a symmetric complex Jastrow.

    Every determinant is embedded in a fixed ``capacity × capacity`` matrix with
    an identity inactive block. This keeps JIT shapes fixed while evaluating the
    exact determinant of the occupied orbital block. Exchanging two particles of
    the same species exchanges two determinant rows and reverses the phase.
    """

    space: VariableSectorSpace
    orbital_bias: Array
    orbital_weights: Array
    orbital_centers: Array
    orbital_precisions: Array
    orbital_momenta: Array
    pair_cusp: Array
    pair_range: Array
    sector_log_weights: Array
    amplitude_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        orbital_bias: ArrayLike,
        orbital_weights: ArrayLike,
        *,
        orbital_centers: ArrayLike | None = None,
        orbital_precisions: ArrayLike | float = 1.0,
        orbital_momenta: ArrayLike | None = None,
        pair_cusp: ArrayLike | float = 0.0,
        pair_range: ArrayLike | float = 0.0,
        sector_log_weights: ArrayLike | None = None,
        amplitude_id: str | None = None,
    ):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        bias = np.asarray(orbital_bias)
        weights = np.asarray(orbital_weights)
        orbital_shape = (space.species_count, space.capacity)
        if bias.shape != orbital_shape or weights.shape != orbital_shape + (
            space.dimension,
        ):
            raise ValueError(
                "orbital_bias/weights require shapes (species,capacity) and "
                "(species,capacity,dimension)."
            )
        centers = (
            np.zeros(orbital_shape + (space.dimension,), dtype=float)
            if orbital_centers is None
            else np.asarray(orbital_centers, dtype=float)
        )
        precisions = np.broadcast_to(
            np.asarray(orbital_precisions, dtype=float), orbital_shape
        ).copy()
        momenta = (
            np.zeros(orbital_shape + (space.dimension,), dtype=float)
            if orbital_momenta is None
            else np.asarray(orbital_momenta, dtype=float)
        )
        cusp = np.broadcast_to(
            np.asarray(pair_cusp), (space.species_count, space.species_count)
        ).copy()
        ranges = np.broadcast_to(
            np.asarray(pair_range, dtype=float),
            (space.species_count, space.species_count),
        ).copy()
        sector = (
            np.zeros((space.species_count, space.capacity + 1), dtype=complex)
            if sector_log_weights is None
            else np.asarray(sector_log_weights)
        )
        cusp = 0.5 * (cusp + cusp.T)
        ranges = 0.5 * (ranges + ranges.T)
        arrays = (bias, weights, centers, precisions, momenta, cusp, ranges, sector)
        if (
            centers.shape != orbital_shape + (space.dimension,)
            or momenta.shape != centers.shape
            or sector.shape != (space.species_count, space.capacity + 1)
            or any(np.any(~np.isfinite(value)) for value in arrays)
            or np.any(precisions <= 0)
            or np.any(ranges < 0)
        ):
            raise ValueError("Fermionic determinant/Jastrow parameters are invalid.")
        dtype = np.result_type(
            bias.dtype, weights.dtype, cusp.dtype, sector.dtype, complex
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "fermionic-variable-sector-determinant-jastrow-amplitude",
                    "space": space.space_id,
                    "orbital_bias": array_tree_fingerprint(bias),
                    "orbital_weights": array_tree_fingerprint(weights),
                    "orbital_centers": array_tree_fingerprint(centers),
                    "orbital_precisions": array_tree_fingerprint(precisions),
                    "orbital_momenta": array_tree_fingerprint(momenta),
                    "pair_cusp": array_tree_fingerprint(cusp),
                    "pair_range": array_tree_fingerprint(ranges),
                    "sector_log_weights": array_tree_fingerprint(sector),
                }
            )
            if amplitude_id is None
            else str(amplitude_id)
        )
        if not identifier:
            raise ValueError("amplitude_id must be non-empty.")
        self.space = space
        self.orbital_bias = jnp.asarray(bias, dtype=dtype)
        self.orbital_weights = jnp.asarray(weights, dtype=dtype)
        self.orbital_centers = jnp.asarray(centers)
        self.orbital_precisions = jnp.asarray(precisions)
        self.orbital_momenta = jnp.asarray(momenta)
        self.pair_cusp = jnp.asarray(cusp, dtype=dtype)
        self.pair_range = jnp.asarray(ranges)
        self.sector_log_weights = jnp.asarray(sector, dtype=dtype)
        self.amplitude_id = identifier

    def _species_determinant(
        self,
        coordinate: Array,
        active: Array,
        labels: Array,
        species: int,
        /,
    ) -> Array:
        selected = active & (labels == species)
        order = jnp.argsort(~selected, stable=True)
        positions = coordinate[order]
        count = jnp.sum(selected, dtype=jnp.int32)
        row_active = jnp.arange(self.space.capacity) < count
        column_active = row_active
        displacement = positions[:, None, :] - self.orbital_centers[species][None, :, :]
        polynomial = self.orbital_bias[species][None, :] + contract(
            "od,rod->ro", self.orbital_weights[species], displacement
        )
        radius_squared = jnp.sum(displacement**2, axis=-1)
        phase = contract("od,rod->ro", self.orbital_momenta[species], displacement)
        orbitals = polynomial * jnp.exp(
            -0.5 * self.orbital_precisions[species][None, :] * radius_squared + 1j * phase
        )
        occupied = row_active[:, None] & column_active[None, :]
        inactive = ~row_active[:, None] & ~column_active[None, :]
        matrix = jnp.where(
            occupied,
            orbitals,
            jnp.where(inactive, jnp.eye(self.space.capacity, dtype=orbitals.dtype), 0.0),
        )
        return jnp.linalg.det(matrix)

    def __call__(self, configuration: VariableParticleConfiguration, /) -> LogAmplitude:
        coordinate, active, labels, valid = _configuration_arrays(
            self.space, configuration
        )
        determinants = jnp.stack(
            tuple(
                self._species_determinant(coordinate, active, labels, species)
                for species in range(self.space.species_count)
            )
        )
        determinant_nonzero = jnp.all(jnp.abs(determinants) > 0)
        determinant_log_abs = jnp.sum(jnp.log(jnp.abs(determinants)))
        determinant_phase = jnp.prod(
            jnp.where(
                jnp.abs(determinants) > 0,
                determinants / jnp.abs(determinants),
                1.0 + 0.0j,
            )
        )
        jastrow, pair_valid = _pair_jastrow(
            coordinate,
            active,
            labels,
            self.pair_cusp,
            self.pair_range,
        )
        counts = configuration.sector_counts(self.space.species_count)
        sector = jnp.sum(
            self.sector_log_weights[
                jnp.arange(self.space.species_count),
                jnp.clip(counts, 0, self.space.capacity),
            ]
        )
        log_abs = determinant_log_abs + jnp.real(jastrow + sector)
        phase = determinant_phase * jnp.exp(1j * jnp.imag(jastrow + sector))
        return LogAmplitude(
            log_abs,
            phase,
            valid=valid
            & pair_valid
            & determinant_nonzero
            & jnp.all(jnp.isfinite(determinants)),
        )


__all__ = ["BosonicJastrowAmplitude", "FermionicDeterminantJastrowAmplitude"]
