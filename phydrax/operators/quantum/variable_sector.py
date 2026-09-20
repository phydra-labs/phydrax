#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-capacity continuum Fock sectors and reversible-jump operators."""

from __future__ import annotations

import abc
import math
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

import phydrax.axes as cx
from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._sampling import AbstractProposal
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...integration import discrete, integrate
from ._amplitude import amplitude_ratio, LogAmplitude


class VariableParticleConfiguration(StrictModule):
    """Fixed-capacity particle coordinates with explicit activity and species.

    Inactive slots are canonical zero padding.  Consequently every numerical
    operation is insensitive to caller-supplied values in inactive slots.
    """

    coordinates: Array
    active_mask: Array
    species: Array

    def __init__(
        self,
        coordinates: ArrayLike,
        active_mask: ArrayLike,
        species: ArrayLike,
        /,
    ):
        coordinate = jnp.asarray(coordinates)
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        labels = jnp.asarray(species, dtype=jnp.int32)
        if coordinate.ndim != 2 or coordinate.shape[0] < 1 or coordinate.shape[1] < 1:
            raise ValueError("coordinates must have shape (capacity, dimension).")
        if active.shape != (coordinate.shape[0],) or labels.shape != active.shape:
            raise ValueError("active_mask/species must have shape (capacity,).")
        if not jnp.issubdtype(coordinate.dtype, jnp.inexact):
            coordinate = coordinate.astype("float64")
        self.coordinates = jnp.where(active[:, None], coordinate, 0.0)
        self.active_mask = active
        self.species = jnp.where(active, labels, 0)

    @property
    def capacity(self) -> int:
        return self.coordinates.shape[0]

    @property
    def dimension(self) -> int:
        return self.coordinates.shape[1]

    @property
    def particle_count(self) -> Array:
        return jnp.sum(self.active_mask, dtype=jnp.int32)

    def canonicalized(self) -> VariableParticleConfiguration:
        order = jnp.argsort(~self.active_mask, stable=True)
        return VariableParticleConfiguration(
            self.coordinates[order], self.active_mask[order], self.species[order]
        )

    def sector_counts(self, species_count: int, /) -> Array:
        count = int(species_count)
        if count < 1:
            raise ValueError("species_count must be positive.")
        labels = jnp.where(self.active_mask, self.species, count)
        return jnp.sum(
            labels[:, None] == jnp.arange(count, dtype=jnp.int32)[None, :], axis=0
        ).astype(jnp.int32)

    def remove(self, index: ArrayLike, /) -> VariableParticleConfiguration:
        """Remove one slot and return canonical packed padding."""
        index_ = jnp.asarray(index, dtype=jnp.int32)
        keep = self.active_mask & (jnp.arange(self.capacity) != index_)
        order = jnp.argsort(~keep, stable=True)
        return VariableParticleConfiguration(
            self.coordinates[order], keep[order], self.species[order]
        )

    def remove_pair(
        self, first: ArrayLike, second: ArrayLike, /
    ) -> VariableParticleConfiguration:
        first_ = jnp.asarray(first, dtype=jnp.int32)
        second_ = jnp.asarray(second, dtype=jnp.int32)
        slots = jnp.arange(self.capacity)
        keep = self.active_mask & (slots != first_) & (slots != second_)
        order = jnp.argsort(~keep, stable=True)
        return VariableParticleConfiguration(
            self.coordinates[order], keep[order], self.species[order]
        )

    def append(
        self, coordinate: ArrayLike, species: ArrayLike, /
    ) -> VariableParticleConfiguration:
        packed = self.canonicalized()
        slot = jnp.minimum(packed.particle_count, self.capacity - 1)
        coordinate_ = jnp.asarray(coordinate, dtype=packed.coordinates.dtype)
        if coordinate_.shape != (self.dimension,):
            raise ValueError(f"coordinate must have shape ({self.dimension},).")
        return VariableParticleConfiguration(
            packed.coordinates.at[slot].set(coordinate_),
            packed.active_mask.at[slot].set(True),
            packed.species.at[slot].set(jnp.asarray(species, dtype=jnp.int32)),
        )

    def append_pair(
        self,
        first_coordinate: ArrayLike,
        first_species: ArrayLike,
        second_coordinate: ArrayLike,
        second_species: ArrayLike,
        /,
    ) -> VariableParticleConfiguration:
        return self.append(first_coordinate, first_species).append(
            second_coordinate, second_species
        )


class VariableSectorSpace(StrictModule, NonTrainableState):
    """Finite resource envelope for a disjoint union of particle sectors."""

    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    species_count: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        capacity: int,
        dimension: int,
        species_count: int = 1,
        /,
        *,
        space_id: str | None = None,
    ):
        capacity_, dimension_, species_ = (
            int(capacity),
            int(dimension),
            int(species_count),
        )
        if capacity_ < 1 or dimension_ < 1 or species_ < 1:
            raise ValueError("capacity, dimension, and species_count must be positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "variable-particle-sector-space",
                    "capacity": capacity_,
                    "dimension": dimension_,
                    "species_count": species_,
                }
            )
            if space_id is None
            else str(space_id)
        )
        if not identifier:
            raise ValueError("space_id must be non-empty.")
        self.capacity = capacity_
        self.dimension = dimension_
        self.species_count = species_
        self.space_id = identifier

    def empty(self, *, dtype: Any = jnp.float64) -> VariableParticleConfiguration:
        return VariableParticleConfiguration(
            jnp.zeros((self.capacity, self.dimension), dtype=dtype),
            jnp.zeros((self.capacity,), dtype=jnp.bool_),
            jnp.zeros((self.capacity,), dtype=jnp.int32),
        )

    def valid(self, configuration: VariableParticleConfiguration, /) -> Array:
        if not isinstance(configuration, VariableParticleConfiguration):
            raise TypeError("configuration must be VariableParticleConfiguration.")
        if configuration.coordinates.shape != (self.capacity, self.dimension):
            raise ValueError(
                "configuration coordinates do not match the variable-sector space."
            )
        active = configuration.active_mask
        finite = jnp.all(
            jnp.isfinite(jnp.where(active[:, None], configuration.coordinates, 0))
        )
        labels = configuration.species
        species_valid = jnp.all(~active | ((labels >= 0) & (labels < self.species_count)))
        return finite & species_valid


class VariableSectorMeasure(StrictModule, NonTrainableState):
    """Exact indistinguishable-particle disjoint-union reference measure.

    Sector ``(n_0, ..., n_S)`` carries the product Lebesgue measure divided by
    ``prod_s n_s!``.  No normalization over the unbounded coordinate domain is
    claimed.
    """

    space: VariableSectorSpace
    measure_id: str = eqx.field(static=True)

    def __init__(self, space: VariableSectorSpace, /, *, measure_id: str | None = None):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "indistinguishable-variable-sector-lebesgue-measure",
                    "space": space.space_id,
                    "sector_factor": "product-inverse-factorial",
                }
            )
            if measure_id is None
            else str(measure_id)
        )
        if not identifier:
            raise ValueError("measure_id must be non-empty.")
        self.space = space
        self.measure_id = identifier

    def log_sector_factor(self, configuration: VariableParticleConfiguration, /) -> Array:
        counts = configuration.sector_counts(self.space.species_count)
        valid = self.space.valid(configuration)
        value = -jnp.sum(jsp.special.gammaln(counts.astype("float64") + 1.0))
        return jnp.where(valid, value, -jnp.inf)

    def sector_factor(self, configuration: VariableParticleConfiguration, /) -> Array:
        return jnp.exp(self.log_sector_factor(configuration))


class VariableSectorProposalPlan(StrictModule, NonTrainableState):
    """Immutable proposal resources before runtime binding."""

    space: VariableSectorSpace
    kind_weights: Array
    birth_species_probabilities: Array
    pair_species: Array
    pair_probabilities: Array
    location: Array
    birth_scale: Array
    displacement_scale: float = eqx.field(static=True)
    pair_center_scale: float = eqx.field(static=True)
    pair_relative_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        /,
        *,
        move_weights: Sequence[float] = (1.0, 1.0, 4.0, 1.0, 0.5, 0.5),
        birth_species_probabilities: ArrayLike | None = None,
        pair_species: ArrayLike | None = None,
        pair_probabilities: ArrayLike | None = None,
        location: ArrayLike | None = None,
        birth_scale: ArrayLike | float = 1.0,
        displacement_scale: float = 0.5,
        pair_center_scale: float = 1.0,
        pair_relative_scale: float = 0.5,
    ):
        prepared = VariableSectorProposal(
            space,
            move_weights=move_weights,
            birth_species_probabilities=birth_species_probabilities,
            pair_species=pair_species,
            pair_probabilities=pair_probabilities,
            location=location,
            birth_scale=birth_scale,
            displacement_scale=displacement_scale,
            pair_center_scale=pair_center_scale,
            pair_relative_scale=pair_relative_scale,
        )
        self.space = space
        self.kind_weights = prepared.kind_weights
        self.birth_species_probabilities = prepared.birth_species_probabilities
        self.pair_species = prepared.pair_species
        self.pair_probabilities = prepared.pair_probabilities
        self.location = prepared.location
        self.birth_scale = prepared.birth_scale
        self.displacement_scale = prepared.displacement_scale
        self.pair_center_scale = prepared.pair_center_scale
        self.pair_relative_scale = prepared.pair_relative_scale
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-sector-proposal-plan",
                "proposal_semantics": prepared.proposal_id,
            }
        )


class VariableSectorProposalPayload(StrictModule):
    count_delta: Array
    changed_coordinates: Array
    changed_species: Array
    pair_log_abs_jacobian: Array


class VariableSectorProposal(AbstractProposal):
    """Normalized mixed RJ proposal on one fixed-capacity sector space.

    Birth/death probabilities include their exact species and particle-choice
    combinatorics. Pair coordinates use ``x=c+r, y=c-r``; the density therefore
    includes the ``2**dimension`` absolute Jacobian. Move-kind probabilities are
    renormalized over moves feasible at the current state, including the reverse.
    """

    space: VariableSectorSpace
    kind_weights: Array
    birth_species_probabilities: Array
    pair_species: Array
    pair_probabilities: Array
    location: Array
    birth_scale: Array
    displacement_scale: float = eqx.field(static=True)
    pair_center_scale: float = eqx.field(static=True)
    pair_relative_scale: float = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        /,
        *,
        move_weights: Sequence[float] = (1.0, 1.0, 4.0, 1.0, 0.5, 0.5),
        birth_species_probabilities: ArrayLike | None = None,
        pair_species: ArrayLike | None = None,
        pair_probabilities: ArrayLike | None = None,
        location: ArrayLike | None = None,
        birth_scale: ArrayLike | float = 1.0,
        displacement_scale: float = 0.5,
        pair_center_scale: float = 1.0,
        pair_relative_scale: float = 0.5,
        proposal_id: str | None = None,
    ):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        weights = np.asarray(tuple(move_weights), dtype=np.float64)
        if weights.shape != (6,) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("move_weights must contain six finite nonnegative values.")
        if not np.any(weights > 0):
            raise ValueError("At least one move weight must be positive.")
        species_probability = (
            np.full((space.species_count,), 1.0 / space.species_count)
            if birth_species_probabilities is None
            else np.asarray(birth_species_probabilities, dtype=np.float64)
        )
        if (
            species_probability.shape != (space.species_count,)
            or np.any(~np.isfinite(species_probability))
            or np.any(species_probability <= 0)
        ):
            raise ValueError("birth species probabilities must be finite and positive.")
        species_probability = species_probability / np.sum(species_probability)
        if pair_species is None:
            pair_labels = np.asarray(
                [
                    (first, second)
                    for first in range(space.species_count)
                    for second in range(space.species_count)
                ],
                dtype=np.int32,
            )
        else:
            pair_labels = np.asarray(pair_species, dtype=np.int32)
        if (
            pair_labels.ndim != 2
            or pair_labels.shape[1] != 2
            or pair_labels.shape[0] < 1
            or np.any(pair_labels < 0)
            or np.any(pair_labels >= space.species_count)
            or np.unique(pair_labels, axis=0).shape[0] != pair_labels.shape[0]
        ):
            raise ValueError(
                "pair_species must contain unique valid ordered species pairs."
            )
        pair_probability = (
            np.full((pair_labels.shape[0],), 1.0 / pair_labels.shape[0])
            if pair_probabilities is None
            else np.asarray(pair_probabilities, dtype=np.float64)
        )
        if (
            pair_probability.shape != (pair_labels.shape[0],)
            or np.any(~np.isfinite(pair_probability))
            or np.any(pair_probability <= 0)
        ):
            raise ValueError("pair probabilities must be finite and positive.")
        pair_probability = pair_probability / np.sum(pair_probability)
        center = (
            np.zeros((space.species_count, space.dimension), dtype=np.float64)
            if location is None
            else np.asarray(location, dtype=np.float64)
        )
        if center.shape != (space.species_count, space.dimension) or np.any(
            ~np.isfinite(center)
        ):
            raise ValueError("location must have shape (species_count, dimension).")
        scale = np.asarray(birth_scale, dtype=np.float64)
        scale = np.broadcast_to(scale, (space.species_count, space.dimension)).copy()
        scalar_values = np.asarray(
            (displacement_scale, pair_center_scale, pair_relative_scale), dtype=np.float64
        )
        if (
            np.any(~np.isfinite(scale))
            or np.any(scale <= 0)
            or np.any(~np.isfinite(scalar_values))
            or np.any(scalar_values <= 0)
        ):
            raise ValueError("All proposal scales must be finite and positive.")
        payload = {
            "kind": "variable-sector-reversible-jump-proposal",
            "space": space.space_id,
            "move_weights": weights.tolist(),
            "birth_species_probabilities": species_probability.tolist(),
            "pair_species": pair_labels.tolist(),
            "pair_probabilities": pair_probability.tolist(),
            "location": center.tolist(),
            "birth_scale": scale.tolist(),
            "displacement_scale": float(displacement_scale),
            "pair_center_scale": float(pair_center_scale),
            "pair_relative_scale": float(pair_relative_scale),
        }
        identifier = (
            canonical_fingerprint(payload) if proposal_id is None else str(proposal_id)
        )
        if not identifier:
            raise ValueError("proposal_id must be non-empty.")
        self.space = space
        self.kind_weights = jnp.asarray(weights)
        self.birth_species_probabilities = jnp.asarray(species_probability)
        self.pair_species = jnp.asarray(pair_labels)
        self.pair_probabilities = jnp.asarray(pair_probability)
        self.location = jnp.asarray(center)
        self.birth_scale = jnp.asarray(scale)
        self.displacement_scale = float(displacement_scale)
        self.pair_center_scale = float(pair_center_scale)
        self.pair_relative_scale = float(pair_relative_scale)
        self.proposal_id = identifier

    @property
    def pair_log_abs_jacobian(self) -> float:
        return self.space.dimension * math.log(2.0)

    def _pair_candidate_mask(self, configuration: VariableParticleConfiguration) -> Array:
        labels = configuration.species
        ordered = jnp.triu(
            jnp.ones((self.space.capacity, self.space.capacity), dtype=jnp.bool_), k=1
        )
        active = configuration.active_mask[:, None] & configuration.active_mask[None, :]
        direct = (labels[:, None, None] == self.pair_species[None, None, :, 0]) & (
            labels[None, :, None] == self.pair_species[None, None, :, 1]
        )
        reverse = (labels[:, None, None] == self.pair_species[None, None, :, 1]) & (
            labels[None, :, None] == self.pair_species[None, None, :, 0]
        )
        allowed = jnp.any(direct | reverse, axis=-1)
        return ordered & active & allowed

    def _exchange_candidate_mask(
        self, configuration: VariableParticleConfiguration
    ) -> Array:
        labels = configuration.species
        return (
            jnp.triu(
                jnp.ones((self.space.capacity, self.space.capacity), dtype=jnp.bool_), k=1
            )
            & configuration.active_mask[:, None]
            & configuration.active_mask[None, :]
            & (labels[:, None] != labels[None, :])
        )

    def _feasible(self, configuration: VariableParticleConfiguration) -> Array:
        count = configuration.particle_count
        return jnp.asarray(
            (
                count < self.space.capacity,
                count > 0,
                count > 0,
                jnp.any(self._exchange_candidate_mask(configuration)),
                count <= self.space.capacity - 2,
                jnp.any(self._pair_candidate_mask(configuration)),
            ),
            dtype=jnp.bool_,
        )

    def _kind_log_probabilities(
        self, configuration: VariableParticleConfiguration
    ) -> Array:
        weighted = jnp.where(self._feasible(configuration), self.kind_weights, 0.0)
        total = jnp.sum(weighted)
        return jnp.where(weighted > 0, jnp.log(weighted) - jnp.log(total), -jnp.inf)

    @staticmethod
    def _normal_log_density(value: Array, mean: Array, scale: Array) -> Array:
        standardized = (value - mean) / scale
        return -0.5 * jnp.sum(standardized**2 + jnp.log(2.0 * jnp.pi * scale**2))

    def sample(
        self, key: Key[Array, ""], current: VariableParticleConfiguration, /
    ) -> VariableParticleConfiguration:
        if not isinstance(current, VariableParticleConfiguration):
            raise TypeError("current must be VariableParticleConfiguration.")
        configuration = current.canonicalized()
        kind_key, choice_key, first_key, second_key = jr.split(key, 4)
        kind = jr.categorical(kind_key, self._kind_log_probabilities(configuration))
        count = configuration.particle_count

        def birth(_):
            species = jr.categorical(
                choice_key, jnp.log(self.birth_species_probabilities)
            )
            coordinate = self.location[species] + self.birth_scale[species] * jr.normal(
                first_key, (self.space.dimension,), dtype=configuration.coordinates.dtype
            )
            return configuration.append(coordinate, species)

        def death(_):
            index = jr.randint(choice_key, (), 0, jnp.maximum(count, 1))
            return configuration.remove(index)

        def displacement(_):
            index = jr.randint(choice_key, (), 0, jnp.maximum(count, 1))
            delta = self.displacement_scale * jr.normal(
                first_key, (self.space.dimension,), dtype=configuration.coordinates.dtype
            )
            return VariableParticleConfiguration(
                configuration.coordinates.at[index].add(delta),
                configuration.active_mask,
                configuration.species,
            )

        def exchange(_):
            candidate = self._exchange_candidate_mask(configuration).reshape((-1,))
            flat = jr.categorical(choice_key, jnp.where(candidate, 0.0, -jnp.inf))
            first = flat // self.space.capacity
            second = flat % self.space.capacity
            labels = configuration.species
            swapped = labels.at[first].set(labels[second]).at[second].set(labels[first])
            return VariableParticleConfiguration(
                configuration.coordinates, configuration.active_mask, swapped
            )

        def pair_birth(_):
            pair_index = jr.categorical(choice_key, jnp.log(self.pair_probabilities))
            labels = self.pair_species[pair_index]
            pair_center = 0.5 * (self.location[labels[0]] + self.location[labels[1]])
            center = pair_center + self.pair_center_scale * jr.normal(
                first_key, (self.space.dimension,), dtype=configuration.coordinates.dtype
            )
            relative = self.pair_relative_scale * jr.normal(
                second_key, (self.space.dimension,), dtype=configuration.coordinates.dtype
            )
            return configuration.append_pair(
                center + relative, labels[0], center - relative, labels[1]
            )

        def pair_death(_):
            candidate = self._pair_candidate_mask(configuration).reshape((-1,))
            flat = jr.categorical(choice_key, jnp.where(candidate, 0.0, -jnp.inf))
            first = flat // self.space.capacity
            second = flat % self.space.capacity
            return configuration.remove_pair(first, second)

        return jax.lax.switch(
            kind,
            (birth, death, displacement, exchange, pair_birth, pair_death),
            operand=None,
        )

    def _same(
        self, left: VariableParticleConfiguration, right: VariableParticleConfiguration
    ) -> Array:
        return (
            jnp.all(left.active_mask == right.active_mask)
            & jnp.all(left.species == right.species)
            & jnp.all(left.coordinates == right.coordinates)
        )

    def log_prob(
        self,
        proposed: VariableParticleConfiguration,
        current: VariableParticleConfiguration,
        /,
    ) -> Array:
        if not isinstance(proposed, VariableParticleConfiguration) or not isinstance(
            current, VariableParticleConfiguration
        ):
            raise TypeError(
                "proposed/current must be VariableParticleConfiguration values."
            )
        source, target = current.canonicalized(), proposed.canonicalized()
        source_count, target_count = source.particle_count, target.particle_count
        kind_log = self._kind_log_probabilities(source)
        capacity = self.space.capacity
        slots = jnp.arange(capacity, dtype=jnp.int32)

        birth_match = (target_count == source_count + 1) & self._same(
            source,
            target.remove(jnp.maximum(target_count - 1, 0)),
        )
        birth_slot = jnp.minimum(source_count, capacity - 1)
        birth_species = target.species[birth_slot]
        safe_birth_species = jnp.clip(birth_species, 0, self.space.species_count - 1)
        birth_log = (
            kind_log[0]
            + jnp.log(self.birth_species_probabilities[safe_birth_species])
            + self._normal_log_density(
                target.coordinates[birth_slot],
                self.location[safe_birth_species],
                self.birth_scale[safe_birth_species],
            )
        )
        birth_log = jnp.where(
            birth_match
            & (birth_species >= 0)
            & (birth_species < self.space.species_count),
            birth_log,
            -jnp.inf,
        )

        removed = jax.vmap(source.remove)(slots)
        death_matches = jax.vmap(lambda candidate: self._same(candidate, target))(removed)
        death_paths = jnp.sum(death_matches & source.active_mask)
        death_log = jnp.where(
            (target_count == source_count - 1) & (death_paths > 0),
            kind_log[1] + jnp.log(death_paths) - jnp.log(source_count.astype("float64")),
            -jnp.inf,
        )

        same_sector = (target_count == source_count) & jnp.all(
            source.species == target.species
        )
        same_activity = jnp.all(source.active_mask == target.active_mask)
        coordinate_difference = target.coordinates - source.coordinates
        other_equal = jax.vmap(
            lambda index: jnp.all(
                jnp.where((slots != index)[:, None], coordinate_difference == 0, True)
            )
        )(slots)
        displacement_matches = (
            source.active_mask & other_equal & same_sector & same_activity
        )
        displacement_logs = jax.vmap(
            lambda delta: self._normal_log_density(
                delta,
                jnp.zeros((self.space.dimension,), dtype=delta.dtype),
                jnp.full(
                    (self.space.dimension,), self.displacement_scale, dtype=delta.dtype
                ),
            )
        )(coordinate_difference)
        displacement_path_logs = jnp.where(
            displacement_matches,
            displacement_logs - jnp.log(source_count.astype("float64")),
            -jnp.inf,
        )
        displacement_log = kind_log[2] + jsp.special.logsumexp(displacement_path_logs)

        pair_slots = jnp.stack(
            jnp.meshgrid(slots, slots, indexing="ij"), axis=-1
        ).reshape((-1, 2))

        def exchanged(pair):
            first, second = pair[0], pair[1]
            labels = source.species
            swapped = labels.at[first].set(labels[second]).at[second].set(labels[first])
            return VariableParticleConfiguration(
                source.coordinates, source.active_mask, swapped
            )

        exchange_candidates = self._exchange_candidate_mask(source).reshape((-1,))
        exchange_matches = jax.vmap(lambda pair: self._same(exchanged(pair), target))(
            pair_slots
        )
        exchange_paths = jnp.sum(exchange_candidates & exchange_matches)
        exchange_total = jnp.sum(exchange_candidates)
        exchange_log = jnp.where(
            exchange_paths > 0,
            kind_log[3] + jnp.log(exchange_paths) - jnp.log(exchange_total),
            -jnp.inf,
        )

        pair_birth_match = (target_count == source_count + 2) & self._same(
            source,
            target.remove_pair(
                jnp.maximum(target_count - 2, 0), jnp.maximum(target_count - 1, 0)
            ),
        )
        first_slot = jnp.minimum(source_count, capacity - 1)
        second_slot = jnp.minimum(source_count + 1, capacity - 1)
        pair_labels = jnp.stack((target.species[first_slot], target.species[second_slot]))
        pair_type_matches = jnp.all(self.pair_species == pair_labels[None, :], axis=1)
        pair_type_index = jnp.argmax(pair_type_matches.astype(jnp.int32))
        first_coordinate = target.coordinates[first_slot]
        second_coordinate = target.coordinates[second_slot]
        center = 0.5 * (first_coordinate + second_coordinate)
        relative = 0.5 * (first_coordinate - second_coordinate)
        selected_labels = self.pair_species[pair_type_index]
        pair_center = 0.5 * (
            self.location[selected_labels[0]] + self.location[selected_labels[1]]
        )
        center_scale = jnp.full(
            (self.space.dimension,), self.pair_center_scale, dtype=center.dtype
        )
        relative_scale = jnp.full(
            (self.space.dimension,), self.pair_relative_scale, dtype=center.dtype
        )
        pair_birth_log = (
            kind_log[4]
            + jnp.log(self.pair_probabilities[pair_type_index])
            + self._normal_log_density(center, pair_center, center_scale)
            + self._normal_log_density(relative, jnp.zeros_like(relative), relative_scale)
            - self.pair_log_abs_jacobian
        )
        pair_birth_log = jnp.where(
            pair_birth_match & jnp.any(pair_type_matches), pair_birth_log, -jnp.inf
        )

        removed_pairs = jax.vmap(lambda pair: source.remove_pair(pair[0], pair[1]))(
            pair_slots
        )
        pair_death_matches = jax.vmap(lambda candidate: self._same(candidate, target))(
            removed_pairs
        )
        pair_candidates = self._pair_candidate_mask(source).reshape((-1,))
        pair_death_paths = jnp.sum(pair_candidates & pair_death_matches)
        pair_death_total = jnp.sum(pair_candidates)
        pair_death_log = jnp.where(
            (target_count == source_count - 2) & (pair_death_paths > 0),
            kind_log[5] + jnp.log(pair_death_paths) - jnp.log(pair_death_total),
            -jnp.inf,
        )
        valid = self.space.valid(source) & self.space.valid(target)
        total = jsp.special.logsumexp(
            jnp.stack(
                (
                    birth_log,
                    death_log,
                    displacement_log,
                    exchange_log,
                    pair_birth_log,
                    pair_death_log,
                )
            )
        )
        return jnp.where(valid, total, -jnp.inf)

    def payload(
        self,
        key: Key[Array, ""],
        current: VariableParticleConfiguration,
        proposed: VariableParticleConfiguration,
        /,
    ) -> VariableSectorProposalPayload:
        del key
        current_, proposed_ = current.canonicalized(), proposed.canonicalized()
        return VariableSectorProposalPayload(
            count_delta=proposed_.particle_count - current_.particle_count,
            changed_coordinates=jnp.sum(
                jnp.any(proposed_.coordinates != current_.coordinates, axis=-1),
                dtype=jnp.int32,
            ),
            changed_species=jnp.sum(
                proposed_.species != current_.species, dtype=jnp.int32
            ),
            pair_log_abs_jacobian=jnp.asarray(self.pair_log_abs_jacobian),
        )


def prepare_variable_sector_proposal(
    plan: VariableSectorProposalPlan,
    /,
    *,
    proposal_id: str | None = None,
) -> VariableSectorProposal:
    if not isinstance(plan, VariableSectorProposalPlan):
        raise TypeError("plan must be VariableSectorProposalPlan.")
    return VariableSectorProposal(
        plan.space,
        move_weights=np.asarray(plan.kind_weights),
        birth_species_probabilities=np.asarray(plan.birth_species_probabilities),
        pair_species=np.asarray(plan.pair_species),
        pair_probabilities=np.asarray(plan.pair_probabilities),
        location=np.asarray(plan.location),
        birth_scale=np.asarray(plan.birth_scale),
        displacement_scale=plan.displacement_scale,
        pair_center_scale=plan.pair_center_scale,
        pair_relative_scale=plan.pair_relative_scale,
        proposal_id=proposal_id,
    )


class VariableSectorLocalStatus:
    SUCCESS = 0
    INVALID_CONFIGURATION = 1
    INVALID_AMPLITUDE = 2
    SINGULAR_CONFIGURATION = 3
    NONFINITE = 4
    CAPACITY_SATURATED = 5


class VariableSectorLocalEstimate(StrictModule):
    value: Array
    valid: Array
    status: Array
    work_count: Array
    cutoff_saturated: Array
    operator_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class AbstractVariableSectorLocalOperator(StrictModule):
    space: eqx.AbstractVar[VariableSectorSpace]
    operator_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def local_value(
        self,
        model: Callable[[VariableParticleConfiguration], LogAmplitude],
        configuration: VariableParticleConfiguration,
        /,
    ) -> VariableSectorLocalEstimate:
        raise NotImplementedError


def _complex_log_amplitude(
    model: Callable[[VariableParticleConfiguration], LogAmplitude],
    configuration: VariableParticleConfiguration,
    /,
) -> tuple[Array, Array]:
    amplitude = model(configuration)
    if not isinstance(amplitude, LogAmplitude) or amplitude.log_abs.shape != ():
        raise TypeError("Variable-sector models must return one scalar LogAmplitude.")
    value = amplitude.log_abs + 1j * jnp.angle(amplitude.phase)
    return value, amplitude.valid & amplitude.nonzero


def _estimate(
    value: Array,
    valid: Array,
    status: Array,
    work_count: Array,
    saturated: Array,
    operator_id: str,
    method_id: str,
) -> VariableSectorLocalEstimate:
    finite = jnp.isfinite(value)
    final_valid = valid & finite
    final_status = jnp.where(valid & ~finite, VariableSectorLocalStatus.NONFINITE, status)
    return VariableSectorLocalEstimate(
        value=jnp.where(final_valid, value, jnp.asarray(jnp.nan + 0.0j)),
        valid=final_valid,
        status=jnp.asarray(final_status, dtype=jnp.int32),
        work_count=jnp.asarray(work_count, dtype=jnp.int32),
        cutoff_saturated=jnp.asarray(saturated, dtype=jnp.bool_),
        operator_id=operator_id,
        method_id=method_id,
    )


class ContinuumKineticPlan(StrictModule, NonTrainableState):
    space: VariableSectorSpace
    coordinate_chunk_size: int = eqx.field(static=True)
    maximum_coordinate_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        /,
        *,
        coordinate_chunk_size: int = 16,
        maximum_coordinate_count: int = 4096,
    ):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        chunk, maximum = int(coordinate_chunk_size), int(maximum_coordinate_count)
        coordinate_count = space.capacity * space.dimension
        if chunk < 1 or maximum < 1 or coordinate_count > maximum:
            raise ValueError("The exact kinetic trace exceeds maximum_coordinate_count.")
        self.space = space
        self.coordinate_chunk_size = min(chunk, coordinate_count)
        self.maximum_coordinate_count = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "continuum-variable-sector-kinetic-plan",
                "space": space.space_id,
                "coordinate_chunk_size": self.coordinate_chunk_size,
                "maximum_coordinate_count": maximum,
                "exact_coordinate_count": coordinate_count,
            }
        )


class ContinuumKineticOperator(AbstractVariableSectorLocalOperator):
    space: VariableSectorSpace
    masses: Array
    plan: ContinuumKineticPlan
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        masses: ArrayLike,
        /,
        *,
        plan: ContinuumKineticPlan | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        mass = np.asarray(masses, dtype=np.float64)
        if (
            mass.shape != (space.species_count,)
            or np.any(~np.isfinite(mass))
            or np.any(mass <= 0)
        ):
            raise ValueError("masses must be finite positive values for every species.")
        plan_ = ContinuumKineticPlan(space) if plan is None else plan
        if (
            not isinstance(plan_, ContinuumKineticPlan)
            or plan_.space.space_id != space.space_id
        ):
            raise ValueError("plan must be a ContinuumKineticPlan for this space.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "continuum-variable-sector-kinetic",
                    "space": space.space_id,
                    "masses": mass.tolist(),
                    "plan": plan_.plan_id,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.space = space
        self.masses = jnp.asarray(mass)
        self.plan = plan_
        self.operator_id = identifier

    def local_value(self, model, configuration, /) -> VariableSectorLocalEstimate:
        if not callable(model):
            raise TypeError("model must be callable.")
        shape = (self.space.capacity, self.space.dimension)
        flat = configuration.coordinates.reshape((-1,))

        def components(coordinates):
            candidate = VariableParticleConfiguration(
                coordinates.reshape(shape),
                configuration.active_mask,
                configuration.species,
            )
            value, _ = _complex_log_amplitude(model, candidate)
            return jnp.stack((jnp.real(value), jnp.imag(value)))

        jacobian = jax.jacrev(components)
        component_gradient = jacobian(flat)

        def diagonal(direction):
            _, tangent = jax.jvp(jacobian, (flat,), (direction,))
            return contract("ad,d->a", tangent, direction)

        coordinate_count = self.space.capacity * self.space.dimension
        blocks = []
        for start in range(0, coordinate_count, self.plan.coordinate_chunk_size):
            stop = min(start + self.plan.coordinate_chunk_size, coordinate_count)
            directions = jax.nn.one_hot(
                jnp.arange(start, stop), coordinate_count, dtype=flat.dtype
            )
            blocks.append(jax.vmap(diagonal)(directions))
        diagonal_components = jnp.concatenate(blocks, axis=0)
        gradient = component_gradient[0] + 1j * component_gradient[1]
        hessian_diagonal = diagonal_components[:, 0] + 1j * diagonal_components[:, 1]
        inverse_mass = jnp.where(
            configuration.active_mask,
            1.0
            / self.masses[
                jnp.clip(configuration.species, 0, self.space.species_count - 1)
            ],
            0.0,
        )
        coefficient = jnp.repeat(inverse_mass, self.space.dimension)
        value = -0.5 * contract(
            "d,d->", coefficient, hessian_diagonal + gradient * gradient
        )
        _, amplitude_valid = _complex_log_amplitude(model, configuration)
        configuration_valid = self.space.valid(configuration)
        status = jnp.where(
            configuration_valid,
            jnp.where(
                amplitude_valid,
                VariableSectorLocalStatus.SUCCESS,
                VariableSectorLocalStatus.INVALID_AMPLITUDE,
            ),
            VariableSectorLocalStatus.INVALID_CONFIGURATION,
        )
        return _estimate(
            value,
            configuration_valid & amplitude_valid,
            status,
            jnp.sum(configuration.active_mask) * self.space.dimension,
            configuration.particle_count == self.space.capacity,
            self.operator_id,
            "exact-chunked-log-amplitude-laplacian",
        )


class ExternalPotentialOperator(AbstractVariableSectorLocalOperator):
    """Exact active-particle sum of a named caller-supplied one-body potential."""

    space: VariableSectorSpace
    potential: Callable[[Array, Array], Array] = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        potential: Callable[[Array, Array], Array],
        /,
        *,
        potential_id: str,
        operator_id: str | None = None,
    ):
        if not isinstance(space, VariableSectorSpace) or not callable(potential):
            raise TypeError("space/potential types are invalid.")
        potential_identifier = str(potential_id)
        if not potential_identifier:
            raise ValueError("potential_id must be non-empty.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "variable-sector-external-potential",
                    "space": space.space_id,
                    "potential": potential_identifier,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.space = space
        self.potential = potential
        self.potential_id = potential_identifier
        self.operator_id = identifier

    def local_value(self, model, configuration, /):
        del model
        values = jax.vmap(self.potential)(
            configuration.coordinates, configuration.species
        )
        if values.shape != (self.space.capacity,):
            raise ValueError("potential must return one scalar per particle.")
        value = jnp.sum(jnp.where(configuration.active_mask, values, 0.0))
        valid = self.space.valid(configuration) & jnp.all(
            ~configuration.active_mask | jnp.isfinite(values)
        )
        return _estimate(
            value.astype(jnp.result_type(value, 1j)),
            valid,
            jnp.where(valid, 0, VariableSectorLocalStatus.INVALID_CONFIGURATION),
            configuration.particle_count,
            configuration.particle_count == self.space.capacity,
            self.operator_id,
            "exact-active-caller-one-body-sum",
        )


class QuadraticExternalPotential(AbstractVariableSectorLocalOperator):
    space: VariableSectorSpace
    centers: Array
    stiffness: Array
    offsets: Array
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        stiffness: ArrayLike,
        /,
        *,
        centers: ArrayLike | None = None,
        offsets: ArrayLike | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        matrix = np.asarray(stiffness, dtype=np.float64)
        if matrix.shape == (space.species_count,):
            matrix = np.asarray([np.eye(space.dimension) * value for value in matrix])
        if matrix.shape != (space.species_count, space.dimension, space.dimension):
            raise ValueError("stiffness must be per-species scalars or square matrices.")
        matrix = 0.5 * (matrix + np.swapaxes(matrix, -1, -2))
        center = (
            np.zeros((space.species_count, space.dimension))
            if centers is None
            else np.asarray(centers, dtype=np.float64)
        )
        offset = (
            np.zeros((space.species_count,))
            if offsets is None
            else np.asarray(offsets, dtype=np.float64)
        )
        if (
            center.shape != (space.species_count, space.dimension)
            or offset.shape != (space.species_count,)
            or np.any(~np.isfinite(matrix))
            or np.any(~np.isfinite(center))
            or np.any(~np.isfinite(offset))
        ):
            raise ValueError("Quadratic potential coefficients are invalid.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "quadratic-variable-sector-potential",
                    "space": space.space_id,
                    "stiffness": matrix.tolist(),
                    "centers": center.tolist(),
                    "offsets": offset.tolist(),
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.space = space
        self.centers = jnp.asarray(center)
        self.stiffness = jnp.asarray(matrix)
        self.offsets = jnp.asarray(offset)
        self.operator_id = identifier

    def local_value(self, model, configuration, /):
        del model
        labels = jnp.clip(configuration.species, 0, self.space.species_count - 1)
        delta = configuration.coordinates - self.centers[labels]
        values = (
            0.5 * contract("ni,nij,nj->n", delta, self.stiffness[labels], delta)
            + self.offsets[labels]
        )
        value = jnp.sum(jnp.where(configuration.active_mask, values, 0.0))
        valid = self.space.valid(configuration)
        return _estimate(
            value.astype(jnp.result_type(value, 1j)),
            valid,
            jnp.where(valid, 0, VariableSectorLocalStatus.INVALID_CONFIGURATION),
            configuration.particle_count,
            configuration.particle_count == self.space.capacity,
            self.operator_id,
            "exact-active-one-body-sum",
        )


class PairPotentialOperator(AbstractVariableSectorLocalOperator):
    space: VariableSectorSpace
    coupling: Array
    softening: float = eqx.field(static=True)
    power: float = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        coupling: ArrayLike,
        /,
        *,
        power: float = 1.0,
        softening: float = 0.0,
        operator_id: str | None = None,
    ):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        matrix = np.asarray(coupling, dtype=np.float64)
        if matrix.shape != (space.species_count, space.species_count):
            raise ValueError("coupling must have shape (species_count, species_count).")
        matrix = 0.5 * (matrix + matrix.T)
        power_, softening_ = float(power), float(softening)
        if (
            np.any(~np.isfinite(matrix))
            or not np.isfinite(power_)
            or power_ <= 0
            or not np.isfinite(softening_)
            or softening_ < 0
        ):
            raise ValueError("Pair-potential parameters are invalid.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "radial-variable-sector-pair-potential",
                    "space": space.space_id,
                    "coupling": matrix.tolist(),
                    "power": power_,
                    "softening": softening_,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.space = space
        self.coupling = jnp.asarray(matrix)
        self.softening = softening_
        self.power = power_
        self.operator_id = identifier

    def local_value(self, model, configuration, /):
        del model
        pair = (
            configuration.active_mask[:, None]
            & configuration.active_mask[None, :]
            & jnp.triu(
                jnp.ones((self.space.capacity, self.space.capacity), dtype=jnp.bool_), k=1
            )
        )
        delta = (
            configuration.coordinates[:, None, :] - configuration.coordinates[None, :, :]
        )
        squared = jnp.sum(delta * delta, axis=-1) + self.softening**2
        singular = jnp.any(pair & (squared == 0))
        safe = jnp.where(pair & (squared > 0), squared, 1.0)
        labels = jnp.clip(configuration.species, 0, self.space.species_count - 1)
        coefficients = self.coupling[labels[:, None], labels[None, :]]
        value = jnp.sum(jnp.where(pair, coefficients / safe ** (0.5 * self.power), 0.0))
        configuration_valid = self.space.valid(configuration)
        valid = configuration_valid & ~singular
        status = jnp.where(
            ~configuration_valid,
            VariableSectorLocalStatus.INVALID_CONFIGURATION,
            jnp.where(singular, VariableSectorLocalStatus.SINGULAR_CONFIGURATION, 0),
        )
        return _estimate(
            value.astype(jnp.result_type(value, 1j)),
            valid,
            status,
            jnp.sum(pair),
            configuration.particle_count == self.space.capacity,
            self.operator_id,
            "exact-active-radial-pair-sum",
        )


class ContactInteractionOperator(AbstractVariableSectorLocalOperator):
    """Finite Gaussian-regulated contact interaction with explicit cutoff."""

    space: VariableSectorSpace
    coupling: Array
    width: float = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        coupling: ArrayLike,
        width: float,
        /,
        *,
        operator_id: str | None = None,
    ):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        matrix = np.asarray(coupling, dtype=np.float64)
        width_ = float(width)
        if matrix.shape != (space.species_count, space.species_count):
            raise ValueError("coupling must have shape (species_count, species_count).")
        matrix = 0.5 * (matrix + matrix.T)
        if np.any(~np.isfinite(matrix)) or not np.isfinite(width_) or width_ <= 0:
            raise ValueError("Contact coupling/width must be finite with positive width.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "gaussian-regulated-contact-interaction",
                    "space": space.space_id,
                    "coupling": matrix.tolist(),
                    "width": width_,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.space = space
        self.coupling = jnp.asarray(matrix)
        self.width = width_
        self.operator_id = identifier

    def local_value(self, model, configuration, /):
        del model
        pair = (
            configuration.active_mask[:, None]
            & configuration.active_mask[None, :]
            & jnp.triu(
                jnp.ones((self.space.capacity, self.space.capacity), dtype=jnp.bool_), k=1
            )
        )
        delta = (
            configuration.coordinates[:, None, :] - configuration.coordinates[None, :, :]
        )
        squared = jnp.sum(delta * delta, axis=-1)
        normalization = (2.0 * jnp.pi * self.width**2) ** (-0.5 * self.space.dimension)
        labels = jnp.clip(configuration.species, 0, self.space.species_count - 1)
        coefficients = self.coupling[labels[:, None], labels[None, :]]
        value = jnp.sum(
            jnp.where(
                pair,
                coefficients * normalization * jnp.exp(-0.5 * squared / self.width**2),
                0.0,
            )
        )
        valid = self.space.valid(configuration)
        return _estimate(
            value.astype(jnp.result_type(value, 1j)),
            valid,
            jnp.where(valid, 0, VariableSectorLocalStatus.INVALID_CONFIGURATION),
            jnp.sum(pair),
            configuration.particle_count == self.space.capacity,
            self.operator_id,
            "exact-gaussian-regulated-contact-sum",
        )


def _constant_particle_mode(coordinate: Array, species: Array, /) -> Array:
    del coordinate, species
    return jnp.asarray(1.0 + 0.0j)


class ParticleChangingLocalOperator(AbstractVariableSectorLocalOperator):
    """Bosonic linear source on the declared ``1/n!`` sector measure.

    With symmetric coordinate amplitudes normalized against ``d x**n / n!``,
    creation is the exact sum over deletions and annihilation is the exact
    insertion integral; no additional square-root convention is mixed in.
    """

    space: VariableSectorSpace
    quadrature_nodes: Array
    quadrature_weights: Array
    source_coupling: Array
    mode: Callable[[Array, Array], Array] = eqx.field(static=True)
    mode_id: str = eqx.field(static=True)
    quadrature_targets: tuple[Any, ...]
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: VariableSectorSpace,
        quadrature_nodes: ArrayLike,
        quadrature_weights: ArrayLike,
        source_coupling: ArrayLike,
        /,
        *,
        mode: Callable[[Array, Array], Array] | None = None,
        mode_id: str | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(space, VariableSectorSpace):
            raise TypeError("space must be VariableSectorSpace.")
        nodes = np.asarray(quadrature_nodes, dtype=np.float64)
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        coupling = np.asarray(source_coupling)
        if (
            nodes.ndim != 3
            or nodes.shape[0] != space.species_count
            or nodes.shape[2] != space.dimension
        ):
            raise ValueError(
                "quadrature_nodes must have shape (species, point, dimension)."
            )
        if weights.shape != nodes.shape[:2] or coupling.shape != (space.species_count,):
            raise ValueError("quadrature weights/couplings have incompatible shapes.")
        if (
            np.any(~np.isfinite(nodes))
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0)
            or np.any(~np.isfinite(coupling))
        ):
            raise ValueError("Particle-changing quadrature data are invalid.")
        if mode is None:
            mode_ = _constant_particle_mode
            mode_identifier = "constant-unit-particle-mode"
            if mode_id is not None:
                raise ValueError("mode_id must be omitted for the constant default mode.")
        else:
            if not callable(mode):
                raise TypeError("mode must be callable or None.")
            mode_ = mode
            mode_identifier = "" if mode_id is None else str(mode_id)
            if not mode_identifier:
                raise ValueError("A caller-supplied mode requires a non-empty mode_id.")
        axis = "variable_sector_insertion_point"
        targets = tuple(
            discrete(
                nodes[species],
                cx.AxisArray(weights[species], dims=(axis,)),
                axes=axis,
                provenance="variable-sector-particle-changing-quadrature",
            )
            for species in range(space.species_count)
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "finite-quadrature-particle-changing-local-operator",
                    "space": space.space_id,
                    "nodes": array_tree_fingerprint(nodes),
                    "weights": array_tree_fingerprint(weights),
                    "coupling": array_tree_fingerprint(coupling),
                    "mode": mode_identifier,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.space = space
        self.quadrature_nodes = jnp.asarray(nodes)
        self.quadrature_weights = jnp.asarray(weights)
        self.source_coupling = jnp.asarray(coupling)
        self.mode = mode_
        self.mode_id = mode_identifier
        self.quadrature_targets = targets
        self.operator_id = identifier

    def local_value(self, model, configuration, /):
        current = model(configuration)
        if not isinstance(current, LogAmplitude):
            raise TypeError("model must return LogAmplitude.")
        counts = configuration.sector_counts(self.space.species_count)
        total = jnp.zeros((), dtype=jnp.result_type(current.phase, self.source_coupling))
        all_valid = self.space.valid(configuration) & current.valid & current.nonzero
        work = jnp.asarray(0, dtype=jnp.int32)
        for species in range(self.space.species_count):
            count = counts[species]
            species_mask = configuration.active_mask & (configuration.species == species)
            deletion_terms = []
            for slot in range(self.space.capacity):
                removed = model(configuration.remove(jnp.asarray(slot)))
                ratio = amplitude_ratio(removed, current)
                mode_value = self.mode(
                    configuration.coordinates[slot], jnp.asarray(species)
                )
                coefficient = jnp.where(species_mask[slot] & (count > 0), mode_value, 0.0)
                deletion_terms.append(coefficient * ratio.value)
                all_valid = all_valid & jnp.where(
                    species_mask[slot],
                    ratio.valid & jnp.isfinite(mode_value),
                    True,
                )
            creation = jnp.sum(jnp.stack(deletion_terms))

            def insertion_integrand(points):
                def at_point(point):
                    inserted = model(configuration.append(point, jnp.asarray(species)))
                    ratio = amplitude_ratio(inserted, current)
                    return ratio.value * jnp.conj(self.mode(point, jnp.asarray(species)))

                return jax.vmap(at_point)(points)

            insertion_estimate = integrate(
                insertion_integrand, self.quadrature_targets[species]
            )
            annihilation = insertion_estimate.value.data
            capacity_open = configuration.particle_count < self.space.capacity
            total = total + self.source_coupling[species] * (
                creation + jnp.where(capacity_open, annihilation, 0.0)
            )
            all_valid = all_valid & jnp.where(
                capacity_open, insertion_estimate.successful, True
            )
            work = (
                work
                + self.space.capacity
                + jnp.where(capacity_open, self.quadrature_nodes.shape[1], 0)
            )
        saturated = configuration.particle_count == self.space.capacity
        status = jnp.where(
            ~self.space.valid(configuration),
            VariableSectorLocalStatus.INVALID_CONFIGURATION,
            jnp.where(
                ~(current.valid & current.nonzero),
                VariableSectorLocalStatus.INVALID_AMPLITUDE,
                jnp.where(saturated, VariableSectorLocalStatus.CAPACITY_SATURATED, 0),
            ),
        )
        return _estimate(
            total,
            all_valid,
            status,
            work,
            saturated,
            self.operator_id,
            "finite-discrete-measure-creation-annihilation-quadrature",
        )


class VariableSectorHamiltonian(AbstractVariableSectorLocalOperator):
    space: VariableSectorSpace
    terms: tuple[AbstractVariableSectorLocalOperator, ...]
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: Sequence[AbstractVariableSectorLocalOperator],
        /,
        *,
        operator_id: str | None = None,
    ):
        terms_ = tuple(terms)
        if not terms_ or any(
            not isinstance(term, AbstractVariableSectorLocalOperator) for term in terms_
        ):
            raise TypeError("terms must contain variable-sector local operators.")
        space = terms_[0].space
        if any(term.space.space_id != space.space_id for term in terms_[1:]):
            raise ValueError("All Hamiltonian terms must use one variable-sector space.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "variable-sector-hamiltonian",
                    "space": space.space_id,
                    "terms": [term.operator_id for term in terms_],
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.space = space
        self.terms = terms_
        self.operator_id = identifier

    def local_value(self, model, configuration, /):
        estimates = tuple(term.local_value(model, configuration) for term in self.terms)
        valid = jnp.all(jnp.stack(tuple(value.valid for value in estimates)))
        status = jnp.max(jnp.stack(tuple(value.status for value in estimates)))
        return _estimate(
            sum(
                (value.value for value in estimates), jnp.zeros((), dtype=jnp.complex128)
            ),
            valid,
            status,
            sum(
                (value.work_count for value in estimates), jnp.asarray(0, dtype=jnp.int32)
            ),
            jnp.any(jnp.stack(tuple(value.cutoff_saturated for value in estimates))),
            self.operator_id,
            "additive-variable-sector-local-hamiltonian",
        )


__all__ = [
    "AbstractVariableSectorLocalOperator",
    "ContactInteractionOperator",
    "ContinuumKineticOperator",
    "ContinuumKineticPlan",
    "ExternalPotentialOperator",
    "PairPotentialOperator",
    "ParticleChangingLocalOperator",
    "QuadraticExternalPotential",
    "VariableParticleConfiguration",
    "VariableSectorHamiltonian",
    "VariableSectorLocalEstimate",
    "VariableSectorLocalStatus",
    "VariableSectorMeasure",
    "VariableSectorProposal",
    "VariableSectorProposalPlan",
    "VariableSectorProposalPayload",
    "VariableSectorSpace",
    "prepare_variable_sector_proposal",
]
