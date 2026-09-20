#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact Lorentz-invariant two-body and recursive phase-space maps."""

from __future__ import annotations

import abc
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._kinematics import minkowski_dot


def kallen(x: ArrayLike, y: ArrayLike, z: ArrayLike, /) -> Array:
    """Källén triangle polynomial."""
    x_, y_, z_ = jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)
    return x_**2 + y_**2 + z_**2 - 2.0 * (x_ * y_ + y_ * z_ + z_ * x_)


def _active_boost(momentum: Array, velocity: Array, /) -> Array:
    speed_squared = jnp.sum(velocity * velocity)
    gamma = 1.0 / jnp.sqrt(1.0 - speed_squared)
    beta_dot_p = jnp.sum(velocity * momentum[1:])
    factor = jnp.where(
        speed_squared > 0.0,
        ((gamma - 1.0) * beta_dot_p / speed_squared + gamma * momentum[0]),
        momentum[0],
    )
    return jnp.concatenate(
        (
            (gamma * (momentum[0] + beta_dot_p)).reshape((1,)),
            momentum[1:] + factor * velocity,
        )
    )


def _two_body_rest(
    parent_mass_squared: Array,
    first_mass: Array,
    second_mass: Array,
    cosine: Array,
    azimuth: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    root_s = jnp.sqrt(jnp.maximum(parent_mass_squared, 0.0))
    triangle = kallen(parent_mass_squared, first_mass**2, second_mass**2)
    magnitude = jnp.sqrt(jnp.maximum(triangle, 0.0)) / (2.0 * root_s)
    sine = jnp.sqrt(jnp.maximum(1.0 - cosine**2, 0.0))
    spatial = magnitude * jnp.asarray(
        [sine * jnp.cos(azimuth), sine * jnp.sin(azimuth), cosine]
    )
    first_energy = (parent_mass_squared + first_mass**2 - second_mass**2) / (2.0 * root_s)
    second_energy = (parent_mass_squared + second_mass**2 - first_mass**2) / (
        2.0 * root_s
    )
    first = jnp.concatenate((first_energy.reshape((1,)), spatial))
    second = jnp.concatenate((second_energy.reshape((1,)), -spatial))
    jacobian = jnp.sqrt(jnp.maximum(triangle, 0.0)) / (8.0 * jnp.pi * parent_mass_squared)
    valid = (
        (parent_mass_squared > 0.0)
        & (root_s >= first_mass + second_mass)
        & (first_mass >= 0.0)
        & (second_mass >= 0.0)
    )
    return first, second, jacobian, valid


class PhaseSpacePoint(StrictModule):
    """One fixed-multiplicity phase-space point and its mapping evidence."""

    momenta: Array
    jacobian: Array
    valid: Array
    map_id: str = eqx.field(static=True)


class AbstractPhaseSpaceMap(StrictModule):
    """Contract for normalized-coordinate maps onto Lorentz-invariant phase space."""

    __strict_abstract__ = True

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def multiplicity(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def map_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def map(self, unit: ArrayLike, total: ArrayLike, /) -> PhaseSpacePoint:
        raise NotImplementedError

    @abc.abstractmethod
    def density(self, momenta: ArrayLike, total: ArrayLike, /) -> Array:
        """Return proposal density with respect to invariant phase-space measure."""
        raise NotImplementedError


class TwoBodyPhaseSpaceMap(AbstractPhaseSpaceMap, NonTrainableState):
    """Exact isotropic two-body Lorentz-invariant phase-space map."""

    masses: Array
    _map_id: str = eqx.field(static=True)

    def __init__(self, first_mass: float, second_mass: float, /):
        masses = np.asarray([first_mass, second_mass], dtype=np.float64)
        if np.any(~np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("Two-body masses must be finite and nonnegative.")
        self.masses = jnp.asarray(masses)
        self._map_id = canonical_fingerprint(
            {"kind": "two-body-phase-space", "masses": array_tree_fingerprint(masses)}
        )

    @property
    def dimension(self) -> int:
        return 2

    @property
    def multiplicity(self) -> int:
        return 2

    @property
    def map_id(self) -> str:
        return self._map_id

    def map(self, unit: ArrayLike, total: ArrayLike, /) -> PhaseSpacePoint:
        coordinates = jnp.asarray(unit)
        parent = jnp.asarray(total)
        if coordinates.shape != (2,) or parent.shape != (4,):
            raise ValueError(
                "Two-body mapping requires unit shape (2,) and total shape (4,)."
            )
        s = minkowski_dot(parent, parent)
        first, second, jacobian, valid = _two_body_rest(
            s,
            self.masses[0],
            self.masses[1],
            2.0 * coordinates[0] - 1.0,
            2.0 * jnp.pi * coordinates[1],
        )
        velocity = parent[1:] / parent[0]
        momenta = jnp.stack(
            (_active_boost(first, velocity), _active_boost(second, velocity))
        )
        in_unit_cube = jnp.all((coordinates >= 0.0) & (coordinates <= 1.0))
        return PhaseSpacePoint(momenta, jacobian, valid & in_unit_cube, self.map_id)

    def density(self, momenta: ArrayLike, total: ArrayLike, /) -> Array:
        values = jnp.asarray(momenta)
        parent = jnp.asarray(total)
        if values.shape != (2, 4) or parent.shape != (4,):
            raise ValueError("Two-body density requires momenta shape (2, 4).")
        s = minkowski_dot(parent, parent)
        triangle = kallen(s, self.masses[0] ** 2, self.masses[1] ** 2)
        jacobian = jnp.sqrt(jnp.maximum(triangle, 0.0)) / (8.0 * jnp.pi * s)
        conservation = jnp.max(jnp.abs(jnp.sum(values, axis=0) - parent))
        shell = jnp.max(
            jnp.abs(
                jnp.asarray(
                    [
                        minkowski_dot(values[0], values[0]) - self.masses[0] ** 2,
                        minkowski_dot(values[1], values[1]) - self.masses[1] ** 2,
                    ]
                )
            )
        )
        scale = jnp.maximum(1.0, jnp.abs(s))
        valid = (
            (triangle >= 0.0)
            & (conservation <= 1.0e-8 * scale)
            & (shell <= 1.0e-8 * scale)
        )
        return jnp.where(valid, 1.0 / jacobian, 0.0)


class RecursivePhaseSpaceMap(AbstractPhaseSpaceMap, NonTrainableState):
    """Exact sequential factorization of fixed-multiplicity invariant phase space."""

    masses: Array
    _dimension: int = eqx.field(static=True)
    _multiplicity: int = eqx.field(static=True)
    max_multiplicity: int = eqx.field(static=True)
    _map_id: str = eqx.field(static=True)

    def __init__(
        self,
        masses: Sequence[float],
        /,
        *,
        max_multiplicity: int = 16,
    ):
        masses_ = np.asarray(tuple(masses), dtype=np.float64)
        maximum = int(max_multiplicity)
        if masses_.ndim != 1 or masses_.size < 2:
            raise ValueError("Recursive phase space requires at least two final masses.")
        if maximum < 2 or masses_.size > maximum:
            raise ValueError("Final-state multiplicity exceeds max_multiplicity.")
        if np.any(~np.isfinite(masses_)) or np.any(masses_ < 0.0):
            raise ValueError("Final-state masses must be finite and nonnegative.")
        self.masses = jnp.asarray(masses_)
        self._multiplicity = masses_.size
        self._dimension = 3 * masses_.size - 4
        self.max_multiplicity = maximum
        self._map_id = canonical_fingerprint(
            {
                "kind": "recursive-phase-space",
                "masses": array_tree_fingerprint(masses_),
                "max_multiplicity": maximum,
            }
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def multiplicity(self) -> int:
        return self._multiplicity

    @property
    def map_id(self) -> str:
        return self._map_id

    def map(self, unit: ArrayLike, total: ArrayLike, /) -> PhaseSpacePoint:
        coordinates = jnp.asarray(unit)
        parent = jnp.asarray(total)
        if coordinates.shape != (self.dimension,) or parent.shape != (4,):
            raise ValueError(
                "Recursive phase-space coordinate or total shape is invalid."
            )
        cursor = 0
        jacobian = jnp.asarray(1.0, dtype=jnp.result_type(coordinates, parent))
        valid = jnp.all((coordinates >= 0.0) & (coordinates <= 1.0))
        emitted: list[Array] = []
        current_parent = parent
        for particle_index in range(self.multiplicity - 1):
            current_s = minkowski_dot(current_parent, current_parent)
            current_root = jnp.sqrt(jnp.maximum(current_s, 0.0))
            current_mass = self.masses[particle_index]
            if particle_index < self.multiplicity - 2:
                residual_minimum = jnp.sum(self.masses[particle_index + 1 :])
                residual_maximum = current_root - current_mass
                lower_squared = residual_minimum**2
                upper_squared = residual_maximum**2
                residual_squared = lower_squared + coordinates[cursor] * (
                    upper_squared - lower_squared
                )
                cursor += 1
                jacobian = jacobian * (upper_squared - lower_squared) / (2.0 * jnp.pi)
                residual_mass = jnp.sqrt(jnp.maximum(residual_squared, 0.0))
                valid = valid & (upper_squared >= lower_squared)
            else:
                residual_mass = self.masses[-1]
            cosine = 2.0 * coordinates[cursor] - 1.0
            azimuth = 2.0 * jnp.pi * coordinates[cursor + 1]
            cursor += 2
            first_rest, residual_rest, local_jacobian, local_valid = _two_body_rest(
                current_s, current_mass, residual_mass, cosine, azimuth
            )
            velocity = current_parent[1:] / current_parent[0]
            emitted.append(_active_boost(first_rest, velocity))
            current_parent = _active_boost(residual_rest, velocity)
            jacobian = jacobian * local_jacobian
            valid = valid & local_valid
        emitted.append(current_parent)
        return PhaseSpacePoint(jnp.stack(emitted), jacobian, valid, self.map_id)

    def density(self, momenta: ArrayLike, total: ArrayLike, /) -> Array:
        values = jnp.asarray(momenta)
        parent = jnp.asarray(total)
        if values.shape != (self.multiplicity, 4) or parent.shape != (4,):
            raise ValueError("Recursive density received incompatible momentum shape.")
        jacobian = jnp.asarray(1.0, dtype=jnp.result_type(values, parent))
        valid = jnp.asarray(True)
        for particle_index in range(self.multiplicity - 1):
            current_parent = jnp.sum(values[particle_index:], axis=0)
            current_s = minkowski_dot(current_parent, current_parent)
            current_root = jnp.sqrt(jnp.maximum(current_s, 0.0))
            first_mass = self.masses[particle_index]
            if particle_index < self.multiplicity - 2:
                residual = jnp.sum(values[particle_index + 1 :], axis=0)
                residual_squared = minkowski_dot(residual, residual)
                lower = jnp.sum(self.masses[particle_index + 1 :]) ** 2
                upper = (current_root - first_mass) ** 2
                jacobian = jacobian * (upper - lower) / (2.0 * jnp.pi)
                residual_mass = jnp.sqrt(jnp.maximum(residual_squared, 0.0))
                valid = (
                    valid
                    & (residual_squared >= lower - 1.0e-8)
                    & (residual_squared <= upper + 1.0e-8)
                )
            else:
                residual_mass = self.masses[-1]
            triangle = kallen(current_s, first_mass**2, residual_mass**2)
            local = jnp.sqrt(jnp.maximum(triangle, 0.0)) / (8.0 * jnp.pi * current_s)
            jacobian = jacobian * local
            valid = valid & (triangle >= 0.0)
        conservation = jnp.max(jnp.abs(jnp.sum(values, axis=0) - parent))
        shells = jnp.asarray(
            [
                minkowski_dot(values[index], values[index]) - self.masses[index] ** 2
                for index in range(self.multiplicity)
            ]
        )
        scale = jnp.maximum(1.0, jnp.abs(minkowski_dot(parent, parent)))
        valid = (
            valid
            & (conservation <= 1.0e-8 * scale)
            & (jnp.max(jnp.abs(shells)) <= 1.0e-8 * scale)
        )
        return jnp.where(valid, 1.0 / jacobian, 0.0)


class MultiChannelPhaseSpacePoint(StrictModule):
    """Phase-space point weighted by the full proposal mixture."""

    point: PhaseSpacePoint
    proposal_density: Array
    integration_weight: Array
    channel_index: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class MultiChannelPhaseSpacePlan(StrictModule, NonTrainableState):
    """Fixed set of normalized phase-space proposal channels."""

    channels: tuple[AbstractPhaseSpaceMap, ...]
    probabilities: Array
    multiplicity: int = eqx.field(static=True)
    max_channels: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        channels: Sequence[AbstractPhaseSpaceMap],
        probabilities: Sequence[float],
        /,
        *,
        max_channels: int = 16,
    ):
        channels_ = tuple(channels)
        probabilities_ = np.asarray(tuple(probabilities), dtype=np.float64)
        maximum = int(max_channels)
        if not channels_ or probabilities_.shape != (len(channels_),):
            raise ValueError(
                "Multi-channel maps require aligned nonempty channels and probabilities."
            )
        if maximum < 1 or len(channels_) > maximum:
            raise ValueError("Channel count exceeds max_channels.")
        if np.any(~np.isfinite(probabilities_)) or np.any(probabilities_ <= 0.0):
            raise ValueError(
                "Every multi-channel probability must be finite and positive."
            )
        if not all(isinstance(channel, AbstractPhaseSpaceMap) for channel in channels_):
            raise TypeError("Every channel must implement AbstractPhaseSpaceMap.")
        multiplicity = channels_[0].multiplicity
        if any(channel.multiplicity != multiplicity for channel in channels_):
            raise ValueError("All phase-space channels must have equal multiplicity.")
        probabilities_ = probabilities_ / np.sum(probabilities_)
        self.channels = channels_
        self.probabilities = jnp.asarray(probabilities_)
        self.multiplicity = multiplicity
        self.max_channels = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multi-channel-phase-space",
                "channels": [channel.map_id for channel in channels_],
                "probabilities": array_tree_fingerprint(probabilities_),
                "max_channels": maximum,
            }
        )

    def select_channel(self, selector: ArrayLike, /) -> Array:
        """Map unit selectors to channel indices without a host-side branch."""
        selector_ = jnp.asarray(selector)
        clipped = jnp.clip(selector_, 0.0, jnp.nextafter(1.0, 0.0))
        return jnp.searchsorted(jnp.cumsum(self.probabilities), clipped)

    def sample_channel_indices(self, key: Key[Array, ""], count: int, /) -> Array:
        """Draw a fixed-size set of channel indices from the prepared mixture."""
        count_ = int(count)
        if count_ < 1:
            raise ValueError("Multi-channel sampling count must be positive.")
        return jr.choice(
            key,
            len(self.channels),
            shape=(count_,),
            replace=True,
            p=self.probabilities,
        )

    def map_channel(
        self,
        channel_index: int,
        unit: ArrayLike,
        total: ArrayLike,
        /,
    ) -> MultiChannelPhaseSpacePoint:
        index = int(channel_index)
        if index < 0 or index >= len(self.channels):
            raise ValueError("channel_index is outside the prepared channel set.")
        point = self.channels[index].map(unit, total)
        mixture = sum(
            (
                self.probabilities[channel]
                * self.channels[channel].density(point.momenta, total)
                for channel in range(len(self.channels))
            ),
            jnp.asarray(0.0),
        )
        valid = point.valid & jnp.isfinite(mixture) & (mixture > 0.0)
        weight = jnp.where(valid, 1.0 / mixture, 0.0)
        return MultiChannelPhaseSpacePoint(
            point,
            mixture,
            weight,
            jnp.asarray(index, dtype=jnp.int32),
            valid,
            self.plan_id,
        )


__all__ = [
    "AbstractPhaseSpaceMap",
    "MultiChannelPhaseSpacePlan",
    "MultiChannelPhaseSpacePoint",
    "PhaseSpacePoint",
    "RecursivePhaseSpaceMap",
    "TwoBodyPhaseSpaceMap",
    "kallen",
]
