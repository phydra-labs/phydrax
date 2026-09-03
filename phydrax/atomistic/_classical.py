#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._trainable import NonTrainableState
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from ._system import PreparedAtomisticSystem


LennardJonesCombiningRule: TypeAlias = Literal[
    "lorentz-berthelot", "geometric", "explicit"
]


def _parameters(name: str, value: ArrayLike, /, *, positive: bool = False) -> Array:
    host = np.asarray(value, dtype=float)
    if host.ndim != 1 or host.size == 0 or np.any(~np.isfinite(host)):
        raise ValueError(f"{name} must be a non-empty finite vector.")
    if positive and np.any(host <= 0.0):
        raise ValueError(f"{name} must be positive.")
    return jnp.asarray(host)


def _term_identity(kind: str, name: str, group: int, arrays, /, **extra) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "name": name,
            "force_group": group,
            "arrays": array_tree_fingerprint(arrays),
            **extra,
        }
    )


def _validate_name_group(name: str, force_group: int, /) -> tuple[str, int]:
    identifier = str(name).strip()
    group = int(force_group)
    if not identifier or group < 0:
        raise ValueError("Potential name must be non-empty and force_group non-negative.")
    return identifier, group


class HarmonicBondPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    stiffness: Array
    equilibrium_distance: Array
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        stiffness: ArrayLike,
        equilibrium_distance: ArrayLike,
        /,
        *,
        name: str = "harmonic-bond",
        force_group: int = 0,
    ):
        k = _parameters("stiffness", stiffness, positive=True)
        distance = _parameters(
            "equilibrium_distance", equilibrium_distance, positive=True
        )
        if k.shape != distance.shape:
            raise ValueError("Bond stiffness and equilibrium distance tables must match.")
        identifier, group = _validate_name_group(name, force_group)
        self.stiffness = k
        self.equilibrium_distance = distance
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=True,
        )
        self.requirements = AtomisticPotentialRequirements(bonded_geometry=True)
        self.term_id = _term_identity(
            "harmonic-bond-potential",
            identifier,
            group,
            {"stiffness": np.asarray(k), "equilibrium_distance": np.asarray(distance)},
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedHarmonicBondPotential":
        types = np.asarray(system.topology.bond_type_ids)
        if types.size and int(np.max(types)) >= self.stiffness.size:
            raise ValueError("Bond type ID exceeds the harmonic parameter table.")
        return PreparedHarmonicBondPotential(self, system)


class PreparedHarmonicBondPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: HarmonicBondPotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: HarmonicBondPotential, system: PreparedAtomisticSystem, /):
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-harmonic-bond",
                "term": plan.term_id,
                "system": system.prepared_id,
            }
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        indices = self.system.topology.bond_indices
        count = int(indices.shape[0])
        if count == 0:
            zero = jnp.zeros((), dtype=context.positions.dtype)
            return AtomisticTermEvaluation(
                zero, jnp.zeros((self.system.capacity,), dtype=zero.dtype), True
            )
        displacement = (
            context.unwrapped_positions[indices[:, 0]]
            - context.unwrapped_positions[indices[:, 1]]
        )
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        types = self.system.topology.bond_type_ids
        delta = distance - self.plan.equilibrium_distance[types]
        interaction = 0.5 * self.plan.stiffness[types] * delta * delta
        atom_energy = jnp.zeros((self.system.capacity,), dtype=interaction.dtype)
        atom_energy = atom_energy.at[indices[:, 0]].add(0.5 * interaction)
        atom_energy = atom_energy.at[indices[:, 1]].add(0.5 * interaction)
        successful = jnp.all(jnp.isfinite(distance) & (distance > 0.0))
        energy = jnp.where(successful, jnp.sum(interaction), jnp.nan)
        return AtomisticTermEvaluation(energy, atom_energy, successful)


class HarmonicAnglePotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    stiffness: Array
    equilibrium_angle: Array
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        stiffness: ArrayLike,
        equilibrium_angle: ArrayLike,
        /,
        *,
        name: str = "harmonic-angle",
        force_group: int = 0,
    ):
        k = _parameters("stiffness", stiffness, positive=True)
        angle = _parameters("equilibrium_angle", equilibrium_angle)
        if k.shape != angle.shape or bool(jnp.any((angle <= 0.0) | (angle >= jnp.pi))):
            raise ValueError(
                "Angle tables must match with targets strictly between zero and pi."
            )
        identifier, group = _validate_name_group(name, force_group)
        self.stiffness = k
        self.equilibrium_angle = angle
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=True,
        )
        self.requirements = AtomisticPotentialRequirements(bonded_geometry=True)
        self.term_id = _term_identity(
            "harmonic-angle-potential",
            identifier,
            group,
            {"stiffness": np.asarray(k), "equilibrium_angle": np.asarray(angle)},
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedHarmonicAnglePotential":
        types = np.asarray(system.topology.angle_type_ids)
        if types.size and int(np.max(types)) >= self.stiffness.size:
            raise ValueError("Angle type ID exceeds the harmonic parameter table.")
        return PreparedHarmonicAnglePotential(self, system)


class PreparedHarmonicAnglePotential(AbstractPreparedAtomisticEnergyTerm):
    plan: HarmonicAnglePotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: HarmonicAnglePotential, system: PreparedAtomisticSystem, /):
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-harmonic-angle",
                "term": plan.term_id,
                "system": system.prepared_id,
            }
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        indices = self.system.topology.angle_indices
        count = int(indices.shape[0])
        if count == 0:
            zero = jnp.zeros((), dtype=context.positions.dtype)
            return AtomisticTermEvaluation(
                zero, jnp.zeros((self.system.capacity,), dtype=zero.dtype), True
            )
        left = (
            context.unwrapped_positions[indices[:, 0]]
            - context.unwrapped_positions[indices[:, 1]]
        )
        right = (
            context.unwrapped_positions[indices[:, 2]]
            - context.unwrapped_positions[indices[:, 1]]
        )
        left_norm = jnp.sqrt(jnp.sum(left * left, axis=-1))
        right_norm = jnp.sqrt(jnp.sum(right * right, axis=-1))
        cross_norm = jnp.sqrt(jnp.sum(jnp.cross(left, right) ** 2, axis=-1))
        dot = jnp.sum(left * right, axis=-1)
        angle = jnp.arctan2(cross_norm, dot)
        types = self.system.topology.angle_type_ids
        delta = angle - self.plan.equilibrium_angle[types]
        interaction = 0.5 * self.plan.stiffness[types] * delta * delta
        atom_energy = jnp.zeros((self.system.capacity,), dtype=interaction.dtype)
        share = interaction / 3.0
        for endpoint in range(3):
            atom_energy = atom_energy.at[indices[:, endpoint]].add(share)
        successful = jnp.all(jnp.isfinite(angle) & (left_norm > 0.0) & (right_norm > 0.0))
        return AtomisticTermEvaluation(
            jnp.where(successful, jnp.sum(interaction), jnp.nan),
            atom_energy,
            successful,
        )


class PeriodicTorsionPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    amplitude: Array
    periodicity: Array
    phase: Array
    improper: bool = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        amplitude: ArrayLike,
        periodicity: ArrayLike,
        phase: ArrayLike,
        /,
        *,
        improper: bool = False,
        name: str = "periodic-torsion",
        force_group: int = 0,
    ):
        amplitude_ = _parameters("amplitude", amplitude)
        periodicity_host = np.asarray(periodicity)
        phase_ = _parameters("phase", phase)
        if (
            periodicity_host.shape != amplitude_.shape
            or not np.issubdtype(periodicity_host.dtype, np.integer)
            or phase_.shape != amplitude_.shape
            or np.any(periodicity_host <= 0)
        ):
            raise ValueError(
                "Torsion amplitude, positive integer periodicity, and phase tables must match."
            )
        identifier, group = _validate_name_group(name, force_group)
        self.amplitude = amplitude_
        self.periodicity = jnp.asarray(periodicity_host, dtype=jnp.int32)
        self.phase = phase_
        self.improper = bool(improper)
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=True,
        )
        self.requirements = AtomisticPotentialRequirements(bonded_geometry=True)
        self.term_id = _term_identity(
            "periodic-torsion-potential",
            identifier,
            group,
            {
                "amplitude": np.asarray(amplitude_),
                "periodicity": periodicity_host,
                "phase": np.asarray(phase_),
            },
            improper=bool(improper),
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedPeriodicTorsionPotential":
        types = (
            np.asarray(system.topology.improper_type_ids)
            if self.improper
            else np.asarray(system.topology.torsion_type_ids)
        )
        if types.size and int(np.max(types)) >= self.amplitude.size:
            raise ValueError("Torsion type ID exceeds the periodic parameter table.")
        return PreparedPeriodicTorsionPotential(self, system)


class PreparedPeriodicTorsionPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: PeriodicTorsionPotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: PeriodicTorsionPotential, system: PreparedAtomisticSystem, /
    ):
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-torsion",
                "term": plan.term_id,
                "system": system.prepared_id,
            }
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        indices = (
            self.system.topology.improper_indices
            if self.plan.improper
            else self.system.topology.torsion_indices
        )
        types = (
            self.system.topology.improper_type_ids
            if self.plan.improper
            else self.system.topology.torsion_type_ids
        )
        count = int(indices.shape[0])
        if count == 0:
            zero = jnp.zeros((), dtype=context.positions.dtype)
            return AtomisticTermEvaluation(
                zero, jnp.zeros((self.system.capacity,), dtype=zero.dtype), True
            )
        q = context.unwrapped_positions
        b0 = q[indices[:, 0]] - q[indices[:, 1]]
        b1 = q[indices[:, 2]] - q[indices[:, 1]]
        b2 = q[indices[:, 3]] - q[indices[:, 2]]
        b1_norm = jnp.sqrt(jnp.sum(b1 * b1, axis=-1))
        safe_b1 = jnp.where(b1_norm[:, None] > 0.0, b1 / b1_norm[:, None], 0.0)
        v = b0 - contract("ni,ni->n", b0, safe_b1)[:, None] * safe_b1
        w = b2 - contract("ni,ni->n", b2, safe_b1)[:, None] * safe_b1
        v_norm = jnp.sqrt(jnp.sum(v * v, axis=-1))
        w_norm = jnp.sqrt(jnp.sum(w * w, axis=-1))
        x = contract("ni,ni->n", v, w)
        y = contract("ni,ni->n", jnp.cross(safe_b1, v), w)
        angle = jnp.arctan2(y, x)
        interaction = self.plan.amplitude[types] * (
            1.0
            + jnp.cos(
                self.plan.periodicity[types].astype(angle.dtype) * angle
                - self.plan.phase[types]
            )
        )
        atom_energy = jnp.zeros((self.system.capacity,), dtype=interaction.dtype)
        share = interaction / 4.0
        for endpoint in range(4):
            atom_energy = atom_energy.at[indices[:, endpoint]].add(share)
        successful = jnp.all(
            jnp.isfinite(angle) & (b1_norm > 0.0) & (v_norm > 0.0) & (w_norm > 0.0)
        )
        return AtomisticTermEvaluation(
            jnp.where(successful, jnp.sum(interaction), jnp.nan),
            atom_energy,
            successful,
        )


class LennardJonesPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    epsilon: Array
    sigma: Array
    explicit_epsilon: Array | None
    explicit_sigma: Array | None
    cutoff: float = eqx.field(static=True)
    switch_distance: float | None = eqx.field(static=True)
    combining_rule: LennardJonesCombiningRule = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        epsilon: ArrayLike,
        sigma: ArrayLike,
        cutoff: float,
        /,
        *,
        switch_distance: float | None = None,
        combining_rule: LennardJonesCombiningRule = "lorentz-berthelot",
        explicit_epsilon: ArrayLike | None = None,
        explicit_sigma: ArrayLike | None = None,
        name: str = "lennard-jones",
        force_group: int = 0,
    ):
        epsilon_ = _parameters("epsilon", epsilon)
        sigma_ = _parameters("sigma", sigma, positive=True)
        if epsilon_.shape != sigma_.shape or bool(jnp.any(epsilon_ < 0.0)):
            raise ValueError(
                "Lennard-Jones epsilon must be non-negative and match positive sigma."
            )
        cutoff_ = float(cutoff)
        switch_ = None if switch_distance is None else float(switch_distance)
        if (
            not math.isfinite(cutoff_)
            or cutoff_ <= 0.0
            or (
                switch_ is not None
                and (not math.isfinite(switch_) or not 0.0 <= switch_ < cutoff_)
            )
        ):
            raise ValueError(
                "Lennard-Jones cutoff must be positive and switching must lie below it."
            )
        if combining_rule not in ("lorentz-berthelot", "geometric", "explicit"):
            raise ValueError("Unknown Lennard-Jones combining rule.")
        explicit_epsilon_ = None
        explicit_sigma_ = None
        if combining_rule == "explicit":
            if explicit_epsilon is None or explicit_sigma is None:
                raise ValueError(
                    "Explicit combining requires epsilon and sigma matrices."
                )
            explicit_epsilon_host = np.asarray(explicit_epsilon, dtype=float)
            explicit_sigma_host = np.asarray(explicit_sigma, dtype=float)
            expected = (epsilon_.size, epsilon_.size)
            if (
                explicit_epsilon_host.shape != expected
                or explicit_sigma_host.shape != expected
                or np.any(~np.isfinite(explicit_epsilon_host))
                or np.any(~np.isfinite(explicit_sigma_host))
                or np.any(explicit_epsilon_host < 0.0)
                or np.any(explicit_sigma_host <= 0.0)
                or not np.allclose(explicit_epsilon_host, explicit_epsilon_host.T)
                or not np.allclose(explicit_sigma_host, explicit_sigma_host.T)
            ):
                raise ValueError(
                    "Explicit Lennard-Jones epsilon must be non-negative; sigma must be "
                    "positive, finite, and symmetric."
                )
            explicit_epsilon_ = jnp.asarray(explicit_epsilon_host)
            explicit_sigma_ = jnp.asarray(explicit_sigma_host)
        identifier, group = _validate_name_group(name, force_group)
        self.epsilon = epsilon_
        self.sigma = sigma_
        self.explicit_epsilon = explicit_epsilon_
        self.explicit_sigma = explicit_sigma_
        self.cutoff = cutoff_
        self.switch_distance = switch_
        self.combining_rule = combining_rule
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=True,
            local_energy_delta=True,
            dynamic_species=True,
        )
        self.requirements = AtomisticPotentialRequirements(
            cutoff=cutoff_, pair_geometry=True
        )
        self.term_id = _term_identity(
            "lennard-jones-potential",
            identifier,
            group,
            {
                "epsilon": np.asarray(epsilon_),
                "sigma": np.asarray(sigma_),
                "explicit_epsilon": None
                if explicit_epsilon_ is None
                else np.asarray(explicit_epsilon_),
                "explicit_sigma": None
                if explicit_sigma_ is None
                else np.asarray(explicit_sigma_),
            },
            cutoff=cutoff_,
            switch_distance=switch_,
            combining_rule=combining_rule,
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedLennardJonesPotential":
        types = np.asarray(system.plan.atom_type_ids)[np.asarray(system.active_mask)]
        if types.size and int(np.max(types)) >= self.epsilon.size:
            raise ValueError("Atom type ID exceeds the Lennard-Jones parameter table.")
        return PreparedLennardJonesPotential(self, system)


class PreparedLennardJonesPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: LennardJonesPotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: LennardJonesPotential, system: PreparedAtomisticSystem, /):
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-lennard-jones",
                "term": plan.term_id,
                "system": system.prepared_id,
            }
        )

    def _mixed(self, left_type: Array, right_type: Array, /) -> tuple[Array, Array]:
        if self.plan.combining_rule == "explicit":
            epsilon = self.plan.explicit_epsilon
            sigma = self.plan.explicit_sigma
            if epsilon is None or sigma is None:
                raise RuntimeError(
                    "Validated explicit Lennard-Jones matrices are absent."
                )
            return epsilon[left_type, right_type], sigma[left_type, right_type]
        left_epsilon = self.plan.epsilon[left_type]
        right_epsilon = self.plan.epsilon[right_type]
        left_sigma = self.plan.sigma[left_type]
        right_sigma = self.plan.sigma[right_type]
        epsilon = jnp.sqrt(left_epsilon * right_epsilon)
        sigma = (
            jnp.sqrt(left_sigma * right_sigma)
            if self.plan.combining_rule == "geometric"
            else 0.5 * (left_sigma + right_sigma)
        )
        return epsilon, sigma

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        raw_left_type = context.species[context.pair_left]
        raw_right_type = context.species[context.pair_right]
        type_count = int(self.plan.epsilon.size)
        valid_types = (
            (raw_left_type >= 0)
            & (raw_left_type < type_count)
            & (raw_right_type >= 0)
            & (raw_right_type < type_count)
        )
        left_type = jnp.clip(raw_left_type, 0, type_count - 1)
        right_type = jnp.clip(raw_right_type, 0, type_count - 1)
        epsilon, sigma = self._mixed(left_type, right_type)
        distance = context.pair_distance
        active = context.pair_valid & valid_types & (distance < self.plan.cutoff)
        valid_geometry = jnp.all(~context.pair_valid | (valid_types & (distance > 0.0)))
        safe_distance = jnp.where(active & (distance > 0.0), distance, 1.0)
        ratio6 = (sigma / safe_distance) ** 6
        raw = 4.0 * epsilon * (ratio6 * ratio6 - ratio6)
        if self.plan.switch_distance is None:
            switch = jnp.where(distance < self.plan.cutoff, 1.0, 0.0)
        else:
            width = self.plan.cutoff - self.plan.switch_distance
            scaled = (distance - self.plan.switch_distance) / width
            smooth = 1.0 - 10.0 * scaled**3 + 15.0 * scaled**4 - 6.0 * scaled**5
            switch = jnp.where(
                distance <= self.plan.switch_distance,
                1.0,
                jnp.where(distance < self.plan.cutoff, smooth, 0.0),
            )
        interaction = jnp.where(
            active,
            raw * switch * context.lennard_jones_scales,
            0.0,
        )
        atom_energy = jnp.zeros((self.system.capacity,), dtype=interaction.dtype)
        atom_energy = atom_energy.at[context.pair_left].add(0.5 * interaction)
        atom_energy = atom_energy.at[context.pair_right].add(0.5 * interaction)
        successful = valid_geometry & jnp.all(jnp.isfinite(interaction))
        return AtomisticTermEvaluation(
            jnp.where(successful, jnp.sum(interaction), jnp.nan),
            atom_energy,
            successful,
        )


__all__ = [
    "HarmonicAnglePotential",
    "HarmonicBondPotential",
    "LennardJonesPotential",
    "PeriodicTorsionPotential",
]
