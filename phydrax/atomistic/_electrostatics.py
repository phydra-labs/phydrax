#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from itertools import product
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jaxtyping import Array
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._trainable import NonTrainableState
from ..discretization import (
    ParticleGridSplatPlan,
    PreparedParticleGridSplat,
    TensorBSplineSplatAssignment,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from ._system import PreparedAtomisticSystem


ChargeNeutralityPolicy: TypeAlias = Literal["require-neutral", "uniform-background"]


def _electrostatic_identity(kind: str, values: dict) -> str:
    return canonical_fingerprint({"kind": kind, **values})


class DirectCoulombPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, *, name: str = "direct-coulomb", force_group: int = 0):
        identifier = str(name).strip()
        group = int(force_group)
        if not identifier or group < 0:
            raise ValueError(
                "Coulomb name must be non-empty and force_group non-negative."
            )
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            conservative_energy=True,
            finite_geometry=True,
            local_energy=True,
            local_energy_delta=True,
            dynamic_species=True,
        )
        self.requirements = AtomisticPotentialRequirements(pair_geometry=True)
        self.term_id = _electrostatic_identity(
            "direct-coulomb-potential", {"name": identifier, "force_group": group}
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedDirectCoulombPotential":
        if system.cell is not None:
            raise ValueError("DirectCoulombPotential is finite and nonperiodic.")
        return PreparedDirectCoulombPotential(self, system)


class PreparedDirectCoulombPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: DirectCoulombPotential
    system: PreparedAtomisticSystem
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: DirectCoulombPotential, system: PreparedAtomisticSystem, /):
        self.plan = plan
        self.system = system
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = _electrostatic_identity(
            "prepared-direct-coulomb",
            {"term": plan.term_id, "system": system.prepared_id},
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        distance = context.pair_distance
        valid_geometry = jnp.all(~context.pair_valid | (distance > 0.0))
        safe = jnp.where(context.pair_valid & (distance > 0.0), distance, 1.0)
        charge = self.system.plan.charges
        pair = (
            self.system.plan.units.coulomb_constant
            * charge[context.pair_left]
            * charge[context.pair_right]
            / safe
            * context.electrostatic_scales
        )
        pair = jnp.where(context.pair_valid, pair, 0.0)
        atom_energy = jnp.zeros((self.system.capacity,), dtype=pair.dtype)
        atom_energy = atom_energy.at[context.pair_left].add(0.5 * pair)
        atom_energy = atom_energy.at[context.pair_right].add(0.5 * pair)
        successful = valid_geometry & jnp.all(jnp.isfinite(pair))
        return AtomisticTermEvaluation(
            jnp.where(successful, jnp.sum(pair), jnp.nan), atom_energy, successful
        )


class EwaldReferencePotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    alpha: float = eqx.field(static=True)
    real_cutoff: float = eqx.field(static=True)
    reciprocal_extent: int = eqx.field(static=True)
    neutrality: ChargeNeutralityPolicy = eqx.field(static=True)
    charge_tolerance: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        alpha: float,
        real_cutoff: float,
        reciprocal_extent: int,
        /,
        *,
        neutrality: ChargeNeutralityPolicy = "require-neutral",
        charge_tolerance: float = 1.0e-10,
        name: str = "ewald-reference",
        force_group: int = 0,
    ):
        alpha_ = float(alpha)
        cutoff = float(real_cutoff)
        extent = int(reciprocal_extent)
        tolerance = float(charge_tolerance)
        identifier = str(name).strip()
        group = int(force_group)
        if (
            not math.isfinite(alpha_)
            or alpha_ <= 0.0
            or not math.isfinite(cutoff)
            or cutoff <= 0.0
            or extent <= 0
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
            or not identifier
            or group < 0
        ):
            raise ValueError("Ewald parameters, name, and force group are invalid.")
        if neutrality not in ("require-neutral", "uniform-background"):
            raise ValueError("Unknown charge-neutrality policy.")
        self.alpha = alpha_
        self.real_cutoff = cutoff
        self.reciprocal_extent = extent
        self.neutrality = neutrality
        self.charge_tolerance = tolerance
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=False,
        )
        self.requirements = AtomisticPotentialRequirements(
            cutoff=cutoff, pair_geometry=True, reciprocal_grid=True
        )
        self.term_id = _electrostatic_identity(
            "ewald-reference-potential",
            {
                "alpha": alpha_,
                "real_cutoff": cutoff,
                "reciprocal_extent": extent,
                "neutrality": neutrality,
                "charge_tolerance": tolerance,
                "name": identifier,
                "force_group": group,
            },
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedEwaldReferencePotential":
        if system.cell is None or not system.cell.fully_periodic:
            raise ValueError("EwaldReferencePotential requires a fully periodic cell.")
        system.cell.require_unique_image(self.real_cutoff)
        return PreparedEwaldReferencePotential(self, system)


class PreparedEwaldReferencePotential(AbstractPreparedAtomisticEnergyTerm):
    plan: EwaldReferencePotential
    system: PreparedAtomisticSystem
    reciprocal_modes: Array
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: EwaldReferencePotential, system: PreparedAtomisticSystem, /):
        cell = system.cell
        if cell is None:
            raise RuntimeError("Validated periodic cell unexpectedly absent.")
        extent = plan.reciprocal_extent
        modes = np.asarray(
            [
                value
                for value in product(range(-extent, extent + 1), repeat=3)
                if value != (0, 0, 0)
            ],
            dtype=float,
        )
        self.plan = plan
        self.system = system
        self.reciprocal_modes = jnp.asarray(modes)
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = _electrostatic_identity(
            "prepared-ewald-reference",
            {"term": plan.term_id, "system": system.prepared_id},
        )

    def _common_energy(
        self,
        context: AtomisticPotentialContext,
        structure_factor: Array,
        reciprocal_weights: Array,
        volume: Array,
        /,
    ) -> Array:
        cell = self.system.cell
        if cell is None:
            raise RuntimeError("Periodic Ewald context has no cell.")
        charges = self.system.plan.charges
        constant = self.system.plan.units.coulomb_constant
        distance = context.pair_distance
        active = context.pair_valid & (distance < self.plan.real_cutoff)
        valid_geometry = jnp.all(~context.pair_valid | (distance > 0.0))
        safe = jnp.where(active & (distance > 0.0), distance, 1.0)
        pair_charge = charges[context.pair_left] * charges[context.pair_right]
        real = constant * pair_charge * jsp.erfc(self.plan.alpha * safe) / safe
        real = jnp.sum(jnp.where(active, real * context.electrostatic_scales, 0.0))
        reciprocal = (
            2.0
            * jnp.pi
            * constant
            / volume
            * jnp.sum(
                reciprocal_weights
                * jnp.real(structure_factor * jnp.conj(structure_factor))
            )
        )
        self_energy = -constant * self.plan.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges**2)
        exception_indices = self.system.topology.exception_indices
        correction = jnp.zeros((), dtype=context.positions.dtype)
        if int(exception_indices.shape[0]):
            displacement = (
                context.positions[exception_indices[:, 0]]
                - context.positions[exception_indices[:, 1]]
            )
            displacement = cell.minimum_image_with_vectors(
                displacement, context.cell_vectors
            )
            exception_distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
            safe_exception = jnp.where(exception_distance > 0.0, exception_distance, 1.0)
            exception_charge = (
                charges[exception_indices[:, 0]] * charges[exception_indices[:, 1]]
            )
            correction = -jnp.sum(
                (1.0 - self.system.topology.electrostatic_scales)
                * constant
                * exception_charge
                * jsp.erf(self.plan.alpha * safe_exception)
                / safe_exception
            )
        total_charge = jnp.sum(jnp.where(self.system.active_mask, charges, 0.0))
        neutral = jnp.abs(total_charge) <= self.plan.charge_tolerance
        background = (
            -jnp.pi * constant * total_charge**2 / (2.0 * self.plan.alpha**2 * volume)
            if self.plan.neutrality == "uniform-background"
            else jnp.zeros((), dtype=context.positions.dtype)
        )
        successful = valid_geometry & (
            neutral if self.plan.neutrality == "require-neutral" else jnp.asarray(True)
        )
        total = real + reciprocal + self_energy + correction + background
        return jnp.where(successful & jnp.isfinite(total), total, jnp.nan)

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        cell = self.system.cell
        if cell is None:
            raise RuntimeError("Periodic Ewald context has no cell.")
        inverse = cell.inverse_for_vectors(context.cell_vectors)
        reciprocal = 2.0 * jnp.pi * inverse.T
        wavevectors = contract("mi,ij->mj", self.reciprocal_modes, reciprocal)
        squared = jnp.sum(wavevectors * wavevectors, axis=-1)
        reciprocal_weights = jnp.exp(-squared / (4.0 * self.plan.alpha**2)) / squared
        phase = contract("kd,nd->kn", wavevectors, context.positions)
        charges = self.system.plan.charges
        structure_factor = contract("n,kn->k", charges, jnp.exp(1.0j * phase))
        volume = jnp.abs(
            jnp.sum(
                context.cell_vectors[0]
                * jnp.cross(context.cell_vectors[1], context.cell_vectors[2])
            )
        )
        energy = self._common_energy(
            context, structure_factor, reciprocal_weights, volume
        )
        successful = jnp.isfinite(energy)
        return AtomisticTermEvaluation(
            energy,
            jnp.zeros((self.system.capacity,), dtype=context.positions.dtype),
            successful,
        )


class ParticleMeshEwaldPotential(AbstractAtomisticEnergyTerm, NonTrainableState):
    alpha: float = eqx.field(static=True)
    real_cutoff: float = eqx.field(static=True)
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    spline_degree: int = eqx.field(static=True)
    neutrality: ChargeNeutralityPolicy = eqx.field(static=True)
    charge_tolerance: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        alpha: float,
        real_cutoff: float,
        grid_shape: tuple[int, int, int],
        /,
        *,
        spline_degree: int = 3,
        neutrality: ChargeNeutralityPolicy = "require-neutral",
        charge_tolerance: float = 1.0e-10,
        name: str = "particle-mesh-ewald",
        force_group: int = 0,
    ):
        alpha_ = float(alpha)
        cutoff = float(real_cutoff)
        shape = tuple(int(value) for value in grid_shape)
        degree = int(spline_degree)
        tolerance = float(charge_tolerance)
        identifier = str(name).strip()
        group = int(force_group)
        if (
            not math.isfinite(alpha_)
            or alpha_ <= 0.0
            or not math.isfinite(cutoff)
            or cutoff <= 0.0
            or len(shape) != 3
            or any(value < 4 for value in shape)
            or degree not in (1, 2, 3)
            or neutrality not in ("require-neutral", "uniform-background")
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
            or not identifier
            or group < 0
        ):
            raise ValueError("Particle-mesh Ewald parameters are invalid.")
        self.alpha = alpha_
        self.real_cutoff = cutoff
        self.grid_shape = shape
        self.spline_degree = degree
        self.neutrality = neutrality
        self.charge_tolerance = tolerance
        self.name = identifier
        self.force_group = group
        self.capabilities = AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=False,
        )
        self.requirements = AtomisticPotentialRequirements(
            cutoff=cutoff, pair_geometry=True, reciprocal_grid=True
        )
        self.term_id = _electrostatic_identity(
            "particle-mesh-ewald-potential",
            {
                "alpha": alpha_,
                "real_cutoff": cutoff,
                "grid_shape": list(shape),
                "spline_degree": degree,
                "neutrality": neutrality,
                "charge_tolerance": tolerance,
                "name": identifier,
                "force_group": group,
            },
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> PreparedParticleMeshEwaldPotential:
        if system.cell is None or not system.cell.fully_periodic:
            raise ValueError("ParticleMeshEwaldPotential requires a fully periodic cell.")
        system.cell.require_unique_image(self.real_cutoff)
        return PreparedParticleMeshEwaldPotential(self, system)


class PreparedParticleMeshEwaldPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: ParticleMeshEwaldPotential
    system: PreparedAtomisticSystem
    splat: PreparedParticleGridSplat
    reciprocal_modes: Array
    inverse_window_squared: Array
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, plan: ParticleMeshEwaldPotential, system: PreparedAtomisticSystem, /
    ):
        cell = system.cell
        if cell is None:
            raise RuntimeError("Validated PME cell unexpectedly absent.")
        grid = TensorGridPlan(
            tuple(UniformCellAxisSpec(size, periodic=True) for size in plan.grid_shape),
            axis_names=("fractional_x", "fractional_y", "fractional_z"),
        ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        splat = ParticleGridSplatPlan(
            grid,
            assignment=TensorBSplineSplatAssignment(plan.spline_degree),
        ).prepare(system.particles)
        mode_axes = tuple(jnp.fft.fftfreq(size) * size for size in plan.grid_shape)
        modes = jnp.stack(jnp.meshgrid(*mode_axes, indexing="ij"), axis=-1)
        window = jnp.ones(plan.grid_shape, dtype=cell.vectors.dtype)
        for axis, size in enumerate(plan.grid_shape):
            mode = mode_axes[axis]
            factor = jnp.sinc(mode / size) ** (plan.spline_degree + 1)
            reshape = [1, 1, 1]
            reshape[axis] = size
            window = window * factor.reshape(tuple(reshape))
        inverse_window_squared = jnp.where(window != 0.0, 1.0 / (window * window), 0.0)
        self.plan = plan
        self.system = system
        self.splat = splat
        self.reciprocal_modes = modes
        self.inverse_window_squared = inverse_window_squared
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = _electrostatic_identity(
            "prepared-particle-mesh-ewald",
            {
                "term": plan.term_id,
                "system": system.prepared_id,
                "splat": splat.prepared_id,
            },
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        cell = self.system.cell
        if cell is None:
            raise RuntimeError("PME context has no periodic cell.")
        fractional = cell.fractional_with_vectors(context.positions, context.cell_vectors)
        fractional = fractional - jnp.floor(fractional)
        routes = self.splat.build(fractional)
        deposited = self.splat.deposit_content(routes, self.system.plan.charges)
        spectrum = jnp.fft.fftn(deposited.content)
        constant = self.system.plan.units.coulomb_constant
        inverse = cell.inverse_for_vectors(context.cell_vectors)
        reciprocal_vectors = 2.0 * jnp.pi * inverse.T
        wavevectors = contract("...i,ij->...j", self.reciprocal_modes, reciprocal_vectors)
        squared = jnp.sum(wavevectors * wavevectors, axis=-1)
        reciprocal_weights = jnp.where(
            squared > 0.0,
            jnp.exp(-squared / (4.0 * self.plan.alpha**2))
            / squared
            * self.inverse_window_squared,
            0.0,
        )
        volume = jnp.abs(
            jnp.sum(
                context.cell_vectors[0]
                * jnp.cross(context.cell_vectors[1], context.cell_vectors[2])
            )
        )
        reciprocal = (
            2.0
            * jnp.pi
            * constant
            / volume
            * jnp.sum(reciprocal_weights * jnp.real(spectrum * jnp.conj(spectrum)))
        )
        distance = context.pair_distance
        active = context.pair_valid & (distance < self.plan.real_cutoff)
        safe = jnp.where(active & (distance > 0.0), distance, 1.0)
        charges = self.system.plan.charges
        pair_charge = charges[context.pair_left] * charges[context.pair_right]
        real = jnp.sum(
            jnp.where(
                active,
                constant
                * pair_charge
                * jsp.erfc(self.plan.alpha * safe)
                / safe
                * context.electrostatic_scales,
                0.0,
            )
        )
        self_energy = -constant * self.plan.alpha / jnp.sqrt(jnp.pi) * jnp.sum(charges**2)
        exception_indices = self.system.topology.exception_indices
        correction = jnp.zeros((), dtype=context.positions.dtype)
        if int(exception_indices.shape[0]):
            displacement = (
                context.positions[exception_indices[:, 0]]
                - context.positions[exception_indices[:, 1]]
            )
            displacement = cell.minimum_image_with_vectors(
                displacement, context.cell_vectors
            )
            exception_distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
            safe_exception = jnp.where(exception_distance > 0.0, exception_distance, 1.0)
            exception_charge = (
                charges[exception_indices[:, 0]] * charges[exception_indices[:, 1]]
            )
            correction = -jnp.sum(
                (1.0 - self.system.topology.electrostatic_scales)
                * constant
                * exception_charge
                * jsp.erf(self.plan.alpha * safe_exception)
                / safe_exception
            )
        total_charge = jnp.sum(jnp.where(self.system.active_mask, charges, 0.0))
        neutral = jnp.abs(total_charge) <= self.plan.charge_tolerance
        background = (
            -jnp.pi * constant * total_charge**2 / (2.0 * self.plan.alpha**2 * volume)
            if self.plan.neutrality == "uniform-background"
            else jnp.zeros((), dtype=context.positions.dtype)
        )
        successful = (
            routes.successful
            & deposited.successful
            & jnp.all(~context.pair_valid | (distance > 0.0))
            & (neutral if self.plan.neutrality == "require-neutral" else True)
        )
        energy = real + reciprocal + self_energy + correction + background
        successful = successful & jnp.isfinite(energy)
        return AtomisticTermEvaluation(
            jnp.where(successful, energy, jnp.nan),
            jnp.zeros((self.system.capacity,), dtype=context.positions.dtype),
            successful,
        )


__all__ = [
    "DirectCoulombPotential",
    "EwaldReferencePotential",
    "ParticleMeshEwaldPotential",
]
