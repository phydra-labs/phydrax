#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._lattice import LatticeBoltzmannVelocitySet
from ._precision import LatticeBoltzmannPrecisionPolicy


class ThermalBoundaryKind(StrEnum):
    ADIABATIC = "adiabatic"
    TEMPERATURE = "temperature"
    HEAT_FLUX = "heat_flux"


class ThermalBoundaryCondition(StrictModule, NonTrainableState):
    """One outward-normal thermal boundary on lattice nodes.

    ``value`` is temperature for ``TEMPERATURE`` and outward sensible-energy
    flux for ``HEAT_FLUX``. It is ignored for ``ADIABATIC``.
    """

    kind: ThermalBoundaryKind = eqx.field(static=True)
    node_mask: Array
    outward_normal: Array
    value: Array
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ThermalBoundaryKind,
        node_mask: ArrayLike,
        outward_normal: ArrayLike,
        /,
        *,
        value: ArrayLike = 0.0,
        boundary_id: str | None = None,
    ):
        if not isinstance(kind, ThermalBoundaryKind):
            raise TypeError("kind must be a ThermalBoundaryKind.")
        mask = np.asarray(node_mask)
        normal = np.asarray(outward_normal, dtype=float)
        prescribed = np.asarray(value, dtype=float)
        if (
            mask.ndim == 0
            or normal.ndim != mask.ndim + 1
            or normal.shape[:-1] != mask.shape
        ):
            raise ValueError("Boundary mask and outward-normal shapes are invalid.")
        if normal.shape[-1] == 0 or np.any(~np.isfinite(normal)):
            raise ValueError("Boundary normals must be finite nonempty vectors.")
        selected_norm = np.linalg.norm(normal[mask.astype(bool)], axis=-1)
        if selected_norm.size and np.any(selected_norm <= 0.0):
            raise ValueError("Every selected boundary node must have a nonzero normal.")
        if prescribed.shape not in ((), mask.shape) or np.any(~np.isfinite(prescribed)):
            raise ValueError("Boundary values must be scalar or match the node mask.")
        generated = canonical_fingerprint(
            {
                "kind": "thermal-lattice-boundary",
                "boundary_kind": kind.value,
                "mask": array_tree_fingerprint(mask.astype(bool)),
                "normal": array_tree_fingerprint(normal),
                "value": array_tree_fingerprint(prescribed),
            }
        )
        self.kind = kind
        self.node_mask = jnp.asarray(mask, dtype=bool)
        self.outward_normal = jnp.asarray(normal)
        self.value = jnp.asarray(prescribed)
        self.boundary_id = generated if boundary_id is None else str(boundary_id)
        if not self.boundary_id:
            raise ValueError("boundary_id must be nonempty.")


class ThermalLatticeBoltzmannPlan(StrictModule, NonTrainableState):
    """Constant-property advection-diffusion plan for sensible energy.

    This is a passive sensible-energy distribution. It does not represent the
    total-energy equation of a compressible flow.
    """

    volumetric_heat_capacity: Array
    thermal_conductivity: Array
    reference_temperature: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        volumetric_heat_capacity: ArrayLike,
        thermal_conductivity: ArrayLike,
        /,
        *,
        reference_temperature: float = 0.0,
        plan_id: str | None = None,
    ):
        capacity = np.asarray(volumetric_heat_capacity, dtype=float)
        conductivity = np.asarray(thermal_conductivity, dtype=float)
        reference = float(reference_temperature)
        if capacity.shape != () or not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("volumetric_heat_capacity must be finite and positive.")
        if (
            conductivity.shape != ()
            or not np.isfinite(conductivity)
            or conductivity <= 0.0
        ):
            raise ValueError("thermal_conductivity must be finite and positive.")
        if not np.isfinite(reference):
            raise ValueError("reference_temperature must be finite.")
        generated = canonical_fingerprint(
            {
                "kind": "thermal-lattice-boltzmann-plan",
                "volumetric_heat_capacity": float(capacity),
                "thermal_conductivity": float(conductivity),
                "reference_temperature": reference,
            }
        )
        self.volumetric_heat_capacity = jnp.asarray(capacity)
        self.thermal_conductivity = jnp.asarray(conductivity)
        self.reference_temperature = reference
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def thermal_diffusivity(self) -> Array:
        return self.thermal_conductivity / self.volumetric_heat_capacity


class BoussinesqCouplingPlan(StrictModule, NonTrainableState):
    reference_density: Array
    thermal_expansion: Array
    reference_temperature: float = eqx.field(static=True)
    gravity: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_density: ArrayLike,
        thermal_expansion: ArrayLike,
        gravity: ArrayLike,
        /,
        *,
        reference_temperature: float,
        plan_id: str | None = None,
    ):
        density = np.asarray(reference_density, dtype=float)
        expansion = np.asarray(thermal_expansion, dtype=float)
        acceleration = np.asarray(gravity, dtype=float)
        reference = float(reference_temperature)
        if density.shape != () or not np.isfinite(density) or density <= 0.0:
            raise ValueError("reference_density must be finite and positive.")
        if expansion.shape != () or not np.isfinite(expansion) or expansion < 0.0:
            raise ValueError("thermal_expansion must be finite and nonnegative.")
        if (
            acceleration.ndim != 1
            or acceleration.size == 0
            or np.any(~np.isfinite(acceleration))
        ):
            raise ValueError("gravity must be a finite nonempty vector.")
        if not np.isfinite(reference):
            raise ValueError("reference_temperature must be finite.")
        generated = canonical_fingerprint(
            {
                "kind": "boussinesq-coupling-plan",
                "reference_density": float(density),
                "thermal_expansion": float(expansion),
                "reference_temperature": reference,
                "gravity": array_tree_fingerprint(acceleration),
            }
        )
        self.reference_density = jnp.asarray(density)
        self.thermal_expansion = jnp.asarray(expansion)
        self.reference_temperature = reference
        self.gravity = jnp.asarray(acceleration)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")


class ThermalEnergyLedger(StrictModule):
    initial_sensible_energy: Array
    boundary_energy: Array
    source_energy: Array
    reaction_energy: Array
    energy_residual: Array


class ThermalLatticeBoltzmannState(StrictModule):
    populations: Array
    ledger: ThermalEnergyLedger
    successful: Array
    step_index: Array
    state_id: str = eqx.field(static=True)


class ThermalCollisionResult(StrictModule):
    populations: Array
    sensible_energy: Array
    source: Array
    relaxation_rate: Array
    successful: Array


def sensible_energy_from_temperature(
    temperature: ArrayLike, plan: ThermalLatticeBoltzmannPlan, /
) -> Array:
    if not isinstance(plan, ThermalLatticeBoltzmannPlan):
        raise TypeError("plan must be a ThermalLatticeBoltzmannPlan.")
    value = jnp.asarray(temperature)
    return plan.volumetric_heat_capacity.astype(value.dtype) * (
        value - jnp.asarray(plan.reference_temperature, dtype=value.dtype)
    )


def temperature_from_sensible_energy(
    sensible_energy: ArrayLike, plan: ThermalLatticeBoltzmannPlan, /
) -> Array:
    if not isinstance(plan, ThermalLatticeBoltzmannPlan):
        raise TypeError("plan must be a ThermalLatticeBoltzmannPlan.")
    value = jnp.asarray(sensible_energy)
    return jnp.asarray(plan.reference_temperature, dtype=value.dtype) + value / (
        plan.volumetric_heat_capacity.astype(value.dtype)
    )


def thermal_equilibrium(
    sensible_energy: ArrayLike,
    velocity: ArrayLike,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    """Return the first-order scalar equilibrium with exact lattice moments."""
    _validate_lattice_precision(lattice, precision)
    energy = precision.compute(jnp.asarray(sensible_energy))
    flow = precision.compute(jnp.asarray(velocity))
    if flow.shape != energy.shape + (lattice.dimension,):
        raise ValueError(
            "velocity must extend sensible-energy shape by lattice dimension."
        )
    velocities = precision.compute(lattice.velocities)
    weights = precision.compute(lattice.weights)
    projection = contract("...d,qd->...q", flow, velocities)
    equilibrium = (
        energy[..., None]
        * weights
        * (1.0 + projection / precision.compute(lattice.sound_speed_squared))
    )
    return precision.population(equilibrium)


def thermal_source_distribution(
    sensible_energy_source: ArrayLike,
    velocity: ArrayLike,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    """Lift a scalar source to populations, preserving zeroth and first moments."""
    return thermal_equilibrium(sensible_energy_source, velocity, lattice, precision)


def thermal_raw_moments(
    populations: ArrayLike,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> tuple[Array, Array]:
    _validate_lattice_precision(lattice, precision)
    value = precision.accumulation(jnp.asarray(populations))
    if value.ndim == 0 or value.shape[-1] != lattice.population_count:
        raise ValueError("populations must have a trailing lattice-direction axis.")
    energy = jnp.sum(value, axis=-1)
    flux = contract(
        "...q,qd->...d",
        value,
        precision.accumulation(lattice.velocities),
    )
    return energy, flux


def thermal_relaxation_rate(
    plan: ThermalLatticeBoltzmannPlan,
    lattice: LatticeBoltzmannVelocitySet,
    step_size: ArrayLike,
    spacing: ArrayLike,
    /,
) -> Array:
    if not isinstance(plan, ThermalLatticeBoltzmannPlan):
        raise TypeError("plan must be a ThermalLatticeBoltzmannPlan.")
    if not isinstance(lattice, LatticeBoltzmannVelocitySet):
        raise TypeError("lattice must be a LatticeBoltzmannVelocitySet.")
    dt = jnp.asarray(step_size)
    dx = jnp.asarray(spacing, dtype=dt.dtype)
    lattice_diffusivity = plan.thermal_diffusivity.astype(dt.dtype) * dt / (dx * dx)
    return 1.0 / (0.5 + lattice_diffusivity / lattice.sound_speed_squared)


def collide_thermal(
    populations: ArrayLike,
    velocity: ArrayLike,
    sensible_energy_source: ArrayLike,
    plan: ThermalLatticeBoltzmannPlan,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    step_size: ArrayLike,
    spacing: ArrayLike,
    /,
) -> ThermalCollisionResult:
    """Collide one thermal field and inject the source with exact zeroth moment."""
    value = precision.population(jnp.asarray(populations))
    source = precision.compute(jnp.asarray(sensible_energy_source))
    energy, _ = thermal_raw_moments(value, lattice, precision)
    if source.shape != energy.shape:
        raise ValueError("sensible_energy_source must match the population field.")
    equilibrium = precision.compute(
        thermal_equilibrium(energy, velocity, lattice, precision)
    )
    lifted_source = precision.compute(
        thermal_source_distribution(source, velocity, lattice, precision)
    )
    rate = precision.compute(thermal_relaxation_rate(plan, lattice, step_size, spacing))
    dt = precision.compute(jnp.asarray(step_size))
    computed = precision.compute(value)
    postcollision = computed - rate * (computed - equilibrium) + dt * lifted_source
    successful = (
        jnp.all(jnp.isfinite(postcollision))
        & jnp.all(jnp.isfinite(energy))
        & jnp.all(jnp.isfinite(source))
        & jnp.all(jnp.isfinite(rate) & (rate > 0.0) & (rate < 2.0))
    )
    return ThermalCollisionResult(
        precision.population(postcollision),
        energy,
        source,
        rate,
        successful,
    )


def apply_thermal_boundary(
    populations: ArrayLike,
    boundary: ThermalBoundaryCondition,
    plan: ThermalLatticeBoltzmannPlan,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    """Apply linkwise halfway bounce-back, anti-bounce-back, or normal flux."""
    if not isinstance(boundary, ThermalBoundaryCondition):
        raise TypeError("boundary must be a ThermalBoundaryCondition.")
    if not isinstance(plan, ThermalLatticeBoltzmannPlan):
        raise TypeError("plan must be a ThermalLatticeBoltzmannPlan.")
    _validate_lattice_precision(lattice, precision)
    value = precision.compute(jnp.asarray(populations))
    if (
        value.shape[:-1] != boundary.node_mask.shape
        or value.shape[-1] != lattice.population_count
    ):
        raise ValueError("Boundary mask must match the population node shape.")
    if boundary.outward_normal.shape != value.shape[:-1] + (lattice.dimension,):
        raise ValueError("Boundary normals do not match lattice dimension.")
    velocities = precision.compute(lattice.velocities)
    weights = precision.compute(lattice.weights)
    normal_projection = contract(
        "...d,qd->...q",
        precision.compute(boundary.outward_normal),
        velocities,
    )
    incoming = boundary.node_mask[..., None] & (normal_projection < 0.0)
    opposite = value[..., lattice.opposite]
    prescribed = opposite
    boundary_value = precision.compute(
        jnp.broadcast_to(boundary.value, boundary.node_mask.shape)
    )
    if boundary.kind is ThermalBoundaryKind.TEMPERATURE:
        wall_energy = sensible_energy_from_temperature(boundary_value, plan)
        prescribed = -opposite + 2.0 * weights * wall_energy[..., None]
    elif boundary.kind is ThermalBoundaryKind.HEAT_FLUX:
        denominator = jnp.sum(
            jnp.where(incoming, weights * normal_projection * normal_projection, 0.0),
            axis=-1,
            keepdims=True,
        )
        correction = jnp.where(
            denominator > 0.0,
            boundary_value[..., None] * weights * normal_projection / denominator,
            0.0,
        )
        prescribed = opposite + correction
    return precision.population(jnp.where(incoming, prescribed, value))


def boussinesq_force(temperature: ArrayLike, plan: BoussinesqCouplingPlan, /) -> Array:
    """Return ``rho_ref beta (T-T_ref) g`` for injection into momentum forcing."""
    if not isinstance(plan, BoussinesqCouplingPlan):
        raise TypeError("plan must be a BoussinesqCouplingPlan.")
    value = jnp.asarray(temperature)
    buoyancy = (
        plan.reference_density.astype(value.dtype)
        * plan.thermal_expansion.astype(value.dtype)
        * (value - jnp.asarray(plan.reference_temperature, dtype=value.dtype))
    )
    return buoyancy[..., None] * plan.gravity.astype(value.dtype)


def initialize_thermal_ledger(
    sensible_energy: ArrayLike, cell_measure: ArrayLike, /
) -> ThermalEnergyLedger:
    energy = jnp.asarray(sensible_energy)
    measure = jnp.asarray(cell_measure, dtype=energy.dtype)
    if measure.shape not in ((), energy.shape):
        raise ValueError("cell_measure must be scalar or match sensible energy.")
    total = jnp.sum(energy * measure)
    zero = jnp.zeros_like(total)
    return ThermalEnergyLedger(total, zero, zero, zero, zero)


def _validate_lattice_precision(lattice, precision):
    if not isinstance(lattice, LatticeBoltzmannVelocitySet):
        raise TypeError("lattice must be a LatticeBoltzmannVelocitySet.")
    if not isinstance(precision, LatticeBoltzmannPrecisionPolicy):
        raise TypeError("precision must be a LatticeBoltzmannPrecisionPolicy.")


__all__ = [
    "BoussinesqCouplingPlan",
    "ThermalBoundaryCondition",
    "ThermalBoundaryKind",
    "ThermalCollisionResult",
    "ThermalEnergyLedger",
    "ThermalLatticeBoltzmannPlan",
    "ThermalLatticeBoltzmannState",
    "apply_thermal_boundary",
    "boussinesq_force",
    "collide_thermal",
    "initialize_thermal_ledger",
    "sensible_energy_from_temperature",
    "temperature_from_sensible_energy",
    "thermal_equilibrium",
    "thermal_raw_moments",
    "thermal_relaxation_rate",
    "thermal_source_distribution",
]
