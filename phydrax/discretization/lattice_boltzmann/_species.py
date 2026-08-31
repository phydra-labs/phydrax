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


class SpeciesBoundaryKind(StrEnum):
    NO_FLUX = "no_flux"
    CONCENTRATION = "concentration"
    MOLAR_FLUX = "molar_flux"


class SpeciesBoundaryCondition(StrictModule, NonTrainableState):
    """One outward-normal boundary for all passive species.

    ``value`` has a trailing species axis. It is concentration for
    ``CONCENTRATION`` and outward molar flux for ``MOLAR_FLUX``.
    """

    kind: SpeciesBoundaryKind = eqx.field(static=True)
    node_mask: Array
    outward_normal: Array
    value: Array
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SpeciesBoundaryKind,
        node_mask: ArrayLike,
        outward_normal: ArrayLike,
        /,
        *,
        value: ArrayLike,
        boundary_id: str | None = None,
    ):
        if not isinstance(kind, SpeciesBoundaryKind):
            raise TypeError("kind must be a SpeciesBoundaryKind.")
        mask = np.asarray(node_mask)
        normal = np.asarray(outward_normal, dtype=float)
        prescribed = np.asarray(value, dtype=float)
        if (
            mask.ndim == 0
            or normal.ndim != mask.ndim + 1
            or normal.shape[:-1] != mask.shape
            or normal.shape[-1] == 0
        ):
            raise ValueError("Boundary mask and outward-normal shapes are invalid.")
        if np.any(~np.isfinite(normal)):
            raise ValueError("Boundary normals must be finite.")
        selected_norm = np.linalg.norm(normal[mask.astype(bool)], axis=-1)
        if selected_norm.size and np.any(selected_norm <= 0.0):
            raise ValueError("Every selected boundary node must have a nonzero normal.")
        if prescribed.ndim == 0 or np.any(~np.isfinite(prescribed)):
            raise ValueError("Boundary value must have a finite trailing species axis.")
        if prescribed.ndim > 1 and prescribed.shape[:-1] != mask.shape:
            raise ValueError("A spatial boundary value must match the node mask.")
        generated = canonical_fingerprint(
            {
                "kind": "species-lattice-boundary",
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


class SpeciesLatticeBoltzmannPlan(StrictModule, NonTrainableState):
    """Independent passive-scalar Fickian distributions in trailing-Q layout."""

    diffusivity: Array
    species_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, diffusivity: ArrayLike, /, *, plan_id: str | None = None):
        coefficients = np.asarray(diffusivity, dtype=float)
        if (
            coefficients.ndim != 1
            or coefficients.size == 0
            or np.any(~np.isfinite(coefficients))
            or np.any(coefficients <= 0.0)
        ):
            raise ValueError("diffusivity must be a finite positive species vector.")
        generated = canonical_fingerprint(
            {
                "kind": "species-lattice-boltzmann-plan",
                "diffusivity": array_tree_fingerprint(coefficients),
            }
        )
        self.diffusivity = jnp.asarray(coefficients)
        self.species_count = coefficients.size
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")


class SpeciesLedger(StrictModule):
    initial_species_amount: Array
    boundary_species_amount: Array
    source_species_amount: Array
    reaction_species_amount: Array
    species_residual: Array
    initial_element_amount: Array
    boundary_element_amount: Array
    source_element_amount: Array
    element_residual: Array


class SpeciesLatticeBoltzmannState(StrictModule):
    populations: Array
    ledger: SpeciesLedger
    successful: Array
    step_index: Array
    state_id: str = eqx.field(static=True)


class SpeciesCollisionResult(StrictModule):
    populations: Array
    concentration: Array
    source: Array
    relaxation_rate: Array
    successful: Array


def species_equilibrium(
    concentration: ArrayLike,
    velocity: ArrayLike,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    """Return ``(..., species, Q)`` equilibria with exact scalar moments."""
    _validate_lattice_precision(lattice, precision)
    amount = precision.compute(jnp.asarray(concentration))
    flow = precision.compute(jnp.asarray(velocity))
    if amount.ndim == 0:
        raise ValueError("concentration must have a trailing species axis.")
    if flow.shape != amount.shape[:-1] + (lattice.dimension,):
        raise ValueError("velocity must match concentration nodes and lattice dimension.")
    projection = contract("...d,qd->...q", flow, precision.compute(lattice.velocities))
    scalar_equilibrium = precision.compute(lattice.weights) * (
        1.0 + projection / precision.compute(lattice.sound_speed_squared)
    )
    return precision.population(amount[..., :, None] * scalar_equilibrium[..., None, :])


def species_source_distribution(
    concentration_source: ArrayLike,
    velocity: ArrayLike,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    """Lift species sources while preserving each zeroth and first moment."""
    return species_equilibrium(concentration_source, velocity, lattice, precision)


def species_raw_moments(
    populations: ArrayLike,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> tuple[Array, Array]:
    _validate_lattice_precision(lattice, precision)
    value = precision.accumulation(jnp.asarray(populations))
    if value.ndim < 2 or value.shape[-1] != lattice.population_count:
        raise ValueError("populations must have trailing species and direction axes.")
    concentration = jnp.sum(value, axis=-1)
    flux = contract(
        "...sq,qd->...sd",
        value,
        precision.accumulation(lattice.velocities),
    )
    return concentration, flux


def species_relaxation_rate(
    plan: SpeciesLatticeBoltzmannPlan,
    lattice: LatticeBoltzmannVelocitySet,
    step_size: ArrayLike,
    spacing: ArrayLike,
    /,
) -> Array:
    if not isinstance(plan, SpeciesLatticeBoltzmannPlan):
        raise TypeError("plan must be a SpeciesLatticeBoltzmannPlan.")
    if not isinstance(lattice, LatticeBoltzmannVelocitySet):
        raise TypeError("lattice must be a LatticeBoltzmannVelocitySet.")
    dt = jnp.asarray(step_size)
    dx = jnp.asarray(spacing, dtype=dt.dtype)
    lattice_diffusivity = plan.diffusivity.astype(dt.dtype) * dt / (dx * dx)
    return 1.0 / (0.5 + lattice_diffusivity / lattice.sound_speed_squared)


def collide_species(
    populations: ArrayLike,
    velocity: ArrayLike,
    concentration_source: ArrayLike,
    plan: SpeciesLatticeBoltzmannPlan,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    step_size: ArrayLike,
    spacing: ArrayLike,
    /,
) -> SpeciesCollisionResult:
    """Collide all Fickian scalars and add sources with exact species moments."""
    value = precision.population(jnp.asarray(populations))
    concentration, _ = species_raw_moments(value, lattice, precision)
    source = precision.compute(jnp.asarray(concentration_source))
    if concentration.shape[-1] != plan.species_count:
        raise ValueError("Population species axis does not match the species plan.")
    if source.shape != concentration.shape:
        raise ValueError("concentration_source must match macroscopic concentration.")
    equilibrium = precision.compute(
        species_equilibrium(concentration, velocity, lattice, precision)
    )
    lifted_source = precision.compute(
        species_source_distribution(source, velocity, lattice, precision)
    )
    rate = precision.compute(species_relaxation_rate(plan, lattice, step_size, spacing))
    dt = precision.compute(jnp.asarray(step_size))
    computed = precision.compute(value)
    postcollision = (
        computed - rate[..., None] * (computed - equilibrium) + dt * lifted_source
    )
    successful = (
        jnp.all(jnp.isfinite(postcollision))
        & jnp.all(jnp.isfinite(concentration) & (concentration >= 0.0))
        & jnp.all(jnp.isfinite(source))
        & jnp.all(jnp.isfinite(rate) & (rate > 0.0) & (rate < 2.0))
    )
    return SpeciesCollisionResult(
        precision.population(postcollision), concentration, source, rate, successful
    )


def apply_species_boundary(
    populations: ArrayLike,
    boundary: SpeciesBoundaryCondition,
    plan: SpeciesLatticeBoltzmannPlan,
    lattice: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    """Apply no-flux bounce-back, concentration, or outward molar flux data."""
    if not isinstance(boundary, SpeciesBoundaryCondition):
        raise TypeError("boundary must be a SpeciesBoundaryCondition.")
    if not isinstance(plan, SpeciesLatticeBoltzmannPlan):
        raise TypeError("plan must be a SpeciesLatticeBoltzmannPlan.")
    _validate_lattice_precision(lattice, precision)
    value = precision.compute(jnp.asarray(populations))
    if (
        value.ndim < 2
        or value.shape[:-2] != boundary.node_mask.shape
        or value.shape[-2] != plan.species_count
        or value.shape[-1] != lattice.population_count
    ):
        raise ValueError("Boundary mask/species count must match population shape.")
    if boundary.outward_normal.shape != value.shape[:-2] + (lattice.dimension,):
        raise ValueError("Boundary normals do not match lattice dimension.")
    if boundary.value.shape not in (
        (plan.species_count,),
        boundary.node_mask.shape + (plan.species_count,),
    ):
        raise ValueError("Boundary value must end in the configured species axis.")
    velocities = precision.compute(lattice.velocities)
    weights = precision.compute(lattice.weights)
    normal_projection = contract(
        "...d,qd->...q",
        precision.compute(boundary.outward_normal),
        velocities,
    )
    incoming = boundary.node_mask[..., None] & (normal_projection < 0.0)
    incoming_species = incoming[..., None, :]
    opposite = value[..., :, lattice.opposite]
    prescribed = opposite
    boundary_value = precision.compute(
        jnp.broadcast_to(boundary.value, boundary.node_mask.shape + (plan.species_count,))
    )
    if boundary.kind is SpeciesBoundaryKind.CONCENTRATION:
        prescribed = (
            -opposite + 2.0 * boundary_value[..., :, None] * weights[..., None, :]
        )
    elif boundary.kind is SpeciesBoundaryKind.MOLAR_FLUX:
        denominator = jnp.sum(
            jnp.where(
                incoming,
                weights * normal_projection * normal_projection,
                0.0,
            ),
            axis=-1,
            keepdims=True,
        )
        correction = jnp.where(
            denominator[..., None, :] > 0.0,
            boundary_value[..., :, None]
            * weights[..., None, :]
            * normal_projection[..., None, :]
            / denominator[..., None, :],
            0.0,
        )
        prescribed = opposite + correction
    return precision.population(jnp.where(incoming_species, prescribed, value))


def initialize_species_ledger(
    concentration: ArrayLike,
    cell_measure: ArrayLike,
    element_composition: ArrayLike,
    /,
) -> SpeciesLedger:
    amount = jnp.asarray(concentration)
    measure = jnp.asarray(cell_measure, dtype=amount.dtype)
    composition = jnp.asarray(element_composition, dtype=amount.dtype)
    if amount.ndim == 0:
        raise ValueError("concentration must have a trailing species axis.")
    if measure.shape not in ((), amount.shape[:-1]):
        raise ValueError("cell_measure must be scalar or match concentration nodes.")
    if composition.ndim != 2 or composition.shape[1] != amount.shape[-1]:
        raise ValueError("element_composition must have shape (element, species).")
    spatial_axes = tuple(range(amount.ndim - 1))
    species_total = jnp.sum(amount * measure[..., None], axis=spatial_axes)
    element_total = contract("es,s->e", composition, species_total)
    species_zero = jnp.zeros_like(species_total)
    element_zero = jnp.zeros_like(element_total)
    return SpeciesLedger(
        species_total,
        species_zero,
        species_zero,
        species_zero,
        species_zero,
        element_total,
        element_zero,
        element_zero,
        element_zero,
    )


def species_element_amount(
    species_amount: ArrayLike, element_composition: ArrayLike, /
) -> Array:
    amount = jnp.asarray(species_amount)
    composition = jnp.asarray(element_composition, dtype=amount.dtype)
    if composition.ndim != 2 or amount.shape[-1] != composition.shape[1]:
        raise ValueError("Species and element-composition axes are inconsistent.")
    return contract("es,...s->...e", composition, amount)


def _validate_lattice_precision(lattice, precision):
    if not isinstance(lattice, LatticeBoltzmannVelocitySet):
        raise TypeError("lattice must be a LatticeBoltzmannVelocitySet.")
    if not isinstance(precision, LatticeBoltzmannPrecisionPolicy):
        raise TypeError("precision must be a LatticeBoltzmannPrecisionPolicy.")


__all__ = [
    "SpeciesBoundaryCondition",
    "SpeciesBoundaryKind",
    "SpeciesCollisionResult",
    "SpeciesLatticeBoltzmannPlan",
    "SpeciesLatticeBoltzmannState",
    "SpeciesLedger",
    "apply_species_boundary",
    "collide_species",
    "initialize_species_ledger",
    "species_element_amount",
    "species_equilibrium",
    "species_raw_moments",
    "species_relaxation_rate",
    "species_source_distribution",
]
