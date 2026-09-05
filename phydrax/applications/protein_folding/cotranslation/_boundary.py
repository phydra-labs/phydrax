#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit reduced ribosome geometry, not an atomistic ribosome model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....atomistic._potential import (
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
)
from ....atomistic._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
)
from ....atomistic._system import PreparedAtomisticSystem


class RibosomeBoundaryPotential(AbstractAtomisticEnergyTerm):
    """Harmonic nascent-end tether plus caller-declared soft excluded spheres.

    All lengths/energies are numeric values in the bound atomistic unit system.
    Sphere centers are fixed external supports; their force reactions are minus
    the total boundary force. A tether is optional for an explicitly released
    stage. Different stages must be separated by a work-accounted epoch switch.
    """

    anchor: Array
    centers: Array
    radii: Array
    tether_particle_id: int | None = eqx.field(static=True)
    tether_stiffness: float = eqx.field(static=True)
    exclusion_stiffness: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        *,
        tether_particle_id: int | None,
        anchor: ArrayLike,
        tether_stiffness: float,
        sphere_centers: ArrayLike = (),
        sphere_radii: ArrayLike = (),
        exclusion_stiffness: float = 0.0,
    ):
        anchor_ = np.asarray(anchor, dtype=float)
        centers = np.asarray(sphere_centers, dtype=float).reshape((-1, 3))
        radii = np.asarray(sphere_radii, dtype=float)
        if anchor_.shape != (3,) or radii.shape != (len(centers),):
            raise ValueError(
                "Anchor and excluded sphere arrays have incompatible shapes."
            )
        if any(np.any(~np.isfinite(x)) for x in (anchor_, centers, radii)) or np.any(
            radii <= 0
        ):
            raise ValueError(
                "Boundary geometry must be finite with positive sphere radii."
            )
        if (
            not np.isfinite(tether_stiffness)
            or tether_stiffness < 0
            or not np.isfinite(exclusion_stiffness)
            or exclusion_stiffness < 0
        ):
            raise ValueError("Boundary stiffnesses must be finite and nonnegative.")
        if tether_particle_id is None and tether_stiffness != 0:
            raise ValueError("A released tether must have zero stiffness.")
        self.anchor, self.centers, self.radii = map(
            jnp.asarray, (anchor_, centers, radii)
        )
        self.tether_particle_id = tether_particle_id
        self.tether_stiffness = float(tether_stiffness)
        self.exclusion_stiffness = float(exclusion_stiffness)
        self.name, self.force_group = "ribosome-boundary", 0
        self.capabilities = AtomisticPotentialCapabilities()
        self.requirements = AtomisticPotentialRequirements()
        self.term_id = canonical_fingerprint(
            {
                "kind": self.name,
                "tether": tether_particle_id,
                "stiffness": [tether_stiffness, exclusion_stiffness],
                "geometry": array_tree_fingerprint((anchor_, centers, radii)),
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedRibosomeBoundaryPotential":
        if system.cell is not None:
            raise ValueError(
                "The fixed ribosome boundary requires nonperiodic coordinates."
            )
        ids = np.asarray(system.plan.particle_ids)
        active = np.asarray(system.active_mask)
        matches = (
            np.flatnonzero((ids == self.tether_particle_id) & active)
            if self.tether_particle_id is not None
            else ()
        )
        if self.tether_particle_id is not None and len(matches) != 1:
            raise ValueError(
                "The tether must bind exactly one active stable particle ID."
            )
        return PreparedRibosomeBoundaryPotential(
            self, system, -1 if not len(matches) else int(matches[0])
        )


class PreparedRibosomeBoundaryPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: RibosomeBoundaryPotential
    system: PreparedAtomisticSystem
    active_slots: Array
    tether_slot: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan, system, tether_slot, /):
        self.plan, self.system, self.tether_slot = plan, system, tether_slot
        self.active_slots = jnp.asarray(
            np.flatnonzero(np.asarray(system.active_mask)), dtype=jnp.int32
        )
        self.name, self.force_group, self.term_id = (
            plan.name,
            plan.force_group,
            plan.term_id,
        )
        self.capabilities, self.requirements = plan.capabilities, plan.requirements
        self.prepared_id = canonical_fingerprint(
            {"term": plan.term_id, "system": system.prepared_id}
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        p = self.plan
        atom_energy = jnp.zeros((self.system.capacity,), dtype=context.positions.dtype)
        successful = jnp.asarray(True)
        if p.centers.shape[0] and p.exclusion_stiffness > 0:
            displacement = (
                context.positions[self.active_slots, None, :] - p.centers[None, :, :]
            )
            squared = jnp.sum(displacement**2, axis=-1)
            distance = jnp.sqrt(jnp.where(squared > 0, squared, 1.0))
            overlap = jnp.maximum(p.radii[None, :] - distance, 0.0)
            energy = 0.5 * p.exclusion_stiffness * jnp.sum(overlap**2, axis=-1)
            atom_energy = atom_energy.at[self.active_slots].add(energy)
            successful = jnp.all(squared > 0)
        if self.tether_slot >= 0:
            displacement = context.positions[self.tether_slot] - p.anchor
            atom_energy = atom_energy.at[self.tether_slot].add(
                0.5 * p.tether_stiffness * jnp.sum(displacement**2)
            )
        energy = jnp.where(successful, jnp.sum(atom_energy), jnp.nan)
        return AtomisticTermEvaluation(energy, atom_energy, successful)


__all__ = ["RibosomeBoundaryPotential", "PreparedRibosomeBoundaryPotential"]
