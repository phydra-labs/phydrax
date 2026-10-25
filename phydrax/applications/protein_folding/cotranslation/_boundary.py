#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit reduced ribosome geometry, not an atomistic ribosome model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from ...._fingerprint import array_tree_fingerprint
from ...._identity import IdentityWords
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

    PyTree layout: whether the stage is tethered (``tether_particle_id`` is
    ``None`` for a released stage) and the excluded-sphere count are the
    declared structure. The tethered stable particle ID (int64 scalar), both
    stiffnesses, the anchor and the sphere geometry are dynamic leaves, and the
    exact canonical term digest is the dynamic ``term_identity`` (``term_id``
    decodes it on the host), so stage revisions of one structure share one
    executable.
    """

    anchor: Array
    centers: Array
    radii: Array
    tether_particle_id: Array | None
    tether_stiffness: Array
    exclusion_stiffness: Array
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        *,
        tether_particle_id: int | None,
        anchor: npt.ArrayLike,
        tether_stiffness: float,
        sphere_centers: npt.ArrayLike = (),
        sphere_radii: npt.ArrayLike = (),
        exclusion_stiffness: float = 0.0,
    ) -> None:
        anchor_ = np.asarray(anchor, dtype=np.float64)
        centers = np.asarray(sphere_centers, dtype=np.float64).reshape((-1, 3))
        radii = np.asarray(sphere_radii, dtype=np.float64)
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
        if tether_particle_id is not None and (
            isinstance(tether_particle_id, bool)
            or not isinstance(tether_particle_id, (int, np.integer))
        ):
            raise TypeError("tether_particle_id must be an integer stable particle ID.")
        name = "ribosome-boundary"
        identity = IdentityWords.of(
            {
                "kind": name,
                "tether": tether_particle_id,
                "stiffness": [tether_stiffness, exclusion_stiffness],
                "geometry": array_tree_fingerprint((anchor_, centers, radii)),
            }
        )
        self.anchor, self.centers, self.radii = map(
            jnp.asarray, (anchor_, centers, radii)
        )
        self.tether_particle_id = (
            None
            if tether_particle_id is None
            else jnp.asarray(int(tether_particle_id), dtype=jnp.int64)
        )
        self.tether_stiffness = jnp.asarray(float(tether_stiffness), dtype=jnp.float64)
        self.exclusion_stiffness = jnp.asarray(
            float(exclusion_stiffness), dtype=jnp.float64
        )
        self.name, self.force_group = name, 0
        self.capabilities = AtomisticPotentialCapabilities()
        self.requirements = AtomisticPotentialRequirements()
        self.term_identity = identity

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedRibosomeBoundaryPotential":
        if system.cell is not None:
            raise ValueError(
                "The fixed ribosome boundary requires nonperiodic coordinates."
            )
        if self.tether_particle_id is None:
            return PreparedRibosomeBoundaryPotential(self, system, None)
        ids = np.asarray(system.plan.particle_ids)
        active = np.asarray(system.active_mask)
        matches = np.flatnonzero(
            (ids == int(np.asarray(self.tether_particle_id))) & active
        )
        if len(matches) != 1:
            raise ValueError(
                "The tether must bind exactly one active stable particle ID."
            )
        return PreparedRibosomeBoundaryPotential(self, system, int(matches[0]))


class PreparedRibosomeBoundaryPotential(AbstractPreparedAtomisticEnergyTerm):
    """Prepared ribosome boundary bound to one atomistic system.

    PyTree layout: the tethered slot is a dynamic int32 scalar (``None`` for a
    released stage); active sites are masked by the system's dynamic
    ``active_mask`` over the full capacity, so no shape depends on the active
    count. ``term_identity`` and ``prepared_identity`` are exact canonical
    digests as dynamic words (``term_id``/``prepared_id`` decode them on the
    host).
    """

    plan: RibosomeBoundaryPotential
    system: PreparedAtomisticSystem
    tether_slot: Array | None
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        plan: RibosomeBoundaryPotential,
        system: PreparedAtomisticSystem,
        tether_slot: int | None,
        /,
    ) -> None:
        if (tether_slot is None) != (plan.tether_particle_id is None):
            raise ValueError("A tethered boundary binds exactly one tether slot.")
        if tether_slot is not None and not 0 <= tether_slot < system.capacity:
            raise ValueError("The tether slot must index the system capacity.")
        self.plan, self.system = plan, system
        self.tether_slot = (
            None if tether_slot is None else jnp.asarray(tether_slot, dtype=jnp.int32)
        )
        self.name, self.force_group, self.term_identity = (
            plan.name,
            plan.force_group,
            plan.term_identity,
        )
        self.capabilities, self.requirements = plan.capabilities, plan.requirements
        self.prepared_identity = IdentityWords.of(
            {"term": plan.term_id, "system": system.prepared_id}
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        p = self.plan
        dtype = context.positions.dtype
        active = self.system.active_mask
        atom_energy = jnp.zeros((self.system.capacity,), dtype=dtype)
        successful = jnp.asarray(True)
        if p.centers.shape[0]:
            positions = jnp.where(active[:, None], context.positions, 0.0)
            displacement = positions[:, None, :] - p.centers[None, :, :]
            squared = jnp.sum(displacement**2, axis=-1)
            distance = jnp.sqrt(jnp.where(squared > 0, squared, 1.0))
            overlap = jnp.maximum(p.radii[None, :] - distance, 0.0)
            stiffness = p.exclusion_stiffness.astype(dtype)
            energy = 0.5 * stiffness * jnp.sum(overlap**2, axis=-1)
            atom_energy = jnp.where(active, energy, atom_energy)
            separated = jnp.all(jnp.where(active[:, None], squared > 0, True))
            successful = (stiffness <= 0) | separated
        if self.tether_slot is not None:
            displacement = context.positions[self.tether_slot] - p.anchor
            atom_energy = atom_energy.at[self.tether_slot].add(
                0.5 * p.tether_stiffness.astype(dtype) * jnp.sum(displacement**2)
            )
        energy = jnp.where(successful, jnp.sum(atom_energy), jnp.nan)
        return AtomisticTermEvaluation(energy, atom_energy, successful)


__all__ = ["RibosomeBoundaryPotential", "PreparedRibosomeBoundaryPotential"]
