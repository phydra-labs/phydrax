#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._potential import AbstractAtomisticPotential
from ._types import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticScaleContract,
    AtomisticStatus,
)


_AtomisticPotential = AbstractAtomisticPotential


class AtomisticProvenance(StrictModule, NonTrainableState):
    """Identity and mathematical guarantees of one energy/force evaluation."""

    architecture_id: str = eqx.field(static=True)
    parameter_state_id: str = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)
    candidate_topology_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    conservative_forces: bool = eqx.field(static=True)
    frozen_candidate_topology: bool = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    stress_available: bool = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(self, potential: _AtomisticPotential, batch: AtomisticBatch, /):
        self.architecture_id = potential.architecture_id
        self.parameter_state_id = potential.parameter_state_id
        self.potential_id = potential.potential_id
        self.batch_id = batch.batch_id
        self.candidate_topology_id = batch.candidate_topology_id
        self.scale_id = batch.scale.scale_id
        self.precision_id = potential.precision.policy_id
        self.method_id = potential.method_id
        self.conservative_forces = True
        self.frozen_candidate_topology = True
        self.periodic = False
        self.stress_available = False
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "atomistic-prediction-provenance",
                "architecture": potential.architecture_id,
                "parameter_state": potential.parameter_state_id,
                "potential": potential.potential_id,
                "batch": batch.batch_id,
                "candidate_topology": batch.candidate_topology_id,
                "scale": batch.scale.scale_id,
                "precision": potential.precision.policy_id,
                "method": self.method_id,
            }
        )


class AtomisticPrediction(StrictModule, NonTrainableState):
    """Typed finite-molecule energies, conservative forces, and diagnostics."""

    energy: Array
    forces: Array
    atom_energy: Array
    valid: Array
    status: Array
    neighbor_overflow: Array
    maximum_neighbor_count: Array
    net_force: Array
    net_torque: Array
    scale: AtomisticScaleContract
    provenance: AtomisticProvenance
    energy_axes: tuple[str] = eqx.field(static=True)
    force_axes: tuple[str, str, str] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


def energy_and_forces(
    potential: _AtomisticPotential,
    structure: AtomicStructure | AtomisticBatch,
    /,
) -> AtomisticPrediction:
    """Evaluate energy once and derive forces as its negative position gradient.

    Provenance is supported for constructor-created models, native training
    results, and models returned by
    ``phydrax.atomistic.checkpoint_atomistic_potential``. External Equinox or
    Optax tree updates must be checkpointed before prediction; otherwise their
    preserved static identity is intentionally not a valid provenance claim.
    """

    if not isinstance(potential, AbstractAtomisticPotential):
        raise TypeError("potential must implement AbstractAtomisticPotential.")
    if isinstance(structure, AtomicStructure):
        batch = AtomisticBatch.from_structure(structure)
    elif isinstance(structure, AtomisticBatch):
        batch = structure
    else:
        raise TypeError("structure must be AtomicStructure or AtomisticBatch.")
    potential._validate_batch(batch)

    def scalar_energy(position: Array) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        energy, atom_energy, graph = potential._energy_unchecked(batch, position)
        return jnp.sum(energy), (
            energy,
            atom_energy,
            graph.overflow,
            graph.maximum_neighbor_count,
        )

    (_, auxiliary), gradient = jax.value_and_grad(scalar_energy, has_aux=True)(
        batch.positions
    )
    energy, atom_energy, overflow, maximum_neighbor_count = auxiliary
    forces = (-gradient).astype(potential.precision.output_dtype)
    mask = batch.atom_mask
    forces = jnp.where(mask[:, :, None], forces, 0.0)
    finite_energy = jnp.isfinite(energy)
    finite_forces = jnp.all(
        jnp.isfinite(jnp.where(mask[:, :, None], forces, 0.0)), axis=(1, 2)
    )
    valid = (~overflow) & finite_energy & finite_forces
    status = jnp.where(
        overflow,
        int(AtomisticStatus.NEIGHBOR_OVERFLOW),
        jnp.where(
            finite_energy & finite_forces,
            int(AtomisticStatus.SUCCESS),
            int(AtomisticStatus.NONFINITE),
        ),
    ).astype(jnp.int32)
    nan = jnp.asarray(jnp.nan, dtype=energy.dtype)
    energy = jnp.where(valid, energy, nan)
    atom_energy = jnp.where(valid[:, None], jnp.where(mask, atom_energy, 0.0), nan)
    forces = jnp.where(
        valid[:, None, None],
        jnp.where(mask[:, :, None], forces, 0.0),
        nan,
    )
    diagnostic_forces = jnp.where(mask[:, :, None], forces, 0.0)
    net_force = jnp.sum(diagnostic_forces, axis=1)
    torque_dtype = jnp.dtype(potential.precision.reduction_dtype)
    mass = jnp.where(mask, batch.masses, 0.0).astype(torque_dtype)
    torque_positions = jnp.where(mask[:, :, None], batch.positions, 0.0).astype(
        torque_dtype
    )
    center = (
        contract("ba,bad->bd", mass, torque_positions) / jnp.sum(mass, axis=1)[:, None]
    )
    lever = torque_positions - center[:, None, :]
    torque_force = diagnostic_forces.astype(torque_dtype)
    net_torque = jnp.sum(jnp.cross(lever, torque_force), axis=1).astype(
        potential.precision.output_dtype
    )
    return AtomisticPrediction(
        energy=energy,
        forces=forces,
        atom_energy=atom_energy,
        valid=valid,
        status=status,
        neighbor_overflow=overflow,
        maximum_neighbor_count=maximum_neighbor_count,
        net_force=net_force,
        net_torque=net_torque,
        scale=batch.scale,
        provenance=AtomisticProvenance(potential, batch),
        energy_axes=("case",),
        force_axes=("case", "atom", "cartesian"),
    )


__all__ = [
    "AtomisticPrediction",
    "AtomisticProvenance",
    "energy_and_forces",
]
