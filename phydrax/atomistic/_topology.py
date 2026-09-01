#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleDiscretization


def _interaction_table(
    name: str,
    value: ArrayLike | None,
    width: int,
    /,
) -> np.ndarray:
    if value is None:
        return np.zeros((0, width), dtype=np.int64)
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (count, {width}).")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain stable integer particle IDs.")
    result = array.astype(np.int64, copy=False)
    if result.size and np.any(
        np.asarray([np.unique(row).size != width for row in result], dtype=bool)
    ):
        raise ValueError(f"Every {name} interaction requires distinct endpoints.")
    if result.size and np.unique(result, axis=0).shape[0] != result.shape[0]:
        raise ValueError(f"{name} contains duplicate ordered interactions.")
    return result


def _type_ids(name: str, value: ArrayLike | None, count: int, /) -> np.ndarray:
    if value is None:
        return np.zeros((count,), dtype=np.int32)
    array = np.asarray(value)
    if array.shape != (count,) or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"{name} must be an integer vector with one value per interaction."
        )
    result = array.astype(np.int32, copy=False)
    if np.any(result < 0):
        raise ValueError(f"{name} values must be non-negative.")
    return result


def _canonical_pairs(name: str, pairs: np.ndarray, /) -> np.ndarray:
    if not pairs.size:
        return pairs
    canonical = np.sort(pairs, axis=1)
    if np.unique(canonical, axis=0).shape[0] != canonical.shape[0]:
        raise ValueError(f"{name} contains duplicate unordered pairs.")
    return canonical


class MolecularTopologyPlan(StrictModule, NonTrainableState):
    """Stable-ID molecular connectivity and explicit nonbonded exceptions."""

    bonds: Array
    angles: Array
    torsions: Array
    impropers: Array
    constraints: Array
    constraint_distances: Array
    pair_exceptions: Array
    lennard_jones_scales: Array
    electrostatic_scales: Array
    bond_type_ids: Array
    angle_type_ids: Array
    torsion_type_ids: Array
    improper_type_ids: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        bonds: ArrayLike | None = None,
        angles: ArrayLike | None = None,
        torsions: ArrayLike | None = None,
        impropers: ArrayLike | None = None,
        constraints: ArrayLike | None = None,
        constraint_distances: ArrayLike | None = None,
        pair_exceptions: ArrayLike | None = None,
        lennard_jones_scales: ArrayLike | None = None,
        electrostatic_scales: ArrayLike | None = None,
        bond_type_ids: ArrayLike | None = None,
        angle_type_ids: ArrayLike | None = None,
        torsion_type_ids: ArrayLike | None = None,
        improper_type_ids: ArrayLike | None = None,
        plan_id: str | None = None,
    ):
        bonds_ = _canonical_pairs("bonds", _interaction_table("bonds", bonds, 2))
        angles_ = _interaction_table("angles", angles, 3)
        torsions_ = _interaction_table("torsions", torsions, 4)
        impropers_ = _interaction_table("impropers", impropers, 4)
        constraints_ = _canonical_pairs(
            "constraints", _interaction_table("constraints", constraints, 2)
        )
        exceptions_ = _canonical_pairs(
            "pair_exceptions", _interaction_table("pair_exceptions", pair_exceptions, 2)
        )
        if constraint_distances is None:
            if constraints_.shape[0]:
                raise ValueError(
                    "constraint_distances are required when constraints are present."
                )
            distances = np.zeros((0,), dtype=float)
        else:
            distances = np.asarray(constraint_distances, dtype=float)
            if distances.shape != (constraints_.shape[0],):
                raise ValueError(
                    "constraint_distances must provide one target per constraint."
                )
            if np.any(~np.isfinite(distances)) or np.any(distances <= 0.0):
                raise ValueError("Constraint distances must be finite and positive.")
        exception_count = exceptions_.shape[0]
        lj = (
            np.ones((exception_count,), dtype=float)
            if lennard_jones_scales is None
            else np.asarray(lennard_jones_scales, dtype=float)
        )
        electrostatic = (
            np.ones((exception_count,), dtype=float)
            if electrostatic_scales is None
            else np.asarray(electrostatic_scales, dtype=float)
        )
        if lj.shape != (exception_count,) or electrostatic.shape != (exception_count,):
            raise ValueError("Exception scales must align with pair_exceptions.")
        if (
            np.any(~np.isfinite(lj))
            or np.any(~np.isfinite(electrostatic))
            or np.any(lj < 0.0)
            or np.any(electrostatic < 0.0)
        ):
            raise ValueError("Exception scales must be finite and non-negative.")
        bond_types = _type_ids("bond_type_ids", bond_type_ids, bonds_.shape[0])
        angle_types = _type_ids("angle_type_ids", angle_type_ids, angles_.shape[0])
        torsion_types = _type_ids(
            "torsion_type_ids", torsion_type_ids, torsions_.shape[0]
        )
        improper_types = _type_ids(
            "improper_type_ids", improper_type_ids, impropers_.shape[0]
        )
        arrays: dict[str, Any] = {
            "bonds": bonds_,
            "angles": angles_,
            "torsions": torsions_,
            "impropers": impropers_,
            "constraints": constraints_,
            "constraint_distances": distances,
            "pair_exceptions": exceptions_,
            "lennard_jones_scales": lj,
            "electrostatic_scales": electrostatic,
            "bond_type_ids": bond_types,
            "angle_type_ids": angle_types,
            "torsion_type_ids": torsion_types,
            "improper_type_ids": improper_types,
        }
        generated = canonical_fingerprint(
            {"kind": "molecular-topology-plan", "arrays": array_tree_fingerprint(arrays)}
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        for name, value in arrays.items():
            setattr(self, name, jnp.asarray(value))
        self.plan_id = identifier

    @classmethod
    def empty(cls) -> "MolecularTopologyPlan":
        return cls()

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> "PreparedMolecularTopology":
        return PreparedMolecularTopology(self, particles)


class PreparedMolecularTopology(StrictModule, NonTrainableState):
    """Slot-resolved molecular topology bound to one particle support."""

    plan: MolecularTopologyPlan
    bond_indices: Array
    angle_indices: Array
    torsion_indices: Array
    improper_indices: Array
    constraint_indices: Array
    constraint_distances: Array
    exception_keys: Array
    exception_indices: Array
    lennard_jones_scales: Array
    electrostatic_scales: Array
    bond_type_ids: Array
    angle_type_ids: Array
    torsion_type_ids: Array
    improper_type_ids: Array
    particle_discretization_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(self, plan: MolecularTopologyPlan, particles: ParticleDiscretization, /):
        if not isinstance(plan, MolecularTopologyPlan):
            raise TypeError("plan must be a MolecularTopologyPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        particle_ids = np.asarray(particles.particle_ids, dtype=np.int64)
        active = np.asarray(particles.active_mask, dtype=bool)
        if np.unique(particle_ids).size != particle_ids.size:
            raise ValueError("Molecular topology requires unique stable particle IDs.")
        slot_by_id = {
            int(identifier): index for index, identifier in enumerate(particle_ids)
        }

        def resolve(name: str, values: Array) -> np.ndarray:
            host = np.asarray(values, dtype=np.int64)
            resolved = np.zeros(host.shape, dtype=np.int32)
            for interaction_index, row in enumerate(host):
                for endpoint_index, identifier in enumerate(row):
                    key = int(identifier)
                    if key not in slot_by_id:
                        raise ValueError(f"{name} references unknown particle ID {key}.")
                    slot = slot_by_id[key]
                    if not active[slot]:
                        raise ValueError(f"{name} references inactive particle ID {key}.")
                    resolved[interaction_index, endpoint_index] = slot
            return resolved

        bonds = resolve("bonds", plan.bonds)
        angles = resolve("angles", plan.angles)
        torsions = resolve("torsions", plan.torsions)
        impropers = resolve("impropers", plan.impropers)
        constraints = resolve("constraints", plan.constraints)
        exception_slots = resolve("pair_exceptions", plan.pair_exceptions)
        sorted_ids = np.sort(particle_ids)
        rank_by_id = {int(identifier): rank for rank, identifier in enumerate(sorted_ids)}
        exception_pairs = np.asarray(plan.pair_exceptions, dtype=np.int64)
        exception_keys = np.zeros((exception_pairs.shape[0], 5), dtype=np.int64)
        if exception_pairs.size:
            exception_keys[:, 0] = 0
            exception_keys[:, 1:3] = exception_pairs
        order = (
            np.lexsort((exception_keys[:, 2], exception_keys[:, 1]))
            if exception_keys.shape[0]
            else np.zeros((0,), dtype=np.int64)
        )
        exception_keys = exception_keys[order]
        if exception_keys.shape[0] > 1 and np.any(
            np.all(exception_keys[1:] == exception_keys[:-1], axis=-1)
        ):
            raise ValueError("Pair exceptions resolve to duplicate stable pair keys.")
        self.plan = plan
        self.bond_indices = jnp.asarray(bonds)
        self.angle_indices = jnp.asarray(angles)
        self.torsion_indices = jnp.asarray(torsions)
        self.improper_indices = jnp.asarray(impropers)
        self.constraint_indices = jnp.asarray(constraints)
        self.constraint_distances = plan.constraint_distances
        self.exception_keys = jnp.asarray(exception_keys)
        self.exception_indices = jnp.asarray(exception_slots[order])
        self.lennard_jones_scales = plan.lennard_jones_scales[order]
        self.electrostatic_scales = plan.electrostatic_scales[order]
        self.bond_type_ids = plan.bond_type_ids
        self.angle_type_ids = plan.angle_type_ids
        self.torsion_type_ids = plan.torsion_type_ids
        self.improper_type_ids = plan.improper_type_ids
        self.particle_discretization_id = particles.prepared_id
        self.topology_id = canonical_fingerprint(
            {
                "kind": "prepared-molecular-topology",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "exception_keys": exception_keys.tolist(),
            }
        )

    @property
    def constraint_count(self) -> int:
        return int(self.constraint_indices.shape[0])

    def pair_scales(self, pair_keys: ArrayLike, /) -> tuple[Array, Array]:
        keys = jnp.asarray(pair_keys, dtype=jnp.int64)
        if keys.ndim != 2 or keys.shape[1] != 5:
            raise ValueError("Pair keys must have shape (routes, 5).")
        count = int(self.exception_keys.shape[0])
        if count == 0:
            one = jnp.ones(keys.shape[:1], dtype=self.lennard_jones_scales.dtype)
            return one, one
        comparisons = jnp.all(
            keys[:, None, :] == self.exception_keys[None, :, :], axis=-1
        )
        matched = jnp.any(comparisons, axis=1)
        index = jnp.argmax(comparisons, axis=1)
        lj = jnp.where(matched, self.lennard_jones_scales[index], 1.0)
        electrostatic = jnp.where(matched, self.electrostatic_scales[index], 1.0)
        return lj, electrostatic


__all__ = ["MolecularTopologyPlan", "PreparedMolecularTopology"]
