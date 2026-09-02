#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity endpoint alchemy and reduced-potential cross-evaluation."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


@jax.custom_jvp
def _safe_norm(value: Array, /) -> Array:
    """Euclidean norm with a defined zero tangent for collapsed routes."""

    return jnp.sqrt(jnp.sum(value * value, axis=-1))


@_safe_norm.defjvp
def _safe_norm_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    norm = _safe_norm(value)
    derivative = jnp.where(
        norm > 0.0,
        jnp.sum(value * tangent, axis=-1) / jnp.where(norm > 0.0, norm, 1.0),
        0.0,
    )
    return norm, derivative


class AlchemicalEndpointPlan(StrictModule, NonTrainableState):
    """One stable-ID endpoint with atom and harmonic-bond parameters."""

    particle_ids: Array
    atom_type_ids: Array
    charges: Array
    sigma: Array
    epsilon: Array
    dummy_mask: Array
    bond_particle_ids: Array
    bond_stiffness: Array
    bond_equilibrium_lengths: Array
    bond_mask: Array
    unit_system_id: str = eqx.field(static=True)
    endpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_ids: ArrayLike,
        atom_type_ids: ArrayLike,
        charges: ArrayLike,
        sigma: ArrayLike,
        epsilon: ArrayLike,
        /,
        *,
        dummy_mask: ArrayLike | None = None,
        bond_particle_ids: ArrayLike | None = None,
        bond_stiffness: ArrayLike | None = None,
        bond_equilibrium_lengths: ArrayLike | None = None,
        bond_mask: ArrayLike | None = None,
        unit_system_id: str,
        endpoint_id: str | None = None,
    ):
        ids = np.asarray(particle_ids)
        types = np.asarray(atom_type_ids)
        charge = np.asarray(charges)
        sigma_ = np.asarray(sigma)
        epsilon_ = np.asarray(epsilon)
        if ids.ndim != 1 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("Alchemical particle_ids must be a non-empty integer vector.")
        if types.shape != ids.shape or not np.issubdtype(types.dtype, np.integer):
            raise TypeError("Alchemical atom_type_ids must be an aligned integer vector.")
        if any(
            value.shape != ids.shape or value.dtype.kind != "f"
            for value in (charge, sigma_, epsilon_)
        ):
            raise TypeError(
                "Alchemical charge and Lennard-Jones parameters must be aligned floating vectors."
            )
        ids = ids.astype(np.int64, copy=False)
        types = types.astype(np.int32, copy=False)
        if np.unique(ids).size != ids.size:
            raise ValueError("Alchemical endpoint particle IDs must be unique.")
        dummy = (
            np.zeros(ids.shape, dtype=bool)
            if dummy_mask is None
            else np.asarray(dummy_mask, dtype=bool)
        )
        if dummy.shape != ids.shape:
            raise ValueError("dummy_mask must align with particle_ids.")
        if (
            np.any(types < 0)
            or np.any(~np.isfinite(charge))
            or np.any(~np.isfinite(sigma_))
            or np.any(~np.isfinite(epsilon_))
            or np.any(sigma_ <= 0.0)
            or np.any(epsilon_ < 0.0)
        ):
            raise ValueError(
                "Endpoint types and nonbonded parameters are outside physical bounds."
            )
        if np.any((charge != 0.0) & dummy) or np.any((epsilon_ != 0.0) & dummy):
            raise ValueError("Explicit dummy atoms require zero charge and epsilon.")
        bonds = (
            np.zeros((0, 2), dtype=np.int64)
            if bond_particle_ids is None
            else np.asarray(bond_particle_ids)
        )
        if (
            bonds.ndim != 2
            or bonds.shape[1] != 2
            or not np.issubdtype(bonds.dtype, np.integer)
        ):
            raise TypeError("bond_particle_ids must have integer shape (bond_count, 2).")
        bonds = np.sort(bonds.astype(np.int64, copy=False), axis=1)
        if bonds.size and (
            np.any(bonds[:, 0] == bonds[:, 1])
            or np.unique(bonds, axis=0).shape[0] != bonds.shape[0]
            or not np.all(np.isin(bonds, ids))
        ):
            raise ValueError("Endpoint bonds must be unique pairs of known particle IDs.")
        count = bonds.shape[0]
        stiffness = (
            np.zeros((count,), dtype=charge.dtype)
            if bond_stiffness is None
            else np.asarray(bond_stiffness)
        )
        lengths = (
            np.ones((count,), dtype=charge.dtype)
            if bond_equilibrium_lengths is None
            else np.asarray(bond_equilibrium_lengths)
        )
        active_bonds = (
            np.ones((count,), dtype=bool)
            if bond_mask is None
            else np.asarray(bond_mask, dtype=bool)
        )
        if (
            stiffness.shape != (count,)
            or lengths.shape != (count,)
            or stiffness.dtype.kind != "f"
            or lengths.dtype.kind != "f"
            or active_bonds.shape != (count,)
        ):
            raise TypeError(
                "Bond parameters and masks must align with bond_particle_ids."
            )
        if (
            np.any(~np.isfinite(stiffness))
            or np.any(~np.isfinite(lengths))
            or np.any(stiffness < 0.0)
            or np.any(lengths <= 0.0)
        ):
            raise ValueError(
                "Bond stiffnesses must be non-negative and lengths positive."
            )
        dummy_by_id = {
            int(identifier): bool(dummy[index]) for index, identifier in enumerate(ids)
        }
        if any(
            active_bonds[index]
            and (dummy_by_id[int(pair[0])] or dummy_by_id[int(pair[1])])
            for index, pair in enumerate(bonds)
        ):
            raise ValueError("Active endpoint bonds cannot reference dummy atoms.")
        order = (
            np.lexsort((bonds[:, 1], bonds[:, 0])) if count else np.zeros((0,), dtype=int)
        )
        bonds, stiffness, lengths, active_bonds = (
            bonds[order],
            stiffness[order],
            lengths[order],
            active_bonds[order],
        )
        unit = str(unit_system_id).strip()
        if not unit:
            raise ValueError("unit_system_id must be non-empty.")
        arrays = {
            "particle_ids": ids,
            "atom_type_ids": types,
            "charges": charge,
            "sigma": sigma_,
            "epsilon": epsilon_,
            "dummy_mask": dummy,
            "bond_particle_ids": bonds,
            "bond_stiffness": stiffness,
            "bond_equilibrium_lengths": lengths,
            "bond_mask": active_bonds,
        }
        generated = canonical_fingerprint(
            {
                "kind": "alchemical-endpoint",
                "unit_system": unit,
                "arrays": array_tree_fingerprint(arrays),
            }
        )
        identifier = generated if endpoint_id is None else str(endpoint_id).strip()
        if not identifier:
            raise ValueError("endpoint_id must be non-empty.")
        for name, value in arrays.items():
            setattr(self, name, jnp.asarray(value))
        self.unit_system_id = unit
        self.endpoint_id = identifier


class SoftCorePolicy(StrictModule, NonTrainableState):
    """Endpoint-exact soft-core regularization for changing pair interactions."""

    lennard_jones_alpha: float = eqx.field(static=True)
    electrostatic_alpha: float = eqx.field(static=True)
    coupling_power: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        lennard_jones_alpha: float = 0.5,
        electrostatic_alpha: float = 0.5,
        coupling_power: int = 2,
    ):
        lj = float(lennard_jones_alpha)
        electrostatic = float(electrostatic_alpha)
        if not isinstance(coupling_power, (int, np.integer)) or isinstance(
            coupling_power, bool
        ):
            raise TypeError("coupling_power must be an integer.")
        power = int(coupling_power)
        if (
            not np.isfinite(lj)
            or not np.isfinite(electrostatic)
            or lj <= 0.0
            or electrostatic <= 0.0
        ):
            raise ValueError("Soft-core alpha values must be finite and positive.")
        if power < 1:
            raise ValueError("Soft-core coupling_power must be positive.")
        self.lennard_jones_alpha = lj
        self.electrostatic_alpha = electrostatic
        self.coupling_power = power
        self.policy_id = canonical_fingerprint(
            {
                "kind": "alchemical-soft-core",
                "lj_alpha": lj.hex(),
                "electrostatic_alpha": electrostatic.hex(),
                "power": power,
            }
        )


class LambdaSchedulePlan(StrictModule, NonTrainableState):
    """Monotone piecewise-linear conversion from protocol lambda to coupling."""

    lambda_values: Array
    coupling_values: Array
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self, lambda_values: ArrayLike, coupling_values: ArrayLike | None = None, /
    ):
        lambdas = np.asarray(lambda_values)
        coupling = (
            lambdas.copy() if coupling_values is None else np.asarray(coupling_values)
        )
        if lambdas.ndim != 1 or lambdas.size < 2 or lambdas.dtype.kind != "f":
            raise TypeError(
                "Lambda schedule coordinates must be a floating vector with at least two entries."
            )
        if coupling.shape != lambdas.shape or coupling.dtype.kind != "f":
            raise TypeError("Lambda coupling values must be an aligned floating vector.")
        if (
            np.any(~np.isfinite(lambdas))
            or np.any(~np.isfinite(coupling))
            or lambdas[0] != 0.0
            or lambdas[-1] != 1.0
            or coupling[0] != 0.0
            or coupling[-1] != 1.0
            or np.any(np.diff(lambdas) <= 0.0)
            or np.any(np.diff(coupling) < 0.0)
        ):
            raise ValueError(
                "Lambda schedules must be finite monotone paths with exact zero and one endpoints."
            )
        self.lambda_values = jnp.asarray(lambdas)
        self.coupling_values = jnp.asarray(coupling)
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "lambda-schedule",
                "arrays": array_tree_fingerprint((lambdas, coupling)),
            }
        )

    @classmethod
    def linear(cls, state_count: int, /) -> "LambdaSchedulePlan":
        if not isinstance(state_count, (int, np.integer)) or isinstance(
            state_count, bool
        ):
            raise TypeError("state_count must be an integer.")
        count = int(state_count)
        if count < 2:
            raise ValueError("A linear lambda schedule requires at least two states.")
        return cls(np.linspace(0.0, 1.0, count))

    def coupling(self, lambda_value: ArrayLike, /) -> Array:
        value = jnp.asarray(lambda_value, dtype=self.lambda_values.dtype)
        clipped = jnp.clip(value, 0.0, 1.0)
        upper = jnp.clip(
            jnp.searchsorted(self.lambda_values, clipped, side="right"),
            1,
            self.lambda_values.size - 1,
        )
        lower = upper - 1
        fraction = (clipped - self.lambda_values[lower]) / (
            self.lambda_values[upper] - self.lambda_values[lower]
        )
        return self.coupling_values[lower] + fraction * (
            self.coupling_values[upper] - self.coupling_values[lower]
        )


class AlchemicalPreparationEvidence(StrictModule, NonTrainableState):
    """Required and reserved atom/bond capacities established at preparation."""

    atom_count: int = eqx.field(static=True)
    atom_capacity: int = eqx.field(static=True)
    bond_count: int = eqx.field(static=True)
    bond_capacity: int = eqx.field(static=True)
    topology_compatible: bool = eqx.field(static=True)
    successful: bool = eqx.field(static=True)


class AlchemicalInterpolatedParameters(StrictModule):
    """Fixed-shape parameters at one schedule coordinate."""

    lambda_value: Array
    coupling: Array
    atom_type_ids: Array
    charges: Array
    sigma: Array
    epsilon: Array
    atom_weights: Array
    bond_stiffness: Array
    bond_equilibrium_lengths: Array
    bond_weights: Array


class AlchemicalEvaluation(StrictModule):
    """Component-resolved energy, force, and thermodynamic derivative."""

    energy: Array
    forces: Array
    component_energies: Array
    dudlambda: Array
    component_dudlambda: Array
    lambda_value: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class AlchemicalWorkEvaluation(StrictModule):
    """Instantaneous endpoint work resolved as bond, Coulomb, and LJ terms."""

    work: Array
    component_work: Array
    initial_lambda: Array
    final_lambda: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class AlchemicalCycleEvaluation(StrictModule):
    """Telescoping cycle work and closure evidence."""

    work: Array
    component_work: Array
    closure_error: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class ReducedPotentialCrossEvaluation(StrictModule):
    """State-by-sample reduced-potential matrix for FEP, BAR, and MBAR."""

    values: Array
    energies: Array
    component_energies: Array
    lambda_values: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class AlchemicalTransformationPlan(StrictModule, NonTrainableState):
    """Two endpoints, stable atom mapping, capacities, schedule, and soft-core policy."""

    endpoint_a: AlchemicalEndpointPlan
    endpoint_b: AlchemicalEndpointPlan
    atom_mapping: Array
    atom_capacity: int = eqx.field(static=True)
    bond_capacity: int = eqx.field(static=True)
    soft_core: SoftCorePolicy
    schedule: LambdaSchedulePlan
    beta: float = eqx.field(static=True)
    coulomb_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        endpoint_a: AlchemicalEndpointPlan,
        endpoint_b: AlchemicalEndpointPlan,
        /,
        *,
        atom_mapping: ArrayLike | None = None,
        atom_capacity: int,
        bond_capacity: int,
        soft_core: SoftCorePolicy | None = None,
        schedule: LambdaSchedulePlan | None = None,
        beta: float = 1.0,
        coulomb_constant: float = 1.0,
    ):
        if not isinstance(endpoint_a, AlchemicalEndpointPlan) or not isinstance(
            endpoint_b, AlchemicalEndpointPlan
        ):
            raise TypeError(
                "Alchemical endpoints must be AlchemicalEndpointPlan instances."
            )
        if endpoint_a.unit_system_id != endpoint_b.unit_system_id:
            raise ValueError(
                "Alchemical endpoints must use the same exact unit system identity."
            )
        ids_a = np.asarray(endpoint_a.particle_ids, dtype=np.int64)
        ids_b = np.asarray(endpoint_b.particle_ids, dtype=np.int64)
        if atom_mapping is None:
            common = np.intersect1d(ids_a, ids_b)
            mapping = np.stack((common, common), axis=-1)
        else:
            mapping = np.asarray(atom_mapping)
        if (
            mapping.ndim != 2
            or mapping.shape[1] != 2
            or not np.issubdtype(mapping.dtype, np.integer)
        ):
            raise TypeError("atom_mapping must have integer shape (mapped_count, 2).")
        mapping = mapping.astype(np.int64, copy=False)
        if mapping.shape[0] and (
            np.unique(mapping[:, 0]).size != mapping.shape[0]
            or np.unique(mapping[:, 1]).size != mapping.shape[0]
            or not np.all(np.isin(mapping[:, 0], ids_a))
            or not np.all(np.isin(mapping[:, 1], ids_b))
        ):
            raise ValueError(
                "Atom mapping must be one-to-one and reference known endpoint IDs."
            )
        order = (
            np.lexsort((mapping[:, 1], mapping[:, 0]))
            if mapping.shape[0]
            else np.zeros((0,), dtype=int)
        )
        mapping = mapping[order]
        if not isinstance(atom_capacity, (int, np.integer)) or isinstance(
            atom_capacity, bool
        ):
            raise TypeError("atom_capacity must be an integer.")
        if not isinstance(bond_capacity, (int, np.integer)) or isinstance(
            bond_capacity, bool
        ):
            raise TypeError("bond_capacity must be an integer.")
        capacity = int(atom_capacity)
        bond_capacity_ = int(bond_capacity)
        mapped_a, mapped_b = set(mapping[:, 0].tolist()), set(mapping[:, 1].tolist())
        required_atoms = (
            mapping.shape[0]
            + sum(int(value) not in mapped_a for value in ids_a)
            + sum(int(value) not in mapped_b for value in ids_b)
        )
        if capacity < required_atoms:
            raise ValueError(
                f"Alchemical atom capacity {capacity} is smaller than required count {required_atoms}."
            )
        if bond_capacity_ < 0:
            raise ValueError("Alchemical bond_capacity must be non-negative.")
        soft = SoftCorePolicy() if soft_core is None else soft_core
        schedule_ = LambdaSchedulePlan.linear(2) if schedule is None else schedule
        if not isinstance(soft, SoftCorePolicy) or not isinstance(
            schedule_, LambdaSchedulePlan
        ):
            raise TypeError("soft_core and schedule must use their dedicated plan types.")
        beta_, coulomb = float(beta), float(coulomb_constant)
        if not np.isfinite(beta_) or beta_ <= 0.0 or not np.isfinite(coulomb):
            raise ValueError(
                "beta must be finite and positive and coulomb_constant finite."
            )
        self.endpoint_a, self.endpoint_b = endpoint_a, endpoint_b
        self.atom_mapping = jnp.asarray(mapping)
        self.atom_capacity, self.bond_capacity = capacity, bond_capacity_
        self.soft_core, self.schedule = soft, schedule_
        self.beta, self.coulomb_constant = beta_, coulomb
        self.plan_id = canonical_fingerprint(
            {
                "kind": "alchemical-transformation",
                "endpoints": [endpoint_a.endpoint_id, endpoint_b.endpoint_id],
                "mapping": array_tree_fingerprint(mapping),
                "atom_capacity": capacity,
                "bond_capacity": bond_capacity_,
                "soft_core": soft.policy_id,
                "schedule": schedule_.schedule_id,
                "beta": beta_.hex(),
                "coulomb_constant": coulomb.hex(),
            }
        )

    def prepare(self, /) -> "PreparedAlchemicalTransformation":
        return PreparedAlchemicalTransformation(self)


class PreparedAlchemicalTransformation(StrictModule, NonTrainableState):
    """Aligned endpoint tensors and fixed pair/bond routes for compiled evaluation."""

    plan: AlchemicalTransformationPlan
    endpoint_particle_ids: Array
    atom_type_ids: Array
    charges: Array
    sigma: Array
    epsilon: Array
    active_mask: Array
    dummy_mask: Array
    bond_indices: Array
    bond_stiffness: Array
    bond_equilibrium_lengths: Array
    bond_mask: Array
    pair_indices: Array
    preparation: AlchemicalPreparationEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: AlchemicalTransformationPlan, /):
        if not isinstance(plan, AlchemicalTransformationPlan):
            raise TypeError("plan must be an AlchemicalTransformationPlan.")
        endpoint_a, endpoint_b = plan.endpoint_a, plan.endpoint_b
        ids_a, ids_b = (
            np.asarray(endpoint_a.particle_ids),
            np.asarray(endpoint_b.particle_ids),
        )
        mapping = np.asarray(plan.atom_mapping)
        mapped_a, mapped_b = set(mapping[:, 0].tolist()), set(mapping[:, 1].tolist())
        slots: list[tuple[int | None, int | None]] = [
            (int(a), int(b)) for a, b in mapping
        ]
        slots.extend(
            (int(value), None)
            for value in sorted(ids_a.tolist())
            if int(value) not in mapped_a
        )
        slots.extend(
            (None, int(value))
            for value in sorted(ids_b.tolist())
            if int(value) not in mapped_b
        )
        index_a = {int(value): index for index, value in enumerate(ids_a)}
        index_b = {int(value): index for index, value in enumerate(ids_b)}
        endpoint_ids = np.full((2, plan.atom_capacity), -1, dtype=np.int64)
        atom_types = np.zeros((2, plan.atom_capacity), dtype=np.int32)
        dtype = np.result_type(
            *(
                np.asarray(value).dtype
                for endpoint in (endpoint_a, endpoint_b)
                for value in (
                    endpoint.charges,
                    endpoint.sigma,
                    endpoint.epsilon,
                    endpoint.bond_stiffness,
                    endpoint.bond_equilibrium_lengths,
                )
            )
        )
        charges = np.zeros((2, plan.atom_capacity), dtype=dtype)
        sigma = np.ones((2, plan.atom_capacity), dtype=dtype)
        epsilon = np.zeros((2, plan.atom_capacity), dtype=dtype)
        active = np.zeros((2, plan.atom_capacity), dtype=bool)
        for slot, pair in enumerate(slots):
            for endpoint, identifier, source, lookup in (
                (0, pair[0], endpoint_a, index_a),
                (1, pair[1], endpoint_b, index_b),
            ):
                if identifier is None:
                    continue
                source_index = lookup[identifier]
                endpoint_ids[endpoint, slot] = identifier
                atom_types[endpoint, slot] = int(source.atom_type_ids[source_index])
                charges[endpoint, slot] = float(source.charges[source_index])
                sigma[endpoint, slot] = float(source.sigma[source_index])
                epsilon[endpoint, slot] = float(source.epsilon[source_index])
                active[endpoint, slot] = not bool(source.dummy_mask[source_index])
        slot_a = {
            pair[0]: index for index, pair in enumerate(slots) if pair[0] is not None
        }
        slot_b = {
            pair[1]: index for index, pair in enumerate(slots) if pair[1] is not None
        }
        bond_maps: list[dict[tuple[int, int], tuple[float, float]]] = []
        for endpoint, slot_by_id in ((endpoint_a, slot_a), (endpoint_b, slot_b)):
            result: dict[tuple[int, int], tuple[float, float]] = {}
            for index, pair in enumerate(np.asarray(endpoint.bond_particle_ids)):
                if bool(endpoint.bond_mask[index]):
                    key = tuple(
                        sorted((slot_by_id[int(pair[0])], slot_by_id[int(pair[1])]))
                    )
                    result[key] = (
                        float(endpoint.bond_stiffness[index]),
                        float(endpoint.bond_equilibrium_lengths[index]),
                    )
            bond_maps.append(result)
        union_bonds = sorted(set(bond_maps[0]) | set(bond_maps[1]))
        for key in union_bonds:
            common_core = all(
                active[endpoint, key[0]] and active[endpoint, key[1]]
                for endpoint in range(2)
            )
            if common_core and ((key in bond_maps[0]) != (key in bond_maps[1])):
                raise ValueError(
                    "Mapped non-dummy core atoms require compatible bonded topology at both endpoints."
                )
        if len(union_bonds) > plan.bond_capacity:
            raise ValueError(
                f"Alchemical bond capacity {plan.bond_capacity} is smaller than required count {len(union_bonds)}."
            )
        bond_indices = np.zeros((plan.bond_capacity, 2), dtype=np.int32)
        bond_k = np.zeros((2, plan.bond_capacity), dtype=dtype)
        bond_r0 = np.ones((2, plan.bond_capacity), dtype=dtype)
        bond_mask = np.zeros((2, plan.bond_capacity), dtype=bool)
        for slot, key in enumerate(union_bonds):
            bond_indices[slot] = key
            for endpoint in range(2):
                if key in bond_maps[endpoint]:
                    bond_k[endpoint, slot], bond_r0[endpoint, slot] = bond_maps[endpoint][
                        key
                    ]
                    bond_mask[endpoint, slot] = True
            if bond_mask[0, slot] and not bond_mask[1, slot]:
                bond_r0[1, slot] = bond_r0[0, slot]
            elif bond_mask[1, slot] and not bond_mask[0, slot]:
                bond_r0[0, slot] = bond_r0[1, slot]
        pair_indices = np.asarray(
            [
                (left, right)
                for left in range(plan.atom_capacity)
                for right in range(left + 1, plan.atom_capacity)
            ],
            dtype=np.int32,
        ).reshape((-1, 2))
        self.plan = plan
        for name, value in {
            "endpoint_particle_ids": endpoint_ids,
            "atom_type_ids": atom_types,
            "charges": charges,
            "sigma": sigma,
            "epsilon": epsilon,
            "active_mask": active,
            "dummy_mask": ~active,
            "bond_indices": bond_indices,
            "bond_stiffness": bond_k,
            "bond_equilibrium_lengths": bond_r0,
            "bond_mask": bond_mask,
            "pair_indices": pair_indices,
        }.items():
            setattr(self, name, jnp.asarray(value))
        self.preparation = AlchemicalPreparationEvidence(
            len(slots),
            plan.atom_capacity,
            len(union_bonds),
            plan.bond_capacity,
            True,
            True,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-alchemical-transformation",
                "plan": plan.plan_id,
                "aligned": array_tree_fingerprint(
                    {
                        "endpoint_ids": endpoint_ids,
                        "bonds": bond_indices,
                        "bond_mask": bond_mask,
                    }
                ),
            }
        )

    def interpolate(self, lambda_value: ArrayLike, /) -> AlchemicalInterpolatedParameters:
        value = jnp.asarray(lambda_value, dtype=self.charges.dtype)
        if value.shape != ():
            raise ValueError("Alchemical interpolation lambda must be scalar.")
        coupling = self.plan.schedule.coupling(value)
        weights = jnp.stack((1.0 - coupling, coupling))
        atom_types = jnp.where(
            coupling < 0.5, self.atom_type_ids[0], self.atom_type_ids[1]
        )
        return AlchemicalInterpolatedParameters(
            value,
            coupling,
            atom_types,
            jnp.sum(weights[:, None] * self.charges, axis=0),
            jnp.sum(weights[:, None] * self.sigma, axis=0),
            jnp.sum(weights[:, None] * self.epsilon, axis=0),
            jnp.sum(
                weights[:, None] * self.active_mask.astype(self.charges.dtype), axis=0
            ),
            jnp.sum(weights[:, None] * self.bond_stiffness, axis=0),
            jnp.sum(weights[:, None] * self.bond_equilibrium_lengths, axis=0),
            jnp.sum(weights[:, None] * self.bond_mask.astype(self.charges.dtype), axis=0),
        )

    def _components(self, positions: Array, lambda_value: Array, /) -> Array:
        coupling = self.plan.schedule.coupling(lambda_value)
        weights = jnp.stack((1.0 - coupling, coupling))
        left_b, right_b = self.bond_indices[:, 0], self.bond_indices[:, 1]
        bond_present = jnp.any(self.bond_mask, axis=0)
        bond_displacement = jnp.where(
            bond_present[:, None],
            positions[left_b] - positions[right_b],
            0.0,
        )
        bond_distance = _safe_norm(bond_displacement)
        bond_energy_by_endpoint = (
            0.5
            * self.bond_stiffness
            * (bond_distance[None, :] - self.bond_equilibrium_lengths) ** 2
        )
        bonded = jnp.sum(
            weights[:, None] * jnp.where(self.bond_mask, bond_energy_by_endpoint, 0.0)
        )
        left, right = self.pair_indices[:, 0], self.pair_indices[:, 1]
        displacement = positions[left] - positions[right]
        r2 = jnp.sum(displacement * displacement, axis=-1)
        pair_active = self.active_mask[:, left] & self.active_mask[:, right]
        sigma_pair = 0.5 * (self.sigma[:, left] + self.sigma[:, right])
        epsilon_pair = jnp.sqrt(self.epsilon[:, left] * self.epsilon[:, right])
        charge_product = self.charges[:, left] * self.charges[:, right]
        lj_active = pair_active & (epsilon_pair > 0.0)
        electrostatic_active = pair_active & (charge_product != 0.0)
        opposing = jnp.stack((coupling, 1.0 - coupling))[:, None]
        lj_gate = (lj_active[0] != lj_active[1])[None, :].astype(positions.dtype)
        electrostatic_gate = (electrostatic_active[0] != electrostatic_active[1])[
            None, :
        ].astype(positions.dtype)
        softened_r6 = (
            r2[None, :] ** 3
            + self.plan.soft_core.lennard_jones_alpha
            * opposing**self.plan.soft_core.coupling_power
            * sigma_pair**6
            * lj_gate
        )
        safe_r6 = jnp.where(lj_active, softened_r6, 1.0)
        sigma6 = sigma_pair**6
        ratio6 = sigma6 / safe_r6
        lj_by_endpoint = 4.0 * epsilon_pair * (ratio6 * ratio6 - ratio6)
        softened_r2 = (
            r2[None, :]
            + self.plan.soft_core.electrostatic_alpha
            * opposing**self.plan.soft_core.coupling_power
            * sigma_pair**2
            * electrostatic_gate
        )
        safe_r2 = jnp.where(electrostatic_active, softened_r2, 1.0)
        coulomb_by_endpoint = (
            self.plan.coulomb_constant * charge_product / jnp.sqrt(safe_r2)
        )
        lj = jnp.sum(weights[:, None] * jnp.where(lj_active, lj_by_endpoint, 0.0))
        coulomb = jnp.sum(
            weights[:, None] * jnp.where(electrostatic_active, coulomb_by_endpoint, 0.0)
        )
        return jnp.stack((bonded, coulomb, lj))

    def _bond_geometry_valid(self, positions: Array, lambda_value: Array, /) -> Array:
        coupling = self.plan.schedule.coupling(lambda_value)
        endpoint_weight = jnp.stack((1.0 - coupling, coupling))
        weighted = jnp.any(self.bond_mask & (endpoint_weight[:, None] > 0.0), axis=0)
        present = jnp.any(self.bond_mask, axis=0)
        left, right = self.bond_indices[:, 0], self.bond_indices[:, 1]
        displacement = jnp.where(
            present[:, None], positions[left] - positions[right], 0.0
        )
        squared = jnp.sum(displacement * displacement, axis=-1)
        return jnp.all(~weighted | (squared > 0.0))

    def evaluate(
        self, positions: ArrayLike, lambda_value: ArrayLike, /
    ) -> AlchemicalEvaluation:
        coordinate = jnp.asarray(positions)
        if (
            coordinate.shape != (self.plan.atom_capacity, 3)
            or coordinate.dtype.kind != "f"
        ):
            raise ValueError(
                "Alchemical positions must have floating shape (atom_capacity, 3)."
            )
        value = jnp.asarray(lambda_value, dtype=coordinate.dtype)
        if value.shape != ():
            raise ValueError("Alchemical lambda must be scalar.")
        components = self._components(coordinate, value)
        energy = jnp.sum(components)
        forces = -jax.grad(lambda x: jnp.sum(self._components(x, value)))(coordinate)
        component_derivative = jax.jacfwd(lambda lam: self._components(coordinate, lam))(
            value
        )
        derivative = jnp.sum(component_derivative)
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(forces))
            & jnp.all(jnp.isfinite(component_derivative))
        )
        in_range = jnp.isfinite(value) & (value >= 0.0) & (value <= 1.0)
        geometry_valid = self._bond_geometry_valid(coordinate, value)
        return AlchemicalEvaluation(
            energy,
            forces,
            components,
            derivative,
            component_derivative,
            value,
            finite,
            finite & in_range & geometry_valid,
            self.prepared_id,
        )

    def work(
        self, positions: ArrayLike, initial_lambda: ArrayLike, final_lambda: ArrayLike, /
    ) -> AlchemicalWorkEvaluation:
        coordinate = jnp.asarray(positions)
        if coordinate.shape != (self.plan.atom_capacity, 3):
            raise ValueError("Alchemical work positions must match atom_capacity.")
        if coordinate.dtype.kind != "f":
            raise TypeError("Alchemical work positions must be floating point.")
        initial = jnp.asarray(initial_lambda, dtype=coordinate.dtype)
        final = jnp.asarray(final_lambda, dtype=coordinate.dtype)
        if initial.shape != () or final.shape != ():
            raise ValueError("Alchemical work endpoints must be scalar.")
        component_work = self._components(coordinate, final) - self._components(
            coordinate, initial
        )
        work = jnp.sum(component_work)
        finite = jnp.isfinite(work) & jnp.all(jnp.isfinite(component_work))
        in_range = (
            jnp.isfinite(initial)
            & jnp.isfinite(final)
            & (initial >= 0.0)
            & (initial <= 1.0)
            & (final >= 0.0)
            & (final <= 1.0)
        )
        geometry_valid = self._bond_geometry_valid(
            coordinate, initial
        ) & self._bond_geometry_valid(coordinate, final)
        return AlchemicalWorkEvaluation(
            work,
            component_work,
            initial,
            final,
            finite,
            finite & in_range & geometry_valid,
            self.prepared_id,
        )

    def cycle_work(
        self, positions: ArrayLike, lambda_cycle: ArrayLike, /
    ) -> AlchemicalCycleEvaluation:
        coordinate = jnp.asarray(positions)
        if coordinate.shape != (self.plan.atom_capacity, 3):
            raise ValueError("Alchemical cycle positions must match atom_capacity.")
        if coordinate.dtype.kind != "f":
            raise TypeError("Alchemical cycle positions must be floating point.")
        path = jnp.asarray(lambda_cycle, dtype=coordinate.dtype)
        if path.ndim != 1 or path.size < 2:
            raise ValueError(
                "An alchemical cycle must contain at least two lambda values."
            )
        component_values = jax.vmap(lambda value: self._components(coordinate, value))(
            path
        )
        geometry_valid = jnp.all(
            jax.vmap(lambda value: self._bond_geometry_valid(coordinate, value))(path)
        )
        increments = component_values[1:] - component_values[:-1]
        component_work = jnp.sum(increments, axis=0)
        work = jnp.sum(component_work)
        closure = jnp.abs(work)
        path_valid = jnp.all(jnp.isfinite(path) & (path >= 0.0) & (path <= 1.0)) & (
            path[0] == path[-1]
        )
        finite = (
            jnp.isfinite(work)
            & jnp.all(jnp.isfinite(component_work))
            & jnp.all(jnp.isfinite(increments))
        )
        scale = jnp.sum(jnp.abs(increments))
        tolerance = 64.0 * jnp.finfo(component_work.dtype).eps * jnp.maximum(1.0, scale)
        return AlchemicalCycleEvaluation(
            work,
            component_work,
            closure,
            finite,
            finite & path_valid & geometry_valid & (closure <= tolerance),
            self.prepared_id,
        )

    def cross_evaluate(
        self, positions: ArrayLike, lambda_values: ArrayLike | None = None, /
    ) -> ReducedPotentialCrossEvaluation:
        coordinates = jnp.asarray(positions)
        if coordinates.ndim != 3 or coordinates.shape[1:] != (self.plan.atom_capacity, 3):
            raise ValueError(
                "Cross-evaluation positions must have shape (samples, atom_capacity, 3)."
            )
        if coordinates.dtype.kind != "f":
            raise TypeError("Cross-evaluation positions must be floating point.")
        lambdas = (
            self.plan.schedule.lambda_values
            if lambda_values is None
            else jnp.asarray(lambda_values, dtype=coordinates.dtype)
        )
        if lambdas.ndim != 1 or lambdas.size == 0:
            raise ValueError("Cross-evaluation lambda values must be a non-empty vector.")
        component = jax.vmap(
            lambda lam: jax.vmap(lambda coordinate: self._components(coordinate, lam))(
                coordinates
            )
        )(lambdas)
        geometry_valid = jnp.all(
            jax.vmap(
                lambda lam: jax.vmap(
                    lambda coordinate: self._bond_geometry_valid(coordinate, lam)
                )(coordinates)
            )(lambdas)
        )
        energies = jnp.sum(component, axis=-1)
        reduced = self.plan.beta * energies
        finite = jnp.all(jnp.isfinite(reduced))
        in_range = jnp.all(jnp.isfinite(lambdas) & (lambdas >= 0.0) & (lambdas <= 1.0))
        return ReducedPotentialCrossEvaluation(
            reduced,
            energies,
            component,
            lambdas,
            finite,
            finite & in_range & geometry_valid,
            self.prepared_id,
        )


__all__ = [
    "AlchemicalCycleEvaluation",
    "AlchemicalEndpointPlan",
    "AlchemicalEvaluation",
    "AlchemicalInterpolatedParameters",
    "AlchemicalPreparationEvidence",
    "AlchemicalTransformationPlan",
    "AlchemicalWorkEvaluation",
    "LambdaSchedulePlan",
    "PreparedAlchemicalTransformation",
    "ReducedPotentialCrossEvaluation",
    "SoftCorePolicy",
]
