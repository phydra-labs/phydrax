#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit correlated orbital partitions and molecular-orbital integral stores."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....ein import contract
from ....operators.quantum.gaussian import (
    electron_repulsion_tensor,
    FactorizedERITensor,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion_energy,
    PreparedGaussianBasis,
)
from .._mean_field import RestrictedMeanFieldState


class CorrelatedOrbitalPartition(StrictModule, NonTrainableState):
    frozen_occupied: tuple[int, ...] = eqx.field(static=True)
    correlated_occupied: tuple[int, ...] = eqx.field(static=True)
    active: tuple[int, ...] = eqx.field(static=True)
    external_virtual: tuple[int, ...] = eqx.field(static=True)
    frozen_virtual: tuple[int, ...] = eqx.field(static=True)
    orbital_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        orbital_count: int,
        occupied_indices: Sequence[int],
        /,
        *,
        frozen_occupied: Sequence[int] = (),
        active: Sequence[int] = (),
        frozen_virtual: Sequence[int] = (),
    ):
        count = int(orbital_count)
        occupied = tuple(sorted(int(value) for value in occupied_indices))
        frozen_occ = tuple(sorted(int(value) for value in frozen_occupied))
        active_ = tuple(sorted(int(value) for value in active))
        frozen_virt = tuple(sorted(int(value) for value in frozen_virtual))
        groups = (occupied, frozen_occ, active_, frozen_virt)
        if count <= 0 or any(
            len(set(group)) != len(group)
            or any(index < 0 or index >= count for index in group)
            for group in groups
        ):
            raise ValueError(
                "Correlated orbital indices must be unique and within capacity."
            )
        occupied_set = set(occupied)
        if not set(frozen_occ) <= occupied_set:
            raise ValueError("Frozen occupied orbitals must be occupied.")
        virtual_set = set(range(count)) - occupied_set
        if not set(frozen_virt) <= virtual_set:
            raise ValueError("Frozen virtual orbitals must be virtual.")
        if set(active_) & (set(frozen_occ) | set(frozen_virt)):
            raise ValueError("Active orbitals cannot also be frozen.")
        correlated = tuple(index for index in occupied if index not in set(frozen_occ))
        external = tuple(
            index
            for index in range(count)
            if index not in occupied_set
            and index not in set(frozen_virt)
            and index not in set(active_)
        )
        self.frozen_occupied = frozen_occ
        self.correlated_occupied = correlated
        self.active = active_
        self.external_virtual = external
        self.frozen_virtual = frozen_virt
        self.orbital_count = count
        self.partition_id = canonical_fingerprint(
            {
                "kind": "correlated-orbital-partition",
                "orbital_count": count,
                "frozen_occupied": list(frozen_occ),
                "correlated_occupied": list(correlated),
                "active": list(active_),
                "external_virtual": list(external),
                "frozen_virtual": list(frozen_virt),
            }
        )

    @classmethod
    def from_restricted_state(
        cls,
        state: RestrictedMeanFieldState,
        /,
        *,
        frozen_occupied_count: int = 0,
        active: Sequence[int] = (),
        frozen_virtual: Sequence[int] = (),
    ) -> CorrelatedOrbitalPartition:
        if not isinstance(state, RestrictedMeanFieldState):
            raise TypeError("state must be RestrictedMeanFieldState.")
        occupied = tuple(np.flatnonzero(np.asarray(state.occupations) > 1.0))
        frozen_count = int(frozen_occupied_count)
        if frozen_count < 0 or frozen_count > len(occupied):
            raise ValueError("frozen_occupied_count exceeds occupied orbitals.")
        return cls(
            state.coefficients.shape[1],
            occupied,
            frozen_occupied=occupied[:frozen_count],
            active=active,
            frozen_virtual=frozen_virtual,
        )


class MolecularOrbitalIntegralStore(StrictModule, NonTrainableState):
    one_body: Array
    two_body: Array | None
    factorized: FactorizedERITensor | None
    orbital_energies: Array
    coefficients: Array
    nuclear_repulsion: Array
    reference_energy: Array
    partition: CorrelatedOrbitalPartition
    source_basis_id: str = eqx.field(static=True)
    store_id: str = eqx.field(static=True)

    def __init__(
        self,
        one_body: ArrayLike,
        orbital_energies: ArrayLike,
        coefficients: ArrayLike,
        nuclear_repulsion: ArrayLike,
        reference_energy: ArrayLike,
        partition: CorrelatedOrbitalPartition,
        source_basis_id: str,
        /,
        *,
        two_body: ArrayLike | None = None,
        factorized: FactorizedERITensor | None = None,
    ):
        one = jnp.asarray(one_body)
        energies = jnp.asarray(orbital_energies, dtype=one.real.dtype)
        coefficients_ = jnp.asarray(coefficients, dtype=one.dtype)
        nuclear = jnp.asarray(nuclear_repulsion, dtype=one.real.dtype).reshape(())
        reference = jnp.asarray(reference_energy, dtype=one.real.dtype).reshape(())
        source = str(source_basis_id).strip()
        two = None if two_body is None else jnp.asarray(two_body, dtype=one.dtype)
        if (two is None) == (factorized is None):
            raise ValueError(
                "Provide exactly one dense or factorized two-electron representation."
            )
        count = one.shape[0] if one.ndim == 2 else -1
        if (
            one.shape != (count, count)
            or energies.shape != (count,)
            or coefficients_.ndim != 2
            or coefficients_.shape[1] != count
            or partition.orbital_count != count
            or not source
        ):
            raise ValueError("MO integral arrays and orbital partition do not align.")
        if two is not None and two.shape != (count, count, count, count):
            raise ValueError("Dense MO ERIs must have shape (N,N,N,N).")
        if factorized is not None and factorized.orbital_count != count:
            raise ValueError("Factorized MO ERIs do not align with orbitals.")
        self.one_body = one
        self.two_body = two
        self.factorized = factorized
        self.orbital_energies = energies
        self.coefficients = coefficients_
        self.nuclear_repulsion = nuclear
        self.reference_energy = reference
        self.partition = partition
        self.source_basis_id = source
        self.store_id = canonical_fingerprint(
            {
                "kind": "molecular-orbital-integral-store",
                "source_basis": source,
                "partition": partition.partition_id,
                "factorized": None if factorized is None else factorized.tensor_id,
                "arrays": array_tree_fingerprint(
                    {
                        "one_body": np.asarray(one),
                        "two_body": None if two is None else np.asarray(two),
                        "orbital_energies": np.asarray(energies),
                        "coefficients": np.asarray(coefficients_),
                        "nuclear_repulsion": np.asarray(nuclear),
                        "reference_energy": np.asarray(reference),
                    }
                ),
            }
        )

    def dense_two_body(self, /) -> Array:
        if self.two_body is not None:
            return self.two_body
        if self.factorized is None:
            raise RuntimeError(
                "MO integral store lost both two-electron representations."
            )
        return self.factorized.reconstruct()


class MolecularIntegralTransformationPlan(StrictModule, NonTrainableState):
    maximum_dense_orbitals: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_dense_orbitals: int = 32, /):
        maximum = int(maximum_dense_orbitals)
        if maximum <= 0:
            raise ValueError("maximum_dense_orbitals must be positive.")
        self.maximum_dense_orbitals = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-integral-transformation",
                "maximum_dense_orbitals": maximum,
            }
        )

    def transform_restricted(
        self,
        basis: PreparedGaussianBasis,
        positions: ArrayLike,
        nuclear_charges: ArrayLike,
        state: RestrictedMeanFieldState,
        partition: CorrelatedOrbitalPartition,
        /,
        *,
        factorized_ao: FactorizedERITensor | None = None,
    ) -> MolecularOrbitalIntegralStore:
        if state.coefficients.shape[1] != partition.orbital_count:
            raise ValueError("Mean-field state and correlated partition do not align.")
        coordinate = jnp.asarray(positions)
        charges = jnp.asarray(nuclear_charges, dtype=coordinate.dtype)
        core_ao = kinetic_matrix(basis, coordinate) + nuclear_attraction_matrix(
            basis, coordinate, charges
        )
        coefficients = state.coefficients
        one = contract("pi,pq,qj->ij", jnp.conj(coefficients), core_ao, coefficients)
        nuclear = nuclear_repulsion_energy(coordinate, charges)
        if factorized_ao is None:
            if partition.orbital_count > self.maximum_dense_orbitals:
                raise ValueError("Dense MO transformation exceeds its orbital capacity.")
            eri_ao = electron_repulsion_tensor(basis, coordinate)
            two = contract(
                "pi,qj,rk,sl,pqrs->ijkl",
                jnp.conj(coefficients),
                coefficients,
                jnp.conj(coefficients),
                coefficients,
                eri_ao,
            )
            return MolecularOrbitalIntegralStore(
                one,
                state.orbital_energies,
                coefficients,
                nuclear,
                state.total_energy,
                partition,
                basis.prepared_id,
                two_body=two,
            )
        transformed_factors = contract(
            "Ppq,pi,qj->Pij",
            factorized_ao.factors,
            jnp.conj(coefficients),
            coefficients,
        )
        factorized_mo = FactorizedERITensor(
            transformed_factors,
            factorized_ao.residual_bound,
            factorized_ao.tensor_id,
            factorized_ao.representation + "-mo",
        )
        return MolecularOrbitalIntegralStore(
            one,
            state.orbital_energies,
            coefficients,
            nuclear,
            state.total_energy,
            partition,
            basis.prepared_id,
            factorized=factorized_mo,
        )


__all__ = [
    "CorrelatedOrbitalPartition",
    "MolecularIntegralTransformationPlan",
    "MolecularOrbitalIntegralStore",
]
