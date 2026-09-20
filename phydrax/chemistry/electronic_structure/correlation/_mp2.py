#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Restricted molecular MP2 with explicit spin-component and denominator evidence."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._orbital import MolecularOrbitalIntegralStore


class MP2Result(StrictModule, NonTrainableState):
    correlation_energy: Array
    opposite_spin_energy: Array
    same_spin_energy: Array
    total_energy: Array
    amplitudes: Array
    minimum_denominator: Array
    residual_bound: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    store_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        correlation_energy,
        opposite_spin_energy,
        same_spin_energy,
        total_energy,
        amplitudes,
        minimum_denominator,
        residual_bound,
        successful,
        plan_id: str,
        store_id: str,
        /,
    ):
        amplitude = jnp.asarray(amplitudes)
        dtype = amplitude.real.dtype
        self.correlation_energy = jnp.asarray(correlation_energy, dtype=dtype).reshape(())
        self.opposite_spin_energy = jnp.asarray(
            opposite_spin_energy, dtype=dtype
        ).reshape(())
        self.same_spin_energy = jnp.asarray(same_spin_energy, dtype=dtype).reshape(())
        self.total_energy = jnp.asarray(total_energy, dtype=dtype).reshape(())
        self.amplitudes = amplitude
        self.minimum_denominator = jnp.asarray(minimum_denominator, dtype=dtype).reshape(
            ()
        )
        self.residual_bound = jnp.asarray(residual_bound, dtype=dtype).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.plan_id = str(plan_id)
        self.store_id = str(store_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "mp2-result",
                "plan": self.plan_id,
                "store": self.store_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "correlation_energy": np.asarray(self.correlation_energy),
                        "opposite_spin_energy": np.asarray(self.opposite_spin_energy),
                        "same_spin_energy": np.asarray(self.same_spin_energy),
                        "total_energy": np.asarray(self.total_energy),
                        "amplitudes": np.asarray(amplitude),
                        "minimum_denominator": np.asarray(self.minimum_denominator),
                        "residual_bound": np.asarray(self.residual_bound),
                    }
                ),
            }
        )


class MP2Plan(StrictModule, NonTrainableState):
    opposite_spin_scale: float = eqx.field(static=True)
    same_spin_scale: float = eqx.field(static=True)
    denominator_tolerance: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        opposite_spin_scale: float = 1.0,
        same_spin_scale: float = 1.0,
        denominator_tolerance: float = 1.0e-8,
        regularization: float = 0.0,
    ):
        values = tuple(
            float(value)
            for value in (
                opposite_spin_scale,
                same_spin_scale,
                denominator_tolerance,
                regularization,
            )
        )
        if (
            any(not isfinite(value) for value in values)
            or values[0] < 0.0
            or values[1] < 0.0
            or values[2] <= 0.0
            or values[3] < 0.0
        ):
            raise ValueError("MP2 scaling, denominator, or regularization is invalid.")
        (
            self.opposite_spin_scale,
            self.same_spin_scale,
            self.denominator_tolerance,
            self.regularization,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mp2-plan",
                "opposite_spin_scale": values[0],
                "same_spin_scale": values[1],
                "denominator_tolerance": values[2],
                "regularization": values[3],
            }
        )

    def evaluate(self, integrals: MolecularOrbitalIntegralStore, /) -> MP2Result:
        if not isinstance(integrals, MolecularOrbitalIntegralStore):
            raise TypeError("integrals must be MolecularOrbitalIntegralStore.")
        occupied = integrals.partition.correlated_occupied
        occupied_set = set(occupied)
        virtual = tuple(
            sorted(
                set(integrals.partition.external_virtual)
                | {
                    value
                    for value in integrals.partition.active
                    if value not in occupied_set
                }
            )
        )
        if not occupied or not virtual:
            raise ValueError("MP2 requires correlated occupied and virtual orbitals.")
        eri = integrals.dense_two_body()
        energies = integrals.orbital_energies
        i = jnp.asarray(occupied, dtype=jnp.int32)
        a = jnp.asarray(virtual, dtype=jnp.int32)
        denominator = (
            energies[i][:, None, None, None]
            + energies[i][None, :, None, None]
            - energies[a][None, None, :, None]
            - energies[a][None, None, None, :]
        )
        iajb = eri[jnp.ix_(i, a, i, a)].transpose((0, 2, 1, 3))
        ibja = jnp.swapaxes(iajb, 2, 3)
        minimum = jnp.min(jnp.abs(denominator))
        if self.regularization:
            absolute_denominator = jnp.abs(denominator)
            damping = (1.0 - jnp.exp(-self.regularization * absolute_denominator)) ** 2
            inverse = jnp.where(
                absolute_denominator > 0.0,
                damping / jnp.where(absolute_denominator > 0.0, denominator, 1.0),
                0.0,
            )
        else:
            inverse = jnp.reciprocal(denominator)
        amplitudes = iajb * inverse
        opposite = jnp.sum(iajb * iajb * inverse)
        same = jnp.sum((iajb - ibja) * iajb * inverse)
        correlation = self.opposite_spin_scale * opposite + self.same_spin_scale * same
        residual_bound = (
            jnp.asarray(0.0, dtype=correlation.dtype)
            if integrals.factorized is None
            else integrals.factorized.residual_bound
        )
        successful = (
            jnp.all(jnp.isfinite(amplitudes))
            & jnp.isfinite(correlation)
            & ((minimum >= self.denominator_tolerance) | (self.regularization > 0.0))
        )
        return MP2Result(
            correlation,
            opposite,
            same,
            integrals.reference_energy + correlation,
            amplitudes,
            minimum,
            residual_bound,
            successful,
            self.plan_id,
            integrals.store_id,
        )


__all__ = ["MP2Plan", "MP2Result"]
