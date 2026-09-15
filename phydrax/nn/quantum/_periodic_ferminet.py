#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resource-admitted finite periodic determinant amplitude with twist covariance."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ...discretization import PeriodicCell
from ...linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ...operators.quantum._amplitude import LogAmplitude
from ...operators.quantum._electronic_advanced import ElectronicVMCResourcePlan
from ...operators.quantum._periodic_electronic import (
    AbstractPeriodicElectronicAmplitude,
)
from ._periodic_features import PeriodicCellFeatures


class PeriodicFermiNet(AbstractPeriodicElectronicAmplitude):
    """Finite physical-cell determinant amplitude with twist covariance.

    Cartesian electron coordinates are transformed by ``PeriodicCellFeatures``.
    The reciprocal basis is finite and integer-indexed, and the metric-aware
    pair stream uses a stopped minimum-image selection. This remains a bounded
    Slater/Jastrow amplitude, not a thermodynamic-limit architecture.
    """

    cell_features: PeriodicCellFeatures
    orbital_coefficients: Array
    determinant_coefficients: Array
    pair_jastrow_strength: Array
    resource_plan: ElectronicVMCResourcePlan
    electron_count: int = eqx.field(static=True)
    determinant_count: int = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    network_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        reciprocal_modes: ArrayLike,
        orbital_coefficients: ArrayLike,
        determinant_coefficients: ArrayLike,
        /,
        *,
        twist: ArrayLike,
        pair_jastrow_strength: ArrayLike = 0.0,
        resource_plan: ElectronicVMCResourcePlan,
    ):
        features = PeriodicCellFeatures(cell, reciprocal_modes, twist=twist)
        coefficients = jnp.asarray(orbital_coefficients)
        mixing = jnp.asarray(determinant_coefficients)
        if (
            coefficients.ndim != 3
            or coefficients.shape[2] != features.reciprocal_mode_count
        ):
            raise ValueError(
                "orbital_coefficients require (determinants,electrons,modes)."
            )
        determinants, electrons = map(int, coefficients.shape[:2])
        if mixing.shape != (determinants,):
            raise ValueError(
                "determinant_coefficients must have one entry per determinant."
            )
        if not isinstance(resource_plan, ElectronicVMCResourcePlan):
            raise TypeError("resource_plan must be ElectronicVMCResourcePlan.")
        if (
            resource_plan.electron_count != electrons
            or resource_plan.determinant_count != determinants
            or resource_plan.coordinate_dimension
            != electrons * features.ambient_dimension
        ):
            raise ValueError(
                "PeriodicFermiNet counts and physical dimension must match resource_plan."
            )
        strength = jnp.asarray(pair_jastrow_strength)
        if strength.shape != () or not bool(np.isfinite(np.asarray(strength))):
            raise ValueError("pair_jastrow_strength must be a finite scalar.")
        dtype = jnp.result_type(coefficients.dtype, mixing.dtype, 1j)
        self.cell_features = features
        self.orbital_coefficients = coefficients.astype(dtype)
        self.determinant_coefficients = mixing.astype(dtype)
        self.pair_jastrow_strength = strength
        self.resource_plan = resource_plan
        self.electron_count = electrons
        self.determinant_count = determinants
        self.spatial_dimension = features.ambient_dimension
        self.boundary_id = features.boundary_id
        self.network_id = canonical_fingerprint(
            {
                "kind": "periodic-ferminet",
                "boundary": features.boundary_id,
                "modes": features.reciprocal_mode_count,
                "electrons": electrons,
                "determinants": determinants,
                "spatial_dimension": features.ambient_dimension,
            }
        )
        self.claim = "finite-physical-cell-reciprocal-determinant-amplitude"

    @property
    def cell(self) -> PeriodicCell:
        return self.cell_features.cell

    @property
    def reciprocal_modes(self) -> Array:
        return self.cell_features.reciprocal_modes

    @property
    def twist(self) -> Array:
        return self.cell_features.twist

    @property
    def configuration_shape(self) -> tuple[int, int]:
        return (self.electron_count, self.spatial_dimension)

    def _single(self, cartesian_coordinates: Array, /) -> LogAmplitude:
        features = self.cell_features(cartesian_coordinates)
        determinant_logs = []
        determinant_phases = []
        for coefficients in self.orbital_coefficients:
            orbitals = contract("im,jm->ij", features.reciprocal_features, coefficients)
            prepared = factorize(
                DenseLinearOperator(orbitals),
                FactorizationPolicy("lu"),
            )
            determinant_logs.append(prepared.log_abs_determinant())
            determinant_phases.append(prepared.determinant_sign())
        logs = jnp.stack(determinant_logs)
        determinant_phases_ = jnp.stack(determinant_phases)
        reference = jnp.max(logs)
        mixture = jnp.sum(
            self.determinant_coefficients
            * determinant_phases_
            * jnp.exp(logs - reference)
        )
        pair_mask = jnp.triu(
            jnp.ones((self.electron_count, self.electron_count), dtype=bool),
            k=1,
        )
        jastrow = self.pair_jastrow_strength * jnp.sum(
            jnp.where(pair_mask, features.pair_distances, 0.0)
        )
        magnitude = jnp.abs(mixture)
        phase = jnp.where(
            magnitude > 0.0,
            mixture / magnitude * features.twist_phase,
            1.0 + 0.0j,
        )
        log_abs = reference + jnp.log(magnitude) + jnp.real(jastrow)
        valid = (
            self.resource_plan.valid
            & features.valid
            & jnp.isfinite(log_abs)
            & (magnitude > 0.0)
        )
        return LogAmplitude(log_abs, phase, valid=valid)

    def __call__(self, cartesian_coordinates: ArrayLike, /) -> LogAmplitude:
        """Evaluate one configuration or arbitrary fixed-shape walker batches."""
        coordinates = jnp.asarray(cartesian_coordinates)
        if (
            coordinates.ndim < 2
            or tuple(coordinates.shape[-2:]) != self.configuration_shape
        ):
            raise ValueError(
                "PeriodicFermiNet inputs must end in physical Cartesian shape "
                f"{self.configuration_shape}; got {coordinates.shape}."
            )
        if coordinates.ndim == 2:
            return self._single(coordinates)
        batch_shape = tuple(int(size) for size in coordinates.shape[:-2])
        count = math.prod(batch_shape)
        values = jax.vmap(self._single)(
            coordinates.reshape((count,) + self.configuration_shape)
        )
        return LogAmplitude(
            values.log_abs.reshape(batch_shape),
            values.phase.reshape(batch_shape),
            valid=values.valid.reshape(batch_shape),
        )


__all__ = ["PeriodicFermiNet"]
