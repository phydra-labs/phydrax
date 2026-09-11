#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rigid-motion-projected molecular normal modes and stationarity evidence."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomicStructure, AtomisticSystemPlan, AtomisticUnitSystem
from ..linalg import DenseLinearOperator, OperatorProperties
from ..linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ._derivatives import MolecularHessianResult
from ._optimization import _require_structure_matches_system
from ._units import angular_frequency_to_wavenumber


class StationaryPointKind(StrEnum):
    MINIMUM = "minimum"
    FIRST_ORDER_SADDLE = "first-order-saddle"
    HIGHER_ORDER_SADDLE = "higher-order-saddle"
    INCONCLUSIVE = "inconclusive"


class VibrationalAnalysisResult(StrictModule, NonTrainableState):
    eigenvalues: Array
    angular_frequencies: Array
    wavenumbers: Array
    imaginary_mask: Array
    normal_modes: Array
    reduced_masses: Array
    external_mode_count: int = eqx.field(static=True)
    internal_mode_count: int = eqx.field(static=True)
    external_projection_residual: Array
    successful: Array
    stationary_point: StationaryPointKind = eqx.field(static=True)
    units: AtomisticUnitSystem
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        eigenvalues: ArrayLike,
        angular_frequencies: ArrayLike,
        wavenumbers: ArrayLike,
        imaginary_mask: ArrayLike,
        normal_modes: ArrayLike,
        reduced_masses: ArrayLike,
        /,
        *,
        external_mode_count: int,
        external_projection_residual: ArrayLike,
        successful: ArrayLike,
        stationary_point: StationaryPointKind,
        units: AtomisticUnitSystem,
        plan_id: str,
    ):
        values = jnp.asarray(eigenvalues)
        frequencies = jnp.asarray(angular_frequencies, dtype=values.dtype)
        waves = jnp.asarray(wavenumbers, dtype=values.dtype)
        imaginary = jnp.asarray(imaginary_mask, dtype=bool)
        modes = jnp.asarray(normal_modes, dtype=values.dtype)
        masses = jnp.asarray(reduced_masses, dtype=values.dtype)
        count = int(values.size)
        if frequencies.shape != (count,) or waves.shape != (count,) or imaginary.shape != (count,):
            raise ValueError("Vibrational eigenvalue and frequency vectors must align.")
        if modes.ndim != 3 or modes.shape[1:] != (3, count):
            raise ValueError("normal_modes must have shape (atom_capacity, 3, mode).")
        if masses.shape != (count,):
            raise ValueError("reduced_masses must have one value per mode.")
        external = int(external_mode_count)
        if external < 0:
            raise ValueError("external_mode_count must be non-negative.")
        if not isinstance(stationary_point, StationaryPointKind):
            raise TypeError("stationary_point must be StationaryPointKind.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        residual = jnp.asarray(external_projection_residual, dtype=values.dtype).reshape(())
        successful_ = jnp.asarray(successful, dtype=bool).reshape(())
        self.eigenvalues = values
        self.angular_frequencies = frequencies
        self.wavenumbers = waves
        self.imaginary_mask = imaginary
        self.normal_modes = modes
        self.reduced_masses = masses
        self.external_mode_count = external
        self.internal_mode_count = count
        self.external_projection_residual = residual
        self.successful = successful_
        self.stationary_point = stationary_point
        self.units = units
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "vibrational-analysis-result",
                "plan": self.plan_id,
                "stationary_point": stationary_point.value,
                "external_modes": external,
                "internal_modes": count,
                "successful": bool(successful_),
                "arrays": array_tree_fingerprint(
                    {
                        "eigenvalues": np.asarray(values),
                        "angular_frequencies": np.asarray(frequencies),
                        "wavenumbers": np.asarray(waves),
                        "imaginary_mask": np.asarray(imaginary),
                        "normal_modes": np.asarray(modes),
                        "reduced_masses": np.asarray(masses),
                        "projection_residual": np.asarray(residual),
                    }
                ),
            }
        )


class VibrationalAnalysisPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    linearity_tolerance: float = eqx.field(static=True)
    projection_tolerance: float = eqx.field(static=True)
    imaginary_wavenumber_threshold: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        /,
        *,
        linearity_tolerance: float = 1.0e-8,
        projection_tolerance: float = 1.0e-8,
        imaginary_wavenumber_threshold: float = 10.0,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        values = tuple(
            float(value)
            for value in (
                linearity_tolerance,
                projection_tolerance,
                imaginary_wavenumber_threshold,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Vibrational tolerances must be finite and positive.")
        self.system = system
        (
            self.linearity_tolerance,
            self.projection_tolerance,
            self.imaginary_wavenumber_threshold,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vibrational-analysis-plan",
                "system": system.system_id,
                "linearity_tolerance": values[0],
                "projection_tolerance": values[1],
                "imaginary_wavenumber_threshold": values[2],
            }
        )

    def evaluate(
        self,
        structure: AtomicStructure,
        hessian: MolecularHessianResult,
        /,
    ) -> VibrationalAnalysisResult:
        if not isinstance(structure, AtomicStructure):
            raise TypeError("structure must be AtomicStructure.")
        if not isinstance(hessian, MolecularHessianResult):
            raise TypeError("hessian must be MolecularHessianResult.")
        _require_structure_matches_system(structure, self.system)
        if hessian.units.unit_system_id != self.system.units.unit_system_id:
            raise ValueError("Hessian and vibration unit systems differ.")
        active = np.asarray(self.system.active_mask, dtype=bool)
        if not np.array_equal(active, np.asarray(self.system.mobile_mask, dtype=bool)):
            raise ValueError(
                "Constrained or fixed-coordinate vibrational analysis is not supported."
            )
        if int(self.system.topology.constraints.shape[0]):
            raise ValueError(
                "Constrained vibrational analysis requires a projected constraint subspace."
            )
        indices = np.flatnonzero(active)
        positions = np.asarray(structure.positions)[active]
        masses = np.asarray(self.system.masses)[active]
        center = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
        centered = positions - center
        root_mass = np.sqrt(masses)
        dimension = 3 * indices.size
        external = np.zeros((dimension, 6), dtype=positions.dtype)
        axes = np.eye(3, dtype=positions.dtype)
        for axis in range(3):
            external[:, axis] = (root_mass[:, None] * axes[axis]).reshape(-1)
            external[:, 3 + axis] = (
                root_mass[:, None] * np.cross(axes[axis], centered)
            ).reshape(-1)
        left, singular, _ = np.linalg.svd(external, full_matrices=True)
        scale = float(singular[0]) if singular.size else 1.0
        external_count = int(np.count_nonzero(singular > self.linearity_tolerance * scale))
        rank_valid = (
            external_count == 3
            if indices.size == 1
            else external_count in (5, 6)
        )
        internal_basis = left[:, external_count:]
        external_basis = left[:, :external_count]
        projection_residual = float(
            np.max(np.abs(external_basis.T @ internal_basis), initial=0.0)
        )
        full_hessian = np.asarray(hessian.hessian)
        active_hessian = full_hessian[
            np.ix_(indices, np.arange(3), indices, np.arange(3))
        ].reshape((dimension, dimension))
        inverse_root_mass = np.repeat(1.0 / root_mass, 3)
        mass_weighted = (
            inverse_root_mass[:, None]
            * active_hessian
            * inverse_root_mass[None, :]
        )
        reduced = internal_basis.T @ mass_weighted @ internal_basis
        internal_count = int(reduced.shape[0])
        if internal_count:
            solve = eigensolve(
                Eigenproblem(
                    DenseLinearOperator(
                        jnp.asarray(reduced),
                        properties=OperatorProperties(
                            self_adjoint=True,
                            evidence={"self_adjoint": "construction"},
                        ),
                    )
                ),
                policy=EigenSolvePolicy(
                    DenseEigh(), count=internal_count, which="smallest-algebraic"
                ),
            )
            eigenvalues = np.asarray(solve.eigenvalues)
            reduced_vectors = np.asarray(solve.eigenvectors)
            mass_weighted_modes = internal_basis @ reduced_vectors
            cartesian = inverse_root_mass[:, None] * mass_weighted_modes
            inverse_reduced_mass = np.sum(cartesian**2, axis=0)
            reduced_masses = 1.0 / inverse_reduced_mass
            normalized = cartesian * np.sqrt(reduced_masses)[None, :]
            modes = np.zeros((active.size, 3, internal_count), dtype=positions.dtype)
            modes[active] = normalized.reshape((indices.size, 3, internal_count))
            omega_squared = eigenvalues * self.system.units.force_to_momentum_rate
            angular = np.sign(omega_squared) * np.sqrt(np.abs(omega_squared))
            wavenumbers = np.asarray(
                angular_frequency_to_wavenumber(angular, self.system.units)
            )
            eigen_successful = bool(solve.successful)
        else:
            eigenvalues = np.zeros((0,), dtype=positions.dtype)
            angular = np.zeros((0,), dtype=positions.dtype)
            wavenumbers = np.zeros((0,), dtype=positions.dtype)
            reduced_masses = np.zeros((0,), dtype=positions.dtype)
            modes = np.zeros((active.size, 3, 0), dtype=positions.dtype)
            eigen_successful = True
        imaginary = wavenumbers < -self.imaginary_wavenumber_threshold
        imaginary_count = int(np.count_nonzero(imaginary))
        successful = (
            bool(hessian.successful)
            and eigen_successful
            and rank_valid
            and projection_residual <= self.projection_tolerance
            and np.all(np.isfinite(eigenvalues))
        )
        if not successful:
            stationary = StationaryPointKind.INCONCLUSIVE
        elif imaginary_count == 0:
            stationary = StationaryPointKind.MINIMUM
        elif imaginary_count == 1:
            stationary = StationaryPointKind.FIRST_ORDER_SADDLE
        else:
            stationary = StationaryPointKind.HIGHER_ORDER_SADDLE
        return VibrationalAnalysisResult(
            eigenvalues,
            angular,
            wavenumbers,
            imaginary,
            modes,
            reduced_masses,
            external_mode_count=external_count,
            external_projection_residual=projection_residual,
            successful=successful,
            stationary_point=stationary,
            units=self.system.units,
            plan_id=self.plan_id,
        )


__all__ = [
    "StationaryPointKind",
    "VibrationalAnalysisPlan",
    "VibrationalAnalysisResult",
]
