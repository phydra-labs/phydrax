#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Intrinsic three-phonon single-mode RTA from canonical provider IFC3."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticUnitSystem
from ...units import UnitDefinition
from ._lattice_force_constants import (
    third_order_force_constant_unit,
    ThirdOrderForceConstants,
)


class ThreePhononScatteringResult(StrictModule, NonTrainableState):
    decay_rates: Array
    coalescence_rates: Array
    total_rates: Array
    momentum_residual: Array
    detailed_balance_residual: Array
    decay_channel_count: int = eqx.field(static=True)
    coalescence_channel_count: int = eqx.field(static=True)
    successful: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        decay,
        coalescence,
        momentum,
        detailed_balance,
        decay_count,
        coalescence_count,
        successful,
        /,
    ):
        self.decay_rates = jnp.asarray(decay)
        self.coalescence_rates = jnp.asarray(coalescence, dtype=self.decay_rates.dtype)
        self.total_rates = self.decay_rates + self.coalescence_rates
        self.momentum_residual = jnp.asarray(
            momentum, dtype=self.decay_rates.dtype
        ).reshape(())
        self.detailed_balance_residual = jnp.asarray(
            detailed_balance, dtype=self.decay_rates.dtype
        ).reshape(())
        self.decay_channel_count = int(decay_count)
        self.coalescence_channel_count = int(coalescence_count)
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.result_id = canonical_fingerprint(
            {
                "kind": "three-phonon-scattering-result",
                "decay_channels": self.decay_channel_count,
                "coalescence_channels": self.coalescence_channel_count,
                "arrays": array_tree_fingerprint(
                    {
                        "decay": np.asarray(self.decay_rates),
                        "coalescence": np.asarray(self.coalescence_rates),
                        "momentum": np.asarray(self.momentum_residual),
                        "detailed_balance": np.asarray(self.detailed_balance_residual),
                    }
                ),
            }
        )


class ThreePhononModeVertices(StrictModule, NonTrainableState):
    """IFC3-derived decay/coalescence vertices on one exact phonon mesh."""

    decay_vertices: Array
    coalescence_vertices: Array
    angular_frequencies: Array
    qpoint_indices: Array
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    energy_unit: UnitDefinition
    ifc3_id: str = eqx.field(static=True)
    phonon_result_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    vertex_id: str = eqx.field(static=True)

    def __init__(
        self,
        decay_vertices,
        coalescence_vertices,
        angular_frequencies,
        qpoint_indices,
        mesh_shape,
        energy_unit,
        ifc3_id,
        phonon_result_id,
        /,
    ):
        decay = np.asarray(decay_vertices)
        coalescence = np.asarray(coalescence_vertices)
        frequency = np.asarray(angular_frequencies)
        indices = np.asarray(qpoint_indices)
        qpoints, branches = frequency.shape
        expected = (qpoints, branches, qpoints, branches, branches)
        if (
            decay.shape != expected
            or coalescence.shape != expected
            or indices.shape != (qpoints, len(mesh_shape))
            or np.any(~np.isfinite(decay))
            or np.any(~np.isfinite(coalescence))
            or np.any(~np.isfinite(frequency))
            or np.any(frequency <= 0.0)
        ):
            raise ValueError("Three-phonon vertex arrays or mesh axes are invalid.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        self.decay_vertices = jnp.asarray(decay)
        self.coalescence_vertices = jnp.asarray(coalescence)
        self.angular_frequencies = jnp.asarray(frequency)
        self.qpoint_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.mesh_shape = tuple(mesh_shape)
        self.energy_unit = energy_unit
        self.ifc3_id = str(ifc3_id)
        self.phonon_result_id = str(phonon_result_id)
        self.convention_id = "mass-weighted-eigenvectors-columns;decay=(-q1,q2,q1-q2);coalescence=(q1,q2,-q1-q2);sqrt-hbar-over-2Momega"
        self.vertex_id = canonical_fingerprint(
            {
                "kind": "three-phonon-mode-vertices",
                "ifc3": self.ifc3_id,
                "phonon": self.phonon_result_id,
                "mesh_shape": list(self.mesh_shape),
                "unit": energy_unit.unit_id,
                "convention": self.convention_id,
                "arrays": array_tree_fingerprint(
                    {
                        "decay": decay,
                        "coalescence": coalescence,
                        "frequency": frequency,
                        "indices": indices,
                    }
                ),
            }
        )


class IFC3ModeVertexPlan(StrictModule, NonTrainableState):
    """Bounded host transform from canonical real-space IFC3 to mode vertices."""

    ifc3: ThirdOrderForceConstants
    qpoint_indices: Array
    fractional_qpoints: Array
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    angular_frequencies: Array
    eigenvectors: Array
    masses: Array
    units: AtomisticUnitSystem
    phonon_result_id: str = eqx.field(static=True)
    maximum_channels: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ifc3: ThirdOrderForceConstants,
        qpoint_indices: ArrayLike,
        mesh_shape: tuple[int, ...],
        fractional_qpoints: ArrayLike,
        angular_frequencies: ArrayLike,
        eigenvectors: ArrayLike,
        masses: ArrayLike,
        units: AtomisticUnitSystem,
        /,
        *,
        phonon_result_id: str,
        maximum_channels: int = 50_000_000,
        maximum_bytes: int = 2_147_483_648,
    ):
        if not isinstance(ifc3, ThirdOrderForceConstants):
            raise TypeError("ifc3 must be ThirdOrderForceConstants.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        expected_unit = third_order_force_constant_unit(
            units.scale.energy_unit, units.scale.length_unit
        )
        if ifc3.unit.unit_id != expected_unit.unit_id:
            raise ValueError("IFC3 unit differs from the phonon unit system.")
        indices = np.asarray(qpoint_indices)
        shape = tuple(mesh_shape)
        q = np.asarray(fractional_qpoints, dtype=np.float64)
        frequency = np.asarray(angular_frequencies, dtype=np.float64)
        vectors = np.asarray(eigenvectors)
        mass = np.asarray(masses, dtype=np.float64)
        qpoints, branches = frequency.shape if frequency.ndim == 2 else (-1, -1)
        if (
            len(shape) not in (1, 2, 3)
            or any(value <= 0 for value in shape)
            or indices.shape != (qpoints, len(shape))
            or not np.issubdtype(indices.dtype, np.integer)
            or ifc3.translations.shape[2] != len(shape)
            or np.any(indices < 0)
            or np.any(indices >= np.asarray(shape)[None, :])
            or q.shape != (qpoints, len(shape))
            or vectors.shape != (qpoints, branches, branches)
            or branches != 3 * ifc3.atom_count
            or mass.shape != (ifc3.atom_count,)
            or np.any(mass <= 0.0)
            or np.any(frequency <= 0.0)
            or np.any(~np.isfinite(frequency))
            or np.any(~np.isfinite(vectors))
        ):
            raise ValueError(
                "IFC3 mode transform axes, masses, or stable modes are invalid."
            )
        if len({tuple(row) for row in indices}) != int(np.prod(shape)):
            raise ValueError("IFC3 mode transform requires one complete regular mesh.")
        expected_q = indices / np.asarray(shape)[None, :]
        if not np.allclose(np.mod(q, 1.0), expected_q, atol=1.0e-12):
            raise ValueError(
                "Fractional q points do not match their regular-mesh addresses."
            )
        channels = 2 * qpoints * qpoints * branches**3
        bytes_required = (
            channels
            * np.dtype(
                np.result_type(vectors.dtype, np.asarray(ifc3.values).dtype)
            ).itemsize
        )
        if channels > int(maximum_channels) or bytes_required > int(maximum_bytes):
            raise ValueError("IFC3 mode-vertex capacity exceeded before allocation.")
        self.ifc3 = ifc3
        self.qpoint_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.fractional_qpoints = jnp.asarray(q)
        self.mesh_shape = shape
        self.angular_frequencies = jnp.asarray(frequency)
        self.eigenvectors = jnp.asarray(vectors)
        self.masses = jnp.asarray(mass)
        self.units = units
        self.phonon_result_id = str(phonon_result_id)
        if not self.phonon_result_id:
            raise ValueError("phonon_result_id must be non-empty.")
        self.maximum_channels = int(maximum_channels)
        self.maximum_bytes = int(maximum_bytes)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ifc3-mode-vertex-plan",
                "ifc3": ifc3.ifc_id,
                "phonon": self.phonon_result_id,
                "units": units.unit_system_id,
                "mesh_shape": list(shape),
                "maximum_channels": self.maximum_channels,
                "maximum_bytes": self.maximum_bytes,
                "arrays": array_tree_fingerprint(
                    {
                        "indices": indices,
                        "q": q,
                        "frequency": frequency,
                        "eigenvectors": vectors,
                        "masses": mass,
                    }
                ),
            }
        )

    def evaluate(self, /) -> ThreePhononModeVertices:
        frequency = np.asarray(self.angular_frequencies)
        vectors = np.asarray(self.eigenvectors)
        q = np.asarray(self.fractional_qpoints)
        indices = np.asarray(self.qpoint_indices)
        shape = np.asarray(self.mesh_shape)
        qpoints, branches = frequency.shape
        atom_count = self.ifc3.atom_count
        lookup = {tuple(row): index for index, row in enumerate(indices)}
        polarization = vectors.reshape((qpoints, atom_count, 3, branches))
        amplitude = np.sqrt(
            self.units.reduced_planck_constant
            / (2.0 * self.units.kinetic_to_energy * frequency)
        )
        modes = (
            polarization
            / np.sqrt(np.asarray(self.masses))[None, :, None, None]
            * amplitude[:, None, None, :]
        )
        triplets = np.asarray(self.ifc3.atom_triplets)
        translations = np.asarray(self.ifc3.translations)
        values = np.asarray(self.ifc3.values)
        valid = np.asarray(self.ifc3.valid)
        triplets = triplets[valid]
        translations = translations[valid]
        values = values[valid]
        decay = np.zeros(
            (qpoints, branches, qpoints, branches, branches),
            dtype=np.result_type(values, vectors, np.complex64),
        )
        coalescence = np.zeros_like(decay)
        normalization = 1.0 / np.sqrt(qpoints)
        for q1 in range(qpoints):
            first = modes[q1, triplets[:, 0]]
            for q2 in range(qpoints):
                q_decay = lookup[tuple(np.mod(indices[q1] - indices[q2], shape))]
                q_coalescence = lookup[tuple(np.mod(indices[q1] + indices[q2], shape))]
                decay_phase = np.exp(
                    2.0j
                    * np.pi
                    * (translations[:, 0] @ q[q2] + translations[:, 1] @ q[q_decay])
                )
                coalescence_phase = np.exp(
                    2.0j
                    * np.pi
                    * (translations[:, 0] @ q[q2] - translations[:, 1] @ q[q_coalescence])
                )
                second = modes[q2, triplets[:, 1]]
                decay_third = modes[q_decay, triplets[:, 2]]
                coalescence_third = modes[q_coalescence, triplets[:, 2]]
                decay[q1, :, q2, :, :] = normalization * ein.contract(
                    "eabc,eam,ebn,eco,e->mno",
                    values,
                    np.conj(first),
                    second,
                    decay_third,
                    decay_phase,
                )
                coalescence[q1, :, q2, :, :] = normalization * ein.contract(
                    "eabc,eam,ebn,eco,e->mno",
                    values,
                    first,
                    second,
                    np.conj(coalescence_third),
                    coalescence_phase,
                )
        return ThreePhononModeVertices(
            decay,
            coalescence,
            frequency,
            indices,
            self.mesh_shape,
            self.units.scale.energy_unit,
            self.ifc3.ifc_id,
            self.phonon_result_id,
        )


class ThreePhononRTAResult(StrictModule, NonTrainableState):
    scattering: ThreePhononScatteringResult
    lifetimes: Array
    mode_conductivity: Array
    thermal_conductivity: Array
    mean_free_paths: Array
    ballistic_mask: Array
    conductivity_antisymmetry_residual: Array
    conductivity_eigenvalues: Array
    finite_conductivity: Array
    successful: Array
    unit_system_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        scattering,
        lifetimes,
        mode_conductivity,
        conductivity,
        mean_free_paths,
        ballistic,
        antisymmetry,
        eigenvalues,
        finite_conductivity,
        successful,
        unit_system_id,
        /,
    ):
        self.scattering = scattering
        self.lifetimes = jnp.asarray(lifetimes)
        self.mode_conductivity = jnp.asarray(
            mode_conductivity, dtype=self.lifetimes.dtype
        )
        self.thermal_conductivity = jnp.asarray(
            conductivity, dtype=self.lifetimes.dtype
        ).reshape((3, 3))
        self.mean_free_paths = jnp.asarray(mean_free_paths, dtype=self.lifetimes.dtype)
        self.ballistic_mask = jnp.asarray(ballistic, dtype=jnp.bool_)
        self.conductivity_antisymmetry_residual = jnp.asarray(
            antisymmetry, dtype=self.lifetimes.dtype
        ).reshape(())
        self.conductivity_eigenvalues = jnp.asarray(
            eigenvalues, dtype=self.lifetimes.dtype
        ).reshape((3,))
        self.finite_conductivity = jnp.asarray(
            finite_conductivity, dtype=jnp.bool_
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.unit_system_id = str(unit_system_id)
        self.approximation = "intrinsic-three-phonon-single-mode-rta"
        self.result_id = canonical_fingerprint(
            {
                "kind": "three-phonon-rta-result",
                "scattering": scattering.result_id,
                "units": self.unit_system_id,
                "approximation": self.approximation,
                "arrays": array_tree_fingerprint(
                    {
                        "lifetimes": np.asarray(self.lifetimes),
                        "conductivity": np.asarray(self.thermal_conductivity),
                        "ballistic": np.asarray(self.ballistic_mask),
                    }
                ),
            }
        )


class ThreePhononRTAPlan(StrictModule, NonTrainableState):
    """Fixed momentum routes, Bose channels, and normalized Gaussian delta."""

    ifc3: ThirdOrderForceConstants
    qpoint_indices: Array
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    qpoint_weights: Array
    broadening: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    primitive_volume: float = eqx.field(static=True)
    units: AtomisticUnitSystem
    maximum_channels: int = eqx.field(static=True)
    detailed_balance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ifc3: ThirdOrderForceConstants,
        qpoint_indices: ArrayLike,
        mesh_shape: tuple[int, ...],
        qpoint_weights: ArrayLike,
        broadening: float,
        temperature: float,
        primitive_volume: float,
        units: AtomisticUnitSystem,
        /,
        *,
        maximum_channels: int = 50_000_000,
        detailed_balance_tolerance: float = 1.0e-5,
    ):
        if not isinstance(ifc3, ThirdOrderForceConstants):
            raise TypeError("ifc3 must be canonical ThirdOrderForceConstants.")
        if not ifc3.source_kind.startswith("provider-"):
            raise ValueError(
                "Production RTA requires provider-normalized IFC3; native IFC3 remains candidate."
            )
        if not bool(ifc3.constraints_successful):
            raise ValueError(
                "Production RTA requires IFC3 permutation and acoustic constraints."
            )
        indices = np.asarray(qpoint_indices)
        shape = tuple(mesh_shape)
        weights = np.asarray(qpoint_weights, dtype=np.float64)
        if (
            indices.ndim != 2
            or not np.issubdtype(indices.dtype, np.integer)
            or indices.shape[1] != len(shape)
            or len(shape) not in (1, 2, 3)
        ):
            raise ValueError("qpoint_indices must be integer (Q,rank) mesh addresses.")
        if (
            any(value <= 0 for value in shape)
            or np.any(indices < 0)
            or np.any(indices >= np.asarray(shape)[None, :])
        ):
            raise ValueError("qpoint indices lie outside mesh_shape.")
        if np.unique(indices, axis=0).shape[0] != indices.shape[0]:
            raise ValueError("qpoint_indices must be unique.")
        if (
            weights.shape != (indices.shape[0],)
            or np.any(weights <= 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1.0e-12)
        ):
            raise ValueError("RTA q weights must be positive and sum to one.")
        width = float(broadening)
        temperature_ = float(temperature)
        volume = float(primitive_volume)
        balance = float(detailed_balance_tolerance)
        if any(
            not isfinite(value) or value <= 0.0
            for value in (width, temperature_, volume, balance)
        ):
            raise ValueError(
                "RTA broadening, temperature, volume, and tolerance must be positive."
            )
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        self.ifc3 = ifc3
        self.qpoint_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.mesh_shape = shape
        self.qpoint_weights = jnp.asarray(weights)
        self.broadening = width
        self.temperature = temperature_
        self.primitive_volume = volume
        self.units = units
        self.maximum_channels = int(maximum_channels)
        self.detailed_balance_tolerance = balance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "three-phonon-rta-plan",
                "ifc3": ifc3.ifc_id,
                "mesh_shape": list(shape),
                "broadening": width,
                "temperature": temperature_,
                "volume": volume,
                "units": units.unit_system_id,
                "maximum_channels": self.maximum_channels,
                "detailed_balance_tolerance": balance,
                "arrays": array_tree_fingerprint(
                    {"indices": indices, "weights": weights}
                ),
            }
        )

    def evaluate(
        self,
        mode_vertices: ThreePhononModeVertices,
        group_velocities: ArrayLike,
        heat_capacities: ArrayLike,
        /,
    ) -> ThreePhononRTAResult:
        if not isinstance(mode_vertices, ThreePhononModeVertices):
            raise TypeError("mode_vertices must be IFC3-derived ThreePhononModeVertices.")
        if (
            mode_vertices.ifc3_id != self.ifc3.ifc_id
            or mode_vertices.energy_unit.unit_id != self.units.scale.energy_unit.unit_id
            or mode_vertices.mesh_shape != self.mesh_shape
            or not np.array_equal(
                np.asarray(mode_vertices.qpoint_indices),
                np.asarray(self.qpoint_indices),
            )
        ):
            raise ValueError("Mode vertices do not bind this IFC3 and momentum mesh.")
        frequency = np.asarray(mode_vertices.angular_frequencies, dtype=np.float64)
        velocity = np.asarray(group_velocities, dtype=np.float64)
        heat = np.asarray(heat_capacities, dtype=np.float64)
        decay_vertices = np.asarray(mode_vertices.decay_vertices)
        coalescence_vertices = np.asarray(mode_vertices.coalescence_vertices)
        qpoints, branches = frequency.shape
        if velocity.shape != (qpoints, branches, 3) or heat.shape != (qpoints, branches):
            raise ValueError(
                "RTA velocities and heat capacities must align with mode vertices."
            )
        if (
            np.any(~np.isfinite(frequency))
            or np.any(frequency <= 0.0)
            or np.any(~np.isfinite(velocity))
            or np.any(~np.isfinite(heat))
            or np.any(heat < 0.0)
        ):
            raise ValueError(
                "RTA mode inputs must be finite, stable, and nonnegative where required."
            )
        channel_count = 2 * qpoints * qpoints * branches**3
        if channel_count > self.maximum_channels:
            raise ValueError("Three-phonon channel capacity exceeded before allocation.")
        addresses = np.asarray(self.qpoint_indices)
        lookup = {tuple(row): index for index, row in enumerate(addresses)}
        shape = np.asarray(self.mesh_shape)
        if len(lookup) != int(np.prod(shape)):
            raise ValueError(
                "RTA requires the complete regular reciprocal mesh for exact momentum routing."
            )
        beta_hbar = self.units.reduced_planck_constant / (
            self.units.boltzmann_constant * self.temperature
        )
        occupation = 1.0 / np.expm1(beta_hbar * frequency)
        decay = np.zeros_like(frequency)
        coalescence = np.zeros_like(frequency)
        balance_numerator = 0.0
        balance_denominator = 0.0
        width = self.broadening
        gaussian_scale = 1.0 / (width * np.sqrt(2.0 * np.pi))
        weights = np.asarray(self.qpoint_weights)
        for q1 in range(qpoints):
            for q2 in range(qpoints):
                q_decay = lookup[tuple(np.mod(addresses[q1] - addresses[q2], shape))]
                q_coalescence = lookup[
                    tuple(np.mod(addresses[q1] + addresses[q2], shape))
                ]
                for b1 in range(branches):
                    omega1 = frequency[q1, b1]
                    for b2 in range(branches):
                        omega2 = frequency[q2, b2]
                        for b3 in range(branches):
                            omega_decay = frequency[q_decay, b3]
                            delta_decay = gaussian_scale * np.exp(
                                -0.5 * ((omega1 - omega2 - omega_decay) / width) ** 2
                            )
                            amplitude_decay = abs(decay_vertices[q1, b1, q2, b2, b3]) ** 2
                            decay[q1, b1] += (
                                0.5
                                * weights[q2]
                                * amplitude_decay
                                * (occupation[q2, b2] + occupation[q_decay, b3] + 1.0)
                                * delta_decay
                            )
                            omega_coalescence = frequency[q_coalescence, b3]
                            delta_coalescence = gaussian_scale * np.exp(
                                -0.5
                                * ((omega1 + omega2 - omega_coalescence) / width) ** 2
                            )
                            amplitude_coalescence = (
                                abs(coalescence_vertices[q1, b1, q2, b2, b3]) ** 2
                            )
                            population = (
                                occupation[q2, b2] - occupation[q_coalescence, b3]
                            )
                            coalescence[q1, b1] += (
                                weights[q2]
                                * amplitude_coalescence
                                * population
                                * delta_coalescence
                            )
                            forward = (
                                occupation[q1, b1]
                                * occupation[q2, b2]
                                * (occupation[q_coalescence, b3] + 1.0)
                            )
                            reverse = (
                                (occupation[q1, b1] + 1.0)
                                * (occupation[q2, b2] + 1.0)
                                * occupation[q_coalescence, b3]
                            )
                            channel_weight = amplitude_coalescence * delta_coalescence
                            balance_numerator = max(
                                balance_numerator,
                                channel_weight * abs(forward - reverse),
                            )
                            balance_denominator = max(
                                balance_denominator,
                                channel_weight * max(abs(forward), abs(reverse)),
                            )
        prefactor = 2.0 * np.pi / self.units.reduced_planck_constant**2
        decay *= prefactor
        coalescence *= prefactor
        rates = decay + coalescence
        detailed_balance = balance_numerator / max(
            balance_denominator, np.finfo(np.float64).tiny
        )
        negative = rates < -np.finfo(np.float64).eps
        rates = np.where(np.abs(rates) <= np.finfo(np.float64).eps, 0.0, rates)
        ballistic = rates == 0.0
        finite_conductivity = not np.any(
            ballistic & (heat > 0.0) & (np.linalg.norm(velocity, axis=-1) > 0.0)
        )
        lifetimes = np.full_like(rates, np.inf)
        np.divide(1.0, rates, out=lifetimes, where=~ballistic)
        if finite_conductivity and not np.any(negative):
            conductivity_lifetimes = np.where(ballistic, 0.0, lifetimes)
            mode_conductivity = (
                weights[:, None, None, None]
                * heat[:, :, None, None]
                * velocity[:, :, :, None]
                * velocity[:, :, None, :]
                * conductivity_lifetimes[:, :, None, None]
                / self.primitive_volume
            )
            conductivity = np.sum(mode_conductivity, axis=(0, 1))
            eigenvalues = np.linalg.eigvalsh(0.5 * (conductivity + conductivity.T))
        else:
            mode_conductivity = np.full((qpoints, branches, 3, 3), np.nan)
            conductivity = np.full((3, 3), np.nan)
            eigenvalues = np.full((3,), np.nan)
        mean_free_paths = np.zeros_like(velocity)
        np.multiply(
            velocity,
            lifetimes[:, :, None],
            out=mean_free_paths,
            where=~ballistic[:, :, None],
        )
        antisymmetry = float(np.max(np.abs(conductivity - conductivity.T), initial=0.0))
        scattering_success = (
            not np.any(negative)
            and np.all(np.isfinite(rates))
            and detailed_balance <= self.detailed_balance_tolerance
        )
        scattering = ThreePhononScatteringResult(
            decay,
            coalescence,
            0.0,
            detailed_balance,
            channel_count // 2,
            channel_count // 2,
            scattering_success,
        )
        successful = (
            scattering_success and finite_conductivity and np.all(eigenvalues >= -1.0e-12)
        )
        return ThreePhononRTAResult(
            scattering,
            lifetimes,
            mode_conductivity,
            conductivity,
            mean_free_paths,
            ballistic,
            antisymmetry,
            eigenvalues,
            finite_conductivity,
            successful,
            self.units.unit_system_id,
        )


def evaluate_three_phonon_rta(
    prepared: ThreePhononRTAPlan,
    mode_vertices: ThreePhononModeVertices,
    group_velocities,
    heat_capacities,
    /,
) -> ThreePhononRTAResult:
    if not isinstance(prepared, ThreePhononRTAPlan):
        raise TypeError("prepared must be ThreePhononRTAPlan.")
    return prepared.evaluate(mode_vertices, group_velocities, heat_capacities)


__all__ = [
    "IFC3ModeVertexPlan",
    "ThreePhononModeVertices",
    "ThreePhononRTAPlan",
    "ThreePhononRTAResult",
    "ThreePhononScatteringResult",
    "evaluate_three_phonon_rta",
]
