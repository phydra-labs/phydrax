#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded VSCF/VCI, periodic hindered rotors, and conformational ensembles."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._anharmonic import _product_basis_hamiltonian, AnharmonicForceFieldResult


class VibrationalConfigurationResult(StrictModule, NonTrainableState):
    vscf_energy: Array
    vscf_modal_coefficients: tuple[Array, ...]
    vscf_variance: Array
    vscf_iterations: Array
    vci_energies: Array
    vci_coefficients: Array
    vci_residuals: Array
    basis_quanta: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        vscf_energy,
        vscf_modal_coefficients,
        vscf_variance,
        vscf_iterations,
        vci_energies,
        vci_coefficients,
        vci_residuals,
        basis_quanta,
        successful,
        plan_id,
        /,
    ):
        energy = jnp.asarray(vci_energies)
        coefficients = jnp.asarray(vci_coefficients, dtype=energy.dtype)
        residuals = jnp.asarray(vci_residuals, dtype=energy.dtype)
        basis = jnp.asarray(basis_quanta, dtype=jnp.int32)
        roots = energy.size
        if (
            coefficients.ndim != 2
            or coefficients.shape[1] != roots
            or residuals.shape != (roots,)
            or basis.ndim != 2
            or basis.shape[0] != coefficients.shape[0]
        ):
            raise ValueError(
                "VSCF/VCI roots, coefficients, residuals, and basis do not align."
            )
        modals = tuple(
            jnp.asarray(value, dtype=energy.dtype) for value in vscf_modal_coefficients
        )
        if len(modals) != basis.shape[1] or any(value.ndim != 1 for value in modals):
            raise ValueError("VSCF modal coefficients must provide one vector per mode.")
        self.vscf_energy = jnp.asarray(vscf_energy, dtype=energy.dtype).reshape(())
        self.vscf_modal_coefficients = modals
        self.vscf_variance = jnp.asarray(vscf_variance, dtype=energy.dtype).reshape(())
        self.vscf_iterations = jnp.asarray(vscf_iterations, dtype=jnp.int32).reshape(())
        self.vci_energies = energy
        self.vci_coefficients = coefficients
        self.vci_residuals = residuals
        self.basis_quanta = basis
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "vibrational-configuration-result",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "vscf_energy": np.asarray(self.vscf_energy),
                        "vscf_modals": tuple(np.asarray(value) for value in modals),
                        "vscf_variance": np.asarray(self.vscf_variance),
                        "vci_energies": np.asarray(energy),
                        "vci_coefficients": np.asarray(coefficients),
                        "vci_residuals": np.asarray(residuals),
                        "basis": np.asarray(basis),
                    }
                ),
            }
        )


class VibrationalConfigurationPlan(StrictModule, NonTrainableState):
    frequencies: Array
    force_field: AnharmonicForceFieldResult
    maximum_quanta: int = eqx.field(static=True)
    maximum_basis_states: int = eqx.field(static=True)
    root_count: int = eqx.field(static=True)
    vscf_tolerance: float = eqx.field(static=True)
    vscf_maximum_sweeps: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequencies: ArrayLike,
        force_field: AnharmonicForceFieldResult,
        /,
        *,
        maximum_quanta: int = 4,
        maximum_basis_states: int = 4096,
        root_count: int = 8,
        vscf_tolerance: float = 1.0e-10,
        vscf_maximum_sweeps: int = 100,
        residual_tolerance: float = 1.0e-9,
    ):
        if not isinstance(force_field, AnharmonicForceFieldResult):
            raise TypeError("force_field must be AnharmonicForceFieldResult.")
        frequency = jnp.asarray(frequencies)
        quanta = int(maximum_quanta)
        capacity = int(maximum_basis_states)
        roots = int(root_count)
        sweeps = int(vscf_maximum_sweeps)
        vscf_tolerance_ = float(vscf_tolerance)
        residual_tolerance_ = float(residual_tolerance)
        dimension = (quanta + 1) ** frequency.size
        if (
            frequency.shape != (force_field.quadratic.shape[0],)
            or bool(jnp.any(frequency <= 0.0))
            or quanta < 1
            or capacity < 1
            or dimension > capacity
            or roots < 1
            or roots > dimension
            or sweeps < 1
            or not isfinite(vscf_tolerance_)
            or vscf_tolerance_ <= 0.0
            or not isfinite(residual_tolerance_)
            or residual_tolerance_ <= 0.0
        ):
            raise ValueError(
                "VSCF/VCI frequencies, basis, roots, or tolerances are invalid."
            )
        self.frequencies = frequency
        self.force_field = force_field
        self.maximum_quanta = quanta
        self.maximum_basis_states = capacity
        self.root_count = roots
        self.vscf_tolerance = vscf_tolerance_
        self.vscf_maximum_sweeps = sweeps
        self.residual_tolerance = residual_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vibrational-configuration-plan",
                "force_field": force_field.result_id,
                "maximum_quanta": quanta,
                "maximum_basis_states": capacity,
                "root_count": roots,
                "vscf_tolerance": vscf_tolerance_,
                "vscf_maximum_sweeps": sweeps,
                "residual_tolerance": residual_tolerance_,
                "frequencies": array_tree_fingerprint(np.asarray(frequency)),
            }
        )

    @staticmethod
    def _product_vector(modals):
        result = np.asarray([1.0])
        for modal in modals:
            result = np.kron(result, modal)
        return result

    def evaluate(self, /) -> VibrationalConfigurationResult:
        states, harmonic, cubic, quartic = _product_basis_hamiltonian(
            np.asarray(self.frequencies),
            np.asarray(self.force_field.cubic),
            np.asarray(self.force_field.quartic),
            self.maximum_quanta,
            self.maximum_basis_states,
        )
        hamiltonian = np.diag(harmonic) + cubic + quartic
        local_dimension = self.maximum_quanta + 1
        mode_count = self.frequencies.size
        modals = [np.eye(local_dimension)[:, 0] for _ in range(mode_count)]
        previous_energy = np.inf
        converged = False
        completed = 0
        for sweep in range(self.vscf_maximum_sweeps):
            for active_mode in range(mode_count):
                embedding_columns = []
                for local_state in range(local_dimension):
                    factors = list(modals)
                    factors[active_mode] = np.eye(local_dimension)[:, local_state]
                    embedding_columns.append(self._product_vector(factors))
                embedding = np.stack(embedding_columns, axis=1)
                effective = embedding.T @ hamiltonian @ embedding
                _, vectors = np.linalg.eigh(0.5 * (effective + effective.T))
                modals[active_mode] = vectors[:, 0]
            product_state = self._product_vector(modals)
            energy = float(product_state @ hamiltonian @ product_state)
            completed = sweep + 1
            if abs(energy - previous_energy) <= self.vscf_tolerance:
                converged = True
                break
            previous_energy = energy
        image = hamiltonian @ product_state
        variance = float(image @ image - energy**2)
        values, vectors = np.linalg.eigh(0.5 * (hamiltonian + hamiltonian.T))
        values = values[: self.root_count]
        vectors = vectors[:, : self.root_count]
        residuals = np.sqrt(
            np.sum(np.abs(hamiltonian @ vectors - vectors * values[None, :]) ** 2, axis=0)
        )
        successful = (
            bool(self.force_field.successful)
            and converged
            and variance <= self.residual_tolerance
            and np.all(residuals <= self.residual_tolerance)
            and np.all(np.isfinite(values))
        )
        return VibrationalConfigurationResult(
            energy,
            tuple(modals),
            variance,
            completed,
            values,
            vectors,
            residuals,
            states,
            successful,
            self.plan_id,
        )


class HinderedRotorResult(StrictModule, NonTrainableState):
    energies: Array
    wavefunctions: Array
    grid: Array
    partition_function: Array
    free_energy: Array
    orthonormality_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies,
        wavefunctions,
        grid,
        partition_function,
        free_energy,
        orthonormality_residual,
        successful,
        plan_id,
        /,
    ):
        energies_ = jnp.asarray(energies)
        vectors_input = jnp.asarray(wavefunctions)
        vectors = vectors_input.astype(
            jnp.result_type(energies_.dtype, vectors_input.dtype)
        )
        grid_ = jnp.asarray(grid, dtype=energies_.dtype)
        if vectors.shape != (grid_.size, energies_.size):
            raise ValueError("Rotor wavefunctions must align with grid and energies.")
        self.energies = energies_
        self.wavefunctions = vectors
        self.grid = grid_
        self.partition_function = jnp.asarray(
            partition_function, dtype=energies_.dtype
        ).reshape(())
        self.free_energy = jnp.asarray(free_energy, dtype=energies_.dtype).reshape(())
        self.orthonormality_residual = jnp.asarray(
            orthonormality_residual, dtype=energies_.dtype
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "hindered-rotor-result",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energies": np.asarray(energies_),
                        "wavefunctions": np.asarray(vectors),
                        "grid": np.asarray(grid_),
                        "partition_function": np.asarray(self.partition_function),
                        "free_energy": np.asarray(self.free_energy),
                    }
                ),
            }
        )


class HinderedRotorPlan(StrictModule, NonTrainableState):
    potential: Array
    rotational_constant: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    root_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        potential: ArrayLike,
        rotational_constant: float,
        /,
        *,
        temperature: float = 298.15,
        boltzmann_constant: float,
        root_count: int = 16,
    ):
        potential_ = jnp.asarray(potential)
        rotational = float(rotational_constant)
        temperature_ = float(temperature)
        boltzmann = float(boltzmann_constant)
        roots = int(root_count)
        if (
            potential_.ndim != 1
            or potential_.size < 5
            or not isfinite(rotational)
            or rotational <= 0.0
            or not isfinite(temperature_)
            or temperature_ <= 0.0
            or not isfinite(boltzmann)
            or boltzmann <= 0.0
            or roots < 1
            or roots > potential_.size
        ):
            raise ValueError("Hindered-rotor potential, constants, or roots are invalid.")
        self.potential = potential_
        self.rotational_constant = rotational
        self.temperature = temperature_
        self.boltzmann_constant = boltzmann
        self.root_count = roots
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hindered-rotor-plan",
                "rotational_constant": rotational,
                "temperature": temperature_,
                "boltzmann_constant": boltzmann,
                "root_count": roots,
                "potential": array_tree_fingerprint(np.asarray(potential_)),
            }
        )

    def evaluate(self, /) -> HinderedRotorResult:
        potential = np.asarray(self.potential)
        size = potential.size
        grid = 2.0 * np.pi * np.arange(size) / size
        wave_numbers = np.fft.fftfreq(size, d=1.0 / size)
        fourier = np.exp(1.0j * np.outer(grid, wave_numbers)) / np.sqrt(size)
        kinetic = (
            fourier
            @ np.diag(self.rotational_constant * wave_numbers**2)
            @ np.conj(fourier.T)
        )
        hamiltonian = kinetic + np.diag(potential)
        values, vectors = np.linalg.eigh(0.5 * (hamiltonian + np.conj(hamiltonian.T)))
        values = values[: self.root_count].real
        vectors = vectors[:, : self.root_count]
        shifted = values - values[0]
        beta = 1.0 / (self.boltzmann_constant * self.temperature)
        partition = float(np.sum(np.exp(-beta * shifted)))
        free = float(values[0] - np.log(partition) / beta)
        residual = float(
            np.max(
                np.abs(np.conj(vectors.T) @ vectors - np.eye(self.root_count)),
                initial=0.0,
            )
        )
        successful = (
            np.all(np.isfinite(values))
            and np.isfinite(partition)
            and partition > 0.0
            and residual <= 1.0e-10
        )
        return HinderedRotorResult(
            values,
            vectors,
            grid,
            partition,
            free,
            residual,
            successful,
            self.plan_id,
        )


class ConformationalEnsembleResult(StrictModule, NonTrainableState):
    weights: Array
    free_energy: Array
    mean_energy: Array
    entropy: Array
    property_average: Array | None
    effective_sample_size: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class ConformationalEnsemblePlan(StrictModule, NonTrainableState):
    energies: Array
    degeneracies: Array
    temperature: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies: ArrayLike,
        /,
        *,
        degeneracies: ArrayLike | None = None,
        temperature: float = 298.15,
        boltzmann_constant: float,
    ):
        energy = jnp.asarray(energies)
        degeneracy = (
            jnp.ones(energy.shape, dtype=energy.dtype)
            if degeneracies is None
            else jnp.asarray(degeneracies, dtype=energy.dtype)
        )
        temperature_ = float(temperature)
        boltzmann = float(boltzmann_constant)
        if (
            energy.ndim != 1
            or not energy.size
            or degeneracy.shape != energy.shape
            or bool(jnp.any(degeneracy <= 0.0))
            or not isfinite(temperature_)
            or temperature_ <= 0.0
            or not isfinite(boltzmann)
            or boltzmann <= 0.0
        ):
            raise ValueError(
                "Conformer energies, degeneracies, or thermal constants are invalid."
            )
        self.energies = energy
        self.degeneracies = degeneracy
        self.temperature = temperature_
        self.boltzmann_constant = boltzmann
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conformational-ensemble-plan",
                "temperature": temperature_,
                "boltzmann_constant": boltzmann,
                "arrays": array_tree_fingerprint(
                    {
                        "energies": np.asarray(energy),
                        "degeneracies": np.asarray(degeneracy),
                    }
                ),
            }
        )

    def evaluate(
        self, properties: ArrayLike | None = None, /
    ) -> ConformationalEnsembleResult:
        beta = 1.0 / (self.boltzmann_constant * self.temperature)
        minimum = jnp.min(self.energies)
        unnormalized = self.degeneracies * jnp.exp(-beta * (self.energies - minimum))
        partition = jnp.sum(unnormalized)
        weights = unnormalized / partition
        free = minimum - jnp.log(partition) / beta
        mean = jnp.sum(weights * self.energies)
        entropy = (mean - free) / self.temperature
        values = None if properties is None else jnp.asarray(properties)
        if values is not None and values.shape[0] != self.energies.size:
            raise ValueError("Ensemble properties must start with the conformer axis.")
        average = (
            None if values is None else jnp.tensordot(weights, values, axes=((0,), (0,)))
        )
        effective = 1.0 / jnp.sum(weights**2)
        successful = (
            jnp.all(jnp.isfinite(weights))
            & jnp.isfinite(free)
            & jnp.isfinite(entropy)
            & (average is None or jnp.all(jnp.isfinite(average)))
        )
        return ConformationalEnsembleResult(
            weights,
            free,
            mean,
            entropy,
            average,
            effective,
            successful,
            self.plan_id,
            canonical_fingerprint(
                {
                    "kind": "conformational-ensemble-result",
                    "plan": self.plan_id,
                    "properties": None
                    if values is None
                    else array_tree_fingerprint(np.asarray(values)),
                }
            ),
        )


__all__ = [
    "ConformationalEnsemblePlan",
    "ConformationalEnsembleResult",
    "HinderedRotorPlan",
    "HinderedRotorResult",
    "VibrationalConfigurationPlan",
    "VibrationalConfigurationResult",
]
