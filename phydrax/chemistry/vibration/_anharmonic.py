#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dimensionless normal-coordinate force fields and bounded VPT2/GVPT2."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from itertools import product
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


NormalCoordinateEnergy = Callable[[Array], Array]


class AnharmonicForceFieldResult(StrictModule, NonTrainableState):
    equilibrium_energy: Array
    gradient: Array
    quadratic: Array
    cubic: Array
    quartic: Array
    maximum_permutation_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        equilibrium_energy: ArrayLike,
        gradient: ArrayLike,
        quadratic: ArrayLike,
        cubic: ArrayLike,
        quartic: ArrayLike,
        maximum_permutation_residual: ArrayLike,
        successful: ArrayLike,
        plan_id: str,
        /,
    ):
        quadratic_ = jnp.asarray(quadratic)
        gradient_ = jnp.asarray(gradient, dtype=quadratic_.dtype)
        cubic_ = jnp.asarray(cubic, dtype=quadratic_.dtype)
        quartic_ = jnp.asarray(quartic, dtype=quadratic_.dtype)
        count = quadratic_.shape[0] if quadratic_.ndim == 2 else -1
        if (
            quadratic_.shape != (count, count)
            or gradient_.shape != (count,)
            or cubic_.shape != (count, count, count)
            or quartic_.shape != (count, count, count, count)
        ):
            raise ValueError("Anharmonic force tensors must share one mode space.")
        residual = jnp.asarray(
            maximum_permutation_residual, dtype=quadratic_.dtype
        ).reshape(())
        self.equilibrium_energy = jnp.asarray(
            equilibrium_energy, dtype=quadratic_.dtype
        ).reshape(())
        self.gradient = gradient_
        self.quadratic = quadratic_
        self.cubic = cubic_
        self.quartic = quartic_
        self.maximum_permutation_residual = residual
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "anharmonic-force-field-result",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(self.equilibrium_energy),
                        "gradient": np.asarray(gradient_),
                        "quadratic": np.asarray(quadratic_),
                        "cubic": np.asarray(cubic_),
                        "quartic": np.asarray(quartic_),
                        "permutation_residual": np.asarray(residual),
                    }
                ),
            }
        )


class AnharmonicForceFieldPlan(StrictModule, NonTrainableState):
    energy: NormalCoordinateEnergy = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    stationarity_tolerance: float = eqx.field(static=True)
    symmetry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy: NormalCoordinateEnergy,
        mode_count: int,
        /,
        *,
        stationarity_tolerance: float = 1.0e-7,
        symmetry_tolerance: float = 1.0e-8,
    ):
        if not callable(energy):
            raise TypeError("energy must be a differentiable callable.")
        count = int(mode_count)
        stationarity = float(stationarity_tolerance)
        symmetry = float(symmetry_tolerance)
        if (
            count <= 0
            or not isfinite(stationarity)
            or stationarity <= 0.0
            or not isfinite(symmetry)
            or symmetry <= 0.0
        ):
            raise ValueError("Force-field mode count and tolerances must be positive.")
        self.energy = energy
        self.mode_count = count
        self.stationarity_tolerance = stationarity
        self.symmetry_tolerance = symmetry
        self.plan_id = canonical_fingerprint(
            {
                "kind": "anharmonic-force-field-plan",
                "mode_count": count,
                "stationarity_tolerance": stationarity,
                "symmetry_tolerance": symmetry,
            }
        )

    def evaluate(self, origin: ArrayLike | None = None, /) -> AnharmonicForceFieldResult:
        coordinate = (
            jnp.zeros((self.mode_count,), dtype=jnp.float64)
            if origin is None
            else jnp.asarray(origin)
        )
        if coordinate.shape != (self.mode_count,):
            raise ValueError("Normal-coordinate origin does not match mode_count.")
        gradient_function = jax.jacfwd(self.energy)
        quadratic_function = jax.jacfwd(gradient_function)
        cubic_function = jax.jacfwd(quadratic_function)
        quartic_function = jax.jacfwd(cubic_function)
        energy = self.energy(coordinate)
        gradient = gradient_function(coordinate)
        quadratic = quadratic_function(coordinate)
        cubic = cubic_function(coordinate)
        quartic = quartic_function(coordinate)
        cubic_symmetry = jnp.max(
            jnp.abs(cubic - jnp.transpose(cubic, (1, 0, 2))), initial=0.0
        )
        quartic_symmetry = jnp.max(
            jnp.abs(quartic - jnp.transpose(quartic, (1, 0, 2, 3))),
            initial=0.0,
        )
        residual = jnp.maximum(cubic_symmetry, quartic_symmetry)
        successful = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(quartic))
            & (jnp.max(jnp.abs(gradient), initial=0.0) <= self.stationarity_tolerance)
            & (residual <= self.symmetry_tolerance)
        )
        return AnharmonicForceFieldResult(
            energy,
            gradient,
            quadratic,
            cubic,
            quartic,
            residual,
            successful,
            self.plan_id,
        )


class VibrationalPerturbationKind(StrEnum):
    VPT2 = "vpt2"
    GVPT2 = "gvpt2"


class VibrationalPerturbationResult(StrictModule, NonTrainableState):
    state_quanta: Array
    harmonic_energies: Array
    anharmonic_energies: Array
    cubic_corrections: Array
    quartic_corrections: Array
    resonant_pair_indices: Array
    minimum_nonresonant_denominator: Array
    successful: Array
    kind: VibrationalPerturbationKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_quanta: ArrayLike,
        harmonic_energies: ArrayLike,
        anharmonic_energies: ArrayLike,
        cubic_corrections: ArrayLike,
        quartic_corrections: ArrayLike,
        resonant_pair_indices: ArrayLike,
        minimum_nonresonant_denominator: ArrayLike,
        successful: ArrayLike,
        kind: VibrationalPerturbationKind,
        plan_id: str,
        /,
    ):
        states = jnp.asarray(state_quanta, dtype=jnp.int32)
        harmonic = jnp.asarray(harmonic_energies)
        anharmonic = jnp.asarray(anharmonic_energies, dtype=harmonic.dtype)
        cubic = jnp.asarray(cubic_corrections, dtype=harmonic.dtype)
        quartic = jnp.asarray(quartic_corrections, dtype=harmonic.dtype)
        resonances = jnp.asarray(resonant_pair_indices, dtype=jnp.int32)
        count = states.shape[0] if states.ndim == 2 else -1
        if (
            any(
                value.shape != (count,)
                for value in (harmonic, anharmonic, cubic, quartic)
            )
            or resonances.ndim != 2
            or resonances.shape[1] != 2
        ):
            raise ValueError("VPT2 energies, states, and resonance pairs do not align.")
        self.state_quanta = states
        self.harmonic_energies = harmonic
        self.anharmonic_energies = anharmonic
        self.cubic_corrections = cubic
        self.quartic_corrections = quartic
        self.resonant_pair_indices = resonances
        self.minimum_nonresonant_denominator = jnp.asarray(
            minimum_nonresonant_denominator, dtype=harmonic.dtype
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.kind = kind
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "vibrational-perturbation-result",
                "method": kind.value,
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "states": np.asarray(states),
                        "harmonic": np.asarray(harmonic),
                        "anharmonic": np.asarray(anharmonic),
                        "cubic": np.asarray(cubic),
                        "quartic": np.asarray(quartic),
                        "resonances": np.asarray(resonances),
                    }
                ),
            }
        )


def _product_basis_hamiltonian(
    frequencies: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    maximum_quanta: int,
    maximum_basis_states: int,
):
    mode_count = frequencies.size
    states = np.asarray(
        tuple(product(range(maximum_quanta + 1), repeat=mode_count)), dtype=np.int64
    )
    dimension = states.shape[0]
    if dimension > maximum_basis_states:
        raise ValueError("Vibrational product basis exceeds maximum_basis_states.")
    local_dimension = maximum_quanta + 1
    creation = np.diag(np.sqrt(np.arange(1, local_dimension)), k=-1)
    annihilation = creation.T
    coordinate = (creation + annihilation) / np.sqrt(2.0)
    identity = np.eye(local_dimension)
    powers = tuple(np.linalg.matrix_power(coordinate, power) for power in range(5))

    def product_operator(counts):
        result = np.asarray([[1.0]])
        for count in counts:
            result = np.kron(result, powers[count] if count else identity)
        return result

    harmonic = np.sum(frequencies[None, :] * (states + 0.5), axis=1)
    third = np.zeros((dimension, dimension), dtype=np.float64)
    fourth = np.zeros_like(third)
    for indices in np.ndindex(cubic.shape):
        coefficient = cubic[indices]
        if coefficient == 0.0:
            continue
        counts = np.bincount(indices, minlength=mode_count)
        third += coefficient * product_operator(counts) / 6.0
    for indices in np.ndindex(quartic.shape):
        coefficient = quartic[indices]
        if coefficient == 0.0:
            continue
        counts = np.bincount(indices, minlength=mode_count)
        fourth += coefficient * product_operator(counts) / 24.0
    return states, harmonic, third, fourth


class VibrationalPerturbationPlan(StrictModule, NonTrainableState):
    kind: VibrationalPerturbationKind = eqx.field(static=True)
    frequencies: Array
    force_field: AnharmonicForceFieldResult
    maximum_quanta: int = eqx.field(static=True)
    maximum_basis_states: int = eqx.field(static=True)
    resonance_tolerance: float = eqx.field(static=True)
    coupling_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: VibrationalPerturbationKind,
        frequencies: ArrayLike,
        force_field: AnharmonicForceFieldResult,
        /,
        *,
        maximum_quanta: int = 4,
        maximum_basis_states: int = 4096,
        resonance_tolerance: float = 1.0e-3,
        coupling_tolerance: float = 1.0e-10,
    ):
        if not isinstance(kind, VibrationalPerturbationKind) or not isinstance(
            force_field, AnharmonicForceFieldResult
        ):
            raise TypeError("VPT2 requires typed method and anharmonic force field.")
        frequencies_ = jnp.asarray(frequencies)
        quanta = int(maximum_quanta)
        capacity = int(maximum_basis_states)
        resonance = float(resonance_tolerance)
        coupling = float(coupling_tolerance)
        if (
            frequencies_.shape != (force_field.quadratic.shape[0],)
            or bool(jnp.any(frequencies_ <= 0.0))
            or quanta < 2
            or capacity < 1
            or not isfinite(resonance)
            or resonance <= 0.0
            or not isfinite(coupling)
            or coupling < 0.0
        ):
            raise ValueError(
                "VPT2 frequencies, basis capacity, or tolerances are invalid."
            )
        self.kind = kind
        self.frequencies = frequencies_
        self.force_field = force_field
        self.maximum_quanta = quanta
        self.maximum_basis_states = capacity
        self.resonance_tolerance = resonance
        self.coupling_tolerance = coupling
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vibrational-perturbation-plan",
                "method": kind.value,
                "force_field": force_field.result_id,
                "maximum_quanta": quanta,
                "maximum_basis_states": capacity,
                "resonance_tolerance": resonance,
                "coupling_tolerance": coupling,
                "frequencies": array_tree_fingerprint(np.asarray(frequencies_)),
            }
        )

    def evaluate(self, state_quanta: ArrayLike, /) -> VibrationalPerturbationResult:
        requested = np.asarray(state_quanta, dtype=np.int64)
        if (
            requested.ndim != 2
            or requested.shape[1] != self.frequencies.size
            or np.any(requested < 0)
            or np.any(requested > self.maximum_quanta)
        ):
            raise ValueError("Requested vibrational states exceed the product basis.")
        states, harmonic, third, fourth = _product_basis_hamiltonian(
            np.asarray(self.frequencies),
            np.asarray(self.force_field.cubic),
            np.asarray(self.force_field.quartic),
            self.maximum_quanta,
            self.maximum_basis_states,
        )
        lookup = {tuple(state): index for index, state in enumerate(states)}
        target_indices = np.asarray([lookup[tuple(state)] for state in requested])
        cubic_corrections = np.zeros((requested.shape[0],), dtype=np.float64)
        quartic_corrections = np.diag(fourth)[target_indices]
        anharmonic = harmonic[target_indices] + quartic_corrections
        resonant_pairs = []
        minimum_denominator = np.inf
        full = np.diag(harmonic) + third + fourth
        for output_index, basis_index in enumerate(target_indices):
            denominators = harmonic[basis_index] - harmonic
            coupled = np.abs(third[basis_index]) > self.coupling_tolerance
            resonant = coupled & (np.abs(denominators) <= self.resonance_tolerance)
            resonant[basis_index] = False
            for partner in np.flatnonzero(resonant):
                resonant_pairs.append((int(basis_index), int(partner)))
            nonresonant = coupled & ~resonant
            nonresonant[basis_index] = False
            if np.any(nonresonant):
                minimum_denominator = min(
                    minimum_denominator,
                    float(np.min(np.abs(denominators[nonresonant]))),
                )
                correction = np.sum(
                    np.abs(third[basis_index, nonresonant]) ** 2
                    / denominators[nonresonant]
                )
            else:
                correction = 0.0
            cubic_corrections[output_index] = correction
            anharmonic[output_index] += correction
            if self.kind is VibrationalPerturbationKind.GVPT2 and np.any(resonant):
                polyad = np.concatenate(
                    (np.asarray([basis_index]), np.flatnonzero(resonant))
                )
                values, vectors = np.linalg.eigh(full[np.ix_(polyad, polyad)])
                root = int(np.argmax(np.abs(vectors[0]) ** 2))
                anharmonic[output_index] = values[root] + correction
        resonance_array = (
            np.zeros((0, 2), dtype=np.int64)
            if not resonant_pairs
            else np.asarray(sorted(set(resonant_pairs)), dtype=np.int64)
        )
        successful = (
            bool(self.force_field.successful)
            and np.all(np.isfinite(anharmonic))
            and (
                self.kind is VibrationalPerturbationKind.GVPT2
                or resonance_array.shape[0] == 0
            )
        )
        return VibrationalPerturbationResult(
            requested,
            harmonic[target_indices],
            anharmonic,
            cubic_corrections,
            quartic_corrections,
            resonance_array,
            minimum_denominator,
            successful,
            self.kind,
            self.plan_id,
        )


__all__ = [
    "AnharmonicForceFieldPlan",
    "AnharmonicForceFieldResult",
    "VibrationalPerturbationKind",
    "VibrationalPerturbationPlan",
    "VibrationalPerturbationResult",
]
