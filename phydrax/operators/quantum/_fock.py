#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._geometry_precision import GeometryPrecisionPolicy
from ..._strict import StrictModule
from ._open_contracts import (
    ApproximationAxis,
    ApproximationQuantity,
    OpenSystemApproximationEvidence,
)


class FockCutoffEvidence(StrictModule):
    top_level_probability: Array
    boundary_amplitude: Array
    state_norm_residual: Array
    valid: Array
    approximation: OpenSystemApproximationEvidence

    def __init__(
        self,
        top_level_probability: ArrayLike,
        boundary_amplitude: ArrayLike,
        cutoffs: Sequence[int],
        state_norm_residual: ArrayLike,
        /,
        top_probability_tolerance: float = 1e-6,
        precision: GeometryPrecisionPolicy | None = None,
        coordinates: ArrayLike | None = None,
    ):
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be GeometryPrecisionPolicy or None.")
        top = precision_.decision(top_level_probability)
        boundary = precision_.output(boundary_amplitude)
        self.top_level_probability = top
        self.boundary_amplitude = boundary
        self.state_norm_residual = jnp.asarray(state_norm_residual)
        self.valid = (
            jnp.all(jnp.isfinite(top))
            & jnp.all(top >= 0.0)
            & jnp.isfinite(self.state_norm_residual)
            & (self.state_norm_residual <= 1e-8)
        )
        self.approximation = OpenSystemApproximationEvidence(
            "bosonic-fock",
            tuple(
                ApproximationAxis(f"mode-{index}-cutoff", cutoff)
                for index, cutoff in enumerate(cutoffs)
            ),
            (
                ApproximationQuantity(
                    "maximum-top-level-probability",
                    jnp.max(top),
                    top_probability_tolerance,
                    units="probability",
                    norm_id="maximum",
                    estimate_kind="estimate",
                ),
                ApproximationQuantity(
                    "state-norm-residual",
                    self.state_norm_residual,
                    1e-8,
                    units="dimensionless",
                    norm_id="absolute",
                    estimate_kind="bound",
                ),
            ),
            execution_valid=self.valid,
            precision_evidence=precision_.evidence_for(
                boundary if coordinates is None else coordinates
            ),
            precision_policy_ids=(precision_.policy_id,),
        )


class BosonicFockSpace(StrictModule):
    cutoffs: tuple[int, ...] = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)
    precision: GeometryPrecisionPolicy

    def __init__(
        self,
        cutoffs: Sequence[int],
        /,
        *,
        precision: GeometryPrecisionPolicy | None = None,
    ):
        cutoffs_ = tuple(int(value) for value in cutoffs)
        if not cutoffs_ or any(value < 2 for value in cutoffs_):
            raise ValueError("Every Fock cutoff must be at least two.")
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be GeometryPrecisionPolicy or None.")
        self.cutoffs = cutoffs_
        self.mode_count = len(cutoffs_)
        self.dimension = prod(cutoffs_)
        self.space_id = "fock:" + "x".join(str(value) for value in cutoffs_)
        self.precision = precision_

    def occupations(self) -> Array:
        indices = jnp.arange(self.dimension)
        values = []
        remainder = indices
        for cutoff in reversed(self.cutoffs):
            values.append(remainder % cutoff)
            remainder = remainder // cutoff
        return jnp.stack(tuple(reversed(values)), axis=-1)

    def annihilate(self, state: ArrayLike, mode: int, /) -> Array:
        vector = jnp.asarray(state)
        if vector.shape != (self.dimension,):
            raise ValueError("Fock state-vector dimension is invalid.")
        mode_ = int(mode)
        if not 0 <= mode_ < self.mode_count:
            raise ValueError("mode index is out of range.")
        tensor = jnp.moveaxis(vector.reshape(self.cutoffs), mode_, 0)
        cutoff = self.cutoffs[mode_]
        result = jnp.zeros_like(tensor)
        result = result.at[:-1].set(
            jnp.sqrt(jnp.arange(1, cutoff)).reshape(
                (cutoff - 1,) + (1,) * (tensor.ndim - 1)
            )
            * tensor[1:]
        )
        return jnp.moveaxis(result, 0, mode_).reshape(-1)

    def create(self, state: ArrayLike, mode: int, /) -> Array:
        vector = jnp.asarray(state)
        if vector.shape != (self.dimension,):
            raise ValueError("Fock state-vector dimension is invalid.")
        mode_ = int(mode)
        if not 0 <= mode_ < self.mode_count:
            raise ValueError("mode index is out of range.")
        tensor = jnp.moveaxis(vector.reshape(self.cutoffs), mode_, 0)
        cutoff = self.cutoffs[mode_]
        result = jnp.zeros_like(tensor)
        result = result.at[1:].set(
            jnp.sqrt(jnp.arange(1, cutoff)).reshape(
                (cutoff - 1,) + (1,) * (tensor.ndim - 1)
            )
            * tensor[:-1]
        )
        return jnp.moveaxis(result, 0, mode_).reshape(-1)

    def annihilation_matrix(self, mode: int, /) -> Array:
        mode_ = int(mode)
        if not 0 <= mode_ < self.mode_count:
            raise ValueError("mode index is out of range.")
        matrices = []
        for index, cutoff in enumerate(self.cutoffs):
            if index == mode_:
                local = jnp.diag(jnp.sqrt(jnp.arange(1, cutoff)), 1)
            else:
                local = jnp.eye(cutoff)
            matrices.append(local)
        result = matrices[0]
        for matrix in matrices[1:]:
            result = jnp.kron(result, matrix)
        return result.astype(complex)

    def creation_matrix(self, mode: int, /) -> Array:
        return jnp.conj(self.annihilation_matrix(mode).T)

    def number_matrix(self, mode: int, /) -> Array:
        creation = self.creation_matrix(mode)
        annihilation = self.annihilation_matrix(mode)
        return creation @ annihilation

    def cutoff_evidence(self, state: ArrayLike, /) -> FockCutoffEvidence:
        vector = jnp.asarray(state)
        if vector.shape != (self.dimension,):
            raise ValueError("Fock state-vector dimension is invalid.")
        self.precision.validate_coordinates(vector)
        probabilities = jnp.abs(self.precision.accumulation(vector)) ** 2
        occupations = self.occupations()
        top = jnp.stack(
            [
                self.precision.sum(
                    jnp.where(
                        occupations[:, mode] == cutoff - 1,
                        probabilities,
                        0.0,
                    )
                )
                for mode, cutoff in enumerate(self.cutoffs)
            ]
        )
        norm_residual = jnp.abs(jnp.sum(probabilities) - 1.0)
        return FockCutoffEvidence(
            top,
            jnp.sqrt(top),
            self.cutoffs,
            norm_residual,
            precision=self.precision,
            coordinates=vector,
        )

    def embed(self, state: ArrayLike, fine: BosonicFockSpace, /) -> Array:
        vector = jnp.asarray(state)
        if vector.shape != (self.dimension,):
            raise ValueError("Coarse Fock state has the wrong dimension.")
        if fine.mode_count != self.mode_count or any(
            fine_cutoff < cutoff
            for cutoff, fine_cutoff in zip(self.cutoffs, fine.cutoffs, strict=True)
        ):
            raise ValueError("Fine Fock space must contain every coarse occupation.")
        coarse_occupations = self.occupations()
        multipliers = []
        multiplier = 1
        for cutoff in reversed(fine.cutoffs):
            multipliers.append(multiplier)
            multiplier *= cutoff
        multipliers = jnp.asarray(tuple(reversed(multipliers)))
        fine_indices = jnp.sum(coarse_occupations * multipliers, axis=-1)
        return (
            jnp.zeros((fine.dimension,), dtype=vector.dtype).at[fine_indices].set(vector)
        )


def kerr_hamiltonian(
    space: BosonicFockSpace,
    mode: int,
    frequency: float,
    nonlinearity: float,
    /,
) -> Array:
    number = space.number_matrix(mode)
    identity = jnp.eye(space.dimension, dtype=complex)
    return float(frequency) * number + 0.5 * float(nonlinearity) * number @ (
        number - identity
    )


def jaynes_cummings_hamiltonian(
    cutoff: int,
    cavity_frequency: float,
    qubit_frequency: float,
    coupling: float,
    /,
) -> tuple[BosonicFockSpace, Array]:
    cavity = BosonicFockSpace((int(cutoff),))
    annihilation = cavity.annihilation_matrix(0)
    creation = jnp.conj(annihilation.T)
    sigma_plus = jnp.asarray([[0, 1], [0, 0]], dtype=complex)
    sigma_minus = jnp.conj(sigma_plus.T)
    sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=complex)
    number = cavity.number_matrix(0)
    hamiltonian = (
        float(cavity_frequency) * jnp.kron(number, jnp.eye(2))
        + 0.5 * float(qubit_frequency) * jnp.kron(jnp.eye(cavity.dimension), sigma_z)
        + float(coupling)
        * (jnp.kron(annihilation, sigma_plus) + jnp.kron(creation, sigma_minus))
    )
    return cavity, hamiltonian


__all__ = [
    "BosonicFockSpace",
    "FockCutoffEvidence",
    "jaynes_cummings_hamiltonian",
    "kerr_hamiltonian",
]
