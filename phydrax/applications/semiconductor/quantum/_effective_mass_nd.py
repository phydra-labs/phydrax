#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from ...._strict import StrictModule
from ._basis import HBAR, Q, QuantumResources


class DenseHamiltonianND(StrictModule):
    """Bounded complete finite-basis Hamiltonian for ND confinement."""

    matrix: Array
    cell_volumes: Array
    resources: QuantumResources
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    energy_reference: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        cell_volumes: ArrayLike,
        /,
        *,
        grid_shape: Sequence[int],
        energy_reference: str,
        resources: QuantumResources,
    ):
        value = jnp.asarray(matrix, dtype=jnp.float64)
        volumes = jnp.asarray(cell_volumes, dtype=jnp.float64)
        shape = tuple(grid_shape)
        size = int(np.prod(shape))
        if value.shape != (size, size) or volumes.shape != (size,):
            raise ValueError("Dense ND Hamiltonian storage does not match grid_shape.")
        if size > resources.max_nodes or size > resources.max_modes:
            raise ValueError("ND confinement basis exceeds declared quantum resources.")
        if not energy_reference:
            raise ValueError("energy_reference must be non-empty.")
        self.matrix = value
        self.cell_volumes = volumes
        self.grid_shape = shape
        self.energy_reference = energy_reference
        self.resources = resources

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    def with_potential(self, potential: ArrayLike, /) -> DenseHamiltonianND:
        value = jnp.asarray(potential, dtype=jnp.float64)
        if value.shape not in (self.grid_shape, (self.size,)):
            raise ValueError("Potential must match the ND confinement grid.")
        diagonal = -Q * value.reshape((-1,))
        return eqx.tree_at(
            lambda hamiltonian: hamiltonian.matrix,
            self,
            self.matrix + jnp.diag(diagonal),
        )

    def shifted(self, energy: ArrayLike, /) -> DenseHamiltonianND:
        value = jnp.asarray(energy, dtype=self.matrix.dtype).reshape(())
        return eqx.tree_at(
            lambda hamiltonian: hamiltonian.matrix,
            self,
            self.matrix + value * jnp.eye(self.size, dtype=self.matrix.dtype),
        )

    def operator(self, *, scale=Q, shift=0.0):
        matrix = (
            self.matrix - shift * jnp.eye(self.size, dtype=self.matrix.dtype)
        ) / scale
        return la.DenseLinearOperator(
            matrix,
            properties=la.OperatorProperties(
                self_adjoint=True,
                evidence={"self_adjoint": "construction"},
            ),
            operator_id="quantum-real-hermitian-nd-confinement",
        )

    def density(self, mode_vectors: Array, occupations: Array, /) -> Array:
        return (
            jnp.sum(jnp.abs(mode_vectors) ** 2 * occupations[None, :], axis=1)
            / self.cell_volumes
        ).reshape(self.grid_shape)


class EffectiveMassND(StrictModule):
    """Uniform tensor-grid effective-mass confinement with Dirichlet exterior."""

    axes: tuple[Array, ...]
    mass: Array
    base_hamiltonian: DenseHamiltonianND
    dimension: int = eqx.field(static=True)
    grid_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        axes: Sequence[ArrayLike],
        band_edge: ArrayLike,
        mass: ArrayLike,
        /,
        *,
        energy_reference: str,
        resources: QuantumResources | None = None,
    ):
        axis_values = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
        dimension = len(axis_values)
        if dimension < 2:
            raise ValueError("EffectiveMassND requires at least two spatial axes.")
        if any(
            axis.ndim != 1
            or axis.size < 2
            or np.any(~np.isfinite(axis))
            or np.any(np.diff(axis) <= 0.0)
            or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-9, atol=0.0)
            for axis in axis_values
        ):
            raise ValueError(
                "EffectiveMassND axes must be finite uniform interior centers."
            )
        shape = tuple(axis.size for axis in axis_values)
        resources_ = QuantumResources() if resources is None else resources
        if not isinstance(resources_, QuantumResources):
            raise TypeError("resources must be QuantumResources or None.")
        size = int(np.prod(shape))
        if size > resources_.max_nodes or size > resources_.max_modes:
            raise ValueError("EffectiveMassND basis exceeds declared quantum resources.")
        band = np.broadcast_to(np.asarray(band_edge, dtype=np.float64), shape).copy()
        mass_values = np.broadcast_to(np.asarray(mass, dtype=np.float64), shape).copy()
        if (
            np.any(~np.isfinite(band))
            or np.any(~np.isfinite(mass_values))
            or np.any(mass_values <= 0.0)
        ):
            raise ValueError(
                "Band edge and effective mass must be finite; mass must be positive."
            )
        matrix = np.zeros((size, size), dtype=np.float64)
        matrix[np.diag_indices(size)] = band.reshape((-1,))
        spacings = tuple(float(np.diff(axis)[0]) for axis in axis_values)
        for index in product(*(range(count) for count in shape)):
            flat = np.ravel_multi_index(index, shape)
            for axis, spacing in enumerate(spacings):
                for direction in (-1, 1):
                    neighbor = list(index)
                    neighbor[axis] += direction
                    if 0 <= neighbor[axis] < shape[axis]:
                        neighbor_tuple = tuple(neighbor)
                        other = np.ravel_multi_index(neighbor_tuple, shape)
                        coupling = HBAR**2 / (
                            (mass_values[index] + mass_values[neighbor_tuple])
                            * spacing**2
                        )
                        matrix[flat, flat] += coupling
                        matrix[flat, other] -= coupling
                    else:
                        matrix[flat, flat] += HBAR**2 / (
                            2.0 * mass_values[index] * spacing**2
                        )
        volumes = np.full((size,), np.prod(spacings), dtype=np.float64)
        self.axes = tuple(jnp.asarray(axis) for axis in axis_values)
        self.mass = jnp.asarray(mass_values)
        self.base_hamiltonian = DenseHamiltonianND(
            matrix,
            volumes,
            grid_shape=shape,
            energy_reference=energy_reference,
            resources=resources_,
        )
        self.dimension = dimension
        self.grid_shape = shape

    def hamiltonian(self, potential: ArrayLike, /) -> DenseHamiltonianND:
        return self.base_hamiltonian.with_potential(potential)


__all__ = ["DenseHamiltonianND", "EffectiveMassND"]
