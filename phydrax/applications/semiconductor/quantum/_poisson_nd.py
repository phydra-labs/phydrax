#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from ...._strict import StrictModule
from ._effective_mass_nd import EffectiveMassND


class QuantumPoissonND(StrictModule):
    """Prepared tensor-grid Poisson solve sharing an EffectiveMassND cell basis."""

    operator: la.DenseLinearOperator
    prepared: la.PreparedLinearSolve
    fixed_charge_density: Array
    cell_volumes: Array
    boundary_load: Array
    scale: Array
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        basis: EffectiveMassND,
        permittivity: ArrayLike,
        fixed_charge_density: ArrayLike,
        boundary_voltages: ArrayLike,
        /,
    ):
        if not isinstance(basis, EffectiveMassND):
            raise TypeError("basis must be EffectiveMassND.")
        shape = basis.grid_shape
        dimension = basis.dimension
        size = int(np.prod(shape))
        epsilon = np.broadcast_to(
            np.asarray(permittivity, dtype=np.float64), shape
        ).copy()
        fixed = np.broadcast_to(
            np.asarray(fixed_charge_density, dtype=np.float64), shape
        ).copy()
        voltages = np.asarray(boundary_voltages, dtype=np.float64)
        if voltages.shape != (dimension, 2):
            raise ValueError("boundary_voltages must have shape (dimension, 2).")
        if (
            np.any(~np.isfinite(epsilon))
            or np.any(epsilon <= 0.0)
            or np.any(~np.isfinite(fixed))
            or np.any(~np.isfinite(voltages))
        ):
            raise ValueError("Poisson coefficients and boundary data are invalid.")
        spacings = tuple(float(np.diff(np.asarray(axis))[0]) for axis in basis.axes)
        volume = float(np.prod(spacings))
        matrix = np.zeros((size, size), dtype=np.float64)
        boundary_load = np.zeros((size,), dtype=np.float64)
        for index in product(*(range(count) for count in shape)):
            flat = np.ravel_multi_index(index, shape)
            for axis, spacing in enumerate(spacings):
                for direction in (-1, 1):
                    neighbor = list(index)
                    neighbor[axis] += direction
                    if 0 <= neighbor[axis] < shape[axis]:
                        neighbor_tuple = tuple(neighbor)
                        other = np.ravel_multi_index(neighbor_tuple, shape)
                        face_epsilon = 2.0 / (
                            1.0 / epsilon[index] + 1.0 / epsilon[neighbor_tuple]
                        )
                        conductance = face_epsilon * volume / spacing**2
                        matrix[flat, flat] += conductance
                        matrix[flat, other] -= conductance
                    else:
                        conductance = epsilon[index] * volume / spacing**2
                        matrix[flat, flat] += conductance
                        boundary_load[flat] += (
                            conductance * voltages[axis, 0 if direction < 0 else 1]
                        )
        scale = float(np.max(np.abs(np.diag(matrix))))
        operator = la.DenseLinearOperator(
            jnp.asarray(matrix / scale),
            properties=la.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id="quantum-poisson-nd-dirichlet",
        )
        budget = basis.base_hamiltonian.resources.workspace_bytes
        policy = la.LinearSolvePolicy(
            materialization=la.MaterializationPolicy(
                max_entries=size * size,
                max_bytes=budget,
            ),
            resources=la.SolveResourcePolicy(
                factorization_bytes=budget,
                workspace_bytes=budget,
            ),
        )
        self.operator = operator
        self.prepared = la.prepare(la.LinearSystem(operator), policy)
        self.fixed_charge_density = jnp.asarray(fixed.reshape((-1,)))
        self.cell_volumes = jnp.full((size,), volume, dtype=jnp.float64)
        self.boundary_load = jnp.asarray(boundary_load)
        self.scale = jnp.asarray(scale)
        self.grid_shape = shape
        self.dimension = dimension

    def right_hand_side(self, quantum_charge_density: ArrayLike, /) -> Array:
        quantum = jnp.asarray(quantum_charge_density, dtype=jnp.float64)
        if quantum.shape not in (self.grid_shape, self.cell_volumes.shape):
            raise ValueError("Quantum charge density must match the confinement grid.")
        charge = (self.fixed_charge_density + quantum.reshape((-1,))) * self.cell_volumes
        return (charge + self.boundary_load) / self.scale

    def solve(self, quantum_charge_density: ArrayLike, /):
        return la.solve(self.prepared, self.right_hand_side(quantum_charge_density))

    def residual(
        self, potential: ArrayLike, quantum_charge_density: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(potential, dtype=jnp.float64)
        if value.shape not in (self.grid_shape, self.cell_volumes.shape):
            raise ValueError("Potential must match the confinement grid.")
        residual = self.scale * (
            self.operator.mv(value.reshape((-1,)))
            - self.right_hand_side(quantum_charge_density)
        )
        return residual.reshape(self.grid_shape)


__all__ = ["QuantumPoissonND"]
