#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""JAX-compatible numerical kernels separated from host evidence envelopes."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule


class ElectronicKernelEvaluation(StrictModule):
    """Fixed-structure ground-state kernel values safe under JAX transforms."""

    energy: Array
    forces: Array
    successful: Array
    hessian: Array | None
    dipole: Array | None
    polarizability: Array | None
    stress: Array | None
    point_charge_forces: Array | None
    iterations: Array
    residual: Array

    def __init__(
        self,
        energy: ArrayLike,
        forces: ArrayLike,
        successful: ArrayLike,
        /,
        *,
        hessian: ArrayLike | None = None,
        dipole: ArrayLike | None = None,
        polarizability: ArrayLike | None = None,
        stress: ArrayLike | None = None,
        point_charge_forces: ArrayLike | None = None,
        iterations: ArrayLike = 0,
        residual: ArrayLike = jnp.nan,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        forces_ = jnp.asarray(forces, dtype=energy_.dtype)
        if forces_.ndim != 2 or forces_.shape[1] != 3:
            raise ValueError("Kernel forces must have shape (atom_capacity, 3).")
        hessian_ = None if hessian is None else jnp.asarray(hessian, dtype=energy_.dtype)
        if hessian_ is not None and hessian_.shape != forces_.shape + forces_.shape:
            raise ValueError("Kernel Hessian axes must match force axes twice.")
        dipole_ = None if dipole is None else jnp.asarray(dipole, dtype=energy_.dtype)
        if dipole_ is not None and dipole_.shape != (3,):
            raise ValueError("Kernel dipole must have shape (3,).")
        polarizability_ = (
            None
            if polarizability is None
            else jnp.asarray(polarizability, dtype=energy_.dtype)
        )
        if polarizability_ is not None and polarizability_.shape != (3, 3):
            raise ValueError("Kernel polarizability must have shape (3, 3).")
        stress_ = None if stress is None else jnp.asarray(stress, dtype=energy_.dtype)
        if stress_ is not None and stress_.shape != (3, 3):
            raise ValueError("Kernel stress must have shape (3, 3).")
        point_forces = (
            None
            if point_charge_forces is None
            else jnp.asarray(point_charge_forces, dtype=energy_.dtype)
        )
        if point_forces is not None and (
            point_forces.ndim != 2 or point_forces.shape[1] != 3
        ):
            raise ValueError("Point-charge forces must have shape (point_capacity, 3).")
        self.energy = energy_
        self.forces = forces_
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.hessian = hessian_
        self.dipole = dipole_
        self.polarizability = polarizability_
        self.stress = stress_
        self.point_charge_forces = point_forces
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.residual = jnp.asarray(residual, dtype=energy_.dtype).reshape(())


class PotentialEnergyKernelEvaluation(StrictModule):
    """Minimal compiled potential-energy-surface evaluation."""

    energy: Array
    forces: Array
    successful: Array

    def __init__(self, energy: ArrayLike, forces: ArrayLike, successful: ArrayLike, /):
        energy_ = jnp.asarray(energy).reshape(())
        forces_ = jnp.asarray(forces, dtype=energy_.dtype)
        if forces_.ndim != 2 or forces_.shape[1] != 3:
            raise ValueError("Kernel forces must have shape (atom_capacity, 3).")
        self.energy = energy_
        self.forces = forces_
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())


__all__ = ["ElectronicKernelEvaluation", "PotentialEnergyKernelEvaluation"]
