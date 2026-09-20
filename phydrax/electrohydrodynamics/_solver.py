#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conservative operator-based electrohydrodynamic field coupling."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._core import ElectrohydrodynamicLedger


@dataclass(frozen=True, slots=True)
class ElectrohydrodynamicState:
    free_charge_density_c_m3: Array
    electric_potential_v: Array
    velocity_m_s: Array


@dataclass(frozen=True, slots=True)
class ElectrohydrodynamicStep:
    state: ElectrohydrodynamicState
    electric_field_v_m: Array
    electric_body_force_n_m3: Array
    charge_balance_residual_c: Array
    momentum_residual_norm: Array
    ledger: ElectrohydrodynamicLedger
    successful: Array


@dataclass(frozen=True, slots=True)
class CoupledElectrohydrodynamicSolver:
    """Electrostatic, charge-transport, and creeping-flow coupling on one mesh."""

    cell_volumes_m3: Array
    poisson_operator: Array
    electric_field_operator: Array
    charge_transport_generator_s_inv: Array
    fluid_mobility_m3_s_kg: Array
    permittivity_f_m: Array
    conductivity_s_m: Array
    spatial_dimension: int

    @classmethod
    def create(
        cls,
        cell_volumes_m3: ArrayLike,
        poisson_operator: ArrayLike,
        electric_field_operator: ArrayLike,
        charge_transport_generator_s_inv: ArrayLike,
        fluid_mobility_m3_s_kg: ArrayLike,
        permittivity_f_m: ArrayLike,
        conductivity_s_m: ArrayLike,
        spatial_dimension: int,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> CoupledElectrohydrodynamicSolver:
        volumes = np.asarray(cell_volumes_m3, dtype=np.float64)
        poisson = np.asarray(poisson_operator, dtype=np.float64)
        field = np.asarray(electric_field_operator, dtype=np.float64)
        transport = np.asarray(charge_transport_generator_s_inv, dtype=np.float64)
        mobility = np.asarray(fluid_mobility_m3_s_kg, dtype=np.float64)
        permittivity = np.broadcast_to(
            np.asarray(permittivity_f_m, dtype=np.float64), volumes.shape
        )
        conductivity = np.broadcast_to(
            np.asarray(conductivity_s_m, dtype=np.float64), volumes.shape
        )
        cells = volumes.size
        vectors = cells * int(spatial_dimension)
        if volumes.ndim != 1 or cells == 0 or np.any(volumes <= 0):
            raise ValueError("EHD cell volumes must be a positive vector.")
        if spatial_dimension not in (1, 2, 3):
            raise ValueError("EHD spatial dimension must be one, two, or three.")
        if poisson.shape != (cells, cells) or field.shape != (vectors, cells):
            raise ValueError("EHD electrostatic operators have incompatible shapes.")
        if transport.shape != (cells, cells):
            raise ValueError("EHD charge transport generator has incompatible shape.")
        if not np.allclose(volumes @ transport, 0, atol=tolerance, rtol=tolerance):
            raise ValueError("EHD charge transport must conserve volume-weighted charge.")
        off_diagonal = transport - np.diag(np.diag(transport))
        if np.any(off_diagonal < -tolerance):
            raise ValueError(
                "EHD implicit charge generator must be positivity preserving."
            )
        if mobility.shape != (vectors, vectors) or not np.allclose(
            mobility, mobility.T, atol=tolerance, rtol=0
        ):
            raise ValueError(
                "EHD fluid mobility must be symmetric on vector coordinates."
            )
        if np.any(permittivity <= 0) or np.any(conductivity < 0):
            raise ValueError("EHD electric material fields are inadmissible.")
        return cls(
            jnp.asarray(volumes),
            jnp.asarray(poisson),
            jnp.asarray(field),
            jnp.asarray(transport),
            jnp.asarray(mobility),
            jnp.asarray(permittivity),
            jnp.asarray(conductivity),
            int(spatial_dimension),
        )

    def _fields(self, charge: Array, fixed_charge: Array, external_body_force: Array):
        potential = solve(
            LinearSystem(DenseLinearOperator(self.poisson_operator)),
            charge + fixed_charge,
            policy=LinearSolvePolicy(DenseLU()),
        )
        electric = -(self.electric_field_operator @ potential.value).reshape(
            (-1, self.spatial_dimension)
        )
        electric_body_force = charge[:, None] * electric
        total_body_force = electric_body_force + external_body_force
        velocity_flat = self.fluid_mobility_m3_s_kg @ total_body_force.reshape((-1,))
        velocity = velocity_flat.reshape(electric.shape)
        momentum_residual = (
            velocity_flat - self.fluid_mobility_m3_s_kg @ total_body_force.reshape((-1,))
        )
        return potential, electric, electric_body_force, velocity, momentum_residual

    def advance(
        self,
        state: ElectrohydrodynamicState,
        fixed_charge_density_c_m3: ArrayLike,
        step_size_s: float,
        /,
        *,
        external_body_force_n_m3: ArrayLike | None = None,
    ) -> ElectrohydrodynamicStep:
        charge = jnp.asarray(state.free_charge_density_c_m3)
        fixed = jnp.broadcast_to(jnp.asarray(fixed_charge_density_c_m3), charge.shape)
        if charge.shape != self.cell_volumes_m3.shape or step_size_s <= 0:
            raise ValueError("EHD state shape or step size is invalid.")
        external_force = (
            jnp.zeros((charge.size, self.spatial_dimension), dtype=charge.dtype)
            if external_body_force_n_m3 is None
            else jnp.asarray(external_body_force_n_m3)
        )
        if external_force.shape != (charge.size, self.spatial_dimension):
            raise ValueError("External EHD body force has incompatible shape.")
        transport_matrix = jnp.eye(charge.size, dtype=charge.dtype) - float(
            step_size_s
        ) * self.charge_transport_generator_s_inv.astype(charge.dtype)
        transported = solve(
            LinearSystem(DenseLinearOperator(transport_matrix)),
            charge,
            policy=LinearSolvePolicy(DenseLU()),
        )
        next_charge = transported.value
        potential, electric, body_force, velocity, momentum_residual = self._fields(
            next_charge, fixed, external_force
        )
        initial_charge = contract("q,q->", self.cell_volumes_m3, charge)
        final_charge = contract("q,q->", self.cell_volumes_m3, next_charge)
        charge_residual = final_charge - initial_charge
        electric_magnitude2 = contract("qd,qd->q", electric, electric)
        force_power_density = contract("qd,qd->q", body_force, velocity)
        ledger = ElectrohydrodynamicLedger(
            final_charge,
            jnp.asarray(0.0),
            0.5
            * contract(
                "q,q,q->",
                self.cell_volumes_m3,
                self.permittivity_f_m,
                electric_magnitude2,
            ),
            contract(
                "q,q,q->",
                self.cell_volumes_m3,
                self.conductivity_s_m,
                electric_magnitude2,
            ),
            contract("q,q->", self.cell_volumes_m3, force_power_density),
        )
        momentum_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(momentum_residual), momentum_residual))
        )
        successful = (
            transported.successful
            & potential.successful
            & jnp.all(jnp.isfinite(velocity))
            & ledger.finite
        )
        return ElectrohydrodynamicStep(
            ElectrohydrodynamicState(next_charge, potential.value, velocity),
            electric,
            body_force,
            charge_residual,
            momentum_norm,
            ledger,
            successful,
        )


__all__ = [
    "CoupledElectrohydrodynamicSolver",
    "ElectrohydrodynamicState",
    "ElectrohydrodynamicStep",
]
