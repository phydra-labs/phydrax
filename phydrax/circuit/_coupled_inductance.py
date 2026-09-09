#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reciprocal coupled-inductor dynamics with certified magnetic energy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve_checked,
    TolerancePolicy,
)


class CoupledInductanceEnergy(StrictModule):
    stored_energy_j: Array
    resistive_power_w: Array
    flux_linkage_wb: Array
    finite: Array
    successful: Array


class CoupledInductanceLedger(StrictModule):
    initial_energy_j: Array
    candidate_energy_j: Array
    supplied_energy_j: Array
    resistive_loss_j: Array
    numerical_dissipation_j: Array
    closure_residual_j: Array
    finite: Array
    successful: Array


class CoupledInductanceStepResult(StrictModule):
    candidate_current_a: Array
    accepted_current_a: Array
    current_rate_a_s: Array
    energy: CoupledInductanceEnergy
    ledger: CoupledInductanceLedger
    linear_residual_norm: Array
    finite: Array
    domain_valid: Array
    sensitivity_valid: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class CoupledInductancePlan:
    inductance_h: np.ndarray
    resistance_ohm: np.ndarray
    winding_ids: tuple[str, ...]
    symmetry_tolerance: float = 1.0e-12
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        inductance = np.array(self.inductance_h, dtype=np.float64, copy=True)
        resistance = np.array(self.resistance_ohm, dtype=np.float64, copy=True)
        winding_ids = tuple(str(value).strip() for value in self.winding_ids)
        if (
            inductance.ndim != 2
            or inductance.shape[0] < 1
            or inductance.shape[0] != inductance.shape[1]
        ):
            raise ValueError("inductance_h must be a nonempty square matrix.")
        if resistance.shape != inductance.shape:
            raise ValueError("resistance_ohm must match inductance_h.")
        if (
            len(winding_ids) != inductance.shape[0]
            or len(set(winding_ids)) != len(winding_ids)
            or any(not value for value in winding_ids)
        ):
            raise ValueError("winding_ids must uniquely identify every winding.")
        tolerance = float(self.symmetry_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("symmetry_tolerance must be finite and positive.")
        if np.any(~np.isfinite(inductance)) or np.any(~np.isfinite(resistance)):
            raise ValueError("Inductance and resistance matrices must be finite.")
        if not np.allclose(inductance, inductance.T, rtol=0.0, atol=tolerance):
            raise ValueError("Mutual inductance matrix must be reciprocal and symmetric.")
        if not np.allclose(resistance, resistance.T, rtol=0.0, atol=tolerance):
            raise ValueError("Resistance matrix must be symmetric.")
        if np.min(np.linalg.eigvalsh(inductance)) <= tolerance:
            raise ValueError("Inductance matrix must be positive definite.")
        if np.min(np.linalg.eigvalsh(resistance)) < -tolerance:
            raise ValueError("Resistance matrix must be positive semidefinite.")
        inductance.setflags(write=False)
        resistance.setflags(write=False)
        object.__setattr__(self, "inductance_h", inductance)
        object.__setattr__(self, "resistance_ohm", resistance)
        object.__setattr__(self, "winding_ids", winding_ids)
        object.__setattr__(self, "symmetry_tolerance", tolerance)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "coupled-inductance-plan",
                    "inductance_h": array_tree_fingerprint(inductance),
                    "resistance_ohm": array_tree_fingerprint(resistance),
                    "windings": list(winding_ids),
                    "symmetry_tolerance": tolerance,
                }
            ),
        )

    def prepare(self) -> PreparedCoupledInductance:
        return PreparedCoupledInductance(
            jnp.asarray(self.inductance_h),
            jnp.asarray(self.resistance_ohm),
            self.winding_ids,
            self.plan_id,
        )


class PreparedCoupledInductance(StrictModule, NonTrainableState):
    inductance_h: Array
    resistance_ohm: Array
    winding_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def winding_count(self) -> int:
        return len(self.winding_ids)

    def energy(self, current_a: ArrayLike, /) -> CoupledInductanceEnergy:
        current = jnp.asarray(current_a, dtype=self.inductance_h.dtype)
        if current.shape != (self.winding_count,):
            raise ValueError("current_a must match the winding axis.")
        flux = self.inductance_h @ current
        stored = 0.5 * contract("i,i->", current, flux)
        resistive = contract("i,ij,j->", current, self.resistance_ohm, current)
        finite = (
            jnp.all(jnp.isfinite(flux)) & jnp.isfinite(stored) & jnp.isfinite(resistive)
        )
        successful = finite & (stored >= 0.0) & (resistive >= 0.0)
        return CoupledInductanceEnergy(stored, resistive, flux, finite, successful)

    def step_implicit_euler(
        self,
        current_a: ArrayLike,
        voltage_v: ArrayLike,
        dt_s: ArrayLike,
        /,
    ) -> CoupledInductanceStepResult:
        current = jnp.asarray(current_a, dtype=self.inductance_h.dtype)
        voltage = jnp.asarray(voltage_v, dtype=self.inductance_h.dtype)
        dt = jnp.asarray(dt_s, dtype=self.inductance_h.dtype)
        if (
            current.shape != (self.winding_count,)
            or voltage.shape != current.shape
            or dt.shape != ()
        ):
            raise ValueError(
                "Current, voltage, and timestep shapes do not match the coupled-inductor plan."
            )
        domain_valid = (
            jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(voltage))
            & jnp.isfinite(dt)
            & (dt > 0.0)
        )
        safe_dt = jnp.where(domain_valid, dt, 1.0)
        matrix = self.inductance_h + safe_dt * self.resistance_ohm
        rhs = self.inductance_h @ current + safe_dt * voltage
        operator = DenseLinearOperator(matrix, operator_id=self.plan_id)
        policy = LinearSolvePolicy(
            DenseLU(),
            tolerance=TolerancePolicy(relative=1.0e-12, absolute=1.0e-14),
            differentiation=DifferentiationPolicy("mathematical"),
            failure=FailurePolicy("status"),
        )
        linear, evidence = solve_checked(LinearSystem(operator), rhs, policy=policy)
        candidate = linear.value
        initial_energy = self.energy(current)
        candidate_energy = self.energy(candidate)
        increment = candidate - current
        supplied = safe_dt * contract("i,i->", candidate, voltage)
        resistive_loss = safe_dt * candidate_energy.resistive_power_w
        numerical_dissipation = 0.5 * contract(
            "i,ij,j->", increment, self.inductance_h, increment
        )
        closure = (
            candidate_energy.stored_energy_j
            - initial_energy.stored_energy_j
            - supplied
            + resistive_loss
            + numerical_dissipation
        )
        finite = (
            initial_energy.finite
            & candidate_energy.finite
            & jnp.isfinite(closure)
            & evidence.finite
        )
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(candidate_energy.stored_energy_j),
                jnp.abs(initial_energy.stored_energy_j),
            ),
        )
        tolerance = 1024.0 * jnp.finfo(candidate.dtype).eps * scale
        ledger_valid = finite & (jnp.abs(closure) <= tolerance)
        successful = domain_valid & linear.successful & evidence.valid & ledger_valid
        accepted = jnp.where(successful, candidate, current)
        rate = (candidate - current) / safe_dt
        ledger = CoupledInductanceLedger(
            initial_energy.stored_energy_j,
            candidate_energy.stored_energy_j,
            supplied,
            resistive_loss,
            numerical_dissipation,
            closure,
            finite,
            ledger_valid,
        )
        return CoupledInductanceStepResult(
            candidate,
            accepted,
            rate,
            candidate_energy,
            ledger,
            evidence.true_residual_norm,
            finite,
            domain_valid,
            successful,
            successful,
        )


__all__ = [
    "CoupledInductanceEnergy",
    "CoupledInductanceLedger",
    "CoupledInductancePlan",
    "CoupledInductanceStepResult",
    "PreparedCoupledInductance",
]
