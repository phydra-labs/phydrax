#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from ._stress import AtomisticCellEvaluation


AtomisticMomentumFrame: TypeAlias = Literal["peculiar"]


class AtomisticDrivenStressPlan(StrictModule, NonTrainableState):
    momentum_frame: AtomisticMomentumFrame = eqx.field(static=True)
    require_solvent_stress: bool = eqx.field(static=True)
    require_stresslet: bool = eqx.field(static=True)
    require_brownian_stress: bool = eqx.field(static=True)
    cell_virial_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        momentum_frame: AtomisticMomentumFrame = "peculiar",
        require_solvent_stress: bool = False,
        require_stresslet: bool = False,
        require_brownian_stress: bool = False,
        cell_virial_tolerance: float = 1.0e-8,
    ):
        tolerance = float(cell_virial_tolerance)
        if momentum_frame != "peculiar":
            raise ValueError("Driven atomistic stress requires peculiar momenta.")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("cell_virial_tolerance must be finite and non-negative.")
        self.momentum_frame = momentum_frame
        self.require_solvent_stress = bool(require_solvent_stress)
        self.require_stresslet = bool(require_stresslet)
        self.require_brownian_stress = bool(require_brownian_stress)
        self.cell_virial_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-driven-stress-plan",
                "momentum_frame": momentum_frame,
                "require_solvent_stress": self.require_solvent_stress,
                "require_stresslet": self.require_stresslet,
                "require_brownian_stress": self.require_brownian_stress,
                "cell_virial_tolerance": tolerance,
                "pressure_tensor": "compression-positive",
                "cauchy_stress": "tension-positive",
            }
        )


class AtomisticDrivenStressResult(StrictModule):
    kinetic_pressure_tensor: Array
    configurational_pressure_tensor: Array
    particle_pressure_tensor: Array
    particle_cauchy_stress: Array
    solvent_cauchy_stress: Array
    stresslet_cauchy_stress: Array
    brownian_cauchy_stress: Array
    total_cauchy_stress: Array
    rate_of_deformation: Array
    flow_power: Array
    pressure: Array
    cell_virial_residual: Array
    solvent_available: Array
    stresslet_available: Array
    brownian_available: Array
    complete: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _optional_stress(
    value: ArrayLike | None,
    dtype,
    /,
) -> tuple[Array, Array]:
    if value is None:
        return jnp.zeros((3, 3), dtype=dtype), jnp.asarray(False)
    stress = jnp.asarray(value, dtype=dtype)
    if stress.shape != (3, 3):
        raise ValueError("Optional stress contributions must have shape (3, 3).")
    return stress, jnp.asarray(True)


def _atomistic_driven_stress_from_components(
    plan: AtomisticDrivenStressPlan,
    momentum: Array,
    inverse_mass: Array,
    mobile: Array,
    potential_virial: Array,
    cell_vectors: Array,
    velocity_gradient: ArrayLike,
    kinetic_to_energy: float,
    force_successful: Array,
    /,
    *,
    solvent_cauchy_stress: ArrayLike | None = None,
    stresslet_cauchy_stress: ArrayLike | None = None,
    brownian_cauchy_stress: ArrayLike | None = None,
    cell_evaluation: AtomisticCellEvaluation | None = None,
) -> AtomisticDrivenStressResult:
    gradient = jnp.asarray(velocity_gradient, dtype=momentum.dtype)
    if gradient.shape != (3, 3):
        raise ValueError("velocity_gradient must have shape (3, 3).")
    volume = jnp.abs(jnp.linalg.det(cell_vectors))
    weighted_momentum = jnp.where(mobile[:, None], momentum * inverse_mass[:, None], 0.0)
    kinetic_virial = kinetic_to_energy * contract(
        "ni,nj->ij", momentum, weighted_momentum
    )
    kinetic_pressure = kinetic_virial / volume
    configurational_pressure = potential_virial / volume
    particle_pressure = kinetic_pressure + configurational_pressure
    particle_cauchy = -particle_pressure
    solvent, solvent_available = _optional_stress(
        solvent_cauchy_stress, particle_cauchy.dtype
    )
    stresslet, stresslet_available = _optional_stress(
        stresslet_cauchy_stress, particle_cauchy.dtype
    )
    brownian, brownian_available = _optional_stress(
        brownian_cauchy_stress, particle_cauchy.dtype
    )
    total_cauchy = particle_cauchy + solvent + stresslet + brownian
    deformation_rate = 0.5 * (gradient + gradient.T)
    flow_power = volume * contract("ij,ij->", total_cauchy, deformation_rate)
    pressure = -jnp.trace(total_cauchy) / 3.0
    if cell_evaluation is None:
        cell_residual = jnp.asarray(jnp.nan, dtype=particle_cauchy.dtype)
        cell_valid = jnp.asarray(True)
    else:
        if not isinstance(cell_evaluation, AtomisticCellEvaluation):
            raise TypeError("cell_evaluation must be AtomisticCellEvaluation or None.")
        difference = cell_evaluation.stress + 0.5 * (
            configurational_pressure + configurational_pressure.T
        )
        cell_residual = jnp.max(jnp.abs(difference))
        scale = jnp.maximum(
            jnp.max(jnp.abs(cell_evaluation.stress)),
            jnp.max(jnp.abs(configurational_pressure)),
        )
        cell_valid = cell_evaluation.successful & (
            cell_residual <= plan.cell_virial_tolerance * jnp.maximum(scale, 1.0)
        )
    complete = (
        (jnp.asarray(not plan.require_solvent_stress) | solvent_available)
        & (jnp.asarray(not plan.require_stresslet) | stresslet_available)
        & (jnp.asarray(not plan.require_brownian_stress) | brownian_available)
    )
    finite = (
        jnp.isfinite(volume)
        & (volume > 0.0)
        & jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(kinetic_pressure))
        & jnp.all(jnp.isfinite(configurational_pressure))
        & jnp.all(jnp.isfinite(total_cauchy))
        & jnp.isfinite(flow_power)
        & jnp.isfinite(pressure)
    )
    successful = force_successful & finite & complete & cell_valid
    return AtomisticDrivenStressResult(
        kinetic_pressure,
        configurational_pressure,
        particle_pressure,
        particle_cauchy,
        solvent,
        stresslet,
        brownian,
        total_cauchy,
        deformation_rate,
        flow_power,
        pressure,
        cell_residual,
        solvent_available,
        stresslet_available,
        brownian_available,
        complete,
        finite,
        successful,
        plan.plan_id,
    )


def atomistic_driven_stress(
    plan: AtomisticDrivenStressPlan,
    dynamics: PreparedAtomisticDynamics,
    state: AtomisticDynamicsState,
    velocity_gradient: ArrayLike,
    /,
    *,
    solvent_cauchy_stress: ArrayLike | None = None,
    stresslet_cauchy_stress: ArrayLike | None = None,
    brownian_cauchy_stress: ArrayLike | None = None,
    cell_evaluation: AtomisticCellEvaluation | None = None,
) -> AtomisticDrivenStressResult:
    """Assemble pressure- and tension-positive stress conventions for driven flow."""
    if not isinstance(plan, AtomisticDrivenStressPlan):
        raise TypeError("plan must be AtomisticDrivenStressPlan.")
    if not isinstance(dynamics, PreparedAtomisticDynamics):
        raise TypeError("dynamics must be PreparedAtomisticDynamics.")
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError("state must be AtomisticDynamicsState.")
    if state.prepared_dynamics_id != dynamics.prepared_id:
        raise ValueError("State belongs to another atomistic dynamics runtime.")
    if dynamics.system.cell is None or state.cell_vectors.shape != (3, 3):
        raise ValueError("Driven atomistic stress requires one full 3-D periodic cell.")
    return _atomistic_driven_stress_from_components(
        plan,
        state.kinematics.momenta,
        dynamics.system.inverse_masses,
        dynamics.system.mobile_mask,
        state.force.virial,
        state.cell_vectors,
        velocity_gradient,
        dynamics.system.plan.units.kinetic_to_energy,
        state.force.successful,
        solvent_cauchy_stress=solvent_cauchy_stress,
        stresslet_cauchy_stress=stresslet_cauchy_stress,
        brownian_cauchy_stress=brownian_cauchy_stress,
        cell_evaluation=cell_evaluation,
    )


__all__ = [
    "AtomisticDrivenStressPlan",
    "AtomisticDrivenStressResult",
    "AtomisticMomentumFrame",
    "atomistic_driven_stress",
]
