#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._dynamics import (
    AtomisticDynamicsState,
    PreparedAtomisticDynamics,
    VelocityVerletPlan,
)
from ._hydrodynamic_brownian import HydrodynamicBrownianPlan
from ._hydrodynamic_mobility import ConstantIsotropicMobilityPlan
from ._thermal import stable_particle_normals
from ._thermodynamic import PreparedThermodynamicStateTable


class OverdampedAtomisticPlan(StrictModule, NonTrainableState):
    step_size: float = eqx.field(static=True)
    mobility: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_size: float,
        mobility: float,
        temperature: float,
        /,
        *,
        realization_id: int = 0,
    ):
        step = float(step_size)
        mobility_ = float(mobility)
        thermal = float(temperature)
        realization = int(realization_id)
        if (
            not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(mobility_)
            or mobility_ <= 0.0
            or not math.isfinite(thermal)
            or thermal <= 0.0
            or realization < 0
        ):
            raise ValueError("Overdamped atomistic controls are invalid.")
        self.step_size = step
        self.mobility = mobility_
        self.temperature = thermal
        self.realization_id = realization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "overdamped-atomistic-plan",
                "step_size": step,
                "mobility": mobility_,
                "temperature": thermal,
                "realization_id": realization,
            }
        )

    def prepare(self, dynamics: PreparedAtomisticDynamics, /):
        return HydrodynamicBrownianPlan(
            self.step_size,
            self.temperature,
            realization_id=self.realization_id,
        ).prepare(
            dynamics,
            ConstantIsotropicMobilityPlan(
                self.mobility,
                maximum_particles=dynamics.system.capacity,
            ),
        )


class GeneralizedLangevinRuntimePlan(StrictModule, NonTrainableState):
    transition_matrix: Array
    noise_factor: Array
    temperature: float = eqx.field(static=True)
    covariance_tolerance: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition_matrix: ArrayLike,
        noise_factor: ArrayLike,
        temperature: float,
        /,
        *,
        covariance_tolerance: float = 1.0e-10,
        realization_id: int = 0,
    ):
        transition = np.asarray(transition_matrix, dtype=np.float64)
        noise = np.asarray(noise_factor, dtype=np.float64)
        thermal = float(temperature)
        tolerance = float(covariance_tolerance)
        realization = int(realization_id)
        if (
            transition.ndim != 2
            or transition.shape[0] != transition.shape[1]
            or noise.shape != transition.shape
            or np.any(~np.isfinite(transition))
            or np.any(~np.isfinite(noise))
            or not math.isfinite(thermal)
            or thermal <= 0.0
            or not math.isfinite(tolerance)
            or tolerance < 0.0
            or realization < 0
        ):
            raise ValueError("Generalized Langevin runtime controls are invalid.")
        covariance_residual = (
            transition @ transition.T + noise @ noise.T - np.eye(transition.shape[0])
        )
        if np.max(np.abs(covariance_residual)) > tolerance:
            raise ValueError(
                "Generalized Langevin transition and noise violate discrete FDT."
            )
        self.transition_matrix = jnp.asarray(transition)
        self.noise_factor = jnp.asarray(noise)
        self.temperature = thermal
        self.covariance_tolerance = tolerance
        self.realization_id = realization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "generalized-langevin-runtime-plan",
                "transition_matrix": transition.tolist(),
                "noise_factor": noise.tolist(),
                "temperature": thermal,
                "covariance_tolerance": tolerance,
                "realization_id": realization,
            }
        )

    @property
    def auxiliary_count(self) -> int:
        return self.transition_matrix.shape[0] - 1

    def prepare(
        self, dynamics: PreparedAtomisticDynamics, /
    ) -> "PreparedGeneralizedLangevinRuntime":
        return PreparedGeneralizedLangevinRuntime(self, dynamics)


class GeneralizedLangevinRuntimeState(StrictModule):
    atomistic: AtomisticDynamicsState
    auxiliary: Array
    prepared_id: str = eqx.field(static=True)


class GeneralizedLangevinStepResult(StrictModule):
    candidate_state: GeneralizedLangevinRuntimeState
    accepted_state: GeneralizedLangevinRuntimeState
    covariance_residual: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedGeneralizedLangevinRuntime(StrictModule, NonTrainableState):
    plan: GeneralizedLangevinRuntimePlan
    dynamics: PreparedAtomisticDynamics
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: GeneralizedLangevinRuntimePlan,
        dynamics: PreparedAtomisticDynamics,
        /,
    ):
        if not isinstance(plan, GeneralizedLangevinRuntimePlan):
            raise TypeError("plan must be GeneralizedLangevinRuntimePlan.")
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(dynamics.integrator, VelocityVerletPlan):
            raise TypeError("Generalized Langevin composition requires Velocity Verlet.")
        if dynamics.constraints is not None:
            raise ValueError("Initial GLE runtime does not admit constraints.")
        self.plan = plan
        self.dynamics = dynamics
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-generalized-langevin-runtime",
                "plan": plan.plan_id,
                "dynamics": dynamics.prepared_id,
            }
        )

    def initialize(
        self, atomistic: AtomisticDynamicsState, /
    ) -> GeneralizedLangevinRuntimeState:
        if atomistic.prepared_dynamics_id != self.dynamics.prepared_id:
            raise ValueError("Atomistic state belongs to another dynamics runtime.")
        auxiliary = jnp.zeros(
            atomistic.kinematics.momenta.shape + (self.plan.auxiliary_count,),
            dtype=atomistic.kinematics.momenta.dtype,
        )
        return GeneralizedLangevinRuntimeState(atomistic, auxiliary, self.prepared_id)

    def _thermostat(
        self,
        state: GeneralizedLangevinRuntimeState,
        operator_id: int,
        /,
    ) -> tuple[GeneralizedLangevinRuntimeState, Array]:
        atomistic = state.atomistic
        momentum = atomistic.kinematics.momenta
        masses = self.dynamics.system.plan.masses
        scale = jnp.sqrt(
            masses[:, None]
            * self.dynamics.system.plan.units.boltzmann_constant
            * self.plan.temperature
            / self.dynamics.system.plan.units.kinetic_to_energy
        )
        normalized = momentum / scale
        combined = jnp.concatenate((normalized[..., None], state.auxiliary), axis=-1)
        channels = self.plan.transition_matrix.shape[0]
        noise = jnp.stack(
            [
                stable_particle_normals(
                    jr.key_data(atomistic.random_key),
                    self.dynamics.system.plan.particle_ids,
                    atomistic.step_index,
                    operator_id=operator_id * channels + channel,
                    realization_id=self.plan.realization_id,
                    dtype=momentum.dtype,
                )
                for channel in range(channels)
            ],
            axis=-1,
        )
        updated = contract(
            "...i,ji->...j", combined, self.plan.transition_matrix
        ) + contract("...i,ji->...j", noise, self.plan.noise_factor)
        mobile = self.dynamics.system.mobile_mask[:, None]
        new_momentum = jnp.where(mobile, updated[..., 0] * scale, 0.0)
        new_auxiliary = jnp.where(mobile[..., None], updated[..., 1:], 0.0)
        kinematics = eqx.tree_at(
            lambda value: value.momenta, atomistic.kinematics, new_momentum
        )
        new_atomistic = eqx.tree_at(lambda value: value.kinematics, atomistic, kinematics)
        successful = (
            jnp.all(jnp.isfinite(updated))
            & jnp.all(jnp.isfinite(scale))
            & jnp.all(scale > 0.0)
        )
        return (
            GeneralizedLangevinRuntimeState(
                new_atomistic, new_auxiliary, self.prepared_id
            ),
            successful,
        )

    def step(
        self,
        state: GeneralizedLangevinRuntimeState,
        thermodynamic: PreparedThermodynamicStateTable,
        /,
    ) -> GeneralizedLangevinStepResult:
        if not isinstance(state, GeneralizedLangevinRuntimeState):
            raise TypeError("state must be GeneralizedLangevinRuntimeState.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("GLE state belongs to another prepared runtime.")
        first, first_successful = self._thermostat(state, 0)
        mechanical = self.dynamics.step_detailed(first.atomistic, thermodynamic)
        middle = GeneralizedLangevinRuntimeState(
            mechanical.accepted_state, first.auxiliary, self.prepared_id
        )
        candidate, second_successful = self._thermostat(middle, 1)
        covariance = (
            self.plan.transition_matrix @ self.plan.transition_matrix.T
            + self.plan.noise_factor @ self.plan.noise_factor.T
            - jnp.eye(self.plan.transition_matrix.shape[0])
        )
        covariance_residual = jnp.max(jnp.abs(covariance))
        successful = (
            first_successful
            & mechanical.successful
            & second_successful
            & (covariance_residual <= self.plan.covariance_tolerance)
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        return GeneralizedLangevinStepResult(
            candidate,
            accepted,
            covariance_residual,
            successful,
            self.prepared_id,
        )


__all__ = [
    "GeneralizedLangevinRuntimePlan",
    "GeneralizedLangevinRuntimeState",
    "GeneralizedLangevinStepResult",
    "OverdampedAtomisticPlan",
    "PreparedGeneralizedLangevinRuntime",
]
