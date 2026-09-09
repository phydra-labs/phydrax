#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class BiotState(StrictModule):
    displacement: Array
    pressure_Pa: Array
    time_s: Array
    plan_id: str = eqx.field(static=True)


class BiotStepResult(StrictModule):
    state: BiotState
    mechanical_residual: Array
    mass_residual: Array
    residual_norm: Array
    successful: Array


class MixedBiotPoromechanicsPlan(StrictModule, NonTrainableState):
    """Backward-Euler mixed Biot system over native displacement/pressure operators.

    ``coupling`` maps displacement to pressure-space volumetric strain. Momentum
    uses its exact transpose. Storage and flow act on pressure. All inputs are
    assembled physical residual actions; no MPM-specific state is assumed.
    """

    elasticity: la.AbstractLinearOperator
    coupling: la.AbstractLinearOperator
    storage: la.AbstractLinearOperator
    flow: la.AbstractLinearOperator
    biot_coefficient: float = eqx.field(static=True)
    displacement_space: la.ArraySpace
    pressure_space: la.ArraySpace
    block_space: la.ArraySpace
    policy: la.LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        elasticity: la.AbstractLinearOperator,
        coupling: la.AbstractLinearOperator,
        storage: la.AbstractLinearOperator,
        flow: la.AbstractLinearOperator,
        biot_coefficient: float,
        /,
        *,
        policy: la.LinearSolvePolicy | None = None,
    ):
        if not all(
            isinstance(value, la.AbstractLinearOperator)
            for value in (elasticity, coupling, storage, flow)
        ):
            raise TypeError(
                "Biot operators must be native AbstractLinearOperator values."
            )
        if not isinstance(elasticity.source, la.ArraySpace) or not isinstance(
            storage.source, la.ArraySpace
        ):
            raise TypeError(
                "Biot displacement and pressure spaces must be ArraySpace values."
            )
        displacement, pressure = elasticity.source, storage.source
        if not (
            elasticity.source.compatible(elasticity.target)
            and storage.source.compatible(storage.target)
            and flow.source.compatible(pressure)
            and flow.target.compatible(pressure)
            and coupling.source.compatible(displacement)
            and coupling.target.compatible(pressure)
        ):
            raise ValueError(
                "Biot elasticity/coupling/storage/flow spaces do not compose."
            )
        alpha = float(biot_coefficient)
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Biot coefficient must be finite in [0,1].")
        selected = (
            la.LinearSolvePolicy(
                la.GMRES(restart=50, stagnation_iterations=50),
                tolerance=la.TolerancePolicy(
                    relative=1e-9, absolute=1e-11, max_steps=2000
                ),
                failure=la.FailurePolicy("status"),
            )
            if policy is None
            else policy
        )
        if not isinstance(selected, la.LinearSolvePolicy):
            raise TypeError("Biot solve policy must be LinearSolvePolicy.")
        self.elasticity, self.coupling, self.storage, self.flow = (
            elasticity,
            coupling,
            storage,
            flow,
        )
        self.biot_coefficient = alpha
        self.displacement_space, self.pressure_space = displacement, pressure
        self.block_space = la.ArraySpace(
            (displacement.size + pressure.size,),
            dtype=jnp.result_type(displacement.dtype, pressure.dtype),
        )
        self.policy = selected
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mixed-biot-poromechanics",
                "elasticity": elasticity.operator_id,
                "coupling": coupling.operator_id,
                "storage": storage.operator_id,
                "flow": flow.operator_id,
                "biot_coefficient": alpha,
            }
        )

    def initial_state(
        self, displacement: ArrayLike = 0.0, pressure_Pa: ArrayLike = 0.0, /
    ) -> BiotState:
        displacement_ = jnp.broadcast_to(
            jnp.asarray(displacement), self.displacement_space.shape
        )
        pressure = jnp.broadcast_to(jnp.asarray(pressure_Pa), self.pressure_space.shape)
        displacement_ = eqx.error_if(
            displacement_,
            jnp.any(~jnp.isfinite(displacement_)) | jnp.any(~jnp.isfinite(pressure)),
            "Biot initial displacement and pressure must be finite.",
        )
        return BiotState(displacement_, pressure, jnp.asarray(0.0), self.plan_id)

    def _operator(self, dt: Array):
        nu = self.displacement_space.size
        alpha = self.biot_coefficient

        def action(values):
            displacement = self.displacement_space.unflatten(values[:nu])
            pressure = self.pressure_space.unflatten(values[nu:])
            mechanical = self.elasticity.mv(displacement)
            pressure_force = self.coupling.transpose_mv(pressure)
            mechanical = mechanical - alpha * pressure_force
            volumetric = self.coupling.mv(displacement)
            mass = (
                alpha * volumetric / dt
                + self.storage.mv(pressure) / dt
                + self.flow.mv(pressure)
            )
            return jnp.concatenate(
                (
                    self.displacement_space.flatten(mechanical),
                    self.pressure_space.flatten(mass),
                )
            )

        return la.FunctionLinearOperator(
            action,
            source=self.block_space,
            target=self.block_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "biot-backward-euler-operator",
                    "plan": self.plan_id,
                    "dt": float(np.asarray(dt)),
                }
            ),
        )

    def step(
        self,
        previous: BiotState,
        dt_s: ArrayLike,
        /,
        *,
        mechanical_load: ArrayLike = 0.0,
        fluid_source: ArrayLike = 0.0,
    ) -> BiotStepResult:
        if not isinstance(previous, BiotState) or previous.plan_id != self.plan_id:
            raise ValueError("Biot state belongs to a different plan.")
        dt = jnp.asarray(dt_s)
        if dt.shape != ():
            raise ValueError("Biot timestep must be scalar.")
        dt = eqx.error_if(
            dt, ~jnp.isfinite(dt) | (dt <= 0), "Biot timestep must be positive."
        )
        load = jnp.broadcast_to(
            jnp.asarray(mechanical_load), self.displacement_space.shape
        )
        source = jnp.broadcast_to(jnp.asarray(fluid_source), self.pressure_space.shape)
        load = eqx.error_if(
            load,
            jnp.any(~jnp.isfinite(load)) | jnp.any(~jnp.isfinite(source)),
            "Biot mechanical load and fluid source must be finite.",
        )
        old_volumetric = self.coupling.mv(previous.displacement)
        old_storage = self.storage.mv(previous.pressure_Pa)
        rhs_pressure = (
            self.biot_coefficient * old_volumetric / dt + old_storage / dt + source
        )
        rhs = jnp.concatenate(
            (
                self.displacement_space.flatten(load),
                self.pressure_space.flatten(rhs_pressure),
            )
        )
        operator = self._operator(dt)
        result = la.solve(la.LinearSystem(operator), rhs, policy=self.policy)
        nu = self.displacement_space.size
        displacement = self.displacement_space.unflatten(result.value[:nu])
        pressure = self.pressure_space.unflatten(result.value[nu:])
        residual = operator.mv(result.value) - rhs
        mechanical = residual[:nu]
        mass = residual[nu:]
        norm = jnp.sqrt(jnp.real(jnp.vdot(residual, residual)))
        successful = result.successful & jnp.isfinite(norm)
        candidate = BiotState(displacement, pressure, previous.time_s + dt, self.plan_id)
        state = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, previous
        )
        return BiotStepResult(state, mechanical, mass, norm, successful)


__all__ = ["BiotState", "BiotStepResult", "MixedBiotPoromechanicsPlan"]
