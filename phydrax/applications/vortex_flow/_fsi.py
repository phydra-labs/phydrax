#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite
from ...discretization.particle import (
    PreparedRigidBodySet,
    rigid_body_kick_drift_kick,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ...dynamics import (
    SecondOrderDifferentialProblem,
    SecondOrderDifferentialSystem,
    TimeGrid,
)
from ...solver import (
    GeneralizedAlphaMethod,
    GeneralizedAlphaSolution,
    solve_generalized_alpha,
)


VortexRigidFluidCoupler = Callable[
    [Array, Any, RigidBodyKinematics, Any], tuple[Any, RigidBodyLoad, Array]
]


class VortexRigidCouplingEvidence(StrictModule):
    iterations: Array
    load_residual: Array
    velocity_residual: Array
    work_residual: Array
    aitken_factor: Array
    converged: Array
    finite: Array


class VortexRigidCouplingResult(StrictModule):
    fluid_state: Any
    kinematics: RigidBodyKinematics
    load: RigidBodyLoad
    evidence: VortexRigidCouplingEvidence
    successful: Array
    coupling_id: str = eqx.field(static=True)


class VortexRigidCouplingPlan(StrictModule, NonTrainableState):
    bodies: PreparedRigidBodySet
    mode: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        bodies: PreparedRigidBodySet,
        mode: str = "strong",
        /,
        *,
        tolerance: float = 1.0e-8,
        maximum_iterations: int = 12,
    ):
        if (
            not isinstance(bodies, PreparedRigidBodySet)
            or mode not in ("prescribed", "loose", "aitken", "strong")
            or float(tolerance) <= 0.0
            or int(maximum_iterations) <= 0
        ):
            raise ValueError("Vortex rigid coupling controls are invalid.")
        self.bodies, self.mode, self.tolerance, self.maximum_iterations = (
            bodies,
            mode,
            float(tolerance),
            int(maximum_iterations),
        )
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "vortex-rigid-coupling",
                "bodies": bodies.prepared_id,
                "mode": mode,
                "tolerance": self.tolerance,
                "maximum_iterations": self.maximum_iterations,
            }
        )

    def step(
        self,
        fluid_state: Any,
        kinematics: RigidBodyKinematics,
        load: RigidBodyLoad,
        time: ArrayLike,
        time_step: ArrayLike,
        coupler: VortexRigidFluidCoupler,
        args: Any = None,
        /,
    ) -> VortexRigidCouplingResult:
        if (
            not isinstance(kinematics, RigidBodyKinematics)
            or not isinstance(load, RigidBodyLoad)
            or not callable(coupler)
        ):
            raise TypeError("Rigid coupling requires native kinematics/load and coupler.")
        time_, dt = jnp.asarray(time), jnp.asarray(time_step)
        if time_.shape != () or dt.shape != ():
            raise ValueError("Rigid coupling time and step must be scalar.")
        if self.mode == "prescribed":
            next_fluid, next_load, work = coupler(
                time_ + dt,
                fluid_state,
                kinematics,
                args,
            )
            evidence = VortexRigidCouplingEvidence(
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                work,
                jnp.asarray(1.0),
                jnp.asarray(True),
                tree_allfinite((next_load.force, next_load.torque)),
            )
            return VortexRigidCouplingResult(
                next_fluid,
                kinematics,
                next_load,
                evidence,
                evidence.finite,
                self.coupling_id,
            )
        current_load = load
        current_kinematics, current_fluid = kinematics, fluid_state
        aitken = jnp.asarray(1.0, dtype=dt.dtype)
        previous_residual = jnp.zeros_like(load.force)
        converged = jnp.asarray(False)
        load_residual = jnp.asarray(jnp.inf, dtype=dt.dtype)
        velocity_residual = jnp.asarray(jnp.inf, dtype=dt.dtype)
        work_residual = jnp.asarray(jnp.inf, dtype=dt.dtype)
        iterations = 1 if self.mode == "loose" else self.maximum_iterations
        for iteration in range(iterations):
            fluid_candidate, fluid_load, fluid_work = coupler(
                time_ + dt,
                fluid_state,
                current_kinematics,
                args,
            )
            residual = fluid_load.force - current_load.force
            if self.mode == "aitken" and iteration > 0:
                delta = residual - previous_residual
                denominator = jnp.maximum(
                    jnp.sum(delta * delta),
                    jnp.finfo(dt.dtype).tiny,
                )
                aitken = jnp.clip(
                    -aitken * jnp.sum(previous_residual * delta) / denominator,
                    0.05,
                    1.5,
                )
            relaxed_load = RigidBodyLoad(
                current_load.force + aitken * residual,
                current_load.torque + aitken * (fluid_load.torque - current_load.torque),
            )

            def load_function(next_time, staged, inner_args):
                del inner_args
                _, staged_load, _ = coupler(
                    next_time,
                    fluid_state,
                    staged,
                    args,
                )
                return staged_load

            rigid = rigid_body_kick_drift_kick(
                self.bodies,
                kinematics,
                relaxed_load,
                time_,
                dt,
                load_function,
            )
            load_residual = jnp.linalg.norm(residual)
            velocity_residual = jnp.linalg.norm(
                rigid.kinematics.velocity - current_kinematics.velocity
            )
            body_work = jnp.sum(
                relaxed_load.force * (rigid.kinematics.position - kinematics.position)
            ) + jnp.sum(
                relaxed_load.torque
                * (rigid.kinematics.angular_velocity - kinematics.angular_velocity)
                * dt
            )
            work_residual = jnp.abs(fluid_work + body_work)
            converged = (
                (load_residual <= self.tolerance)
                & (velocity_residual <= self.tolerance)
                & (work_residual <= 10.0 * self.tolerance)
            )
            current_fluid = fluid_candidate
            current_kinematics = rigid.kinematics
            current_load = relaxed_load
            previous_residual = residual
            if self.mode == "loose":
                converged = rigid.successful
        finite = tree_allfinite(
            (
                current_kinematics.position,
                current_kinematics.velocity,
                current_load.force,
                current_load.torque,
            )
        )
        successful = converged & finite
        accepted_fluid = jax.tree_util.tree_map(
            lambda candidate, previous: jnp.where(
                successful,
                candidate,
                previous,
            ),
            current_fluid,
            fluid_state,
        )
        accepted_kinematics = jax.tree_util.tree_map(
            lambda candidate, previous: jnp.where(
                successful,
                candidate,
                previous,
            ),
            current_kinematics,
            kinematics,
        )
        accepted_load = jax.tree_util.tree_map(
            lambda candidate, previous: jnp.where(
                successful,
                candidate,
                previous,
            ),
            current_load,
            load,
        )
        evidence = VortexRigidCouplingEvidence(
            jnp.asarray(iterations, dtype=jnp.int32),
            load_residual,
            velocity_residual,
            work_residual,
            aitken,
            converged,
            finite,
        )
        return VortexRigidCouplingResult(
            accepted_fluid,
            accepted_kinematics,
            accepted_load,
            evidence,
            successful,
            self.coupling_id,
        )


class VortexFlexibleCouplingResult(StrictModule):
    fluid_state: Any
    structural_solution: GeneralizedAlphaSolution
    work_residual: Array
    successful: Array
    coupling_id: str = eqx.field(static=True)


class VortexFlexibleCouplingPlan(StrictModule, NonTrainableState):
    structure: SecondOrderDifferentialSystem
    method: GeneralizedAlphaMethod
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        structure: SecondOrderDifferentialSystem,
        /,
        *,
        method: GeneralizedAlphaMethod | None = None,
    ):
        if not isinstance(structure, SecondOrderDifferentialSystem):
            raise TypeError("structure must be SecondOrderDifferentialSystem.")
        self.structure = structure
        self.method = GeneralizedAlphaMethod() if method is None else method
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "vortex-flexible-coupling",
                "structure": structure.system_id,
                "method": self.method.method_id,
            }
        )

    def step(
        self,
        fluid_state: Any,
        configuration: ArrayLike,
        velocity: ArrayLike,
        acceleration: ArrayLike,
        time: ArrayLike,
        time_step: ArrayLike,
        fluid_load: ArrayLike,
        fluid_step: Callable[[Any, Array, Array, Any], tuple[Any, Array]],
        args: Any = None,
        /,
    ) -> VortexFlexibleCouplingResult:
        time_, dt = jnp.asarray(time), jnp.asarray(time_step)
        load = jnp.asarray(fluid_load)
        problem = SecondOrderDifferentialProblem(
            self.structure,
            configuration,
            velocity,
            initial_acceleration=acceleration,
            t0=time_,
            t1=time_ + dt,
            args={"vortex_load": load, "user_args": args},
            problem_id=f"{self.coupling_id}:structure",
        )
        grid = TimeGrid(
            jnp.stack((time_, time_ + dt)), time_id=f"{self.coupling_id}:step"
        )
        solution = solve_generalized_alpha(problem, grid, method=self.method)
        next_configuration = solution.configurations[-1]
        fluid_candidate, fluid_work = fluid_step(
            fluid_state, next_configuration, dt, args
        )
        structural_work = jnp.sum(
            load * (next_configuration - jnp.asarray(configuration))
        )
        residual = jnp.abs(fluid_work + structural_work)
        successful = solution.successful & jnp.isfinite(residual)
        accepted_fluid = jax.tree_util.tree_map(
            lambda candidate, previous: jnp.where(successful, candidate, previous),
            fluid_candidate,
            fluid_state,
        )
        return VortexFlexibleCouplingResult(
            accepted_fluid, solution, residual, successful, self.coupling_id
        )


__all__ = [
    "VortexFlexibleCouplingPlan",
    "VortexFlexibleCouplingResult",
    "VortexRigidCouplingEvidence",
    "VortexRigidCouplingPlan",
    "VortexRigidCouplingResult",
]
