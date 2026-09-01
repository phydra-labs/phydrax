#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...solver import (
    DifferentialProblem,
    DifferentialSolution,
    SeparableHamiltonianVectorField,
    solve_diffrax,
    StormerVerlet,
)
from ._forces import (
    AbstractAstrodynamicsForce,
    astrodynamics_continuous_system,
    PointMassGravity,
)
from ._state import CartesianOrbitState, CartesianOrbitTrajectory
from ._status import AstrodynamicsStatus
from ._two_body import propagate_universal_kepler, UniversalKeplerPolicy


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value, axis=-1))


class AstrodynamicsPropagationDiagnostics(StrictModule):
    specific_energy: Array
    angular_momentum: Array
    energy_drift: Array
    angular_momentum_drift: Array


class AstrodynamicsPropagationResult(StrictModule):
    solution: DifferentialSolution | None
    trajectory: CartesianOrbitTrajectory
    diagnostics: AstrodynamicsPropagationDiagnostics
    successful: Array
    plan_id: str = eqx.field(static=True)


class _PointMassPotentialGradient(StrictModule):
    mu: Array

    def __call__(self, time, position, args, /):
        del time, args
        radius = jnp.sqrt(jnp.sum(position * position))
        return self.mu * position / jnp.where(radius > 0.0, radius**3, 1.0)


class _UnitKineticGradient(StrictModule):
    def __call__(self, time, momentum, args, /):
        del time, args
        return momentum


class AstrodynamicsPropagationPlan(StrictModule):
    """Thin astrodynamics assembly over native analytic and solver substrates."""

    force: AbstractAstrodynamicsForce
    save_times: Array
    solver: Any
    stepsize_controller: Any
    adjoint: Any
    dt0: Array | None
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    dense: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        force: AbstractAstrodynamicsForce,
        save_times: ArrayLike,
        /,
        *,
        solver: Any = None,
        stepsize_controller: Any = None,
        adjoint: Any = None,
        dt0: ArrayLike | None = None,
        relative_tolerance: float = 1.0e-9,
        absolute_tolerance: float = 1.0e-11,
        max_steps: int = 4096,
        dense: bool = False,
    ):
        if not isinstance(force, AbstractAstrodynamicsForce):
            raise TypeError("force must be an AbstractAstrodynamicsForce.")
        times = jnp.asarray(save_times, dtype=float)
        if times.ndim != 1 or int(times.size) < 2:
            raise ValueError(
                "save_times must be a rank-one array with at least two nodes."
            )
        times_host = np.asarray(times)
        if np.any(~np.isfinite(times_host)) or np.any(np.diff(times_host) <= 0.0):
            raise ValueError("save_times must be finite and strictly increasing.")
        rtol = float(relative_tolerance)
        atol = float(absolute_tolerance)
        steps = int(max_steps)
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol <= 0.0 or atol <= 0.0:
            raise ValueError("Propagation tolerances must be finite and positive.")
        if steps <= 0:
            raise ValueError("max_steps must be positive.")
        if not isinstance(dense, bool):
            raise TypeError("dense must be a bool.")
        self.force = force
        self.save_times = times
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.adjoint = adjoint
        self.dt0 = (
            None if dt0 is None else jnp.asarray(dt0, dtype=times.dtype).reshape(())
        )
        self.relative_tolerance = rtol
        self.absolute_tolerance = atol
        self.max_steps = steps
        self.dense = dense
        self.plan_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-propagation-plan",
                "force": force.force_id,
                "num_times": int(times.size),
                "solver": "default" if solver is None else type(solver).__name__,
                "rtol": rtol,
                "atol": atol,
                "max_steps": steps,
                "dense": dense,
            }
        )

    def _drift(self):
        if isinstance(self.solver, StormerVerlet):
            if not isinstance(self.force, PointMassGravity):
                raise TypeError("StormerVerlet currently requires PointMassGravity.")
            return SeparableHamiltonianVectorField(
                _PointMassPotentialGradient(self.force.mu),
                _UnitKineticGradient(),
                3,
            )
        return astrodynamics_continuous_system(self.force)

    def _diagnostics(
        self, states: Array, args: Any, /
    ) -> AstrodynamicsPropagationDiagnostics:
        times = self.save_times
        evaluations = jax.vmap(
            lambda time, state: self.force.evaluate(time, state, args)
        )(times, states)
        speed_squared = jnp.sum(states[:, 3:] ** 2, axis=-1)
        specific_energy = 0.5 * speed_squared + evaluations.potential
        angular_momentum = jnp.cross(states[:, :3], states[:, 3:])
        energy_drift = specific_energy - specific_energy[0]
        angular_momentum_drift = _norm(angular_momentum - angular_momentum[0])
        return AstrodynamicsPropagationDiagnostics(
            specific_energy,
            angular_momentum,
            energy_drift,
            angular_momentum_drift,
        )

    def solve(
        self,
        initial_state: CartesianOrbitState,
        args: Any = None,
        /,
    ) -> AstrodynamicsPropagationResult:
        if not isinstance(initial_state, CartesianOrbitState):
            raise TypeError("initial_state must be a CartesianOrbitState.")
        self.force.context.require_compatible(initial_state.context)
        drift = self._drift()
        problem = DifferentialProblem(
            drift,
            initial_state.packed(),
            t0=self.save_times[0],
            t1=self.save_times[-1],
            args=args,
            state_geometry=(
                self.solver.geometry if isinstance(self.solver, StormerVerlet) else None
            ),
            problem_id=f"astrodynamics-problem:{self.plan_id}",
        )
        solution = solve_diffrax(
            problem,
            save_times=self.save_times,
            solver=self.solver,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            dt0=self.dt0,
            rtol=self.relative_tolerance,
            atol=self.absolute_tolerance,
            dense=self.dense,
            max_steps=self.max_steps,
            throw=False,
            solver_configuration_id=self.plan_id,
        )
        valid = solution.valid
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.NONCONVERGED),
        ).astype(jnp.int32)
        trajectory = CartesianOrbitTrajectory(
            solution.times,
            solution.states,
            valid,
            status,
            initial_state.context,
            trajectory_id=f"trajectory:{self.plan_id}",
        )
        diagnostics = self._diagnostics(solution.states, args)
        return AstrodynamicsPropagationResult(
            solution,
            trajectory,
            diagnostics,
            solution.successful,
            self.plan_id,
        )

    def solve_analytic_two_body(
        self,
        initial_state: CartesianOrbitState,
        /,
        *,
        policy: UniversalKeplerPolicy | None = None,
    ) -> AstrodynamicsPropagationResult:
        force = self.force
        if not isinstance(force, PointMassGravity):
            raise TypeError("Analytic propagation requires PointMassGravity.")
        force.context.require_compatible(initial_state.context)
        elapsed = self.save_times - self.save_times[0]

        def one(delta):
            result = propagate_universal_kepler(
                initial_state,
                delta,
                force.mu,
                policy=policy,
            )
            return result.state.packed(), result.valid, result.status

        states, valid, status = jax.vmap(one)(elapsed)
        trajectory = CartesianOrbitTrajectory(
            self.save_times,
            states,
            valid,
            status,
            initial_state.context,
            trajectory_id=f"analytic-trajectory:{self.plan_id}",
        )
        diagnostics = self._diagnostics(states, None)
        return AstrodynamicsPropagationResult(
            None,
            trajectory,
            diagnostics,
            jnp.all(valid),
            self.plan_id,
        )


__all__ = [
    "AstrodynamicsPropagationDiagnostics",
    "AstrodynamicsPropagationPlan",
    "AstrodynamicsPropagationResult",
]
