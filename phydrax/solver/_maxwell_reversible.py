#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._maxwell import CompatibleMaxwellState, PreparedCompatibleMaxwell


class MaxwellReconstructionDiagnostics(StrictModule):
    maximum_relative_residual: Array
    reconstructed_steps: Array
    checkpoint_count: int = eqx.field(static=True)
    passed: Array


class MaxwellReversibleArchive(StrictModule):
    checkpoint_steps: Array
    checkpoints: tuple[CompatibleMaxwellState, ...]


class MaxwellReversibleAdjointPlan(StrictModule):
    """Hybrid reversible Maxwell evolution with segmented exact checkpoints."""

    runtime: PreparedCompatibleMaxwell
    steps: int = eqx.field(static=True)
    checkpoint_count: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedCompatibleMaxwell,
        steps: int,
        /,
        *,
        checkpoint_count: int = 0,
        tolerance: float = 1e-8,
    ):
        if not isinstance(runtime, PreparedCompatibleMaxwell):
            raise TypeError("runtime must be PreparedCompatibleMaxwell.")
        count = int(steps)
        checkpoints = int(checkpoint_count)
        tolerance_ = float(tolerance)
        if count <= 0 or checkpoints < 0 or checkpoints >= count:
            raise ValueError("Reversible steps/checkpoint_count are invalid.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Reversible tolerance must be finite and positive.")
        if not runtime.capabilities.reversible:
            raise ValueError("Maxwell runtime is not reversible.")
        if (
            runtime.pml is not None
            or runtime.observers
            or runtime.plan.current_source is not None
        ):
            raise ValueError(
                "Reversible Maxwell requires no PML, observers, or external current."
            )
        if any(boundary.kind == "impedance" for boundary in runtime.boundaries):
            raise ValueError("Impedance boundaries are not reversible.")
        self.runtime = runtime
        self.steps = count
        self.checkpoint_count = checkpoints
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "maxwell-reversible-adjoint",
                "runtime": runtime.prepared_id,
                "steps": count,
                "checkpoint_count": checkpoints,
                "tolerance": tolerance_,
            }
        )

    def inverse_step(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        step_size: ArrayLike,
        /,
    ) -> CompatibleMaxwellState:
        dt = jnp.asarray(step_size)
        state_ = self.runtime._state(state)
        half = 0.5 * dt
        electric_new = self.runtime.electric_field(state_)
        magnetic_half = (
            state_.primary.magnetic_flux
            + half * self.runtime.plan.bridge.exterior_derivative(1, electric_new)
        )
        magnetic_field_half = self.runtime.constitutive.magnetic_field(
            magnetic_half,
            state_.auxiliary.material,
        )
        displacement_old = (
            state_.primary.electric_displacement
            - dt * self.runtime.plan.bridge.codifferential(2, magnetic_field_half)
        )
        electric_old = self.runtime.constitutive.electric_field(
            displacement_old,
            state_.auxiliary.material,
        )
        magnetic_old = (
            magnetic_half
            + half * self.runtime.plan.bridge.exterior_derivative(1, electric_old)
        )
        del time
        return self.runtime.pack(
            displacement_old,
            magnetic_old,
            state_.primary.charge,
            material_state=state_.auxiliary.material,
            boundary_state=state_.auxiliary.boundary,
            observations=state_.observations,
        )

    def _checkpoint_steps(self, /) -> tuple[int, ...]:
        if self.checkpoint_count == 0:
            return ()
        segments = self.checkpoint_count + 1
        return tuple(round(index * self.steps / segments) for index in range(1, segments))

    def forward_with_archive(
        self,
        initial_state: CompatibleMaxwellState,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> tuple[CompatibleMaxwellState, MaxwellReversibleArchive]:
        state = initial_state
        checkpoints = []
        checkpoint_steps = self._checkpoint_steps()
        for index in range(self.steps):
            state = self.runtime.leapfrog_step(
                jnp.asarray(time) + index * jnp.asarray(step_size),
                state,
                step_size,
            )
            if index + 1 in checkpoint_steps:
                checkpoints.append(state)
        return state, MaxwellReversibleArchive(
            checkpoint_steps=jnp.asarray(checkpoint_steps, dtype=jnp.int32),
            checkpoints=tuple(checkpoints),
        )

    def reconstruction_diagnostics(
        self,
        initial_state: CompatibleMaxwellState,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MaxwellReconstructionDiagnostics:
        final, _ = self.forward_with_archive(initial_state, time, step_size)
        reconstructed = final
        maximum = jnp.asarray(0.0)
        for index in range(self.steps - 1, -1, -1):
            previous = self.inverse_step(
                jnp.asarray(time) + index * jnp.asarray(step_size),
                reconstructed,
                step_size,
            )
            forward = self.runtime.leapfrog_step(
                jnp.asarray(time) + index * jnp.asarray(step_size),
                previous,
                step_size,
            )
            leaves_forward = jax.tree.leaves(forward)
            leaves_target = jax.tree.leaves(reconstructed)
            residual = max(
                jnp.linalg.norm(left - right) / jnp.maximum(1.0, jnp.linalg.norm(right))
                for left, right in zip(leaves_forward, leaves_target, strict=True)
                if eqx.is_inexact_array(left)
            )
            maximum = jnp.maximum(maximum, residual)
            reconstructed = previous
        return MaxwellReconstructionDiagnostics(
            maximum,
            jnp.asarray(self.steps, dtype=jnp.int32),
            self.checkpoint_count,
            maximum <= self.tolerance,
        )

    def evolve(
        self,
        initial_state: CompatibleMaxwellState,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> CompatibleMaxwellState:
        plan = self
        t0 = jnp.asarray(time)
        dt = jnp.asarray(step_size)

        @jax.custom_vjp
        def run(state):
            return plan.forward_with_archive(state, t0, dt)[0]

        def forward(state):
            final, archive = plan.forward_with_archive(state, t0, dt)
            return final, (final, archive)

        def backward(residual, cotangent):
            final, archive = residual
            del archive
            state = final
            cot = cotangent
            for index in range(plan.steps - 1, -1, -1):
                previous = plan.inverse_step(t0 + index * dt, state, dt)
                _, pullback = jax.vjp(
                    lambda value, _index=index: plan.runtime.leapfrog_step(
                        t0 + _index * dt,
                        value,
                        dt,
                    ),
                    previous,
                )
                cot = pullback(cot)[0]
                state = previous
            return (cot,)

        run.defvjp(forward, backward)
        return run(initial_state)


__all__ = [
    "MaxwellReconstructionDiagnostics",
    "MaxwellReversibleAdjointPlan",
    "MaxwellReversibleArchive",
]
