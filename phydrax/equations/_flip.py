#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.flip import (
    FLIPDiagnostics,
    FLIPMethodPlan,
    FLIPParticleState,
    FLIPRejectionReason,
    FLIPRunStatus,
    FLIPRuntimeState,
    FLIPStepResult,
    PreparedFLIPParticleTransfer,
)


if TYPE_CHECKING:
    from ..solver._mac_free_surface import MACFreeSurfaceProjectionPlan


class FLIPProblemIR(StrictModule, NonTrainableState):
    """Constant-density inviscid free-surface FLIP problem declaration."""

    name: str = eqx.field(static=True)
    reference_density: float = eqx.field(static=True)
    acceleration: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        reference_density: float,
        acceleration: ArrayLike,
        /,
    ):
        identifier = str(name)
        density = float(reference_density)
        force = jnp.asarray(acceleration)
        if not identifier:
            raise ValueError("FLIP problem name must be nonempty.")
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("reference_density must be positive and finite.")
        if force.ndim != 1 or force.size not in (2, 3):
            raise ValueError("acceleration must be a two- or three-vector.")
        force = eqx.error_if(
            force, jnp.any(~jnp.isfinite(force)), "acceleration must be finite."
        )
        self.name = identifier
        self.reference_density = density
        self.acceleration = force
        self.problem_id = canonical_fingerprint(
            {
                "kind": "flip-problem-ir",
                "name": identifier,
                "reference_density": density,
                "acceleration_shape": list(force.shape),
            }
        )


def _shift_valid(value, valid, axis, offset, periodic):
    shifted_value = jnp.roll(value, offset, axis=axis)
    shifted_valid = jnp.roll(valid, offset, axis=axis)
    if periodic:
        return shifted_value, shifted_valid
    index = [slice(None)] * value.ndim
    index[axis] = 0 if offset > 0 else value.shape[axis] - 1
    shifted_valid = shifted_valid.at[tuple(index)].set(False)
    shifted_value = shifted_value.at[tuple(index)].set(0.0)
    return shifted_value, shifted_valid


def _extrapolate_component(value, valid, axes, layers):
    def body(_, carry):
        current, support = carry
        total = jnp.zeros_like(current)
        count = jnp.zeros_like(current)
        for axis_index, axis in enumerate(axes):
            for offset in (-1, 1):
                shifted, shifted_valid = _shift_valid(
                    current, support, axis_index, offset, axis.periodic
                )
                total = total + jnp.where(shifted_valid, shifted, 0.0)
                count = count + shifted_valid.astype(current.dtype)
        fill = (~support) & (count > 0.0)
        candidate = total / jnp.where(count > 0.0, count, 1.0)
        return jnp.where(fill, candidate, current), support | fill

    return jax.lax.fori_loop(0, layers, body, (value, valid))


class CompiledFLIPProblem(StrictModule, NonTrainableState):
    problem: FLIPProblemIR
    transfer: PreparedFLIPParticleTransfer
    projection: MACFreeSurfaceProjectionPlan
    method: FLIPMethodPlan
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> FLIPRuntimeState:
        position_ = jnp.asarray(position, dtype=self.transfer.particles.safe_masses.dtype)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        expected = (self.transfer.particles.capacity, self.transfer.dimension)
        if position_.shape != expected or velocity_.shape != expected:
            raise ValueError(f"FLIP position and velocity must have shape {expected}.")
        active = self.transfer.particles.active_mask[:, None]
        position_ = jnp.where(active, position_, 0.0)
        velocity_ = jnp.where(active, velocity_, 0.0)
        pressure = jnp.zeros(
            self.projection.operators.discretization.cell_shape, dtype=position_.dtype
        )
        return FLIPRuntimeState(
            FLIPParticleState(position_, velocity_),
            pressure,
            jnp.asarray(time, dtype=position_.dtype).reshape(()),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(FLIPRunStatus.SUCCESS), dtype=jnp.int32),
        )

    def _extrapolate(self, velocity, support):
        axes = self.projection.operators.discretization.grid.structured_axes
        values = []
        valid = []
        for component, mask in zip(velocity, support, strict=True):
            value, current = _extrapolate_component(
                component, mask, axes, self.method.extrapolation_layers
            )
            values.append(value)
            valid.append(current)
        return tuple(values), tuple(valid)

    def step_detailed(
        self, state: FLIPRuntimeState, step_size: ArrayLike, /
    ) -> FLIPStepResult:
        if not isinstance(state, FLIPRuntimeState):
            raise TypeError("state must be FLIPRuntimeState.")
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        routes = self.transfer.build(state.particles.position)
        p2g = self.transfer.particle_to_grid(
            routes, state.particles.velocity, self.problem.reference_density
        )
        threshold = self.method.liquid_fraction_threshold
        liquid = jax.lax.stop_gradient(p2g.liquid_fraction >= threshold)
        margin = jnp.min(jnp.abs(p2g.liquid_fraction - threshold), initial=jnp.inf)
        pre_grid, pre_support = self._extrapolate(p2g.velocity, p2g.face_support)
        axes = self.projection.operators.discretization.grid.structured_axes
        forced = tuple(
            component + dt * self.problem.acceleration[axis]
            for axis, component in enumerate(pre_grid)
        )
        boundary_stage = self.projection.boundaries.evaluate(state.time, None)
        forced = self.projection.boundaries.enforce(forced, boundary_stage)
        projected = self.projection.project(
            forced,
            liquid,
            dt,
            pressure=state.pressure,
            boundary_stage=boundary_stage,
        )
        post_grid, post_support = self._extrapolate(projected.velocity, pre_support)
        g2p = self.transfer.grid_to_particle(routes, pre_grid, post_grid)
        beta = self.method.pic_fraction
        next_velocity = (1.0 - beta) * (
            state.particles.velocity + g2p.flip_increment
        ) + beta * g2p.pic_velocity
        midpoint = state.particles.position + 0.5 * dt * next_velocity
        midpoint_routes = self.transfer.build(midpoint)
        midpoint_sample = self.transfer.grid_to_particle(
            midpoint_routes, post_grid, post_grid
        )
        displacement = dt * midpoint_sample.pic_velocity
        next_position = state.particles.position + displacement
        active = self.transfer.particles.active_mask
        next_position = jnp.where(active[:, None], next_position, 0.0)
        next_velocity = jnp.where(active[:, None], next_velocity, 0.0)
        widths = jnp.asarray(
            [jnp.min(axis.interval_widths) for axis in axes], dtype=dt.dtype
        )
        maximum_fraction = jnp.max(
            jnp.where(active[:, None], jnp.abs(displacement) / widths, 0.0),
            initial=0.0,
        )
        extrapolation_holes = sum(
            (jnp.sum(~mask, dtype=jnp.int32) for mask in post_support),
            jnp.asarray(0, dtype=jnp.int32),
        )
        mass = self.transfer.particles.masses.astype(dt.dtype)
        energy_before = 0.5 * jnp.sum(
            jnp.where(active, mass * jnp.sum(state.particles.velocity**2, axis=-1), 0.0)
        )
        energy_after = 0.5 * jnp.sum(
            jnp.where(active, mass * jnp.sum(next_velocity**2, axis=-1), 0.0)
        )
        finite = (
            jnp.isfinite(dt)
            & (dt > 0.0)
            & jnp.all(jnp.isfinite(next_position))
            & jnp.all(jnp.isfinite(next_velocity))
        )
        stable = maximum_fraction <= self.method.cfl_fraction
        transfer_success = p2g.successful & g2p.successful & midpoint_sample.successful
        extrapolation_success = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(mask | ~original)
                    for mask, original in zip(post_support, p2g.face_support, strict=True)
                )
            )
        )
        successful = (
            transfer_success
            & projected.successful
            & extrapolation_success
            & stable
            & finite
        )
        reason = jnp.asarray(int(FLIPRejectionReason.NONE), dtype=jnp.int32)
        reason = jnp.where(transfer_success, reason, reason | int(FLIPRejectionReason.TRANSFER))
        reason = jnp.where(projected.successful, reason, reason | int(FLIPRejectionReason.PROJECTION))
        reason = jnp.where(extrapolation_success, reason, reason | int(FLIPRejectionReason.EXTRAPOLATION))
        reason = jnp.where(stable, reason, reason | int(FLIPRejectionReason.STABILITY))
        reason = jnp.where(finite, reason, reason | int(FLIPRejectionReason.NONFINITE))
        candidate = FLIPRuntimeState(
            FLIPParticleState(next_position, next_velocity),
            projected.pressure,
            state.time + dt,
            state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            jnp.where(successful, int(FLIPRunStatus.SUCCESS), int(FLIPRunStatus.INVALID_STATE)).astype(jnp.int32),
        )
        accepted = jax.tree.map(
            lambda proposed, current: jnp.where(successful, proposed, current),
            candidate,
            state,
        )
        diagnostics = FLIPDiagnostics(
            projected.liquid_count,
            projected.air_count,
            margin,
            p2g.mass_balance_defect,
            p2g.momentum_balance_defect,
            extrapolation_holes,
            projected.residual_norm,
            projected.active_divergence_norm,
            maximum_fraction,
            energy_before,
            energy_after,
            successful,
            reason,
            projected,
        )
        return FLIPStepResult(
            candidate,
            accepted,
            pre_grid,
            post_grid,
            p2g.liquid_fraction,
            diagnostics,
            successful,
        )



def compile_flip_problem(
    problem: FLIPProblemIR,
    transfer: PreparedFLIPParticleTransfer,
    projection: MACFreeSurfaceProjectionPlan,
    method: FLIPMethodPlan,
    /,
) -> CompiledFLIPProblem:
    from ..solver._mac_free_surface import MACFreeSurfaceProjectionPlan

    if not isinstance(problem, FLIPProblemIR):
        raise TypeError("problem must be FLIPProblemIR.")
    if not isinstance(transfer, PreparedFLIPParticleTransfer):
        raise TypeError("transfer must be PreparedFLIPParticleTransfer.")
    if not isinstance(projection, MACFreeSurfaceProjectionPlan):
        raise TypeError("projection must be MACFreeSurfaceProjectionPlan.")
    if not isinstance(method, FLIPMethodPlan):
        raise TypeError("method must be FLIPMethodPlan.")
    if transfer.plan.operators.prepared_id != projection.operators.prepared_id:
        raise ValueError("FLIP transfer and projection must share one MAC grid.")
    if problem.acceleration.shape != (transfer.dimension,):
        raise ValueError("FLIP acceleration dimension must match particles and grid.")
    particles = transfer.particles
    residual_key = DiscretizationKey(
        "flip_free_surface",
        DiscretizationRole.RESIDUAL,
        domain_labels=projection.operators.discretization.grid.axis_names,
    )
    bundle = DiscretizationBundle(
        (
            DiscretizationRecord(
                particles.key,
                type(particles).__name__,
                particles.prepared_id,
                numeric_version=particles.numeric_version,
                precision_evidence_id=particles.precision_evidence_id,
                resource_evidence_id=particles.resource_evidence_id,
            ),
            DiscretizationRecord(
                residual_key,
                "compiled-flip-free-surface",
                canonical_fingerprint(
                    {
                        "problem": problem.problem_id,
                        "transfer": transfer.prepared_id,
                        "projection": projection.plan_id,
                        "method": method.method_id,
                    }
                ),
                dependency_key_ids=(particles.key.key_id,),
            ),
        )
    )
    identifier = canonical_fingerprint(
        {
            "kind": "compiled-flip-problem",
            "problem": problem.problem_id,
            "transfer": transfer.prepared_id,
            "projection": projection.plan_id,
            "method": method.method_id,
        }
    )
    return CompiledFLIPProblem(problem, transfer, projection, method, bundle, identifier)


__all__ = ["CompiledFLIPProblem", "FLIPProblemIR", "compile_flip_problem"]
