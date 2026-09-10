#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..discretization.flip import (
    FLIPMethodPlan,
    FLIPParticleState,
    FLIPRejectionReason,
    FLIPRunStatus,
    PreparedSparseFLIPParticleTransfer,
    SparseFLIPTransferState,
)
from ._flip import FLIPProblemIR


if TYPE_CHECKING:
    from ..solver._sparse_flip import (
        SparseMACFreeSurfaceProjectionPlan,
        SparseMACFreeSurfaceProjectionResult,
    )


class SparseFLIPRuntimeState(StrictModule):
    particles: FLIPParticleState
    pressure: Array
    transfer: SparseFLIPTransferState
    time: Array
    accepted_step: Array
    status: Array


class SparseFLIPDiagnostics(StrictModule):
    liquid_count: Array
    air_count: Array
    classification_margin: Array
    mass_balance_defect: Array
    momentum_balance_defect: Array
    projection_residual: Array
    divergence_norm: Array
    maximum_displacement_fraction: Array
    energy_before: Array
    energy_after: Array
    topology_generation: Array
    rejection_reason: Array
    successful: Array


class SparseFLIPStepResult(StrictModule):
    candidate_state: SparseFLIPRuntimeState
    accepted_state: SparseFLIPRuntimeState
    pre_grid_velocity: tuple[Array, ...]
    post_grid_velocity: tuple[Array, ...]
    liquid_fraction: Array
    projection: SparseMACFreeSurfaceProjectionResult
    diagnostics: SparseFLIPDiagnostics
    successful: Array


class CompiledSparseFLIPProblem(StrictModule, NonTrainableState):
    """Transactional compact FLIP transfer, pressure projection, and advection."""

    problem: FLIPProblemIR
    transfer: PreparedSparseFLIPParticleTransfer
    projection: SparseMACFreeSurfaceProjectionPlan
    method: FLIPMethodPlan
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: FLIPProblemIR,
        transfer: PreparedSparseFLIPParticleTransfer,
        projection: SparseMACFreeSurfaceProjectionPlan,
        method: FLIPMethodPlan,
        /,
    ) -> None:
        from ..solver._sparse_flip import SparseMACFreeSurfaceProjectionPlan

        if not isinstance(problem, FLIPProblemIR):
            raise TypeError("problem must be FLIPProblemIR.")
        if not isinstance(transfer, PreparedSparseFLIPParticleTransfer):
            raise TypeError("transfer must be PreparedSparseFLIPParticleTransfer.")
        if not isinstance(projection, SparseMACFreeSurfaceProjectionPlan):
            raise TypeError("projection must be SparseMACFreeSurfaceProjectionPlan.")
        if not isinstance(method, FLIPMethodPlan):
            raise TypeError("method must be FLIPMethodPlan.")
        if projection.transfer.prepared_id != transfer.prepared_id:
            raise ValueError("Sparse FLIP transfer and projection must match.")
        if problem.acceleration.shape != (transfer.dimension,):
            raise ValueError("FLIP acceleration dimension must match the transfer.")
        self.problem = problem
        self.transfer = transfer
        self.projection = projection
        self.method = method
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-sparse-flip-problem",
                "problem": problem.problem_id,
                "transfer": transfer.prepared_id,
                "projection": projection.plan_id,
                "method": method.method_id,
            }
        )

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> SparseFLIPRuntimeState:
        position_ = jnp.asarray(position, dtype=self.transfer.particles.safe_masses.dtype)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        expected = (self.transfer.particles.capacity, self.transfer.dimension)
        if position_.shape != expected or velocity_.shape != expected:
            raise ValueError(f"FLIP position and velocity must have shape {expected}.")
        active = self.transfer.particles.active_mask[:, None]
        particles = FLIPParticleState(
            jnp.where(active, position_, 0.0),
            jnp.where(active, velocity_, 0.0),
        )
        routes = self.transfer.build(particles.position)
        pressure = jnp.zeros(
            (self.transfer.plan.cell_topology.storage_capacity,),
            dtype=position_.dtype,
        )
        return SparseFLIPRuntimeState(
            particles=particles,
            pressure=pressure,
            transfer=routes,
            time=jnp.asarray(time, dtype=position_.dtype).reshape(()),
            accepted_step=jnp.asarray(0, dtype=jnp.int32),
            status=jnp.asarray(int(FLIPRunStatus.SUCCESS), dtype=jnp.int32),
        )

    @staticmethod
    def _align_pressure(
        previous: SparseFLIPRuntimeState,
        candidate_transfer: SparseFLIPTransferState,
    ) -> Array:
        logical = candidate_transfer.cell_topology.logical_node_ids.reshape((-1,))
        valid = candidate_transfer.cell_topology.node_valid.reshape((-1,))
        lookup = previous.transfer.cell_topology.lookup(logical, valid)
        retained = (
            lookup.supported
            & previous.transfer.cell_topology.node_valid.reshape((-1,))[
                lookup.storage_slots
            ]
        )
        return jnp.where(retained, previous.pressure[lookup.storage_slots], 0.0)

    def step_detailed(
        self,
        state: SparseFLIPRuntimeState,
        step_size: ArrayLike,
        /,
    ) -> SparseFLIPStepResult:
        if not isinstance(state, SparseFLIPRuntimeState):
            raise TypeError("state must be SparseFLIPRuntimeState.")
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        routes = self.transfer.build(
            state.particles.position,
            previous=state.transfer,
        )
        pressure = self._align_pressure(state, routes)
        p2g = self.transfer.particle_to_grid(
            routes,
            state.particles.velocity,
            self.problem.reference_density,
        )
        liquid = jnp.asarray(
            p2g.liquid_fraction >= self.method.liquid_fraction_threshold,
            dtype=bool,
        )
        classification_margin = jnp.min(
            jnp.abs(p2g.liquid_fraction - self.method.liquid_fraction_threshold),
            initial=jnp.inf,
        )
        pre_grid = tuple(
            jnp.where(support, value, 0.0)
            for value, support in zip(p2g.velocity, p2g.face_support, strict=True)
        )
        forced = tuple(
            value + dt * self.problem.acceleration[axis]
            for axis, value in enumerate(pre_grid)
        )
        projected = self.projection.project(
            routes,
            forced,
            liquid,
            dt,
            pressure=pressure,
        )
        g2p = self.transfer.grid_to_particle(
            routes,
            pre_grid,
            projected.velocity,
        )
        beta = self.method.pic_fraction
        next_velocity = (1.0 - beta) * (
            state.particles.velocity + g2p.flip_increment
        ) + beta * g2p.pic_velocity
        midpoint = state.particles.position + 0.5 * dt * next_velocity
        midpoint_routes = self.transfer.build_fixed(midpoint, routes)
        midpoint_sample = self.transfer.grid_to_particle(
            midpoint_routes,
            projected.velocity,
            projected.velocity,
        )
        displacement = dt * midpoint_sample.pic_velocity
        active = self.transfer.particles.active_mask
        next_particles = FLIPParticleState(
            jnp.where(active[:, None], state.particles.position + displacement, 0.0),
            jnp.where(active[:, None], next_velocity, 0.0),
        )
        widths = jnp.asarray(
            [
                jnp.min(axis.interval_widths)
                for axis in self.transfer.plan.index_space.structured_axes
            ],
            dtype=dt.dtype,
        )
        maximum_fraction = jnp.max(
            jnp.where(active[:, None], jnp.abs(displacement) / widths, 0.0),
            initial=0.0,
        )
        masses = self.transfer.particles.masses.astype(dt.dtype)
        energy_before = 0.5 * jnp.sum(
            jnp.where(
                active,
                masses * jnp.sum(state.particles.velocity**2, axis=-1),
                0.0,
            )
        )
        energy_after = 0.5 * jnp.sum(
            jnp.where(
                active,
                masses * jnp.sum(next_particles.velocity**2, axis=-1),
                0.0,
            )
        )
        finite = (
            jnp.isfinite(dt)
            & (dt > 0.0)
            & jnp.all(jnp.isfinite(next_particles.position))
            & jnp.all(jnp.isfinite(next_particles.velocity))
        )
        transfer_successful = (
            routes.successful
            & p2g.successful
            & g2p.successful
            & midpoint_routes.successful
            & midpoint_sample.successful
        )
        stable = maximum_fraction <= self.method.cfl_fraction
        successful = transfer_successful & projected.successful & stable & finite
        reason = jnp.asarray(int(FLIPRejectionReason.NONE), dtype=jnp.int32)
        reason = jnp.where(
            transfer_successful,
            reason,
            reason | int(FLIPRejectionReason.TRANSFER),
        )
        reason = jnp.where(
            projected.successful,
            reason,
            reason | int(FLIPRejectionReason.PROJECTION),
        )
        reason = jnp.where(stable, reason, reason | int(FLIPRejectionReason.STABILITY))
        reason = jnp.where(finite, reason, reason | int(FLIPRejectionReason.NONFINITE))
        failed_status = jnp.where(
            ~transfer_successful,
            int(FLIPRunStatus.TRANSFER_FAILED),
            jnp.where(
                ~projected.successful,
                int(FLIPRunStatus.PROJECTION_FAILED),
                jnp.where(
                    ~stable,
                    int(FLIPRunStatus.STABILITY_LIMIT_EXCEEDED),
                    int(FLIPRunStatus.NONFINITE_STATE),
                ),
            ),
        ).astype(jnp.int32)
        status = jnp.where(successful, int(FLIPRunStatus.SUCCESS), failed_status).astype(
            jnp.int32
        )
        candidate = SparseFLIPRuntimeState(
            particles=next_particles,
            pressure=projected.pressure,
            transfer=routes,
            time=state.time + dt,
            accepted_step=state.accepted_step + 1,
            status=status,
        )
        accepted = SparseFLIPRuntimeState(
            particles=FLIPParticleState(
                jnp.where(
                    successful, candidate.particles.position, state.particles.position
                ),
                jnp.where(
                    successful, candidate.particles.velocity, state.particles.velocity
                ),
            ),
            pressure=jnp.where(successful, candidate.pressure, state.pressure),
            transfer=tree_where(successful, routes, state.transfer),
            time=jnp.where(successful, candidate.time, state.time),
            accepted_step=jnp.where(
                successful, candidate.accepted_step, state.accepted_step
            ),
            status=status,
        )
        diagnostics = SparseFLIPDiagnostics(
            liquid_count=projected.liquid_count,
            air_count=projected.air_count,
            classification_margin=classification_margin,
            mass_balance_defect=p2g.mass_balance_defect,
            momentum_balance_defect=p2g.momentum_balance_defect,
            projection_residual=projected.residual_norm,
            divergence_norm=projected.active_divergence_norm,
            maximum_displacement_fraction=maximum_fraction,
            energy_before=energy_before,
            energy_after=energy_after,
            topology_generation=routes.cell_topology.generation,
            rejection_reason=reason,
            successful=successful,
        )
        return SparseFLIPStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            pre_grid_velocity=pre_grid,
            post_grid_velocity=projected.velocity,
            liquid_fraction=p2g.liquid_fraction,
            projection=projected,
            diagnostics=diagnostics,
            successful=successful,
        )


def compile_sparse_flip_problem(
    problem: FLIPProblemIR,
    transfer: PreparedSparseFLIPParticleTransfer,
    projection: SparseMACFreeSurfaceProjectionPlan,
    method: FLIPMethodPlan,
    /,
) -> CompiledSparseFLIPProblem:
    return CompiledSparseFLIPProblem(problem, transfer, projection, method)


__all__ = [
    "CompiledSparseFLIPProblem",
    "SparseFLIPDiagnostics",
    "SparseFLIPRuntimeState",
    "SparseFLIPStepResult",
    "compile_sparse_flip_problem",
]
