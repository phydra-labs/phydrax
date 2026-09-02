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
from .._sharp_measures import QualifiedSharpGeometry
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
    FLIPSolidBoundaryPlan,
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
    solid_geometry_source_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        reference_density: float,
        acceleration: ArrayLike,
        /,
        *,
        solid_geometry_source_id: str | None = None,
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
        solid_source = (
            None if solid_geometry_source_id is None else str(solid_geometry_source_id)
        )
        if solid_source == "":
            raise ValueError("solid_geometry_source_id must be nonempty or None.")
        self.reference_density = density
        self.acceleration = force
        self.solid_geometry_source_id = solid_source
        self.problem_id = canonical_fingerprint(
            {
                "kind": "flip-problem-ir",
                "name": identifier,
                "reference_density": density,
                "acceleration_shape": list(force.shape),
                "solid_geometry_source_id": solid_source,
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


def _extrapolate_component(value, valid, allowed, axes, layers):
    def body(_, carry):
        current, support = carry
        current = jnp.where(allowed, current, 0.0)
        support = support & allowed
        total = jnp.zeros_like(current)
        count = jnp.zeros_like(current)
        for axis_index, axis in enumerate(axes):
            for offset in (-1, 1):
                shifted, shifted_valid = _shift_valid(
                    current, support, axis_index, offset, axis.periodic
                )
                total = total + jnp.where(shifted_valid, shifted, 0.0)
                count = count + shifted_valid.astype(current.dtype)
        fill = (~support) & allowed & (count > 0.0)
        candidate = total / jnp.where(count > 0.0, count, 1.0)
        return jnp.where(fill, candidate, current), support | fill

    return jax.lax.fori_loop(
        0, layers, body, (jnp.where(allowed, value, 0.0), valid & allowed)
    )


class CompiledFLIPProblem(StrictModule, NonTrainableState):
    problem: FLIPProblemIR
    transfer: PreparedFLIPParticleTransfer
    projection: MACFreeSurfaceProjectionPlan
    method: FLIPMethodPlan
    discretization_bundle: DiscretizationBundle
    geometry: QualifiedSharpGeometry | None
    solid_boundary: FLIPSolidBoundaryPlan | None
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
        geometry_id = "" if self.geometry is None else self.geometry.realization_id
        geometry_epoch = (
            jnp.asarray(-1, dtype=jnp.int32)
            if self.geometry is None
            else self.geometry.epoch
        )
        return FLIPRuntimeState(
            FLIPParticleState(position_, velocity_),
            pressure,
            jnp.asarray(time, dtype=position_.dtype).reshape(()),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(FLIPRunStatus.SUCCESS), dtype=jnp.int32),
            geometry_epoch,
            geometry_id,
        )

    def _extrapolate(self, velocity, support):
        axes = self.projection.operators.discretization.grid.structured_axes
        allowed = (
            tuple(jnp.ones_like(value, dtype=bool) for value in velocity)
            if self.geometry is None
            else self.geometry.face_active
        )
        values = []
        valid = []
        for component, mask, permitted in zip(velocity, support, allowed, strict=True):
            value, current = _extrapolate_component(
                component,
                mask,
                permitted,
                axes,
                self.method.extrapolation_layers,
            )
            values.append(jnp.where(permitted, value, 0.0))
            valid.append(current & permitted)
        return tuple(values), tuple(valid)

    def step_detailed(
        self, state: FLIPRuntimeState, step_size: ArrayLike, /
    ) -> FLIPStepResult:
        if not isinstance(state, FLIPRuntimeState):
            raise TypeError("state must be FLIPRuntimeState.")
        expected_geometry_id = (
            "" if self.geometry is None else self.geometry.realization_id
        )
        if state.geometry_id != expected_geometry_id:
            raise ValueError("FLIP runtime state belongs to another geometry identity.")
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        routes = self.transfer.build(state.particles.position)
        p2g = self.transfer.particle_to_grid(
            routes,
            state.particles.velocity,
            self.problem.reference_density,
            geometry=self.geometry,
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
            geometry=self.geometry,
        )
        post_grid, post_support = self._extrapolate(projected.velocity, pre_support)
        g2p = self.transfer.grid_to_particle(
            routes, pre_grid, post_grid, geometry=self.geometry
        )
        beta = self.method.pic_fraction
        next_velocity = (1.0 - beta) * (
            state.particles.velocity + g2p.flip_increment
        ) + beta * g2p.pic_velocity
        midpoint = state.particles.position + 0.5 * dt * next_velocity
        midpoint_routes = self.transfer.build(midpoint)
        midpoint_sample = self.transfer.grid_to_particle(
            midpoint_routes, post_grid, post_grid, geometry=self.geometry
        )
        displacement = dt * midpoint_sample.pic_velocity
        proposed_position = state.particles.position + displacement
        active = self.transfer.particles.active_mask
        proposed_position = jnp.where(active[:, None], proposed_position, 0.0)
        next_velocity = jnp.where(active[:, None], next_velocity, 0.0)
        if self.solid_boundary is None:
            solid = None
            next_position = proposed_position
            collision_success = jnp.asarray(True)
            collision_count = jnp.asarray(0, dtype=jnp.int32)
            wall_work = jnp.asarray(0.0, dtype=dt.dtype)
            maximum_penetration = jnp.asarray(0.0, dtype=dt.dtype)
        else:
            solid = self.solid_boundary.apply(
                FLIPParticleState(state.particles.position, next_velocity),
                proposed_position,
                self.transfer.particles.masses,
                active,
                state.time + dt,
            )
            next_position = solid.candidate_particles.position
            next_velocity = solid.candidate_particles.velocity
            collision_success = solid.successful
            collision_count = jnp.sum(solid.collided, dtype=jnp.int32)
            wall_work = solid.wall_work
            maximum_penetration = jnp.max(
                jnp.where(active, solid.penetration, 0.0), initial=0.0
            )
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
            jnp.where(
                active,
                mass * jnp.sum(state.particles.velocity**2, axis=-1),
                0.0,
            )
        )
        energy_after = 0.5 * jnp.sum(
            jnp.where(active, mass * jnp.sum(next_velocity**2, axis=-1), 0.0)
        )
        geometry_accepted = (
            jnp.asarray(True)
            if self.geometry is None
            else self.geometry.accepted & (state.geometry_epoch == self.geometry.epoch)
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
            & geometry_accepted
            & collision_success
            & stable
            & finite
        )
        reason = jnp.asarray(int(FLIPRejectionReason.NONE), dtype=jnp.int32)
        reason = jnp.where(
            transfer_success, reason, reason | int(FLIPRejectionReason.TRANSFER)
        )
        reason = jnp.where(
            projected.successful, reason, reason | int(FLIPRejectionReason.PROJECTION)
        )
        reason = jnp.where(
            extrapolation_success, reason, reason | int(FLIPRejectionReason.EXTRAPOLATION)
        )
        reason = jnp.where(stable, reason, reason | int(FLIPRejectionReason.STABILITY))
        reason = jnp.where(finite, reason, reason | int(FLIPRejectionReason.NONFINITE))
        reason = jnp.where(
            geometry_accepted,
            reason,
            reason | int(FLIPRejectionReason.GEOMETRY),
        )
        reason = jnp.where(
            collision_success,
            reason,
            reason | int(FLIPRejectionReason.COLLISION),
        )
        geometry_epoch = (
            jnp.asarray(-1, dtype=jnp.int32)
            if self.geometry is None
            else self.geometry.epoch
        )
        geometry_id = "" if self.geometry is None else self.geometry.realization_id
        failed_status = jnp.where(
            ~geometry_accepted,
            int(FLIPRunStatus.GEOMETRY_FAILED),
            jnp.where(
                ~collision_success,
                int(FLIPRunStatus.COLLISION_FAILED),
                int(FLIPRunStatus.INVALID_STATE),
            ),
        )
        candidate = FLIPRuntimeState(
            FLIPParticleState(next_position, next_velocity),
            projected.pressure,
            state.time + dt,
            state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            jnp.where(successful, int(FLIPRunStatus.SUCCESS), failed_status).astype(
                jnp.int32
            ),
            geometry_epoch,
            geometry_id,
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
            geometry_accepted,
            collision_count,
            wall_work,
            maximum_penetration,
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
            solid,
            geometry_id,
        )


def compile_flip_problem(
    problem: FLIPProblemIR,
    transfer: PreparedFLIPParticleTransfer,
    projection: MACFreeSurfaceProjectionPlan,
    method: FLIPMethodPlan,
    /,
    *,
    geometry: QualifiedSharpGeometry | None = None,
    solid_boundary: FLIPSolidBoundaryPlan | None = None,
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
    if (geometry is None) != (solid_boundary is None):
        raise ValueError(
            "Qualified FLIP solid geometry and collision plan must be supplied together."
        )
    if geometry is not None:
        if not isinstance(geometry, QualifiedSharpGeometry):
            raise TypeError("geometry must be QualifiedSharpGeometry or None.")
        if not isinstance(solid_boundary, FLIPSolidBoundaryPlan):
            raise TypeError("solid_boundary must be FLIPSolidBoundaryPlan or None.")
        if (
            geometry.operator_id != transfer.plan.operators.prepared_id
            or geometry.source_id != solid_boundary.source_id
            or problem.solid_geometry_source_id != geometry.source_id
        ):
            raise ValueError(
                "FLIP problem, transfer, sharp geometry, and collision source differ."
            )
        if not bool(np.asarray(geometry.accepted)):
            raise ValueError("FLIP compilation rejects failed sharp geometry.")
        if np.any(np.asarray(geometry.swept_cell_measure_rate) != 0.0):
            raise ValueError("FLIP sharp composition currently requires static geometry.")
        from ..solver._mac_sharp_interface import MACSharpInterfaceProjectionPlan

        topology = MACSharpInterfaceProjectionPlan(
            projection.operators, projection.boundaries, geometry
        )
        if topology.component_count != 1:
            raise ValueError(
                "Static FLIP sharp composition currently requires one connected "
                "fluid component."
            )
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
                        "geometry": (
                            None if geometry is None else geometry.realization_id
                        ),
                        "solid_boundary": (
                            None if solid_boundary is None else solid_boundary.plan_id
                        ),
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
            "geometry": None if geometry is None else geometry.realization_id,
            "solid_boundary": (
                None if solid_boundary is None else solid_boundary.plan_id
            ),
        }
    )
    return CompiledFLIPProblem(
        problem,
        transfer,
        projection,
        method,
        bundle,
        geometry,
        solid_boundary,
        identifier,
    )


__all__ = ["CompiledFLIPProblem", "FLIPProblemIR", "compile_flip_problem"]
