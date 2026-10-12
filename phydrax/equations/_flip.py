#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._sharp_measures import QualifiedSharpGeometry
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    StructuredAxis,
)
from ..discretization.finite_volume import MACFreeSurfaceViscousMeasurePlan
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
from ..discretization.flip._viscous import viscous_flip_stage
from ..rheology import GeneralizedNewtonianLaw, MuILaw, MuIParameters
from ..typing import checked


if TYPE_CHECKING:
    from ..solver._mac_free_surface import MACFreeSurfaceProjectionPlan
    from ..solver._mac_variational_viscosity import MACVariationalViscosityPlan


type FLIPViscousLaw = GeneralizedNewtonianLaw | MuILaw


class FLIPProblemIR(StrictModule, NonTrainableState):
    """Constant-density free-surface FLIP with optional implicit viscous stress."""

    name: str = eqx.field(static=True)
    reference_density: float = eqx.field(static=True)
    acceleration: Array
    solid_geometry_source_id: str | None = eqx.field(static=True)
    viscous_law: FLIPViscousLaw | None = eqx.field(static=True)
    mu_i_parameters: MuIParameters | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        reference_density: float,
        acceleration: ArrayLike,
        /,
        *,
        solid_geometry_source_id: str | None = None,
        viscous_law: FLIPViscousLaw | None = None,
        mu_i_parameters: MuIParameters | None = None,
    ) -> None:
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
        if viscous_law is not None and not isinstance(
            viscous_law, (GeneralizedNewtonianLaw, MuILaw)
        ):
            raise TypeError(
                "viscous_law must be GeneralizedNewtonianLaw, MuILaw, or None."
            )
        if isinstance(viscous_law, MuILaw):
            if not isinstance(mu_i_parameters, MuIParameters):
                raise TypeError("MuILaw requires MuIParameters.")
        elif mu_i_parameters is not None:
            raise ValueError("mu_i_parameters requires MuILaw.")
        self.viscous_law = viscous_law
        self.mu_i_parameters = mu_i_parameters
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
                **({} if viscous_law is None else {"viscous_law": viscous_law.law_id}),
            }
        )


def _shift_valid(
    value: Array, valid: Array, axis: int, offset: int, periodic: bool
) -> tuple[Array, Array]:
    shifted_value = jnp.roll(value, offset, axis=axis)
    shifted_valid = jnp.roll(valid, offset, axis=axis)
    if periodic:
        return shifted_value, shifted_valid
    index: list[slice | int] = [slice(None)] * value.ndim
    index[axis] = 0 if offset > 0 else value.shape[axis] - 1
    shifted_valid = shifted_valid.at[tuple(index)].set(False)
    shifted_value = shifted_value.at[tuple(index)].set(0.0)
    return shifted_value, shifted_valid


def _extrapolate_component(
    value: Array,
    valid: Array,
    allowed: Array,
    axes: tuple[StructuredAxis, ...],
    layers: int,
) -> tuple[Array, Array]:
    def body(_: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
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
    viscosity: MACVariationalViscosityPlan | None
    viscous_measures: MACFreeSurfaceViscousMeasurePlan | None

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
        initial_pressure: ArrayLike | None = None,
    ) -> FLIPRuntimeState:
        position_ = jnp.asarray(position, dtype=self.transfer.particles.safe_masses.dtype)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        expected = (self.transfer.particles.capacity, self.transfer.dimension)
        if position_.shape != expected or velocity_.shape != expected:
            raise ValueError(f"FLIP position and velocity must have shape {expected}.")
        active = self.transfer.particles.active_mask[:, None]
        position_ = jnp.where(active, position_, 0.0)
        velocity_ = jnp.where(active, velocity_, 0.0)
        shape = self.projection.operators.discretization.cell_shape
        if isinstance(self.problem.viscous_law, MuILaw) and initial_pressure is None:
            raise ValueError("MuILaw FLIP requires an admitted initial_pressure field.")
        pressure = (
            jnp.zeros(shape, dtype=position_.dtype)
            if initial_pressure is None
            else jnp.asarray(initial_pressure, dtype=position_.dtype)
        )
        if pressure.shape != shape:
            raise ValueError(f"initial_pressure must have cell shape {shape}.")
        pressure = eqx.error_if(
            pressure, jnp.any(~jnp.isfinite(pressure)), "initial_pressure must be finite."
        )
        if isinstance(self.problem.viscous_law, MuILaw):
            initial_transfer = self.transfer.particle_to_grid(
                self.transfer.build(position_),
                velocity_,
                self.problem.reference_density,
                geometry=self.geometry,
            )
            liquid = (
                initial_transfer.liquid_fraction >= self.method.liquid_fraction_threshold
            )
            fluid = (
                jnp.ones_like(liquid)
                if self.geometry is None
                else self.geometry.cell_active
            )
            admitted = (
                initial_transfer.successful
                & jnp.any(liquid & fluid)
                & jnp.any(~liquid & fluid)
                & jnp.all(jnp.where(~liquid, pressure == 0.0, True))
            )
            pressure = eqx.error_if(
                pressure,
                ~admitted,
                "MuILaw requires a free surface and zero atmospheric-gauge pressure in air.",
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

    def _extrapolate(
        self, velocity: tuple[Array, ...], support: tuple[Array, ...]
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        axes = self.projection.operators.discretization.grid.structured_axes
        allowed = (
            tuple(jnp.ones_like(value, dtype=jnp.bool_) for value in velocity)
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

    @checked
    def step_detailed(
        self, state: FLIPRuntimeState, step_size: ArrayLike, /
    ) -> FLIPStepResult:
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
        viscous = None
        viscous_success = jnp.asarray(True)
        if self.problem.viscous_law is not None:
            if self.viscosity is None or self.viscous_measures is None:
                raise ValueError(
                    "Viscous FLIP compilation is missing its prepared stage."
                )
            measures = self.viscous_measures.evaluate_cell_fraction(
                liquid.astype(dt.dtype), 0.0, solid=self.geometry
            )
            fluid = (
                jnp.ones_like(liquid)
                if self.geometry is None
                else self.geometry.cell_active
            )
            pressure_admitted = (
                jnp.any(~liquid & fluid) & jnp.all(jnp.isfinite(state.pressure))
                if isinstance(self.problem.viscous_law, MuILaw)
                else jnp.asarray(True)
            )
            # Incremental pressure correction: viscosity must not absorb the
            # accepted pressure's support of gravity (Goda 1979; van Kan 1986).
            operators = self.projection.operators
            pressure_liquid = liquid & fluid
            previous_pressure = jnp.where(pressure_liquid, state.pressure, 0.0)
            volumes = operators.discretization.cell_volumes
            denominator = jnp.sum(jnp.where(pressure_liquid, volumes, 0.0))
            mean = jnp.sum(volumes * previous_pressure) / jnp.where(
                denominator > 0.0, denominator, 1.0
            )
            previous_pressure = jnp.where(
                jnp.all(pressure_liquid | ~fluid),
                jnp.where(pressure_liquid, previous_pressure - mean, 0.0),
                previous_pressure,
            )
            face_inverse = operators.interpolate_inverse_momentum(
                jnp.full(
                    operators.discretization.cell_shape,
                    dt / self.projection.density,
                    dtype=operators.pressure_space.dtype,
                )
            )
            gradient = self.projection.boundaries.pressure_gradient(
                previous_pressure, boundary_stage, homogeneous=False
            )
            pressure_impulse = tuple(
                coefficient * derivative
                for coefficient, derivative in zip(face_inverse, gradient, strict=True)
            )
            predictor_terms = (forced, pressure_impulse)
            forced = tuple(
                component - impulse
                for component, impulse in zip(forced, pressure_impulse, strict=True)
            )
            if self.geometry is not None:
                forced = tuple(
                    jnp.where(opened > 0.0, component, wall)
                    for component, opened, wall in zip(
                        forced,
                        self.geometry.face_open_fraction,
                        self.geometry.wall_velocity,
                        strict=True,
                    )
                )
            forced = self.projection.boundaries.enforce(forced, boundary_stage)
            forced, viscous = viscous_flip_stage(
                self.viscosity,
                self.problem.viscous_law,
                self.problem.mu_i_parameters,
                self.method,
                measures,
                forced,
                state.pressure,
                liquid,
                dt,
                boundary_stage,
                pressure_admitted,
                predictor_terms=predictor_terms,
            )
            viscous_success = viscous.successful
            projected = self.projection.project(
                forced,
                liquid,
                dt,
                pressure=state.pressure,
                boundary_stage=boundary_stage,
                geometry=self.geometry,
                incremental=True,
            )
        else:
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
            & viscous_success
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
        if viscous is not None:
            reason = jnp.where(
                viscous.successful, reason, reason | int(FLIPRejectionReason.VISCOSITY)
            )
            reason = jnp.where(
                viscous.closure_successful & viscous.pressure_admitted,
                reason,
                reason | int(FLIPRejectionReason.VISCOSITY_CLOSURE),
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
        if viscous is not None:
            failed_status = jnp.where(
                ~viscous.successful, int(FLIPRunStatus.VISCOSITY_FAILED), failed_status
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
            viscous,
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
    if (
        isinstance(problem.viscous_law, MuILaw)
        and projection.density != problem.reference_density
    ):
        raise ValueError("MuILaw physical pressure requires matching projection density.")
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
        if problem.viscous_law is not None:
            raise ValueError(
                "FLIP compilation does not support an embedded-wall viscous stage: "
                "exact cut-cell essential stress affine offsets are not implemented. "
                "Solid geometry is supported only with viscous_law=None."
            )
        from ..solver._mac_sharp_interface import MACSharpInterfaceProjectionPlan

        topology = MACSharpInterfaceProjectionPlan(
            projection.operators, projection.boundaries, geometry
        )
        if topology.component_count != 1:
            raise ValueError(
                "Static FLIP sharp composition currently requires one connected fluid component."
            )
    particles = transfer.particles
    viscosity = None
    viscous_measures = None
    if problem.viscous_law is not None:
        from ..discretization.finite_volume import MACMomentumPlan
        from ..solver._mac_variational_viscosity import MACVariationalViscosityPlan

        cells = np.prod(projection.operators.discretization.cell_shape)
        faces = sum(
            np.prod(layout.shape)
            for layout in projection.operators.discretization.face_layouts
        )
        itemsize = projection.operators.pressure_space.dtype.itemsize
        # Conservative matrix-free Picard/Krylov admission; no dense matrix.
        workspace_bytes = (64 * faces + 32 * cells) * itemsize
        state_bytes = (faces + cells) * itemsize
        if (
            workspace_bytes > method.resource_policy.maximum_workspace_bytes
            or state_bytes > method.resource_policy.maximum_state_bytes
        ):
            raise ValueError("Viscous FLIP exceeds its declared resource policy.")
        momentum = MACMomentumPlan(
            projection.operators, boundaries=projection.boundaries
        ).prepare()
        viscosity = MACVariationalViscosityPlan(
            momentum,
            tolerance=1.0e-4 * method.viscous_tolerance,
            maximum_iterations=method.maximum_viscous_linear_iterations,
        )
        viscous_measures = MACFreeSurfaceViscousMeasurePlan(
            projection.operators, problem.reference_density
        )
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
        viscosity,
        viscous_measures,
    )


__all__ = [
    "CompiledFLIPProblem",
    "FLIPProblemIR",
    "FLIPViscousLaw",
    "compile_flip_problem",
]
