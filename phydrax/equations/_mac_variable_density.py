#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_variable_density import (
    MACDensityUpdateResult,
    MACVariableDensityTransportResult,
    PreparedMACVariableDensityOperators,
)


if TYPE_CHECKING:
    from ..solver._mac_variable_density import (
        MACVariableDensityProjectionPlan,
        MACVariableDensityProjectionResult,
        MACVariableDensityRateProjectionResult,
    )


def _maximum_abs(values: tuple[Array, ...], dtype: jnp.dtype, /) -> Array:
    if not values:
        return jnp.asarray(0.0, dtype=dtype)
    return jnp.max(jnp.stack(tuple(jnp.max(jnp.abs(value)) for value in values)))


class MACVariableDensityFlowProblem(StrictModule):
    """Fixed-grid miscible incompressible flow with dynamic viscosity.

    The optional source callable returns face-centered body acceleration. Density
    is an independently transported positive cell average; no constitutive EOS is
    implied.
    """

    dynamic_viscosity: Array
    body_acceleration: Any
    spatial_dimension: int = eqx.field(static=True)
    body_acceleration_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_dimension: int,
        dynamic_viscosity: ArrayLike,
        /,
        *,
        body_acceleration: Any = None,
        body_acceleration_id: str | None = None,
        problem_id: str | None = None,
    ):
        if isinstance(spatial_dimension, bool):
            raise TypeError("spatial_dimension must be an integer.")
        dimension = index(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError(
                "Variable-density incompressible flow requires dimension two or three."
            )
        raw_viscosity = jnp.asarray(dynamic_viscosity)
        if jnp.iscomplexobj(raw_viscosity):
            raise TypeError("dynamic_viscosity must be real.")
        viscosity = raw_viscosity.astype(float)
        if viscosity.shape != () or not bool(
            jnp.isfinite(viscosity) & (viscosity >= 0.0)
        ):
            raise ValueError("dynamic_viscosity must be one finite nonnegative scalar.")
        if body_acceleration is not None and not callable(body_acceleration):
            raise TypeError("body_acceleration must be callable or None.")
        if body_acceleration is None:
            acceleration_id = "none"
            if body_acceleration_id is not None:
                raise ValueError(
                    "body_acceleration_id must be omitted without body_acceleration."
                )
        else:
            acceleration_id = (
                "" if body_acceleration_id is None else str(body_acceleration_id)
            )
            if not acceleration_id:
                raise ValueError(
                    "A body_acceleration callable requires body_acceleration_id."
                )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mac-variable-density-flow-problem",
                    "dimension": dimension,
                    "dynamic_viscosity": float(viscosity),
                    "body_acceleration": acceleration_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.dynamic_viscosity = viscosity
        self.body_acceleration = body_acceleration
        self.spatial_dimension = dimension
        self.body_acceleration_id = acceleration_id
        self.problem_id = identifier


class MACVariableDensityState(StrictModule):
    """Physical view of density and prognostic face momentum with derived velocity."""

    density: Array
    face_momentum: FaceVelocity
    face_density: FaceVelocity
    velocity: FaceVelocity
    minimum_density: Array
    positive: Array
    finite: Array
    operators_id: str = eqx.field(static=True)


class MACVariableDensityStepRestriction(StrictModule):
    """Donor-cell advective and density-dependent viscous step bounds."""

    advective: Array
    diffusive: Array
    positivity: Array
    selected: Array


class MACVariableDensityRateResult(StrictModule):
    """One projected semidiscrete mass and momentum rate evaluation."""

    density_rate: Array
    momentum_rate: FaceVelocity
    velocity_rate: FaceVelocity
    face_density_rate: FaceVelocity
    momentum_advection: FaceVelocity
    viscous_force: FaceVelocity
    body_force: FaceVelocity
    pressure_force: FaceVelocity
    transport: MACVariableDensityTransportResult
    projection: MACVariableDensityRateProjectionResult
    momentum_rate_identity_residual: Array
    positive: Array
    finite: Array
    converged: Array
    successful: Array
    compilation_id: str = eqx.field(static=True)


class MACVariableDensityDiagnostics(StrictModule):
    """Mass, momentum, kinetic, positivity, and pressure ledgers."""

    mass: Array
    mass_rate: Array
    boundary_mass_flux: Array
    mass_balance_residual: Array
    total_momentum: Array
    total_momentum_rate: Array
    kinetic_energy: Array
    advective_kinetic_energy_rate: Array
    viscous_power: Array
    forcing_power: Array
    pressure_power: Array
    semidiscrete_kinetic_energy_rate: Array
    kinetic_energy_balance_residual: Array
    minimum_density: Array
    minimum_face_density: Array
    constant_density_face_residual: Array
    constant_density_velocity_residual: Array
    divergence_norm: Array
    pressure_residual_norm: Array
    pressure_gauge_residual: Array
    momentum_rate_identity_residual: Array
    positive: Array
    finite: Array
    projection_converged: Array
    successful: Array
    variable_density_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)


class MACVariableDensityStepResult(StrictModule):
    """Fail-closed forward donor stage followed by a pressure impulse."""

    state: Array
    physical_state: MACVariableDensityState
    density_update: MACDensityUpdateResult
    projection: MACVariableDensityProjectionResult
    mass_before: Array
    mass_after: Array
    momentum_before: Array
    momentum_after: Array
    kinetic_energy_before: Array
    kinetic_energy_after: Array
    minimum_density: Array
    positive: Array
    finite: Array
    converged: Array
    successful: Array
    compilation_id: str = eqx.field(static=True)


class CompiledMACVariableDensityDynamics(StrictModule):
    """Compiled fixed-grid miscible MAC dynamics in canonical flat coordinates."""

    problem: MACVariableDensityFlowProblem
    variable_density: PreparedMACVariableDensityOperators
    projection: MACVariableDensityProjectionPlan
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        problem: MACVariableDensityFlowProblem,
        variable_density: PreparedMACVariableDensityOperators,
        projection: MACVariableDensityProjectionPlan,
        /,
        *,
        compilation_id: str,
    ):
        discretization = variable_density.operators.discretization
        residual_key = DiscretizationKey(
            "mac_variable_density_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    discretization.key,
                    type(discretization).__name__,
                    discretization.prepared_id,
                    numeric_version=discretization.numeric_version,
                ),
                DiscretizationRecord(
                    residual_key,
                    "compiled-mac-variable-density-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.problem = problem
        self.variable_density = variable_density
        self.projection = projection
        self.discretization_bundle = bundle
        self.compilation_id = str(compilation_id)
        self.source_hash = problem.problem_id
        self.resolved_method = "mac-donor-variable-density-iterative-projected"

    @property
    def state_shape(self) -> tuple[int, ...]:
        size = (
            self.variable_density.operators.pressure_space.size
            + self.variable_density.operators.velocity_space.size
        )
        return (size,)

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"MAC density-momentum coordinates must have shape {self.state_shape}; "
                f"got {value.shape}."
            )
        dtype = self.variable_density.operators.pressure_space.dtype
        if value.dtype != dtype:
            raise TypeError(f"MAC density-momentum coordinates must have dtype {dtype}.")
        return value

    def pack_state(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        /,
    ) -> Array:
        cell = self.variable_density.validate_density(density)
        momentum = self.variable_density.validate_face_momentum(face_momentum)
        return jnp.concatenate(
            (
                self.variable_density.operators.pressure_space.flatten(cell),
                self.variable_density.operators.velocity_space.flatten(momentum),
            )
        )

    def unpack_state(self, state: ArrayLike, /) -> tuple[Array, FaceVelocity]:
        value = self.validate_state(state)
        pressure_size = self.variable_density.operators.pressure_space.size
        density = self.variable_density.operators.pressure_space.unflatten(
            value[:pressure_size]
        )
        density = self.variable_density.validate_density(density)
        momentum = tuple(
            self.variable_density.operators.velocity_space.unflatten(
                value[pressure_size:]
            )
        )
        momentum = self.variable_density.validate_face_momentum(momentum)
        return density, momentum

    def physical_state(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> MACVariableDensityState:
        del time, args
        density, momentum = self.unpack_state(state)
        face_density = self.variable_density.face_density(density, momentum)
        velocity = tuple(
            component / density_value
            for component, density_value in zip(momentum, face_density, strict=True)
        )
        finite = (
            jnp.all(jnp.isfinite(density))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in momentum))
            )
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in velocity))
            )
        )
        positive = (jnp.min(density) > 0.0) & jnp.all(
            jnp.stack(tuple(jnp.all(value > 0.0) for value in face_density))
        )
        return MACVariableDensityState(
            density=density,
            face_momentum=momentum,
            face_density=face_density,
            velocity=velocity,
            minimum_density=jnp.min(density),
            positive=positive,
            finite=finite,
            operators_id=self.variable_density.prepared_id,
        )

    def project_state(
        self,
        density: ArrayLike,
        face_momentum: FaceVelocity,
        /,
        *,
        pressure: ArrayLike | None = None,
    ) -> MACVariableDensityProjectionResult:
        cell = self.variable_density.validate_density(density)
        momentum = self.variable_density.validate_face_momentum(face_momentum)
        face_density = self.variable_density.face_density(cell, momentum)
        inverse_density = tuple(1.0 / value for value in face_density)
        return self.projection.project(
            momentum,
            inverse_density,
            1.0,
            pressure=pressure,
        )

    def project_coordinates(self, state: ArrayLike, /) -> Array:
        density, momentum = self.unpack_state(state)
        projected = self.project_state(density, momentum)
        coordinates = self.pack_state(density, projected.momentum)
        return eqx.error_if(
            coordinates,
            ~projected.converged,
            "Initial variable-density MAC pressure projection failed.",
        )

    def _body_acceleration(
        self,
        time: Array,
        physical: MACVariableDensityState,
        args: Any,
        /,
    ) -> FaceVelocity:
        if self.problem.body_acceleration is None:
            return tuple(jnp.zeros_like(value) for value in physical.velocity)
        acceleration = self.variable_density.operators.validate_velocity(
            self.problem.body_acceleration(time, physical, args)
        )
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in acceleration))
        )
        acceleration = tuple(
            eqx.error_if(
                value,
                ~finite,
                "MAC body acceleration must be finite.",
            )
            for value in acceleration
        )
        return self.variable_density.momentum.boundaries.homogeneous_rate(acceleration)

    def rate_components(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> MACVariableDensityRateResult:
        time_value = jnp.asarray(time)
        boundary_stage = self.variable_density.momentum.boundaries.evaluate(
            time_value, args
        )
        physical = self.physical_state(time_value, state, args)
        transport = self.variable_density.transport(
            physical.density,
            physical.face_momentum,
            stage=boundary_stage,
        )
        density_rate = transport.density_rate
        face_density_rate = self.variable_density.face_density_rate(
            density_rate, physical.face_momentum
        )
        acceleration = self._body_acceleration(time_value, physical, args)
        body_force = tuple(
            density_value * value
            for density_value, value in zip(
                physical.face_density, acceleration, strict=True
            )
        )
        viscosity = self.problem.dynamic_viscosity.astype(
            self.variable_density.operators.pressure_space.dtype
        )
        viscous_force = tuple(
            viscosity * value
            for value in self.variable_density.momentum.laplacian(
                physical.velocity, stage=boundary_stage
            )
        )
        unconstrained_momentum_rate = tuple(
            -advective + viscous + forcing
            for advective, viscous, forcing in zip(
                transport.momentum_advection,
                viscous_force,
                body_force,
                strict=True,
            )
        )
        unconstrained_momentum_rate = (
            self.variable_density.momentum.boundaries.homogeneous_rate(
                unconstrained_momentum_rate
            )
        )
        velocity_rate_before_pressure = tuple(
            inverse * (momentum_rate - velocity * density_rate_value)
            for inverse, momentum_rate, velocity, density_rate_value in zip(
                transport.face_inverse_density,
                unconstrained_momentum_rate,
                physical.velocity,
                face_density_rate,
                strict=True,
            )
        )
        projected = self.projection.project_velocity_rate(
            velocity_rate_before_pressure,
            transport.face_inverse_density,
        )
        pressure_force = projected.momentum_pressure_rate
        momentum_rate = tuple(
            value + pressure_value
            for value, pressure_value in zip(
                unconstrained_momentum_rate, pressure_force, strict=True
            )
        )
        reconstructed_momentum_rate = tuple(
            density_value * velocity_rate + velocity * density_rate_value
            for density_value, velocity_rate, velocity, density_rate_value in zip(
                physical.face_density,
                projected.velocity_rate,
                physical.velocity,
                face_density_rate,
                strict=True,
            )
        )
        identity = _maximum_abs(
            tuple(
                value - reconstructed
                for value, reconstructed in zip(
                    momentum_rate, reconstructed_momentum_rate, strict=True
                )
            ),
            self.variable_density.operators.pressure_space.dtype,
        )
        finite = (
            transport.finite
            & projected.finite
            & jnp.all(jnp.isfinite(density_rate))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in momentum_rate))
            )
        )
        scale = jnp.maximum(_maximum_abs(momentum_rate, density_rate.dtype), 1.0)
        tolerance = 512.0 * jnp.finfo(density_rate.dtype).eps * scale
        successful = (
            transport.successful
            & projected.successful
            & physical.positive
            & finite
            & (identity <= tolerance)
        )
        return MACVariableDensityRateResult(
            density_rate=density_rate,
            momentum_rate=momentum_rate,
            velocity_rate=projected.velocity_rate,
            face_density_rate=face_density_rate,
            momentum_advection=transport.momentum_advection,
            viscous_force=viscous_force,
            body_force=body_force,
            pressure_force=pressure_force,
            transport=transport,
            projection=projected,
            momentum_rate_identity_residual=identity,
            positive=physical.positive,
            finite=finite,
            converged=projected.converged,
            successful=successful,
            compilation_id=self.compilation_id,
        )

    def step_restriction(self, state: ArrayLike, /) -> MACVariableDensityStepRestriction:
        physical = self.physical_state(0.0, state)
        grid = self.variable_density.operators.discretization.grid
        inverse_advective = jnp.zeros(
            self.variable_density.operators.discretization.cell_shape,
            dtype=jnp.dtype(self.variable_density.momentum.precision.reduction_dtype),
        )
        inverse_diffusive = jnp.zeros_like(inverse_advective)
        for axis_index, axis in enumerate(grid.structured_axes):
            component = physical.velocity[axis_index]
            moved = jnp.moveaxis(component, axis_index, 0)
            cell_velocity = (
                0.5 * (moved + jnp.roll(moved, -1, axis=0))
                if axis.periodic
                else 0.5 * (moved[:-1] + moved[1:])
            )
            cell_velocity = jnp.moveaxis(cell_velocity, 0, axis_index)
            shape = [1] * inverse_advective.ndim
            shape[axis_index] = int(axis.interval_widths.size)
            widths = axis.interval_widths.reshape(tuple(shape))
            inverse_advective = inverse_advective + jnp.abs(cell_velocity) / widths
            inverse_diffusive = inverse_diffusive + 2.0 / widths**2
        advective_rate = jnp.max(inverse_advective)
        kinematic_viscosity_bound = (
            self.problem.dynamic_viscosity.astype(inverse_diffusive.dtype)
            / physical.minimum_density
        )
        diffusive_rate = kinematic_viscosity_bound * jnp.max(inverse_diffusive)
        safe_advective = jnp.where(advective_rate > 0.0, advective_rate, 1.0)
        safe_diffusive = jnp.where(diffusive_rate > 0.0, diffusive_rate, 1.0)
        advective = jnp.where(advective_rate > 0.0, 1.0 / safe_advective, jnp.inf)
        diffusive = jnp.where(diffusive_rate > 0.0, 1.0 / safe_diffusive, jnp.inf)
        positivity = advective
        selected = jnp.minimum(positivity, diffusive)
        reduction = self.variable_density.momentum.precision.reduction
        return MACVariableDensityStepRestriction(
            advective=reduction(advective),
            diffusive=reduction(diffusive),
            positivity=reduction(positivity),
            selected=reduction(selected),
        )

    def diagnostics(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> MACVariableDensityDiagnostics:
        physical = self.physical_state(time, state, args)
        rate = self.rate_components(time, state, args)
        transport = rate.transport
        measures = self.variable_density.operators.face_dual_measures
        total_momentum_rate = jnp.stack(
            tuple(
                jnp.sum(measure * value)
                for measure, value in zip(measures, rate.momentum_rate, strict=True)
            )
        )
        viscous_power = sum(
            jnp.sum(measure * velocity * force)
            for measure, velocity, force in zip(
                measures, physical.velocity, rate.viscous_force, strict=True
            )
        )
        forcing_power = sum(
            jnp.sum(measure * velocity * force)
            for measure, velocity, force in zip(
                measures, physical.velocity, rate.body_force, strict=True
            )
        )
        pressure_power = sum(
            jnp.sum(measure * velocity * force)
            for measure, velocity, force in zip(
                measures, physical.velocity, rate.pressure_force, strict=True
            )
        )
        semidiscrete_energy_rate = sum(
            jnp.sum(
                measure
                * (velocity * momentum_rate - 0.5 * velocity**2 * face_density_rate)
            )
            for measure, velocity, momentum_rate, face_density_rate in zip(
                measures,
                physical.velocity,
                rate.momentum_rate,
                rate.face_density_rate,
                strict=True,
            )
        )
        advective_energy_rate = sum(
            jnp.sum(
                measure
                * (velocity * (-advection) - 0.5 * velocity**2 * face_density_rate)
            )
            for measure, velocity, advection, face_density_rate in zip(
                measures,
                physical.velocity,
                rate.momentum_advection,
                rate.face_density_rate,
                strict=True,
            )
        )
        expected_energy_rate = (
            advective_energy_rate + viscous_power + forcing_power + pressure_power
        )
        volumes = self.variable_density.operators.discretization.cell_volumes
        divergence_norm = jnp.sqrt(jnp.sum(volumes * rate.projection.divergence_after**2))
        pressure_residual_norm = jnp.sqrt(
            jnp.sum(volumes * rate.projection.pressure_residual**2)
        )
        mean_density = transport.mass / jnp.sum(volumes)
        constant_face_residual = _maximum_abs(
            tuple(value - mean_density for value in physical.face_density),
            physical.density.dtype,
        )
        constant_velocity_residual = _maximum_abs(
            tuple(
                density_value * velocity - momentum
                for density_value, velocity, momentum in zip(
                    physical.face_density,
                    physical.velocity,
                    physical.face_momentum,
                    strict=True,
                )
            ),
            physical.density.dtype,
        )
        finite = (
            rate.finite
            & jnp.isfinite(semidiscrete_energy_rate)
            & jnp.isfinite(transport.kinetic_energy)
            & jnp.all(jnp.isfinite(total_momentum_rate))
        )
        successful = rate.successful & finite & physical.positive
        reduction = self.variable_density.momentum.precision.reduction
        return MACVariableDensityDiagnostics(
            mass=reduction(transport.mass),
            mass_rate=reduction(transport.mass_rate),
            boundary_mass_flux=reduction(transport.boundary_mass_flux),
            mass_balance_residual=reduction(transport.mass_balance_residual),
            total_momentum=transport.total_momentum,
            total_momentum_rate=total_momentum_rate,
            kinetic_energy=reduction(transport.kinetic_energy),
            advective_kinetic_energy_rate=reduction(advective_energy_rate),
            viscous_power=reduction(viscous_power),
            forcing_power=reduction(forcing_power),
            pressure_power=reduction(pressure_power),
            semidiscrete_kinetic_energy_rate=reduction(semidiscrete_energy_rate),
            kinetic_energy_balance_residual=reduction(
                semidiscrete_energy_rate - expected_energy_rate
            ),
            minimum_density=physical.minimum_density,
            minimum_face_density=transport.minimum_face_density,
            constant_density_face_residual=reduction(constant_face_residual),
            constant_density_velocity_residual=reduction(constant_velocity_residual),
            divergence_norm=reduction(divergence_norm),
            pressure_residual_norm=reduction(pressure_residual_norm),
            pressure_gauge_residual=reduction(rate.projection.gauge_defect),
            momentum_rate_identity_residual=reduction(
                rate.momentum_rate_identity_residual
            ),
            positive=physical.positive,
            finite=finite,
            projection_converged=rate.projection.converged,
            successful=successful,
            variable_density_id=self.variable_density.prepared_id,
            projection_id=self.projection.plan_id,
            compilation_id=self.compilation_id,
        )

    def step(
        self,
        time: ArrayLike,
        state: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> MACVariableDensityStepResult:
        physical_before = self.physical_state(time, state, args)
        rate = self.rate_components(time, state, args)
        dtype = physical_before.density.dtype
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Variable-density MAC step_size must be positive and finite.",
        )
        density_update = self.variable_density.update_density(
            physical_before.density,
            physical_before.face_momentum,
            step,
        )
        unconstrained_momentum_rate = tuple(
            momentum_rate - pressure_force
            for momentum_rate, pressure_force in zip(
                rate.momentum_rate, rate.pressure_force, strict=True
            )
        )
        momentum_candidate = tuple(
            momentum + step * momentum_rate
            for momentum, momentum_rate in zip(
                physical_before.face_momentum,
                unconstrained_momentum_rate,
                strict=True,
            )
        )
        momentum_candidate = self.variable_density.momentum.boundaries.homogeneous_rate(
            momentum_candidate
        )
        face_density = self.variable_density.face_density(
            density_update.density, momentum_candidate
        )
        projection = self.projection.project(
            momentum_candidate,
            tuple(1.0 / value for value in face_density),
            step,
        )
        successful = density_update.successful & projection.successful
        accepted_density = jnp.where(
            successful, density_update.density, physical_before.density
        )
        accepted_momentum = tuple(
            jnp.where(successful, candidate, original)
            for candidate, original in zip(
                projection.momentum,
                physical_before.face_momentum,
                strict=True,
            )
        )
        accepted_state = self.pack_state(accepted_density, accepted_momentum)
        physical_after = self.physical_state(
            jnp.asarray(time) + step, accepted_state, args
        )
        measures = self.variable_density.operators.face_dual_measures
        momentum_before = jnp.stack(
            tuple(
                jnp.sum(measure * value)
                for measure, value in zip(
                    measures, physical_before.face_momentum, strict=True
                )
            )
        )
        momentum_after = jnp.stack(
            tuple(
                jnp.sum(measure * value)
                for measure, value in zip(
                    measures, physical_after.face_momentum, strict=True
                )
            )
        )
        kinetic_before = 0.5 * sum(
            jnp.sum(measure * momentum * velocity)
            for measure, momentum, velocity in zip(
                measures,
                physical_before.face_momentum,
                physical_before.velocity,
                strict=True,
            )
        )
        kinetic_after = 0.5 * sum(
            jnp.sum(measure * momentum * velocity)
            for measure, momentum, velocity in zip(
                measures,
                physical_after.face_momentum,
                physical_after.velocity,
                strict=True,
            )
        )
        volumes = self.variable_density.operators.discretization.cell_volumes
        mass_before = jnp.sum(volumes * physical_before.density)
        mass_after = jnp.sum(volumes * physical_after.density)
        finite = (
            density_update.finite
            & projection.finite
            & physical_after.finite
            & jnp.isfinite(kinetic_after)
        )
        return MACVariableDensityStepResult(
            state=accepted_state,
            physical_state=physical_after,
            density_update=density_update,
            projection=projection,
            mass_before=mass_before,
            mass_after=mass_after,
            momentum_before=momentum_before,
            momentum_after=momentum_after,
            kinetic_energy_before=kinetic_before,
            kinetic_energy_after=kinetic_after,
            minimum_density=physical_after.minimum_density,
            positive=physical_after.positive,
            finite=finite,
            converged=projection.converged,
            successful=successful & finite & physical_after.positive,
            compilation_id=self.compilation_id,
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        rate = self.rate_components(time, state, args)
        coordinates = jnp.concatenate(
            (
                self.variable_density.operators.pressure_space.flatten(rate.density_rate),
                self.variable_density.operators.velocity_space.flatten(
                    rate.momentum_rate
                ),
            )
        )
        return eqx.error_if(
            coordinates,
            ~rate.successful,
            "Variable-density MAC rate evaluation failed.",
        )


def compile_mac_variable_density_flow(
    problem: MACVariableDensityFlowProblem,
    variable_density: PreparedMACVariableDensityOperators,
    projection: MACVariableDensityProjectionPlan,
    /,
) -> CompiledMACVariableDensityDynamics:
    """Compile explicit donor-cell mass/momentum dynamics with iterative projection."""
    from ..solver._mac_variable_density import MACVariableDensityProjectionPlan

    if not isinstance(problem, MACVariableDensityFlowProblem):
        raise TypeError("problem must be MACVariableDensityFlowProblem.")
    if not isinstance(variable_density, PreparedMACVariableDensityOperators):
        raise TypeError("variable_density must be PreparedMACVariableDensityOperators.")
    if not isinstance(projection, MACVariableDensityProjectionPlan):
        raise TypeError("projection must be MACVariableDensityProjectionPlan.")
    if problem.spatial_dimension != variable_density.dimension:
        raise ValueError("Variable-density problem and prepared MAC dimensions differ.")
    if projection.operators.prepared_id != variable_density.operators.prepared_id:
        raise ValueError(
            "Variable-density transport and pressure projection must share operators."
        )
    if not bool(variable_density.report.passed):
        raise RuntimeError(
            "Variable-density MAC constant-density reduction evidence failed."
        )
    identifier = canonical_fingerprint(
        {
            "kind": "compiled-mac-variable-density-flow",
            "problem": problem.problem_id,
            "variable_density": variable_density.prepared_id,
            "projection": projection.plan_id,
        }
    )
    return CompiledMACVariableDensityDynamics(
        problem,
        variable_density,
        projection,
        compilation_id=identifier,
    )


__all__ = [
    "CompiledMACVariableDensityDynamics",
    "MACVariableDensityDiagnostics",
    "MACVariableDensityFlowProblem",
    "MACVariableDensityRateResult",
    "MACVariableDensityState",
    "MACVariableDensityStepRestriction",
    "MACVariableDensityStepResult",
    "compile_mac_variable_density_flow",
]
