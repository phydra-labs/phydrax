#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._geometry_precision import GeometryPrecisionPolicy
from .._strict import StrictModule
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.finite_volume import (
    FaceVelocity,
    PreparedMACMomentumOperators,
)
from ..discretization.finite_volume._mac_boundary import MACBoundaryStageData
from ._incompressible import IncompressibleFlowProblem


if TYPE_CHECKING:
    from ..solver._structured_incompressible import (
        MACPressureProjectionPlan,
        MACRateProjectionResult,
    )


class MACStepRestriction(StrictModule):
    """Explicit advective and viscous step-size diagnostics."""

    advective: Array
    diffusive: Array
    selected: Array


class MACIncompressibleDiagnostics(StrictModule):
    """Constraint and energy evidence for one bounded MAC velocity state."""

    kinetic_energy: Array
    nonlinear_energy_rate: Array
    forcing_power: Array
    viscous_energy_rate: Array
    dissipation: Array
    boundary_power: Array
    open_backflow_dissipation: Array
    integrated_mass_flux: Array
    semidiscrete_energy_rate: Array
    energy_balance_defect: Array
    divergence_norm: Array
    boundary_defect: Array
    pressure_residual_norm: Array
    pressure_gauge_residual: Array
    finite: Array
    successful: Array
    projection_converged: Array
    pressure_closure: str = eqx.field(static=True)
    pressure_gauge: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)


class CompiledMACIncompressibleDynamics(StrictModule):
    """Velocity-only bounded MAC dynamics in canonical flat coordinates."""

    problem: IncompressibleFlowProblem
    momentum: PreparedMACMomentumOperators
    projection: MACPressureProjectionPlan
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        momentum: PreparedMACMomentumOperators,
        projection: MACPressureProjectionPlan,
        /,
        *,
        compilation_id: str,
    ):
        discretization = momentum.operators.discretization
        residual_key = DiscretizationKey(
            "mac_incompressible_form",
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
                    "compiled-mac-incompressible-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.problem = problem
        self.momentum = momentum
        self.projection = projection
        self.discretization_bundle = bundle
        self.compilation_id = str(compilation_id)
        self.source_hash = problem.problem_id
        self.resolved_method = "mac-symmetry-preserving-projected"

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.momentum.operators.velocity_space.size,)

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"MAC velocity coordinates must have shape {self.state_shape}; "
                f"got {value.shape}."
            )
        dtype = self.momentum.operators.pressure_space.dtype
        if value.dtype != dtype:
            raise TypeError(f"MAC velocity coordinates must have dtype {dtype}.")
        return value

    def boundary_stage(
        self, time: ArrayLike, args: Any = None, /
    ) -> MACBoundaryStageData:
        return self.momentum.boundaries.evaluate(time, args)

    def pack_velocity(
        self,
        velocity: FaceVelocity,
        /,
        *,
        time: ArrayLike = 0.0,
        args: Any = None,
    ) -> Array:
        stage = self.boundary_stage(time, args)
        value = self.momentum.boundaries.enforce(velocity, stage)
        coordinates = self.momentum.operators.velocity_space.flatten(value)
        return eqx.error_if(
            coordinates,
            ~stage.successful,
            "MAC boundary provider evaluation failed.",
        )

    def unpack_velocity(self, state: ArrayLike, /) -> FaceVelocity:
        value = self.validate_state(state)
        return tuple(self.momentum.operators.velocity_space.unflatten(value))

    def project_state(
        self,
        velocity: FaceVelocity,
        /,
        *,
        time: ArrayLike = 0.0,
        args: Any = None,
    ) -> Array:
        stage = self.boundary_stage(time, args)
        value = self.momentum.boundaries.enforce(velocity, stage)
        projected = self.projection.project(value, 1.0, boundary_stage=stage)
        coordinates = self.momentum.operators.velocity_space.flatten(projected.velocity)
        return eqx.error_if(
            coordinates,
            ~stage.successful | ~projected.converged,
            "Initial MAC pressure projection failed.",
        )

    def physical_state(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> FaceVelocity:
        stage = self.boundary_stage(time, args)
        velocity = self.momentum.boundaries.enforce(self.unpack_velocity(state), stage)
        coordinates = self.momentum.operators.velocity_space.flatten(velocity)
        checked = eqx.error_if(
            coordinates,
            ~stage.successful,
            "MAC boundary provider evaluation failed.",
        )
        return tuple(self.momentum.operators.velocity_space.unflatten(checked))

    def _forcing(
        self,
        time: Array,
        velocity: FaceVelocity,
        args: Any,
        /,
    ) -> FaceVelocity:
        if self.problem.forcing is None:
            return tuple(jnp.zeros_like(value) for value in velocity)
        forcing = self.momentum.operators.validate_velocity(
            self.problem.forcing(time, velocity, args)
        )
        return self.momentum.boundaries.homogeneous_rate(forcing)

    def _rate_components(
        self,
        time: Array,
        state: ArrayLike,
        args: Any,
        stage: MACBoundaryStageData,
        /,
    ) -> tuple[FaceVelocity, FaceVelocity, FaceVelocity, FaceVelocity]:
        velocity = self.momentum.boundaries.enforce(self.unpack_velocity(state), stage)
        convection = self.momentum.convection(velocity, stage=stage)
        diffusion = self.momentum.laplacian(velocity, stage=stage)
        forcing = self._forcing(time, velocity, args)
        viscosity = self.problem.viscosity.astype(
            self.momentum.operators.pressure_space.dtype
        )
        unconstrained = tuple(
            -advective + viscosity * viscous + source
            for advective, viscous, source in zip(
                convection, diffusion, forcing, strict=True
            )
        )
        return (
            self.momentum.boundaries.enforce_rate(unconstrained, stage),
            convection,
            diffusion,
            forcing,
        )

    def rate_components(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[FaceVelocity, FaceVelocity, FaceVelocity, FaceVelocity]:
        time_ = jnp.asarray(time)
        stage = self.boundary_stage(time_, args)
        return self._rate_components(time_, state, args, stage)

    def unconstrained_rate(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> FaceVelocity:
        return self.rate_components(time, state, args)[0]

    def rate_projection(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> MACRateProjectionResult:
        time_ = jnp.asarray(time)
        stage = self.boundary_stage(time_, args)
        unconstrained = self._rate_components(time_, state, args, stage)[0]
        return self.projection.project_rate(unconstrained, boundary_stage=stage)

    def pressure_field(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        projected = self.rate_projection(time, state, args)
        return eqx.error_if(
            projected.pressure,
            ~projected.converged,
            "MAC pressure recovery failed.",
        )

    def step_restriction(self, state: ArrayLike, /) -> MACStepRestriction:
        velocity = self.unpack_velocity(state)
        grid = self.momentum.operators.discretization.grid
        inverse_advective = jnp.zeros(
            self.momentum.operators.discretization.cell_shape,
            dtype=jnp.dtype(self.momentum.precision.reduction_dtype),
        )
        inverse_diffusive = jnp.zeros_like(inverse_advective)
        for axis_index, axis in enumerate(grid.structured_axes):
            component = velocity[axis_index]
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
        viscosity = self.problem.viscosity.astype(inverse_diffusive.dtype)
        diffusive_rate = viscosity * jnp.max(inverse_diffusive)
        safe_advective = jnp.where(advective_rate > 0.0, advective_rate, 1.0)
        safe_diffusive = jnp.where(diffusive_rate > 0.0, diffusive_rate, 1.0)
        advective = jnp.where(advective_rate > 0.0, 1.0 / safe_advective, jnp.inf)
        diffusive = jnp.where(diffusive_rate > 0.0, 1.0 / safe_diffusive, jnp.inf)
        return MACStepRestriction(
            advective=self.momentum.precision.reduction(advective),
            diffusive=self.momentum.precision.reduction(diffusive),
            selected=self.momentum.precision.reduction(jnp.minimum(advective, diffusive)),
        )

    def diagnostics(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> MACIncompressibleDiagnostics:
        time_ = jnp.asarray(time)
        stage = self.boundary_stage(time_, args)
        velocity = self.momentum.boundaries.enforce(self.unpack_velocity(state), stage)
        unconstrained, convection, diffusion, forcing = self._rate_components(
            time_, state, args, stage
        )
        projected = self.projection.project_rate(unconstrained, boundary_stage=stage)
        space = self.momentum.operators.velocity_space
        viscosity = self.problem.viscosity.astype(
            self.momentum.operators.pressure_space.dtype
        )
        nonlinear_rate = tuple(-value for value in convection)
        viscous_rate = tuple(viscosity * value for value in diffusion)
        nonlinear_energy_rate = jnp.real(space.inner(velocity, nonlinear_rate))
        forcing_power = jnp.real(space.inner(velocity, forcing))
        viscous_energy_rate = jnp.real(space.inner(velocity, viscous_rate))
        semidiscrete_energy_rate = jnp.real(space.inner(velocity, projected.rate))
        momentum_diagnostics = self.momentum.diagnostics(velocity, stage=stage)
        dissipation = viscosity * momentum_diagnostics.dissipation
        traction_power = self.momentum._boundary_traction_power(velocity, stage)
        diffusive_boundary_power = momentum_diagnostics.boundary_power - traction_power
        boundary_power = viscosity * diffusive_boundary_power + traction_power
        backflow_dissipation = momentum_diagnostics.open_backflow_dissipation
        expected_rate = (
            forcing_power - dissipation + boundary_power - backflow_dissipation
        )
        volumes = self.momentum.operators.discretization.cell_volumes
        pressure_residual_norm = jnp.sqrt(
            jnp.sum(volumes * projected.pressure_residual**2)
        )
        divergence_norm = GeometryPrecisionPolicy().norm(
            projected.divergence_after.reshape((-1,))
        )
        finite = (
            momentum_diagnostics.finite
            & projected.finite
            & jnp.isfinite(pressure_residual_norm)
            & jnp.isfinite(divergence_norm)
            & jnp.isfinite(semidiscrete_energy_rate)
        )
        successful = (
            finite
            & stage.successful
            & momentum_diagnostics.successful
            & projected.converged
        )
        return MACIncompressibleDiagnostics(
            kinetic_energy=momentum_diagnostics.kinetic_energy,
            nonlinear_energy_rate=self.momentum.precision.reduction(
                nonlinear_energy_rate
            ),
            forcing_power=self.momentum.precision.reduction(forcing_power),
            viscous_energy_rate=self.momentum.precision.reduction(viscous_energy_rate),
            dissipation=self.momentum.precision.reduction(dissipation),
            boundary_power=self.momentum.precision.reduction(boundary_power),
            open_backflow_dissipation=self.momentum.precision.reduction(
                backflow_dissipation
            ),
            integrated_mass_flux=momentum_diagnostics.integrated_mass_flux,
            semidiscrete_energy_rate=self.momentum.precision.reduction(
                semidiscrete_energy_rate
            ),
            energy_balance_defect=self.momentum.precision.reduction(
                semidiscrete_energy_rate - expected_rate
            ),
            divergence_norm=self.momentum.precision.reduction(divergence_norm),
            boundary_defect=momentum_diagnostics.boundary_defect,
            pressure_residual_norm=self.momentum.precision.reduction(
                pressure_residual_norm
            ),
            pressure_gauge_residual=self.momentum.precision.reduction(
                projected.gauge_defect
            ),
            finite=finite,
            successful=successful,
            projection_converged=projected.converged,
            pressure_closure=projected.closure.kind,
            pressure_gauge=projected.closure.gauge,
            momentum_id=self.momentum.prepared_id,
            projection_id=self.projection.plan_id,
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        projected = self.rate_projection(time, state, args)
        coordinates = self.momentum.operators.velocity_space.flatten(projected.rate)
        return eqx.error_if(
            coordinates,
            ~projected.converged,
            "MAC momentum-rate projection failed.",
        )


def compile_mac_incompressible_flow(
    problem: IncompressibleFlowProblem,
    momentum: PreparedMACMomentumOperators,
    projection: MACPressureProjectionPlan,
    /,
) -> CompiledMACIncompressibleDynamics:
    """Compile explicit symmetry-preserving MAC dynamics with stage projection."""
    from ..solver._structured_incompressible import MACPressureProjectionPlan

    if not isinstance(problem, IncompressibleFlowProblem):
        raise TypeError("problem must be IncompressibleFlowProblem.")
    if not isinstance(momentum, PreparedMACMomentumOperators):
        raise TypeError("momentum must be PreparedMACMomentumOperators.")
    if not isinstance(projection, MACPressureProjectionPlan):
        raise TypeError("projection must be MACPressureProjectionPlan.")
    if problem.spatial_dimension != momentum.dimension:
        raise ValueError("Incompressible problem and MAC momentum dimensions differ.")
    if projection.operators.prepared_id != momentum.operators.prepared_id:
        raise ValueError("MAC momentum and pressure projection must share operators.")
    if not np.isclose(projection.density, 1.0, rtol=0.0, atol=0.0):
        raise ValueError("MAC compiled dynamics use unit density and reduced pressure.")
    if projection.boundaries.prepared_id != momentum.boundaries.prepared_id:
        raise ValueError("MAC momentum and pressure projection must share boundaries.")
    identifier = canonical_fingerprint(
        {
            "kind": "compiled-mac-incompressible-flow",
            "problem": problem.problem_id,
            "momentum": momentum.prepared_id,
            "projection": projection.plan_id,
        }
    )
    return CompiledMACIncompressibleDynamics(
        problem,
        momentum,
        projection,
        compilation_id=identifier,
    )


__all__ = [
    "CompiledMACIncompressibleDynamics",
    "MACIncompressibleDiagnostics",
    "MACStepRestriction",
    "compile_mac_incompressible_flow",
]
