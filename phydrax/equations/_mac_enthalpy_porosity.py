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
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_boundary import MACBoundaryStageData
from ..discretization.finite_volume._mac_enthalpy import (
    MACEnthalpyDiagnostics,
    MACEnthalpyFluxResult,
    MACEnthalpyStepRestriction,
    PreparedMACEnthalpyTransport,
)
from ..discretization.finite_volume._mac_momentum import PreparedMACMomentumOperators
from ._incompressible import IncompressibleFlowProblem
from ._mac_incompressible import (
    compile_mac_incompressible_flow,
    CompiledMACIncompressibleDynamics,
    MACIncompressibleRateComponents,
    MACLESStepRestriction,
)
from ._solid_liquid_phase_change import (
    SolidLiquidEnthalpyPlan,
    SolidLiquidEnthalpyState,
)


if TYPE_CHECKING:
    from ..solver._structured_incompressible import MACPressureProjectionPlan


class MACEnthalpyPorosityProblem(StrictModule, NonTrainableState):
    material: SolidLiquidEnthalpyPlan
    gravity: Array
    source: Any = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        material: SolidLiquidEnthalpyPlan,
        gravity: ArrayLike,
        /,
        *,
        source: Any = None,
        source_id: str | None = None,
    ):
        if not isinstance(material, SolidLiquidEnthalpyPlan):
            raise TypeError("material must be SolidLiquidEnthalpyPlan.")
        gravity_ = jnp.asarray(gravity, dtype=jnp.float64)
        if (
            gravity_.shape not in ((2,), (3,))
            or jnp.iscomplexobj(gravity_)
            or not bool(jnp.all(jnp.isfinite(gravity_)))
        ):
            raise ValueError("MAC enthalpy gravity must be a finite 2D or 3D vector.")
        if source is not None and not callable(source):
            raise TypeError("MAC enthalpy source must be callable or None.")
        if source is None:
            if source_id is not None:
                raise ValueError("source_id requires an enthalpy source callable.")
            identifier = "none"
        else:
            identifier = "" if source_id is None else str(source_id)
            if not identifier:
                raise ValueError("MAC enthalpy source callable requires source_id.")
        self.material = material
        self.gravity = gravity_
        self.source = source
        self.source_id = identifier
        self.problem_id = canonical_fingerprint(
            {
                "kind": "mac-enthalpy-porosity-problem",
                "material": material.plan_id,
                "gravity": np.asarray(gravity_).tolist(),
                "source": identifier,
            }
        )


class MACEnthalpyPorosityStage(StrictModule):
    boundary_stage: MACBoundaryStageData
    velocity: FaceVelocity
    enthalpy: Array
    thermodynamics: SolidLiquidEnthalpyState
    momentum_components: MACIncompressibleRateComponents
    enthalpy_flux: MACEnthalpyFluxResult
    buoyancy_force: FaceVelocity
    resistance_force: FaceVelocity
    unconstrained_velocity_rate: FaceVelocity
    velocity_rate: FaceVelocity
    enthalpy_rate: Array
    pressure: Array
    pressure_residual: Array
    divergence_before: Array
    divergence_after: Array
    projection_converged: Array
    finite: Array
    successful: Array
    compilation_id: str = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)


class MACEnthalpyPorosityStepRestriction(StrictModule):
    momentum: MACLESStepRestriction
    enthalpy: MACEnthalpyStepRestriction
    resistance_explicit: Array
    selected: Array
    finite: Array
    successful: Array
    compilation_id: str = eqx.field(static=True)


class MACEnthalpyPorosityDiagnostics(StrictModule):
    enthalpy: MACEnthalpyDiagnostics
    kinetic_energy: Array
    momentum_power: Array
    base_power: Array
    buoyancy_power: Array
    resistance_power: Array
    resistance_dissipation: Array
    momentum_balance_defect: Array
    divergence_norm: Array
    pressure_residual_norm: Array
    pressure_gauge_residual: Array
    projection_converged: Array
    finite: Array
    successful: Array
    compilation_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class CompiledMACEnthalpyPorosityDynamics(StrictModule):
    flow_problem: IncompressibleFlowProblem
    problem: MACEnthalpyPorosityProblem
    momentum: PreparedMACMomentumOperators
    projection: MACPressureProjectionPlan
    transport: PreparedMACEnthalpyTransport
    base_dynamics: CompiledMACIncompressibleDynamics
    velocity_size: int = eqx.field(static=True)
    cell_size: int = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        flow_problem: IncompressibleFlowProblem,
        problem: MACEnthalpyPorosityProblem,
        momentum: PreparedMACMomentumOperators,
        projection: MACPressureProjectionPlan,
        transport: PreparedMACEnthalpyTransport,
        base_dynamics: CompiledMACIncompressibleDynamics,
        /,
        *,
        compilation_id: str,
    ):
        self.flow_problem = flow_problem
        self.problem = problem
        self.momentum = momentum
        self.projection = projection
        self.transport = transport
        self.base_dynamics = base_dynamics
        self.velocity_size = momentum.operators.velocity_space.size
        self.cell_size = momentum.operators.pressure_space.size
        self.compilation_id = str(compilation_id)
        self.resolved_method = "mac-projected-enthalpy-porosity"

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.velocity_size + self.cell_size,)

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        dtype = self.momentum.operators.pressure_space.dtype
        if value.shape != self.state_shape:
            raise ValueError(
                f"MAC enthalpy coordinates must have shape {self.state_shape}; got {value.shape}."
            )
        if value.dtype != dtype:
            raise TypeError(f"MAC enthalpy coordinates must have dtype {dtype}.")
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "MAC enthalpy coordinates must be finite.",
        )

    def pack_state(self, velocity: FaceVelocity, enthalpy: ArrayLike, /) -> Array:
        enthalpy_ = jnp.asarray(enthalpy)
        expected = self.momentum.operators.discretization.cell_shape
        if enthalpy_.shape != expected:
            raise ValueError(
                f"MAC enthalpy must have cell shape {expected}; got {enthalpy_.shape}."
            )
        coordinates = jnp.concatenate(
            (self.base_dynamics.pack_velocity(velocity), enthalpy_.reshape((-1,)))
        )
        return self.validate_state(coordinates)

    def unpack_state(self, state: ArrayLike, /) -> tuple[FaceVelocity, Array]:
        value = self.validate_state(state)
        velocity = self.base_dynamics.unpack_velocity(value[: self.velocity_size])
        enthalpy = value[self.velocity_size :].reshape(
            self.momentum.operators.discretization.cell_shape
        )
        return velocity, enthalpy

    def physical_state(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> tuple[FaceVelocity, SolidLiquidEnthalpyState]:
        del time, args
        velocity, enthalpy = self.unpack_state(state)
        return velocity, self.problem.material.evaluate(enthalpy)

    def project_state(self, velocity: FaceVelocity, enthalpy: ArrayLike, /) -> Array:
        projected = self.base_dynamics.project_state(velocity)
        return self.validate_state(
            jnp.concatenate((projected, jnp.asarray(enthalpy).reshape((-1,))))
        )

    def _source(
        self,
        time: Array,
        thermodynamics: SolidLiquidEnthalpyState,
        velocity: FaceVelocity,
        args: Any,
        /,
    ) -> Array:
        shape = self.momentum.operators.discretization.cell_shape
        dtype = self.momentum.operators.pressure_space.dtype
        if self.problem.source is None:
            return jnp.zeros(shape, dtype=dtype)
        source = jnp.asarray(
            self.problem.source(time, thermodynamics, velocity, args), dtype=dtype
        )
        if source.shape != shape:
            raise ValueError(
                f"MAC enthalpy source must have cell shape {shape}; got {source.shape}."
            )
        return eqx.error_if(
            source,
            jnp.any(~jnp.isfinite(source)),
            "MAC enthalpy source must be finite.",
        )

    def stage(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> MACEnthalpyPorosityStage:
        value = self.validate_state(state)
        time_ = jnp.asarray(time)
        velocity_coordinates = value[: self.velocity_size]
        raw_velocity, enthalpy = self.unpack_state(value)
        boundary_stage = self.base_dynamics.boundary_stage(time_, args)
        velocity = self.momentum.boundaries.enforce(raw_velocity, boundary_stage)
        components = self.base_dynamics._rate_components(
            time_, velocity_coordinates, args, boundary_stage
        )
        thermodynamics = self.problem.material.evaluate(enthalpy)
        source = self._source(time_, thermodynamics, velocity, args)
        enthalpy_flux = self.transport.evaluate(
            time_,
            enthalpy,
            thermodynamics.temperature,
            thermodynamics.conductivity,
            velocity,
            source,
            args,
        )
        face_states = tuple(
            self.problem.material.evaluate(face_enthalpy)
            for face_enthalpy in enthalpy_flux.enthalpy_face_values
        )
        beta = self.problem.material.thermal_expansion
        reference = self.problem.material.buoyancy_reference_temperature
        buoyancy_force = tuple(
            -jnp.asarray(beta, dtype=velocity[axis].dtype)
            * self.problem.gravity[axis].astype(velocity[axis].dtype)
            * (face_state.temperature - reference)
            for axis, face_state in enumerate(face_states)
        )
        resistance_force = tuple(
            -face_state.mushy_resistance * component
            for face_state, component in zip(face_states, velocity, strict=True)
        )
        unconstrained = self.momentum.boundaries.homogeneous_rate(
            tuple(
                base + buoyancy + resistance
                for base, buoyancy, resistance in zip(
                    components.unconstrained,
                    buoyancy_force,
                    resistance_force,
                    strict=True,
                )
            )
        )
        projected = self.projection.project_rate(
            unconstrained, boundary_stage=boundary_stage
        )
        projected_finite = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(component)) for component in projected.rate)
            )
        )
        valid = projected.converged & projected_finite
        velocity_rate = tuple(
            jnp.where(valid, component, jnp.zeros_like(component))
            for component in projected.rate
        )
        finite = (
            boundary_stage.finite
            & jnp.all(thermodynamics.finite)
            & enthalpy_flux.finite
            & projected_finite
            & jnp.all(jnp.isfinite(projected.pressure))
            & jnp.all(jnp.isfinite(projected.pressure_residual))
            & jnp.all(jnp.isfinite(projected.divergence_after))
            & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(component))
                        for component in buoyancy_force + resistance_force
                    )
                )
            )
        )
        successful = (
            finite
            & boundary_stage.successful
            & jnp.all(thermodynamics.successful)
            & enthalpy_flux.successful
            & projected.converged
        )
        return MACEnthalpyPorosityStage(
            boundary_stage,
            velocity,
            enthalpy,
            thermodynamics,
            components,
            enthalpy_flux,
            buoyancy_force,
            resistance_force,
            unconstrained,
            velocity_rate,
            enthalpy_flux.rate,
            projected.pressure,
            projected.pressure_residual,
            projected.divergence_before,
            projected.divergence_after,
            projected.converged,
            finite,
            successful,
            self.compilation_id,
            canonical_fingerprint(
                {
                    "kind": "mac-enthalpy-porosity-stage",
                    "compilation": self.compilation_id,
                }
            ),
        )

    def pressure_field(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> Array:
        stage = self.stage(time, state, args)
        return eqx.error_if(
            stage.pressure,
            ~stage.successful,
            "MAC enthalpy pressure recovery failed.",
        )

    def step_restriction(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> MACEnthalpyPorosityStepRestriction:
        value = self.validate_state(state)
        velocity, enthalpy = self.unpack_state(value)
        thermodynamics = self.problem.material.evaluate(enthalpy)
        momentum = self.base_dynamics.step_restriction(
            time, value[: self.velocity_size], args
        )
        thermal = self.transport.step_restriction(
            velocity,
            thermodynamics.conductivity,
            thermodynamics.temperature_enthalpy_derivative,
        )
        resistance_rate = jnp.max(thermodynamics.mushy_resistance)
        resistance = jnp.where(resistance_rate > 0.0, 1.0 / resistance_rate, jnp.inf)
        selected = jnp.minimum(
            momentum.combined, jnp.minimum(thermal.selected, resistance)
        )
        finite = ~jnp.isnan(selected)
        return MACEnthalpyPorosityStepRestriction(
            momentum,
            thermal,
            resistance,
            selected,
            finite,
            finite & thermal.successful & jnp.all(thermodynamics.successful),
            self.compilation_id,
        )

    def diagnostics_from_stage(
        self, stage: MACEnthalpyPorosityStage, /
    ) -> MACEnthalpyPorosityDiagnostics:
        if (
            not isinstance(stage, MACEnthalpyPorosityStage)
            or stage.compilation_id != self.compilation_id
        ):
            raise ValueError("MAC enthalpy diagnostics provenance does not match.")
        enthalpy = self.transport.diagnostics(
            stage.enthalpy,
            stage.thermodynamics.temperature,
            stage.thermodynamics.liquid_fraction,
            stage.enthalpy_flux,
        )
        space = self.momentum.operators.velocity_space
        kinetic = 0.5 * jnp.real(space.inner(stage.velocity, stage.velocity))
        momentum_power = jnp.real(space.inner(stage.velocity, stage.velocity_rate))
        base_power = jnp.real(
            space.inner(stage.velocity, stage.momentum_components.unconstrained)
        )
        buoyancy_power = jnp.real(space.inner(stage.velocity, stage.buoyancy_force))
        resistance_power = jnp.real(space.inner(stage.velocity, stage.resistance_force))
        defect = momentum_power - (base_power + buoyancy_power + resistance_power)
        volumes = self.momentum.operators.discretization.cell_volumes
        divergence_norm = GeometryPrecisionPolicy().norm(
            stage.divergence_after.reshape((-1,))
        )
        pressure_residual_norm = jnp.sqrt(jnp.sum(volumes * stage.pressure_residual**2))
        pressure_gauge = jnp.abs(jnp.sum(volumes * stage.pressure) / jnp.sum(volumes))
        values = jnp.stack(
            (
                kinetic,
                momentum_power,
                base_power,
                buoyancy_power,
                resistance_power,
                -resistance_power,
                defect,
                divergence_norm,
                pressure_residual_norm,
                pressure_gauge,
            )
        )
        finite = stage.finite & enthalpy.finite & jnp.all(jnp.isfinite(values))
        successful = (
            finite & stage.successful & enthalpy.successful & (resistance_power <= 0.0)
        )
        return MACEnthalpyPorosityDiagnostics(
            enthalpy,
            *tuple(values),
            stage.projection_converged,
            finite,
            successful,
            self.compilation_id,
            self.momentum.operators.discretization.grid.prepared_id,
        )

    def diagnostics(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> MACEnthalpyPorosityDiagnostics:
        return self.diagnostics_from_stage(self.stage(time, state, args))

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        stage = self.stage(time, state, args)
        coordinates = jnp.concatenate(
            (
                self.momentum.operators.velocity_space.flatten(stage.velocity_rate),
                stage.enthalpy_rate.reshape((-1,)),
            )
        )
        return eqx.error_if(
            coordinates,
            ~stage.successful | jnp.any(~jnp.isfinite(coordinates)),
            "MAC enthalpy-porosity stage failed.",
        )


def compile_mac_enthalpy_porosity(
    flow_problem: IncompressibleFlowProblem,
    problem: MACEnthalpyPorosityProblem,
    momentum: PreparedMACMomentumOperators,
    projection: MACPressureProjectionPlan,
    transport: PreparedMACEnthalpyTransport,
    /,
) -> CompiledMACEnthalpyPorosityDynamics:
    from ..solver._structured_incompressible import MACPressureProjectionPlan

    if not isinstance(flow_problem, IncompressibleFlowProblem):
        raise TypeError("flow_problem must be IncompressibleFlowProblem.")
    if not isinstance(problem, MACEnthalpyPorosityProblem):
        raise TypeError("problem must be MACEnthalpyPorosityProblem.")
    if not isinstance(momentum, PreparedMACMomentumOperators):
        raise TypeError("momentum must be PreparedMACMomentumOperators.")
    if not isinstance(projection, MACPressureProjectionPlan):
        raise TypeError("projection must be MACPressureProjectionPlan.")
    if not isinstance(transport, PreparedMACEnthalpyTransport):
        raise TypeError("transport must be PreparedMACEnthalpyTransport.")
    if flow_problem.spatial_dimension != momentum.dimension:
        raise ValueError("Flow problem and MAC momentum dimensions differ.")
    if len(problem.gravity) != momentum.dimension:
        raise ValueError("MAC enthalpy gravity and momentum dimensions differ.")
    if projection.operators.prepared_id != momentum.operators.prepared_id:
        raise ValueError("MAC momentum and projection must share operators.")
    if transport.operators.prepared_id != momentum.operators.prepared_id:
        raise ValueError("MAC enthalpy transport and momentum must share operators.")
    if not np.isclose(projection.density, 1.0, rtol=0.0, atol=0.0):
        raise ValueError("MAC enthalpy-porosity requires unit projection density.")
    if not np.isclose(
        float(flow_problem.viscosity),
        problem.material.liquid_kinematic_viscosity,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "Flow viscosity must equal the material liquid kinematic viscosity."
        )
    base = compile_mac_incompressible_flow(flow_problem, momentum, projection)
    identifier = canonical_fingerprint(
        {
            "kind": "compiled-mac-enthalpy-porosity",
            "flow": flow_problem.problem_id,
            "problem": problem.problem_id,
            "momentum": momentum.prepared_id,
            "projection": projection.plan_id,
            "transport": transport.prepared_id,
        }
    )
    return CompiledMACEnthalpyPorosityDynamics(
        flow_problem,
        problem,
        momentum,
        projection,
        transport,
        base,
        compilation_id=identifier,
    )


__all__ = [
    "CompiledMACEnthalpyPorosityDynamics",
    "MACEnthalpyPorosityDiagnostics",
    "MACEnthalpyPorosityProblem",
    "MACEnthalpyPorosityStage",
    "MACEnthalpyPorosityStepRestriction",
    "compile_mac_enthalpy_porosity",
]
