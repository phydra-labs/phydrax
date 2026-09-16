#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_enthalpy import (
    MACEnthalpyFluxResult,
)
from ..discretization.finite_volume._mac_scalar import (
    MACScalarFluxResult,
    MACScalarStepRestriction,
    PreparedMACScalarTransport,
)
from ._mac_enthalpy_porosity import CompiledMACEnthalpyPorosityDynamics
from ._solid_liquid_phase_change import (
    BinaryAlloyEnthalpyState,
    BinaryAlloyPhaseDiagramPlan,
)


class MACBinaryAlloyStage(StrictModule):
    velocity: FaceVelocity
    enthalpy: Array
    total_solute: Array
    thermodynamics: BinaryAlloyEnthalpyState
    enthalpy_flux: MACEnthalpyFluxResult
    solute_flux: MACScalarFluxResult
    buoyancy_force: FaceVelocity
    resistance_force: FaceVelocity
    velocity_rate: FaceVelocity
    enthalpy_rate: Array
    solute_rate: Array
    pressure: Array
    divergence_after: Array
    projection_converged: Array
    finite: Array
    successful: Array
    compilation_id: str = eqx.field(static=True)


class MACBinaryAlloyStepRestriction(StrictModule):
    momentum: Array
    enthalpy: Array
    solute: MACScalarStepRestriction
    resistance: Array
    selected: Array
    finite: Array
    successful: Array
    compilation_id: str = eqx.field(static=True)


class MACBinaryAlloyDiagnostics(StrictModule):
    total_enthalpy: Array
    total_solute: Array
    enthalpy_balance_defect: Array
    solute_balance_defect: Array
    resistance_power: Array
    divergence_norm: Array
    finite: Array
    successful: Array
    compilation_id: str = eqx.field(static=True)


class CompiledMACBinaryAlloyDynamics(StrictModule, NonTrainableState):
    base: CompiledMACEnthalpyPorosityDynamics
    phase_diagram: BinaryAlloyPhaseDiagramPlan
    solute_transport: PreparedMACScalarTransport
    velocity_size: int = eqx.field(static=True)
    cell_size: int = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: CompiledMACEnthalpyPorosityDynamics,
        phase_diagram: BinaryAlloyPhaseDiagramPlan,
        solute_transport: PreparedMACScalarTransport,
        /,
    ):
        if not isinstance(base, CompiledMACEnthalpyPorosityDynamics):
            raise TypeError("base must be CompiledMACEnthalpyPorosityDynamics.")
        if not isinstance(phase_diagram, BinaryAlloyPhaseDiagramPlan):
            raise TypeError("phase_diagram must be BinaryAlloyPhaseDiagramPlan.")
        if not isinstance(solute_transport, PreparedMACScalarTransport):
            raise TypeError("solute_transport must be PreparedMACScalarTransport.")
        if (
            solute_transport.layout.operators.prepared_id
            != base.momentum.operators.prepared_id
        ):
            raise ValueError("Solute and enthalpy transports must share MAC operators.")
        if solute_transport.layout.field_names != ("total_solute",):
            raise ValueError("Binary-alloy transport requires only 'total_solute'.")
        declaration = solute_transport.problem.transports[0]
        if (
            not bool(jnp.all(declaration.diffusivity == 0.0))
            or declaration.source is not None
            or solute_transport.problem.reaction is not None
        ):
            raise ValueError(
                "Binary-alloy solute transport requires zero declared diffusivity, "
                "no independent source, and no reaction; phase-supported diffusivity "
                "is supplied by the phase diagram."
            )
        if base.problem.source is not None:
            raise ValueError("Binary-alloy coupling does not accept a base heat source.")
        if not np.isclose(
            base.problem.material.reference_density,
            phase_diagram.reference_density,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Binary-alloy and base enthalpy reference densities differ.")
        self.base = base
        self.phase_diagram = phase_diagram
        self.solute_transport = solute_transport
        self.velocity_size = base.velocity_size
        self.cell_size = base.cell_size
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-mac-binary-alloy",
                "base": base.compilation_id,
                "phase_diagram": phase_diagram.plan_id,
                "solute_transport": solute_transport.prepared_id,
            }
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.velocity_size + 2 * self.cell_size,)

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        dtype = self.base.momentum.operators.pressure_space.dtype
        if value.shape != self.state_shape:
            raise ValueError(
                f"MAC binary-alloy coordinates must have shape {self.state_shape}."
            )
        if value.dtype != dtype:
            raise TypeError(f"MAC binary-alloy coordinates must have dtype {dtype}.")
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "MAC binary-alloy coordinates must be finite.",
        )

    def pack_state(
        self,
        velocity: FaceVelocity,
        enthalpy: ArrayLike,
        total_solute: ArrayLike,
        /,
    ) -> Array:
        shape = self.base.momentum.operators.discretization.cell_shape
        enthalpy_ = jnp.asarray(enthalpy)
        solute = jnp.asarray(total_solute, dtype=enthalpy_.dtype)
        if enthalpy_.shape != shape or solute.shape != shape:
            raise ValueError("Binary-alloy enthalpy and solute must use MAC cell shape.")
        return self.validate_state(
            jnp.concatenate(
                (
                    self.base.base_dynamics.pack_velocity(velocity),
                    enthalpy_.reshape((-1,)),
                    solute.reshape((-1,)),
                )
            )
        )

    def unpack_state(self, state: ArrayLike, /) -> tuple[FaceVelocity, Array, Array]:
        value = self.validate_state(state)
        velocity = self.base.base_dynamics.unpack_velocity(value[: self.velocity_size])
        shape = self.base.momentum.operators.discretization.cell_shape
        enthalpy = value[
            self.velocity_size : self.velocity_size + self.cell_size
        ].reshape(shape)
        solute = value[self.velocity_size + self.cell_size :].reshape(shape)
        return velocity, enthalpy, solute

    def stage(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> MACBinaryAlloyStage:
        value = self.validate_state(state)
        time_ = jnp.asarray(time)
        velocity_coordinates = value[: self.velocity_size]
        raw_velocity, enthalpy, solute = self.unpack_state(value)
        boundary = self.base.base_dynamics.boundary_stage(time_, args)
        velocity = self.base.momentum.boundaries.enforce(raw_velocity, boundary)
        momentum = self.base.base_dynamics._rate_components(
            time_, velocity_coordinates, args, boundary
        )
        thermodynamics = self.phase_diagram.evaluate(enthalpy, solute)
        zero_source = jnp.zeros_like(enthalpy)
        enthalpy_flux = self.base.transport.evaluate(
            time_,
            enthalpy,
            thermodynamics.temperature,
            thermodynamics.conductivity,
            velocity,
            zero_source,
            args,
        )
        solute_flux = self.solute_transport.evaluate(
            time_,
            {"total_solute": solute},
            velocity,
            args,
            sgs_diffusivities={"total_solute": thermodynamics.solute_diffusivity},
        )["total_solute"]
        face_states = tuple(
            self.phase_diagram.evaluate(enthalpy_face, solute_face)
            for enthalpy_face, solute_face in zip(
                enthalpy_flux.enthalpy_face_values,
                solute_flux.face_values,
                strict=True,
            )
        )
        material = self.base.problem.material
        buoyancy_force = tuple(
            -material.thermal_expansion
            * self.base.problem.gravity[axis].astype(component.dtype)
            * (face_state.temperature - material.buoyancy_reference_temperature)
            for axis, (component, face_state) in enumerate(
                zip(velocity, face_states, strict=True)
            )
        )
        resistance_force = tuple(
            -face_state.mushy_resistance * component
            for face_state, component in zip(face_states, velocity, strict=True)
        )
        unconstrained = self.base.momentum.boundaries.homogeneous_rate(
            tuple(
                base + buoyancy + resistance
                for base, buoyancy, resistance in zip(
                    momentum.unconstrained,
                    buoyancy_force,
                    resistance_force,
                    strict=True,
                )
            )
        )
        projected = self.base.projection.project_rate(
            unconstrained, boundary_stage=boundary
        )
        projected_finite = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(component)) for component in projected.rate)
            )
        )
        velocity_rate = tuple(
            jnp.where(projected.converged, component, jnp.zeros_like(component))
            for component in projected.rate
        )
        finite = (
            boundary.finite
            & jnp.all(thermodynamics.finite)
            & enthalpy_flux.finite
            & solute_flux.finite
            & projected_finite
            & jnp.all(jnp.isfinite(projected.pressure))
        )
        successful = (
            finite
            & boundary.successful
            & jnp.all(thermodynamics.successful)
            & enthalpy_flux.successful
            & solute_flux.success
            & projected.converged
        )
        return MACBinaryAlloyStage(
            velocity,
            enthalpy,
            solute,
            thermodynamics,
            enthalpy_flux,
            solute_flux,
            buoyancy_force,
            resistance_force,
            velocity_rate,
            enthalpy_flux.rate,
            solute_flux.rate,
            projected.pressure,
            projected.divergence_after,
            projected.converged,
            finite,
            successful,
            self.compilation_id,
        )

    def step_restriction(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> MACBinaryAlloyStepRestriction:
        velocity, enthalpy, solute = self.unpack_state(state)
        thermodynamics = self.phase_diagram.evaluate(enthalpy, solute)
        momentum = self.base.base_dynamics.step_restriction(
            time,
            self.base.base_dynamics.pack_velocity(velocity),
            args,
        )
        thermal = self.base.transport.step_restriction(
            velocity,
            thermodynamics.conductivity,
            thermodynamics.temperature_enthalpy_derivative,
        )
        solute_restriction = self.solute_transport.step_restriction(
            velocity,
            sgs_diffusivities={"total_solute": thermodynamics.solute_diffusivity},
        )
        resistance_rate = jnp.max(thermodynamics.mushy_resistance)
        resistance = jnp.where(resistance_rate > 0.0, 1.0 / resistance_rate, jnp.inf)
        selected = jnp.minimum(
            momentum.combined,
            jnp.minimum(
                thermal.selected,
                jnp.minimum(solute_restriction.selected, resistance),
            ),
        )
        finite = ~jnp.isnan(selected)
        return MACBinaryAlloyStepRestriction(
            momentum.combined,
            thermal.selected,
            solute_restriction,
            resistance,
            selected,
            finite,
            finite
            & thermal.successful
            & solute_restriction.success
            & jnp.all(thermodynamics.successful),
            self.compilation_id,
        )

    def diagnostics_from_stage(
        self, stage: MACBinaryAlloyStage, /
    ) -> MACBinaryAlloyDiagnostics:
        if stage.compilation_id != self.compilation_id:
            raise ValueError("Binary-alloy stage provenance does not match.")
        enthalpy = self.base.transport.diagnostics(
            stage.enthalpy,
            stage.thermodynamics.temperature,
            stage.thermodynamics.liquid_fraction,
            stage.enthalpy_flux,
        )
        solute = self.solute_transport.diagnostics_from_fluxes(
            {"total_solute": stage.total_solute},
            {"total_solute": stage.solute_flux},
        ).fields["total_solute"]
        resistance_power = jnp.real(
            self.base.momentum.operators.velocity_space.inner(
                stage.velocity, stage.resistance_force
            )
        )
        divergence_norm = jnp.sqrt(jnp.sum(stage.divergence_after**2))
        values = jnp.stack(
            (
                enthalpy.total_enthalpy,
                solute.content,
                enthalpy.balance_defect,
                solute.content_balance_defect,
                resistance_power,
                divergence_norm,
            )
        )
        finite = stage.finite & jnp.all(jnp.isfinite(values))
        return MACBinaryAlloyDiagnostics(
            *tuple(values),
            finite,
            finite
            & stage.successful
            & enthalpy.successful
            & solute.success
            & (resistance_power <= 0.0),
            self.compilation_id,
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        stage = self.stage(time, state, args)
        coordinates = jnp.concatenate(
            (
                self.base.momentum.operators.velocity_space.flatten(stage.velocity_rate),
                stage.enthalpy_rate.reshape((-1,)),
                stage.solute_rate.reshape((-1,)),
            )
        )
        return eqx.error_if(
            coordinates,
            ~stage.successful | jnp.any(~jnp.isfinite(coordinates)),
            "MAC binary-alloy stage failed.",
        )


def compile_mac_binary_alloy(
    base: CompiledMACEnthalpyPorosityDynamics,
    phase_diagram: BinaryAlloyPhaseDiagramPlan,
    solute_transport: PreparedMACScalarTransport,
    /,
) -> CompiledMACBinaryAlloyDynamics:
    return CompiledMACBinaryAlloyDynamics(base, phase_diagram, solute_transport)


__all__ = [
    "CompiledMACBinaryAlloyDynamics",
    "MACBinaryAlloyDiagnostics",
    "MACBinaryAlloyStage",
    "MACBinaryAlloyStepRestriction",
    "compile_mac_binary_alloy",
]
