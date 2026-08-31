#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._geometry_precision import GeometryPrecisionPolicy
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_momentum import PreparedMACMomentumOperators
from ..discretization.finite_volume._mac_scalar import (
    MACScalarDiagnostics,
    MACScalarFluxResult,
    MACScalarProblem,
    MACScalarStepRestriction,
    PreparedMACScalarTransport,
)
from ._incompressible import IncompressibleFlowProblem
from ._mac_incompressible import (
    compile_mac_incompressible_flow,
    CompiledMACIncompressibleDynamics,
    MACStepRestriction,
)


if TYPE_CHECKING:
    from ..solver._structured_incompressible import MACPressureProjectionPlan


def _canonical_coefficients(
    coefficients: Mapping[str, ArrayLike],
    references: Mapping[str, ArrayLike] | None,
    /,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    supplied = {str(name): jnp.asarray(value) for name, value in coefficients.items()}
    names = tuple(sorted(supplied))
    if not names or any(not name for name in names):
        raise ValueError("MAC buoyancy coefficients require non-empty field names.")
    reference_values = (
        {name: jnp.asarray(0.0) for name in names}
        if references is None
        else {str(name): jnp.asarray(value) for name, value in references.items()}
    )
    if set(reference_values) != set(names):
        raise ValueError("MAC buoyancy references must exactly match coefficient fields.")
    coefficient_values = []
    references_ = []
    for name in names:
        coefficient = supplied[name]
        reference = reference_values[name]
        if (
            coefficient.shape != ()
            or reference.shape != ()
            or jnp.iscomplexobj(coefficient)
            or jnp.iscomplexobj(reference)
            or not bool(jnp.isfinite(coefficient) & jnp.isfinite(reference))
        ):
            raise ValueError(
                "MAC buoyancy coefficients and references must be finite real scalars."
            )
        coefficient_values.append(float(coefficient))
        references_.append(float(reference))
    return names, tuple(coefficient_values), tuple(references_)


class MACBuoyancyLedger(StrictModule):
    """Face-exact kinetic/potential Boussinesq power exchange evidence."""

    force: FaceVelocity
    power_by_field: dict[str, Array]
    potential_energy_rate_by_field: dict[str, Array]
    total_power: Array
    potential_energy_rate: Array
    exchange_defect: Array
    finite: Array
    success: Array
    law_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)


class MACBuoyancyLaw(StrictModule, NonTrainableState):
    """Named Boussinesq acceleration from transported scalar anomalies."""

    gravity: Array
    field_names: tuple[str, ...] = eqx.field(static=True)
    coefficients: tuple[float, ...] = eqx.field(static=True)
    references: tuple[float, ...] = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravity: Sequence[float] | ArrayLike,
        coefficients: Mapping[str, ArrayLike],
        /,
        *,
        references: Mapping[str, ArrayLike] | None = None,
        law_id: str | None = None,
    ):
        gravity_ = jnp.asarray(gravity, dtype=float)
        if (
            gravity_.shape not in ((2,), (3,))
            or jnp.iscomplexobj(gravity_)
            or not bool(jnp.all(jnp.isfinite(gravity_)))
        ):
            raise ValueError(
                "MAC buoyancy gravity must be a finite real 2D or 3D vector."
            )
        names, coefficient_values, reference_values = _canonical_coefficients(
            coefficients, references
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "mac-buoyancy-law",
                    "gravity": np.asarray(gravity_).tolist(),
                    "fields": list(names),
                    "coefficients": list(coefficient_values),
                    "references": list(reference_values),
                }
            )
            if law_id is None
            else str(law_id)
        )
        if not identifier:
            raise ValueError("law_id must be non-empty.")
        self.gravity = gravity_
        self.field_names = names
        self.coefficients = coefficient_values
        self.references = reference_values
        self.law_id = identifier

    def evaluate(
        self,
        velocity: FaceVelocity,
        scalar_fluxes: Mapping[str, MACScalarFluxResult],
        transport: PreparedMACScalarTransport,
        momentum: PreparedMACMomentumOperators,
        /,
        *,
        projection_id: str,
    ) -> MACBuoyancyLedger:
        if not isinstance(transport, PreparedMACScalarTransport):
            raise TypeError("transport must be PreparedMACScalarTransport.")
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        if transport.layout.operators.prepared_id != momentum.operators.prepared_id:
            raise ValueError("MAC buoyancy momentum and scalar transport grids differ.")
        if len(self.gravity) != momentum.dimension:
            raise ValueError("MAC buoyancy gravity and momentum dimensions differ.")
        if not set(self.field_names).issubset(transport.layout.field_names):
            raise ValueError("MAC buoyancy fields must belong to scalar transport.")
        projection_identifier = str(projection_id)
        if not projection_identifier:
            raise ValueError("projection_id must be non-empty.")
        velocity_ = momentum.operators.validate_velocity(velocity)
        fluxes = dict(scalar_fluxes)
        if set(fluxes) != set(transport.layout.field_names):
            raise ValueError("MAC buoyancy requires every named scalar flux result.")
        force = tuple(
            jnp.zeros(layout.shape, dtype=transport.layout.dtype)
            for layout in momentum.operators.discretization.face_layouts
        )
        power_by_field: dict[str, Array] = {}
        potential_by_field: dict[str, Array] = {}
        for name, coefficient, reference in zip(
            self.field_names,
            self.coefficients,
            self.references,
            strict=True,
        ):
            result = fluxes[name]
            if (
                not isinstance(result, MACScalarFluxResult)
                or result.field_name != name
                or result.transport_id != transport.prepared_id
                or result.grid_id != momentum.operators.discretization.grid.prepared_id
            ):
                raise ValueError("MAC buoyancy scalar flux provenance does not match.")
            field_force = tuple(
                jnp.asarray(coefficient, dtype=transport.layout.dtype)
                * self.gravity[axis].astype(transport.layout.dtype)
                * (face_value - jnp.asarray(reference, dtype=transport.layout.dtype))
                for axis, face_value in enumerate(result.face_values)
            )
            force = tuple(
                total + contribution
                for total, contribution in zip(force, field_force, strict=True)
            )
            power = jnp.real(
                momentum.operators.velocity_space.inner(velocity_, field_force)
            )
            potential_rate = -sum(
                jnp.sum(
                    dual_measure
                    * jnp.asarray(coefficient, dtype=transport.layout.dtype)
                    * self.gravity[axis].astype(transport.layout.dtype)
                    * (
                        result.advective_fluxes[axis]
                        - jnp.asarray(reference, dtype=transport.layout.dtype)
                        * velocity_[axis]
                    )
                )
                for axis, dual_measure in enumerate(momentum.operators.face_dual_measures)
            )
            power_by_field[name] = power
            potential_by_field[name] = potential_rate
        force = tuple(
            eqx.error_if(
                component,
                jnp.any(~jnp.isfinite(component)),
                "MAC buoyancy force must be finite.",
            )
            for component in force
        )
        total_power = sum(power_by_field.values())
        potential_energy_rate = sum(potential_by_field.values())
        exchange_defect = total_power + potential_energy_rate
        finite = (
            jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in force)))
            & jnp.isfinite(total_power)
            & jnp.isfinite(potential_energy_rate)
            & jnp.isfinite(exchange_defect)
        )
        return MACBuoyancyLedger(
            force=force,
            power_by_field=power_by_field,
            potential_energy_rate_by_field=potential_by_field,
            total_power=total_power,
            potential_energy_rate=potential_energy_rate,
            exchange_defect=exchange_defect,
            finite=finite,
            success=finite,
            law_id=self.law_id,
            transport_id=transport.prepared_id,
            momentum_id=momentum.prepared_id,
            projection_id=projection_identifier,
            grid_id=momentum.operators.discretization.grid.prepared_id,
            ledger_id=canonical_fingerprint(
                {
                    "kind": "mac-buoyancy-ledger",
                    "law": self.law_id,
                    "transport": transport.prepared_id,
                    "momentum": momentum.prepared_id,
                    "projection": projection_identifier,
                }
            ),
        )


class MACScalarBuoyancyStage(StrictModule):
    """Dynamic coupled stage, separate from all prepared plan data."""

    velocity: FaceVelocity
    scalars: dict[str, Array]
    scalar_fluxes: dict[str, MACScalarFluxResult]
    buoyancy: MACBuoyancyLedger
    unconstrained_velocity_rate: FaceVelocity
    velocity_rate: FaceVelocity
    scalar_rates: dict[str, Array]
    pressure: Array
    pressure_residual: Array
    divergence_before: Array
    divergence_after: Array
    projection_converged: Array
    finite: Array
    success: Array
    compilation_id: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)


class MACScalarBuoyancyDiagnostics(StrictModule):
    """Coupled kinetic, scalar-content, variance, and buoyancy ledgers."""

    scalars: MACScalarDiagnostics
    buoyancy: MACBuoyancyLedger
    kinetic_energy: Array
    nonlinear_energy_rate: Array
    forcing_power: Array
    buoyancy_power: Array
    viscous_energy_rate: Array
    dissipation: Array
    wall_power: Array
    semidiscrete_energy_rate: Array
    energy_balance_defect: Array
    divergence_norm: Array
    pressure_residual_norm: Array
    pressure_gauge_residual: Array
    projection_converged: Array
    finite: Array
    success: Array
    compilation_id: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


class MACScalarBuoyancyStepRestriction(StrictModule):
    """Combined explicit momentum and named scalar stage restriction."""

    momentum: MACStepRestriction
    scalars: MACScalarStepRestriction
    selected: Array
    finite: Array
    success: Array
    compilation_id: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)


class CompiledMACScalarBuoyancyDynamics(StrictModule):
    """Flat velocity-plus-named-scalar Boussinesq MAC dynamics."""

    flow_problem: IncompressibleFlowProblem
    scalar_problem: MACScalarProblem
    momentum: PreparedMACMomentumOperators
    projection: MACPressureProjectionPlan
    transport: PreparedMACScalarTransport
    buoyancy: MACBuoyancyLaw
    base_dynamics: CompiledMACIncompressibleDynamics
    discretization_bundle: DiscretizationBundle
    velocity_size: int = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        flow_problem: IncompressibleFlowProblem,
        scalar_problem: MACScalarProblem,
        momentum: PreparedMACMomentumOperators,
        projection: MACPressureProjectionPlan,
        transport: PreparedMACScalarTransport,
        buoyancy: MACBuoyancyLaw,
        base_dynamics: CompiledMACIncompressibleDynamics,
        /,
        *,
        compilation_id: str,
    ):
        discretization = momentum.operators.discretization
        residual_key = DiscretizationKey(
            "mac_scalar_buoyancy_form",
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
                    "compiled-mac-scalar-buoyancy-form",
                    compilation_id,
                    dependency_key_ids=(discretization.key.key_id,),
                ),
            )
        )
        self.flow_problem = flow_problem
        self.scalar_problem = scalar_problem
        self.momentum = momentum
        self.projection = projection
        self.transport = transport
        self.buoyancy = buoyancy
        self.base_dynamics = base_dynamics
        self.discretization_bundle = bundle
        self.velocity_size = momentum.operators.velocity_space.size
        self.compilation_id = str(compilation_id)
        self.source_hash = canonical_fingerprint(
            {
                "flow": flow_problem.problem_id,
                "scalars": scalar_problem.problem_id,
                "buoyancy": buoyancy.law_id,
            }
        )
        self.resolved_method = "mac-symmetry-preserving-projected-scalar-buoyancy"

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.velocity_size + self.transport.layout.state_size,)

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Coupled MAC coordinates must have shape {self.state_shape}; "
                f"got {value.shape}."
            )
        dtype = self.momentum.operators.pressure_space.dtype
        if value.dtype != dtype:
            raise TypeError(f"Coupled MAC coordinates must have dtype {dtype}.")
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "Coupled MAC coordinates must be finite.",
        )

    def pack_state(
        self,
        velocity: FaceVelocity,
        scalars: Mapping[str, ArrayLike],
        /,
    ) -> Array:
        velocity_coordinates = self.base_dynamics.pack_velocity(velocity)
        scalar_coordinates = self.transport.layout.pack(scalars)
        return self.validate_state(
            jnp.concatenate((velocity_coordinates, scalar_coordinates))
        )

    def unpack_state(self, state: ArrayLike, /) -> tuple[FaceVelocity, dict[str, Array]]:
        value = self.validate_state(state)
        velocity = self.base_dynamics.unpack_velocity(value[: self.velocity_size])
        scalars = self.transport.layout.unpack(value[self.velocity_size :])
        return velocity, scalars

    def project_state(
        self,
        velocity: FaceVelocity,
        scalars: Mapping[str, ArrayLike],
        /,
    ) -> Array:
        velocity_coordinates = self.base_dynamics.project_state(velocity)
        scalar_coordinates = self.transport.layout.pack(scalars)
        return self.validate_state(
            jnp.concatenate((velocity_coordinates, scalar_coordinates))
        )

    def physical_state(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[FaceVelocity, dict[str, Array]]:
        del time, args
        return self.unpack_state(state)

    def stage(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> MACScalarBuoyancyStage:
        value = self.validate_state(state)
        velocity_coordinates = value[: self.velocity_size]
        velocity, scalars = self.unpack_state(value)
        scalar_fluxes = self.transport.evaluate(time, scalars, velocity, args)
        buoyancy = self.buoyancy.evaluate(
            velocity,
            scalar_fluxes,
            self.transport,
            self.momentum,
            projection_id=self.projection.plan_id,
        )
        unconstrained_base, _, _, _ = self.base_dynamics.rate_components(
            time, velocity_coordinates, args
        )
        unconstrained = self.momentum.boundaries.homogeneous_rate(
            tuple(
                base + force
                for base, force in zip(
                    unconstrained_base,
                    buoyancy.force,
                    strict=True,
                )
            )
        )
        projected = self.projection.project_rate(unconstrained)
        projected_rate = tuple(
            eqx.error_if(
                component,
                ~projected.converged | jnp.any(~jnp.isfinite(component)),
                "Coupled MAC momentum-rate projection failed.",
            )
            for component in projected.rate
        )
        scalar_rates = {
            name: scalar_fluxes[name].rate for name in self.transport.layout.field_names
        }
        finite = (
            buoyancy.finite
            & jnp.all(
                jnp.stack(
                    tuple(
                        scalar_fluxes[name].finite
                        for name in self.transport.layout.field_names
                    )
                )
            )
            & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(component)) for component in projected_rate
                    )
                )
            )
            & jnp.all(jnp.isfinite(projected.pressure))
            & jnp.all(jnp.isfinite(projected.pressure_residual))
            & jnp.all(jnp.isfinite(projected.divergence_after))
        )
        success = finite & projected.converged & buoyancy.success
        return MACScalarBuoyancyStage(
            velocity=velocity,
            scalars=scalars,
            scalar_fluxes=scalar_fluxes,
            buoyancy=buoyancy,
            unconstrained_velocity_rate=unconstrained,
            velocity_rate=projected_rate,
            scalar_rates=scalar_rates,
            pressure=projected.pressure,
            pressure_residual=projected.pressure_residual,
            divergence_before=projected.divergence_before,
            divergence_after=projected.divergence_after,
            projection_converged=projected.converged,
            finite=finite,
            success=success,
            compilation_id=self.compilation_id,
            momentum_id=self.momentum.prepared_id,
            transport_id=self.transport.prepared_id,
            projection_id=self.projection.plan_id,
            stage_id=canonical_fingerprint(
                {
                    "kind": "mac-scalar-buoyancy-stage",
                    "compilation": self.compilation_id,
                }
            ),
        )

    def pressure_field(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        stage = self.stage(time, state, args)
        return eqx.error_if(
            stage.pressure,
            ~stage.success,
            "Coupled MAC pressure recovery failed.",
        )

    def step_restriction(self, state: ArrayLike, /) -> MACScalarBuoyancyStepRestriction:
        value = self.validate_state(state)
        velocity, _ = self.unpack_state(value)
        momentum = self.base_dynamics.step_restriction(value[: self.velocity_size])
        scalars = self.transport.step_restriction(velocity)
        selected = jnp.minimum(momentum.selected, scalars.selected)
        finite = ~jnp.isnan(selected)
        return MACScalarBuoyancyStepRestriction(
            momentum=momentum,
            scalars=scalars,
            selected=selected,
            finite=finite,
            success=finite & scalars.success,
            compilation_id=self.compilation_id,
            momentum_id=self.momentum.prepared_id,
            transport_id=self.transport.prepared_id,
            projection_id=self.projection.plan_id,
        )

    def diagnostics(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> MACScalarBuoyancyDiagnostics:
        value = self.validate_state(state)
        velocity_coordinates = value[: self.velocity_size]
        stage = self.stage(time, value, args)
        _, convection, diffusion, forcing = self.base_dynamics.rate_components(
            time, velocity_coordinates, args
        )
        scalar_diagnostics = self.transport.diagnostics_from_fluxes(
            stage.scalars,
            stage.scalar_fluxes,
        )
        space = self.momentum.operators.velocity_space
        viscosity = self.flow_problem.viscosity.astype(
            self.momentum.operators.pressure_space.dtype
        )
        nonlinear_rate = tuple(-component for component in convection)
        viscous_rate = tuple(viscosity * component for component in diffusion)
        kinetic_energy = 0.5 * jnp.real(space.inner(stage.velocity, stage.velocity))
        nonlinear_energy_rate = jnp.real(space.inner(stage.velocity, nonlinear_rate))
        forcing_power = jnp.real(space.inner(stage.velocity, forcing))
        viscous_energy_rate = jnp.real(space.inner(stage.velocity, viscous_rate))
        semidiscrete_energy_rate = jnp.real(
            space.inner(stage.velocity, stage.velocity_rate)
        )
        homogeneous_diffusion = self.momentum.homogeneous_laplacian(stage.velocity)
        homogeneous_viscous_rate = viscosity * jnp.real(
            space.inner(stage.velocity, homogeneous_diffusion)
        )
        dissipation = -homogeneous_viscous_rate
        wall_power = viscous_energy_rate - homogeneous_viscous_rate
        expected = forcing_power + stage.buoyancy.total_power - dissipation + wall_power
        volumes = self.momentum.operators.discretization.cell_volumes
        pressure_residual_norm = jnp.sqrt(jnp.sum(volumes * stage.pressure_residual**2))
        divergence_norm = GeometryPrecisionPolicy().norm(
            stage.divergence_after.reshape((-1,))
        )
        pressure_gauge_residual = jnp.abs(
            jnp.sum(volumes * stage.pressure) / jnp.sum(volumes)
        )
        energy_defect = semidiscrete_energy_rate - expected
        finite = (
            stage.finite
            & scalar_diagnostics.finite
            & jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        (
                            kinetic_energy,
                            nonlinear_energy_rate,
                            forcing_power,
                            stage.buoyancy.total_power,
                            viscous_energy_rate,
                            dissipation,
                            wall_power,
                            semidiscrete_energy_rate,
                            energy_defect,
                            divergence_norm,
                            pressure_residual_norm,
                            pressure_gauge_residual,
                        )
                    )
                )
            )
        )
        success = finite & stage.success & scalar_diagnostics.success
        return MACScalarBuoyancyDiagnostics(
            scalars=scalar_diagnostics,
            buoyancy=stage.buoyancy,
            kinetic_energy=kinetic_energy,
            nonlinear_energy_rate=nonlinear_energy_rate,
            forcing_power=forcing_power,
            buoyancy_power=stage.buoyancy.total_power,
            viscous_energy_rate=viscous_energy_rate,
            dissipation=dissipation,
            wall_power=wall_power,
            semidiscrete_energy_rate=semidiscrete_energy_rate,
            energy_balance_defect=energy_defect,
            divergence_norm=divergence_norm,
            pressure_residual_norm=pressure_residual_norm,
            pressure_gauge_residual=pressure_gauge_residual,
            projection_converged=stage.projection_converged,
            finite=finite,
            success=success,
            compilation_id=self.compilation_id,
            momentum_id=self.momentum.prepared_id,
            transport_id=self.transport.prepared_id,
            projection_id=self.projection.plan_id,
            grid_id=self.momentum.operators.discretization.grid.prepared_id,
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        stage = self.stage(time, state, args)
        velocity_rate = self.momentum.operators.velocity_space.flatten(
            stage.velocity_rate
        )
        scalar_rate = self.transport.layout.pack(stage.scalar_rates)
        coordinates = jnp.concatenate((velocity_rate, scalar_rate))
        return eqx.error_if(
            coordinates,
            ~stage.success | jnp.any(~jnp.isfinite(coordinates)),
            "Coupled MAC scalar-buoyancy stage failed.",
        )


def compile_mac_scalar_buoyancy(
    flow_problem: IncompressibleFlowProblem,
    momentum: PreparedMACMomentumOperators,
    projection: MACPressureProjectionPlan,
    scalar_problem: MACScalarProblem,
    transport: PreparedMACScalarTransport,
    buoyancy: MACBuoyancyLaw,
    /,
) -> CompiledMACScalarBuoyancyDynamics:
    """Compile projected unit-density MAC flow with named explicit scalars."""
    from ..solver._structured_incompressible import MACPressureProjectionPlan

    if not isinstance(flow_problem, IncompressibleFlowProblem):
        raise TypeError("flow_problem must be IncompressibleFlowProblem.")
    if not isinstance(momentum, PreparedMACMomentumOperators):
        raise TypeError("momentum must be PreparedMACMomentumOperators.")
    if not isinstance(projection, MACPressureProjectionPlan):
        raise TypeError("projection must be MACPressureProjectionPlan.")
    if not isinstance(scalar_problem, MACScalarProblem):
        raise TypeError("scalar_problem must be MACScalarProblem.")
    if not isinstance(transport, PreparedMACScalarTransport):
        raise TypeError("transport must be PreparedMACScalarTransport.")
    if not isinstance(buoyancy, MACBuoyancyLaw):
        raise TypeError("buoyancy must be MACBuoyancyLaw.")
    if flow_problem.spatial_dimension != momentum.dimension:
        raise ValueError("Incompressible problem and MAC momentum dimensions differ.")
    if projection.operators.prepared_id != momentum.operators.prepared_id:
        raise ValueError("MAC momentum and pressure projection must share operators.")
    if transport.layout.operators.prepared_id != momentum.operators.prepared_id:
        raise ValueError("MAC scalar transport and momentum must share operators.")
    if transport.problem.problem_id != scalar_problem.problem_id:
        raise ValueError("Prepared MAC scalar transport does not match scalar problem.")
    if tuple(transport.layout.field_names) != tuple(scalar_problem.field_names):
        raise ValueError("MAC scalar problem and prepared layout fields differ.")
    if not set(buoyancy.field_names).issubset(scalar_problem.field_names):
        raise ValueError("MAC buoyancy fields must belong to the scalar problem.")
    if len(buoyancy.gravity) != momentum.dimension:
        raise ValueError("MAC buoyancy gravity and flow dimensions differ.")
    if not np.isclose(projection.density, 1.0, rtol=0.0, atol=0.0):
        raise ValueError("MAC Boussinesq dynamics require unit reference density.")
    base = compile_mac_incompressible_flow(flow_problem, momentum, projection)
    identifier = canonical_fingerprint(
        {
            "kind": "compiled-mac-scalar-buoyancy",
            "flow_problem": flow_problem.problem_id,
            "scalar_problem": scalar_problem.problem_id,
            "momentum": momentum.prepared_id,
            "projection": projection.plan_id,
            "transport": transport.prepared_id,
            "buoyancy": buoyancy.law_id,
        }
    )
    return CompiledMACScalarBuoyancyDynamics(
        flow_problem,
        scalar_problem,
        momentum,
        projection,
        transport,
        buoyancy,
        base,
        compilation_id=identifier,
    )


__all__ = [
    "CompiledMACScalarBuoyancyDynamics",
    "MACBuoyancyLaw",
    "MACBuoyancyLedger",
    "MACScalarBuoyancyDiagnostics",
    "MACScalarBuoyancyStage",
    "MACScalarBuoyancyStepRestriction",
    "compile_mac_scalar_buoyancy",
]
