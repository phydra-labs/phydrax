#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import (
    FiniteVolumeDiscretization,
    MACBoundaryPlan,
    MACMomentumPlan,
    MACOperatorPlan,
    MACScalarAdvection,
    MACScalarBoundaryCondition,
    MACScalarBoundarySet,
    MACScalarLayout,
    MACScalarProblem,
    MACScalarTransport,
    PreparedMACBoundaryPlan,
    PreparedMACMomentumOperators,
    PreparedMACOperators,
    PreparedMACScalarTransport,
)
from ...discretization.finite_volume._mac_ocean import PreparedMACOceanForcing
from ...equations import (
    compile_mac_scalar_buoyancy,
    CompiledMACScalarBuoyancyDynamics,
    IncompressibleFlowProblem,
)
from ...solver import MACPressureProjectionPlan
from ._reference import LinearSeawaterReference, OceanAxisConvention


class OceanStateView(StrictModule):
    """Non-authoritative view of one packed Cartesian ocean state."""

    velocity: tuple[Array, ...]
    temperature: Array
    salinity: Array
    density_anomaly: Array
    state_id: str = eqx.field(static=True)


class CartesianBoussinesqOceanPlan(StrictModule, NonTrainableState):
    """Compile a rigid-lid Cartesian T/S Boussinesq process model."""

    axes: OceanAxisConvention
    reference: LinearSeawaterReference
    viscosity: float = eqx.field(static=True)
    temperature_diffusivity: Array
    salinity_diffusivity: Array
    scalar_advection: MACScalarAdvection = eqx.field(static=True)
    coriolis_parameter: float = eqx.field(static=True)
    surface_stress: Array
    surface_stress_function: Any = eqx.field(static=True)
    surface_stress_id: str = eqx.field(static=True)
    temperature_surface_flux: MACScalarBoundaryCondition
    salinity_surface_flux: MACScalarBoundaryCondition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        axes: OceanAxisConvention,
        reference: LinearSeawaterReference,
        /,
        *,
        viscosity: float = 0.0,
        temperature_diffusivity: ArrayLike = 0.0,
        salinity_diffusivity: ArrayLike = 0.0,
        scalar_advection: MACScalarAdvection = "centered",
        coriolis_parameter: float = 0.0,
        surface_stress: Any = None,
        surface_stress_id: str | None = None,
        temperature_surface_flux: MACScalarBoundaryCondition | None = None,
        salinity_surface_flux: MACScalarBoundaryCondition | None = None,
    ):
        if not isinstance(axes, OceanAxisConvention):
            raise TypeError("axes must be OceanAxisConvention.")
        if not isinstance(reference, LinearSeawaterReference):
            raise TypeError("reference must be LinearSeawaterReference.")
        viscosity_ = float(viscosity)
        f = float(coriolis_parameter)
        if not np.isfinite(viscosity_) or viscosity_ < 0.0 or not np.isfinite(f):
            raise ValueError("Ocean viscosity and Coriolis parameter are invalid.")
        temperature_diffusivity_ = jnp.asarray(temperature_diffusivity, dtype=float)
        salinity_diffusivity_ = jnp.asarray(salinity_diffusivity, dtype=float)
        for name, value in (
            ("temperature", temperature_diffusivity_),
            ("salinity", salinity_diffusivity_),
        ):
            if (
                value.ndim > 1
                or value.size == 0
                or bool(jnp.any(~jnp.isfinite(value)))
                or bool(jnp.any(value < 0.0))
            ):
                raise ValueError(
                    f"Ocean {name} diffusivity must be scalar or one value per axis."
                )
        if scalar_advection not in ("centered", "upwind"):
            raise ValueError("Ocean scalar advection must be centered or upwind.")
        if callable(surface_stress):
            stress_identifier = (
                "" if surface_stress_id is None else str(surface_stress_id)
            )
            if not stress_identifier:
                raise ValueError("Dynamic surface stress requires surface_stress_id.")
            stress = jnp.zeros((3,))
            stress_function = surface_stress
        else:
            if surface_stress_id is not None:
                raise ValueError("surface_stress_id is only valid for dynamic stress.")
            stress = (
                jnp.zeros((3,)) if surface_stress is None else jnp.asarray(surface_stress)
            )
            if stress.shape != (3,) or bool(jnp.any(~jnp.isfinite(stress))):
                raise ValueError(
                    "Ocean surface stress must have three finite components."
                )
            stress_function = None
            stress_identifier = canonical_fingerprint(np.asarray(stress).tolist())
        temperature_flux = (
            MACScalarBoundaryCondition("flux", 0.0)
            if temperature_surface_flux is None
            else temperature_surface_flux
        )
        salinity_flux = (
            MACScalarBoundaryCondition("flux", 0.0)
            if salinity_surface_flux is None
            else salinity_surface_flux
        )
        if temperature_flux.kind != "flux" or salinity_flux.kind != "flux":
            raise ValueError(
                "Ocean surface scalar conditions must be conservative fluxes."
            )
        self.axes = axes
        self.reference = reference
        self.viscosity = viscosity_
        self.temperature_diffusivity = temperature_diffusivity_
        self.salinity_diffusivity = salinity_diffusivity_
        self.scalar_advection = scalar_advection
        self.coriolis_parameter = f
        self.surface_stress = stress
        self.surface_stress_function = stress_function
        self.surface_stress_id = stress_identifier
        self.temperature_surface_flux = temperature_flux
        self.salinity_surface_flux = salinity_flux
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cartesian-boussinesq-ocean-plan",
                "axes": axes.convention_id,
                "reference": reference.reference_id,
                "viscosity": viscosity_,
                "temperature_diffusivity": np.asarray(temperature_diffusivity_).tolist(),
                "salinity_diffusivity": np.asarray(salinity_diffusivity_).tolist(),
                "advection": scalar_advection,
                "f": f,
                "surface_stress": stress_identifier,
                "temperature_surface_flux": temperature_flux.boundary_id,
                "salinity_surface_flux": salinity_flux.boundary_id,
            }
        )

    def prepare(
        self,
        discretization: FiniteVolumeDiscretization,
        /,
        *,
        boundaries: MACBoundaryPlan | PreparedMACBoundaryPlan | None = None,
        projection_tolerance: float = 1.0e-9,
        projection_iterations: int = 500,
    ) -> "PreparedCartesianBoussinesqOcean":
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be FiniteVolumeDiscretization.")
        self.axes.validate_discretization(discretization)
        operators = MACOperatorPlan(discretization).prepare()
        prepared_boundaries = (
            MACBoundaryPlan(operators).prepare()
            if boundaries is None
            else boundaries.prepare()
            if isinstance(boundaries, MACBoundaryPlan)
            else boundaries
        )
        if not isinstance(prepared_boundaries, PreparedMACBoundaryPlan):
            raise TypeError("boundaries must be MACBoundaryPlan or prepared boundaries.")
        momentum = MACMomentumPlan(
            operators,
            boundaries=prepared_boundaries,
        ).prepare()
        projection = MACPressureProjectionPlan(
            operators,
            boundaries=prepared_boundaries,
            density=1.0,
            tolerance=projection_tolerance,
            maximum_iterations=projection_iterations,
        )
        layout = MACScalarLayout(operators, self.reference.field_names)
        vertical_name = discretization.grid.axis_names[self.axes.vertical_axis]
        zero_gradient = MACScalarBoundaryCondition("neumann", 0.0)
        if self.axes.surface_index == -1:
            temperature_pair = (zero_gradient, self.temperature_surface_flux)
            salinity_pair = (zero_gradient, self.salinity_surface_flux)
        else:
            temperature_pair = (self.temperature_surface_flux, zero_gradient)
            salinity_pair = (self.salinity_surface_flux, zero_gradient)
        scalar_boundaries = MACScalarBoundarySet(
            layout,
            walls={
                self.reference.temperature_name: {
                    vertical_name: temperature_pair,
                },
                self.reference.salinity_name: {
                    vertical_name: salinity_pair,
                },
            },
        )
        scalar_problem = MACScalarProblem(
            (
                MACScalarTransport(
                    self.reference.temperature_name,
                    self.temperature_diffusivity,
                    advection=self.scalar_advection,
                ),
                MACScalarTransport(
                    self.reference.salinity_name,
                    self.salinity_diffusivity,
                    advection=self.scalar_advection,
                ),
            )
        )
        transport = PreparedMACScalarTransport(
            scalar_problem,
            layout,
            scalar_boundaries,
        )
        flow_problem = IncompressibleFlowProblem(3, self.viscosity)
        ocean_forcing = PreparedMACOceanForcing(
            operators,
            self.coriolis_parameter,
            vertical_axis=self.axes.vertical_axis,
            surface_at_upper=self.axes.surface_index == -1,
            reference_density=self.reference.reference_density,
            surface_stress=(
                self.surface_stress
                if self.surface_stress_function is None
                else self.surface_stress_function
            ),
            surface_stress_id=(
                self.surface_stress_id
                if self.surface_stress_function is not None
                else None
            ),
        )
        dynamics = compile_mac_scalar_buoyancy(
            flow_problem,
            momentum,
            projection,
            scalar_problem,
            transport,
            self.reference.buoyancy_law(self.axes),
            ocean_forcing=ocean_forcing,
        )
        return PreparedCartesianBoussinesqOcean(
            self,
            operators,
            momentum,
            projection,
            transport,
            dynamics,
            prepared_boundaries,
        )


class PreparedCartesianBoussinesqOcean(StrictModule):
    """Prepared Cartesian rigid-lid ocean process model."""

    plan: CartesianBoussinesqOceanPlan
    operators: PreparedMACOperators
    momentum: PreparedMACMomentumOperators
    projection: MACPressureProjectionPlan
    transport: PreparedMACScalarTransport
    dynamics: CompiledMACScalarBuoyancyDynamics
    boundaries: PreparedMACBoundaryPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CartesianBoussinesqOceanPlan,
        operators: PreparedMACOperators,
        momentum: PreparedMACMomentumOperators,
        projection: MACPressureProjectionPlan,
        transport: PreparedMACScalarTransport,
        dynamics: CompiledMACScalarBuoyancyDynamics,
        boundaries: PreparedMACBoundaryPlan,
        /,
    ):
        self.plan = plan
        self.operators = operators
        self.momentum = momentum
        self.projection = projection
        self.transport = transport
        self.dynamics = dynamics
        self.boundaries = boundaries
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cartesian-boussinesq-ocean",
                "plan": plan.plan_id,
                "dynamics": dynamics.compilation_id,
                "boundaries": boundaries.prepared_id,
            }
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.dynamics.state_shape

    def initial_state(
        self,
        velocity: tuple[ArrayLike, ...],
        temperature: ArrayLike,
        salinity: ArrayLike,
        /,
    ) -> Array:
        return self.dynamics.project_state(
            tuple(jnp.asarray(component) for component in velocity),
            {
                self.plan.reference.temperature_name: temperature,
                self.plan.reference.salinity_name: salinity,
            },
        )

    def state_view(self, state: ArrayLike, /) -> OceanStateView:
        velocity, scalars = self.dynamics.unpack_state(state)
        temperature = scalars[self.plan.reference.temperature_name]
        salinity = scalars[self.plan.reference.salinity_name]
        return OceanStateView(
            velocity=velocity,
            temperature=temperature,
            salinity=salinity,
            density_anomaly=self.plan.reference.density_anomaly(temperature, salinity),
            state_id=self.prepared_id,
        )

    def stable_step(self, state: ArrayLike, /) -> Array:
        return self.dynamics.step_restriction(state).selected


__all__ = [
    "CartesianBoussinesqOceanPlan",
    "OceanStateView",
    "PreparedCartesianBoussinesqOcean",
]
