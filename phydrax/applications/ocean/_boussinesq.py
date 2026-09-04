#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

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
from ...discretization.finite_volume._mac_scalar import MACScalarSGSPlan
from ...equations import (
    compile_mac_scalar_buoyancy,
    CompiledMACScalarBuoyancyDynamics,
    IncompressibleFlowProblem,
)
from ...equations._ksgs import AbstractKSGSPlan, KSGSState
from ...equations._mac_les import MACAlgebraicLESPlan
from ...solver import MACPressureProjectionPlan
from ._reference import LinearSeawaterReference, OceanAxisConvention


if TYPE_CHECKING:
    from ...discretization.finite_volume._mac_scalar import PreparedMACScalarSGS
    from ...equations._mac_les import PreparedMACAlgebraicLES
    from ...equations._mac_scalar_buoyancy import PreparedMACKSGS
    from ._step import OceanBoussinesqContinuationState


class OceanStateView(StrictModule):
    """Non-authoritative view of one packed Cartesian ocean state."""

    velocity: tuple[Array, ...]
    temperature: Array
    salinity: Array
    density_anomaly: Array
    sgs_kinetic_energy: Array | None
    ksgs_state: KSGSState | None
    state_id: str = eqx.field(static=True)


class CartesianBoussinesqOceanPlan(StrictModule, NonTrainableState):
    """Compile a rigid-lid Cartesian T/S Boussinesq process model."""

    axes: OceanAxisConvention
    reference: LinearSeawaterReference
    viscosity: float = eqx.field(static=True)
    temperature_diffusivity: Array
    salinity_diffusivity: Array
    scalar_advection: MACScalarAdvection = eqx.field(static=True)
    algebraic_les: MACAlgebraicLESPlan | None
    scalar_sgs: MACScalarSGSPlan | None
    ksgs: AbstractKSGSPlan | None
    ksgs_field_name: str | None = eqx.field(static=True)
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
        algebraic_les: MACAlgebraicLESPlan | None = None,
        scalar_sgs: MACScalarSGSPlan | None = None,
        ksgs: AbstractKSGSPlan | None = None,
        ksgs_field_name: str | None = None,
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
        if algebraic_les is not None and not isinstance(
            algebraic_les, MACAlgebraicLESPlan
        ):
            raise TypeError("algebraic_les must be MACAlgebraicLESPlan or None.")
        if ksgs is not None and not isinstance(ksgs, AbstractKSGSPlan):
            raise TypeError("ksgs must be AbstractKSGSPlan or None.")
        if algebraic_les is not None and ksgs is not None:
            raise ValueError("Ocean algebraic LES and prognostic KSGS are alternatives.")
        if scalar_sgs is not None and not isinstance(scalar_sgs, MACScalarSGSPlan):
            raise TypeError("scalar_sgs must be MACScalarSGSPlan or None.")
        closure_active = algebraic_les is not None or ksgs is not None
        if closure_active != (scalar_sgs is not None):
            raise ValueError(
                "Ocean LES requires explicit named scalar SGS declarations; no "
                "turbulent Prandtl or Schmidt numbers are defaulted."
            )
        if scalar_sgs is not None and scalar_sgs.field_names != reference.field_names:
            raise ValueError(
                "Ocean scalar SGS declarations must exactly match the named "
                f"temperature/salinity fields {reference.field_names}."
            )
        kinetic_name = None if ksgs_field_name is None else str(ksgs_field_name)
        if ksgs is not None and (
            not kinetic_name or kinetic_name in reference.field_names
        ):
            raise ValueError("Ocean KSGS requires an explicit, distinct ksgs_field_name.")
        if ksgs is None and ksgs_field_name is not None:
            raise ValueError("ksgs_field_name is valid only with a KSGS plan.")
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
        self.algebraic_les = algebraic_les
        self.scalar_sgs = scalar_sgs
        self.ksgs = ksgs
        self.ksgs_field_name = kinetic_name
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
                "algebraic_les": (
                    "none" if algebraic_les is None else algebraic_les.plan_id
                ),
                "scalar_sgs": ("none" if scalar_sgs is None else scalar_sgs.plan_id),
                "ksgs": "none" if ksgs is None else ksgs.plan_id,
                "ksgs_field_name": kinetic_name,
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
        scalar_names = (
            self.reference.field_names
            if self.ksgs_field_name is None
            else (*self.reference.field_names, self.ksgs_field_name)
        )
        layout = MACScalarLayout(operators, scalar_names)
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
        scalar_transports = [
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
        ]
        if self.ksgs_field_name is not None:
            scalar_transports.append(
                MACScalarTransport(
                    self.ksgs_field_name,
                    self.viscosity,
                    advection="upwind",
                )
            )
        scalar_problem = MACScalarProblem(tuple(scalar_transports))
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
            algebraic_les=self.algebraic_les,
            scalar_sgs=self.scalar_sgs,
            ksgs=self.ksgs,
            ksgs_field_name=self.ksgs_field_name,
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
                "scalar_sgs": (
                    "none"
                    if dynamics.scalar_sgs is None
                    else dynamics.scalar_sgs.prepared_id
                ),
                "ksgs": ("none" if dynamics.ksgs is None else dynamics.ksgs.prepared_id),
            }
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.dynamics.state_shape

    @property
    def prepared_algebraic_les(self) -> "PreparedMACAlgebraicLES | None":
        return self.dynamics.base_dynamics.algebraic_les

    @property
    def prepared_scalar_sgs(self) -> "PreparedMACScalarSGS | None":
        return self.dynamics.scalar_sgs

    @property
    def prepared_ksgs(self) -> "PreparedMACKSGS | None":
        return self.dynamics.ksgs

    def initial_state(
        self,
        velocity: tuple[ArrayLike, ...],
        temperature: ArrayLike,
        salinity: ArrayLike,
        /,
        *,
        sgs_kinetic_energy: ArrayLike | None = None,
    ) -> Array:
        if self.plan.ksgs is None and sgs_kinetic_energy is not None:
            raise ValueError("sgs_kinetic_energy is valid only for an ocean KSGS plan.")
        if self.plan.ksgs is not None and sgs_kinetic_energy is None:
            raise ValueError(
                "Ocean KSGS initial state requires explicit nonnegative "
                "sgs_kinetic_energy."
            )
        scalars: dict[str, ArrayLike] = {
            self.plan.reference.temperature_name: temperature,
            self.plan.reference.salinity_name: salinity,
        }
        if self.plan.ksgs_field_name is not None:
            if self.plan.ksgs is None:
                raise ValueError("Ocean KSGS field has no closure plan.")
            ksgs_state = self.plan.ksgs.initialize_state(sgs_kinetic_energy)
            scalars[self.plan.ksgs_field_name] = ksgs_state.kinetic_energy
        return self.dynamics.project_state(
            tuple(jnp.asarray(component) for component in velocity),
            scalars,
        )

    def ksgs_state(self, state: ArrayLike, /) -> KSGSState | None:
        if self.dynamics.ksgs is None:
            return None
        _, scalars = self.dynamics.unpack_state(state)
        return self.dynamics.ksgs.plan.initialize_state(
            scalars[self.dynamics.ksgs.scalar_field_name]
        )

    def state_view(
        self,
        state: ArrayLike | "OceanBoussinesqContinuationState",
        /,
    ) -> OceanStateView:
        from ._step import OceanBoussinesqContinuationState

        continuation = (
            state if isinstance(state, OceanBoussinesqContinuationState) else None
        )
        coordinates = (
            continuation.coordinates if continuation is not None else jnp.asarray(state)
        )
        velocity, scalars = self.dynamics.unpack_state(coordinates)
        temperature = scalars[self.plan.reference.temperature_name]
        salinity = scalars[self.plan.reference.salinity_name]
        kinetic_energy = (
            None
            if self.plan.ksgs_field_name is None
            else scalars[self.plan.ksgs_field_name]
        )
        ksgs_state = (
            continuation.ksgs_state
            if continuation is not None and continuation.ksgs_state is not None
            else self.ksgs_state(coordinates)
        )
        return OceanStateView(
            velocity=velocity,
            temperature=temperature,
            salinity=salinity,
            density_anomaly=self.plan.reference.density_anomaly(temperature, salinity),
            sgs_kinetic_energy=kinetic_energy,
            ksgs_state=ksgs_state,
            state_id=self.prepared_id,
        )

    def stable_step(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        restriction = self.dynamics.step_restriction(time, state, args)
        return eqx.error_if(
            restriction.selected,
            ~restriction.success,
            "Ocean explicit stable-step certification is unavailable.",
        )


__all__ = [
    "CartesianBoussinesqOceanPlan",
    "OceanStateView",
    "PreparedCartesianBoussinesqOcean",
]
