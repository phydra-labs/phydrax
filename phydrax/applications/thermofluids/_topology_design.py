#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...discretization.finite_volume._mac_scalar import (
    MACScalarProblem,
    MACScalarTransport,
)
from ...discretization.finite_volume._viscous import _cell_to_faces
from ...equations._incompressible import IncompressibleFlowProblem
from ...equations._mac_scalar_buoyancy import (
    compile_mac_scalar_buoyancy,
    CompiledMACScalarBuoyancyDynamics,
)
from ...optim._density_transform import PreparedDensityTransform, threshold_density
from ...optim._iterative._types import Bounds, OptimizationDiagnostics, OptimizationStatus
from ...optim._pde_constrained import (
    AbstractStateSolver,
    StateDesignConstraint,
    StateDesignProblem,
    StateEquationResult,
)
from ...solver._fixed_step import (
    FixedStepProblem,
    FixedStepReplayPolicy,
    FixedStepRolloutPlan,
    FixedStepRolloutResult,
    SSPRK33FixedStepMethod,
)


class ThermofluidMaterial(StrictModule):
    """Two-phase fictitious-domain constitutive law; density one denotes solid.

    Resistance is an acceleration coefficient [1/time], conductivity is [power /
    (length temperature)], and volumetric heat capacity is spatially constant.
    The resistance interpolation is convex for ``resistance_penalty > 1``.
    Conductivity uses a linear cell mixture and native harmonic face fluxes.
    """

    fluid_resistance: float = eqx.field(static=True)
    solid_resistance: float = eqx.field(static=True)
    fluid_conductivity: float = eqx.field(static=True)
    solid_conductivity: float = eqx.field(static=True)
    heat_capacity: float = eqx.field(static=True)
    resistance_penalty: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        solid_resistance: float,
        fluid_conductivity: float,
        solid_conductivity: float,
        heat_capacity: float,
        fluid_resistance: float = 0.0,
        resistance_penalty: float = 3.0,
    ):
        values = tuple(
            float(value)
            for value in (
                fluid_resistance,
                solid_resistance,
                fluid_conductivity,
                solid_conductivity,
                heat_capacity,
                resistance_penalty,
            )
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Thermofluid material coefficients must be finite.")
        fluid, solid, k_fluid, k_solid, capacity, penalty = values
        if fluid < 0.0 or solid <= fluid:
            raise ValueError("Resistance requires solid > fluid >= 0.")
        if min(k_fluid, k_solid, capacity) <= 0.0 or penalty < 1.0:
            raise ValueError(
                "Conductivity/capacity must be positive; penalty must be >= 1."
            )
        self.fluid_resistance = fluid
        self.solid_resistance = solid
        self.fluid_conductivity = k_fluid
        self.solid_conductivity = k_solid
        self.heat_capacity = capacity
        self.resistance_penalty = penalty

    def resistance(self, density: Array, /) -> Array:
        fraction = density / (
            self.resistance_penalty + (1.0 - self.resistance_penalty) * density
        )
        return (
            self.fluid_resistance
            + (self.solid_resistance - self.fluid_resistance) * fraction
        )

    def conductivity(self, density: Array, /) -> Array:
        return (
            self.fluid_conductivity
            + (self.solid_conductivity - self.fluid_conductivity) * density
        )


class _MaterialArgs(StrictModule):
    resistance: tuple[Array, ...]
    heat_source_rate: Array
    user_args: Any


class _BrinkmanForcing(StrictModule):
    external: Any

    def __call__(self, time, velocity, args):
        external = (
            tuple(jnp.zeros_like(value) for value in velocity)
            if self.external is None
            else self.external(time, velocity, args.user_args)
        )
        return tuple(
            source - resistance * value
            for source, resistance, value in zip(
                external, args.resistance, velocity, strict=True
            )
        )


def _heat_source(time, fields, velocity, args):
    del time, fields, velocity
    return args.heat_source_rate


class _ThermofluidRate(StrictModule):
    dynamics: CompiledMACScalarBuoyancyDynamics
    step_size: float = eqx.field(static=True)
    diffusive_resistance_rate: float = eqx.field(static=True)

    def __call__(self, time, state, args):
        stage = self.dynamics.stage(time, state, args)
        restriction = self.dynamics.step_restriction(time, state, args)
        # The native coupled bound includes momentum, scalar transport,
        # buoyancy/stratification, SGS, and any ocean forcing. Retain the
        # additional worst-case material diffusion/Brinkman decay bound.
        inverse_bound = (
            2.0 / restriction.scalars.advective["temperature"]
            + self.diffusive_resistance_rate
        )
        rate = jnp.concatenate(
            (
                self.dynamics.momentum.operators.velocity_space.flatten(
                    stage.velocity_rate
                ),
                self.dynamics.transport.layout.pack(stage.scalar_rates),
            )
        )
        stable = (
            restriction.success
            & restriction.finite
            & (self.step_size <= 0.8 * restriction.selected)
            & (self.step_size * inverse_bound <= 0.8)
        )
        return eqx.error_if(
            rate,
            ~stage.success | ~stable,
            "Thermofluid stage failed or fixed step exceeds the declared stability bound.",
        )


class ThermofluidTopologyEvidence(StrictModule):
    """Endpoint semidiscrete balances, not a steady-state convergence claim.

    Thermal powers use the declared volumetric heat capacity; mechanical powers
    use the native unit reference density. ``solid_velocity_energy_fraction`` is
    a dual-volume weighted leakage measure, including interpolated interface faces.
    """

    objective: Array
    solid_fraction: Array
    maximum_temperature: Array
    heat_content: Array
    heat_content_rate: Array
    heat_source_power: Array
    boundary_heat_outflow: Array
    thermal_balance_defect: Array
    divergence_norm: Array
    boundary_mass_flux: Array
    kinetic_energy: Array
    kinetic_energy_rate: Array
    external_power: Array
    buoyancy_power: Array
    viscous_dissipation: Array
    resistance_power: Array
    resistance_work_defect: Array
    kinetic_balance_defect: Array
    solid_velocity_energy_fraction: Array
    endpoint_rate_norm: Array
    successful: Array


class ThermofluidTopologyReanalysis(StrictModule):
    density: Array
    state: Array
    evidence: ThermofluidTopologyEvidence
    integration_successful: Array
    material_bound_satisfied: Array


class ThermofluidTopologyDesign(StrictModule):
    """Spatial material/Brinkman design on one stationary, uniform 2D MAC grid.

    The envelope is laminar incompressible Newtonian/Boussinesq flow, one temperature,
    static impermeable no-slip/free-slip walls and optional periodic axes. A fixed
    volumetric heating field is applied in both phases. Temperature obeys
    C (dT/dt + div(u T)) = div(k(rho) grad(T)) + Q, with a constant C.
    Velocity is the continuous fictitious-domain field, not a porosity-divided
    interstitial velocity. Finite resistance permits solid leakage, reported
    explicitly. No sharp-interface conjugate heat, moving mesh, variable capacity,
    turbulence, phase change, or viscous/Brinkman dissipation heating is claimed.

    The state equation is **terminal_state - SSPRK33_fixed_schedule(design)**.
    This is an exact derivative of the declared finite-horizon numerical algorithm,
    not an implicit derivative of an incompletely converged steady flow. Every
    realization starts from the same initial condition; optimizer warm starts cannot
    change the physical horizon. Native projection, transport and rollout are reused.
    """

    dynamics: CompiledMACScalarBuoyancyDynamics
    transform: PreparedDensityTransform
    material: ThermofluidMaterial
    initial_state: Array
    heat_source: Array
    rollout_plan: FixedStepRolloutPlan
    beta: float = eqx.field(static=True)
    final_time: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    maximum_solid_fraction: float = eqx.field(static=True)
    resistance_weight: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    diffusive_resistance_rate: float = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACScalarBuoyancyDynamics,
        transform: PreparedDensityTransform,
        material: ThermofluidMaterial,
        initial_state: ArrayLike,
        /,
        *,
        heat_source: ArrayLike,
        final_time: float,
        step_size: float,
        maximum_solid_fraction: float,
        beta: float = 2.0,
        resistance_weight: float = 0.0,
        reference_temperature: float = 0.0,
    ):
        if not isinstance(dynamics, CompiledMACScalarBuoyancyDynamics):
            raise TypeError(
                "dynamics must be compiled native MAC scalar/buoyancy dynamics."
            )
        if not isinstance(transform, PreparedDensityTransform):
            raise TypeError("transform must be PreparedDensityTransform.")
        if not isinstance(material, ThermofluidMaterial):
            raise TypeError("material must be ThermofluidMaterial.")
        if (
            dynamics.momentum.dimension != 2
            or dynamics.transport.layout.field_names != ("temperature",)
            or dynamics.scalar_problem.reaction is not None
            or dynamics.scalar_problem.transports[0].source is not None
            or dynamics.scalar_sgs is not None
            or dynamics.ksgs is not None
            or dynamics.ocean_forcing is not None
        ):
            raise ValueError(
                "Topology design requires 2D laminar temperature-only dynamics without independent sources/reactions."
            )
        discretization = dynamics.momentum.operators.discretization
        grid = discretization.grid
        for axis in grid.structured_axes:
            widths = np.asarray(axis.interval_widths)
            if widths.size < 3 or not np.allclose(
                widths, widths[0], rtol=1e-12, atol=1e-14
            ):
                raise ValueError(
                    "Topology design requires at least three uniform cells on every axis."
                )
        for side in dynamics.momentum.boundaries.sides:
            if (
                side.kind not in ("no-slip", "free-slip")
                or side.provider.time_dependent
                or bool(jnp.any(side.provider.value != 0.0))
                or bool(jnp.any(side.provider.rate != 0.0))
            ):
                raise ValueError(
                    "Topology design requires stationary impermeable zero-velocity walls."
                )
        for lower, upper in dynamics.transport.boundaries.field_conditions("temperature"):
            if lower.function is not None or upper.function is not None:
                raise ValueError("Topology temperature boundaries must be static.")
        filter_plan = transform.plan.filter
        coordinates = np.asarray(discretization.cell_centers).reshape((-1, 2))
        volumes = np.asarray(discretization.cell_volumes).reshape((-1,))
        if (
            filter_plan.coordinates.shape != coordinates.shape
            or not np.allclose(np.asarray(filter_plan.coordinates), coordinates)
            or not np.allclose(np.asarray(filter_plan.measures), volumes)
        ):
            raise ValueError(
                "Density transform coordinates and measures must match MAC cells."
            )
        fixed = np.asarray(filter_plan.fixed_density)
        mask = np.asarray(filter_plan.design_mask)
        if not np.any(mask) or np.any((fixed[~mask] != 0.0) & (fixed[~mask] != 1.0)):
            raise ValueError(
                "A nonempty design region and binary fixed regions are required."
            )
        fraction = float(maximum_solid_fraction)
        if not isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError(
                "maximum_solid_fraction must lie strictly between zero and one."
            )
        beta_, weight, reference = (
            float(beta),
            float(resistance_weight),
            float(reference_temperature),
        )
        if (
            not all(isfinite(value) for value in (beta_, weight, reference))
            or beta_ <= 0.0
            or weight < 0.0
        ):
            raise ValueError(
                "Projection beta must be positive; objective weight nonnegative; all must be finite."
            )
        minimum = transform.apply(jnp.zeros_like(filter_plan.fixed_density), beta_)
        if (
            float(jnp.sum(minimum * filter_plan.measures) / jnp.sum(filter_plan.measures))
            > fraction
        ):
            raise ValueError(
                "Fixed material and filtering make the material bound infeasible."
            )
        source = jnp.asarray(heat_source, dtype=dynamics.transport.layout.dtype)
        if (
            source.shape != discretization.cell_shape
            or bool(jnp.any(~jnp.isfinite(source) | (source < 0.0)))
            or not bool(jnp.any(source > 0.0))
        ):
            raise ValueError(
                "heat_source must be a finite nonnegative cell field with positive total heating."
            )
        state = dynamics.validate_state(initial_state)
        velocity, _ = dynamics.unpack_state(state)
        stage = dynamics.momentum.boundaries.evaluate(0.0)
        enforced = dynamics.momentum.boundaries.enforce(velocity, stage)
        divergence = dynamics.momentum.operators.divergence(velocity)
        if (
            not bool(stage.successful)
            or float(jnp.max(jnp.abs(divergence))) > 1e-8
            or any(
                not np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(velocity, enforced, strict=True)
            )
        ):
            raise ValueError(
                "Initial state must satisfy the fixed MAC boundary and mass constraints."
            )
        flow = IncompressibleFlowProblem(
            2,
            dynamics.flow_problem.viscosity,
            forcing=_BrinkmanForcing(dynamics.flow_problem.forcing),
            forcing_id="thermofluid-brinkman:" + dynamics.flow_problem.forcing_id,
        )
        scalars = MACScalarProblem(
            (
                MACScalarTransport(
                    "temperature",
                    material.fluid_conductivity / material.heat_capacity,
                    advection=dynamics.scalar_problem.transports[0].advection,
                    source=_heat_source,
                    source_id="thermofluid-fixed-volumetric-heating",
                ),
            )
        )
        transport = scalars.prepare(
            dynamics.momentum.operators,
            boundaries=dynamics.transport.boundaries,
        )
        compiled = compile_mac_scalar_buoyancy(
            flow,
            dynamics.momentum,
            dynamics.projection,
            scalars,
            transport,
            dynamics.buoyancy,
        )
        inverse_width_squared = sum(
            1.0 / float(axis.interval_widths[0]) ** 2 for axis in grid.structured_axes
        )
        diffusion_rate = (
            4.0
            * max(
                float(flow.viscosity),
                max(material.fluid_conductivity, material.solid_conductivity)
                / material.heat_capacity,
            )
            * inverse_width_squared
            + material.solid_resistance
        )
        schedule = FixedStepProblem(
            SSPRK33FixedStepMethod(compiled),
            state,
            t0=0.0,
            t1=final_time,
            step_size=step_size,
        )
        if schedule.step_size * diffusion_rate > 0.8:
            raise ValueError(
                "Fixed step exceeds the material diffusion/resistance stability bound."
            )
        self.dynamics = compiled
        self.transform = transform
        self.material = material
        self.initial_state = state
        self.heat_source = source
        self.rollout_plan = FixedStepRolloutPlan(replay=FixedStepReplayPolicy("step"))
        self.beta = beta_
        self.final_time = schedule.t1
        self.step_size = schedule.step_size
        self.maximum_solid_fraction = fraction
        self.resistance_weight = weight
        self.reference_temperature = reference
        self.diffusive_resistance_rate = diffusion_rate

    def density(self, design: ArrayLike, /) -> Array:
        """Filter/project raw cell coordinates, preserving declared fixed regions."""
        return self.transform.apply(design, self.beta).reshape(self.heat_source.shape)

    def _validate_density(self, density: ArrayLike, /) -> Array:
        value = jnp.asarray(density, dtype=self.heat_source.dtype)
        if value.shape != self.heat_source.shape:
            raise ValueError("Physical density must have the MAC cell shape.")
        plan = self.transform.plan.filter
        fixed_mismatch = jnp.any(
            ~plan.design_mask & (value.reshape((-1,)) != plan.fixed_density)
        )
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value) | (value < 0.0) | (value > 1.0))
            | fixed_mismatch,
            "Physical density must be finite in [0, 1] and preserve fixed regions.",
        )

    def _realize(self, density, args):
        density = self._validate_density(density)
        grid = self.dynamics.momentum.operators.discretization.grid
        resistance = self.material.resistance(density)
        face_resistance = tuple(
            _cell_to_faces(resistance, axis, grid_axis.periodic)
            for axis, grid_axis in enumerate(grid.structured_axes)
        )
        diffusivity = self.material.conductivity(density) / self.material.heat_capacity
        # Preserve the native tensor coefficient layout and harmonic-face lowering.
        coefficient = diffusivity[..., None, None] * jnp.eye(2, dtype=diffusivity.dtype)
        dynamics = eqx.tree_at(
            lambda item: item.transport.diffusion[0].coefficient,
            self.dynamics,
            coefficient,
        )
        runtime_args = _MaterialArgs(
            face_resistance,
            self.heat_source / self.material.heat_capacity,
            args,
        )
        return dynamics, runtime_args

    def integrate_density(
        self, density: ArrayLike, /, *, args=None
    ) -> FixedStepRolloutResult:
        """Reanalyze physical density, without filtering or differentiating thresholding."""
        dynamics, runtime_args = self._realize(density, args)
        rate = _ThermofluidRate(dynamics, self.step_size, self.diffusive_resistance_rate)
        problem = FixedStepProblem(
            SSPRK33FixedStepMethod(rate),
            self.initial_state,
            t0=0.0,
            t1=self.final_time,
            step_size=self.step_size,
            args=runtime_args,
            discretization_bundle=dynamics.discretization_bundle,
        )
        return self.rollout_plan.rollout(problem)

    def integrate(self, design: ArrayLike, /, *, args=None) -> FixedStepRolloutResult:
        return self.integrate_density(self.density(design), args=args)

    def residual(self, state, design, args=None):
        result = self.integrate(design, args=args)
        terminal = eqx.error_if(
            result.final_state,
            ~result.successful,
            "Thermofluid fixed integration failed.",
        )
        return self.dynamics.validate_state(state) - terminal

    def solid_fraction(self, density: ArrayLike, /) -> Array:
        volumes = self.dynamics.momentum.operators.discretization.cell_volumes
        return jnp.sum(volumes * density) / jnp.sum(volumes)

    def objective_density(self, state, density, args=None):
        dynamics, runtime_args = self._realize(density, args)
        velocity, scalars = dynamics.unpack_state(state)
        volumes = dynamics.momentum.operators.discretization.cell_volumes
        heat_weights = volumes * self.heat_source
        thermal = jnp.sum(
            heat_weights * (scalars["temperature"] - self.reference_temperature)
        ) / jnp.sum(heat_weights)
        drag = tuple(
            alpha * value
            for alpha, value in zip(runtime_args.resistance, velocity, strict=True)
        )
        power = jnp.real(dynamics.momentum.operators.velocity_space.inner(velocity, drag))
        return thermal + self.resistance_weight * power

    def objective(self, state, design, args=None):
        """Heat-source weighted terminal temperature plus resistance power."""
        return self.objective_density(state, self.density(design), args)

    def state_design_problem(self) -> StateDesignProblem:
        """Bridge the native fixed integration to accepted-point state-design APIs."""
        plan = self.transform.plan.filter
        return StateDesignProblem(
            self.residual,
            self.objective,
            state_solver=_FixedIntegrationStateSolver(self),
            design_bounds=Bounds(
                jnp.zeros_like(plan.fixed_density),
                jnp.ones_like(plan.fixed_density),
            ),
            constraints=(
                StateDesignConstraint(
                    lambda _state, design, _args: self.solid_fraction(
                        self.density(design)
                    ),
                    upper=self.maximum_solid_fraction,
                    constraint_id="thermofluid-solid-volume",
                    depends_on_state=False,
                ),
            ),
            problem_id="thermofluid-fixed-integration-topology",
        )

    def evidence(self, state, density, /, *, args=None) -> ThermofluidTopologyEvidence:
        dynamics, runtime_args = self._realize(density, args)
        stage = dynamics.stage(self.final_time, state, runtime_args)
        diagnostics = dynamics.diagnostics_from_stage(stage)
        thermal = diagnostics.scalars.fields["temperature"]
        capacity = self.material.heat_capacity
        velocity = stage.velocity
        space = dynamics.momentum.operators.velocity_space
        drag = tuple(
            alpha * value
            for alpha, value in zip(runtime_args.resistance, velocity, strict=True)
        )
        resistance_power = jnp.real(space.inner(velocity, drag))
        forcing = dynamics.flow_problem.forcing.external
        external = (
            tuple(jnp.zeros_like(value) for value in velocity)
            if forcing is None
            else forcing(self.final_time, velocity, args)
        )
        external = dynamics.momentum.boundaries.homogeneous_rate(external)
        external_power = jnp.real(space.inner(velocity, external))
        grid = dynamics.momentum.operators.discretization.grid
        solid_velocity = tuple(
            _cell_to_faces(density, axis, grid_axis.periodic) * velocity[axis]
            for axis, grid_axis in enumerate(grid.structured_axes)
        )
        velocity_energy = jnp.real(space.inner(velocity, velocity))
        solid_energy = jnp.real(space.inner(velocity, solid_velocity))
        leakage = jnp.where(
            velocity_energy > 0.0,
            solid_energy / jnp.maximum(velocity_energy, jnp.finfo(state.dtype).tiny),
            0.0,
        )
        rate = jnp.concatenate(
            (
                space.flatten(stage.velocity_rate),
                dynamics.transport.layout.pack(stage.scalar_rates),
            )
        )
        return ThermofluidTopologyEvidence(
            objective=self.objective_density(state, density, args),
            solid_fraction=self.solid_fraction(density),
            maximum_temperature=jnp.max(stage.scalars["temperature"]),
            heat_content=capacity * thermal.content,
            heat_content_rate=capacity * thermal.content_rate,
            heat_source_power=capacity * thermal.source_content_rate,
            boundary_heat_outflow=-capacity * thermal.diffusive_content_rate,
            thermal_balance_defect=capacity
            * (
                thermal.content_rate
                - thermal.source_content_rate
                - thermal.diffusive_content_rate
            ),
            divergence_norm=diagnostics.divergence_norm,
            boundary_mass_flux=dynamics.momentum.boundaries.integrated_mass_flux(
                velocity
            ),
            kinetic_energy=diagnostics.kinetic_energy,
            kinetic_energy_rate=diagnostics.semidiscrete_energy_rate,
            external_power=external_power,
            buoyancy_power=diagnostics.buoyancy_power,
            viscous_dissipation=-diagnostics.viscous_energy_rate,
            resistance_power=resistance_power,
            resistance_work_defect=diagnostics.forcing_power
            - external_power
            + resistance_power,
            kinetic_balance_defect=diagnostics.energy_balance_defect,
            solid_velocity_energy_fraction=leakage,
            endpoint_rate_norm=jnp.sqrt(jnp.sum(rate * rate)),
            successful=diagnostics.success & stage.success,
        )

    def binary_reanalysis(
        self, design: ArrayLike, /, *, eta=0.5, args=None
    ) -> ThermofluidTopologyReanalysis:
        """Threshold once, rerun the same physics, and separately check material volume.

        The thresholded design need not remain volume-feasible. The returned flag is
        evidence, not silent volume repair or a claim that finite drag is impermeable.
        """
        density = threshold_density(self.density(design), eta)
        plan = self.transform.plan.filter
        density = jnp.where(
            plan.design_mask.reshape(density.shape),
            density,
            plan.fixed_density.reshape(density.shape),
        )
        result = self.integrate_density(density, args=args)
        evidence = self.evidence(result.final_state, density, args=args)
        return ThermofluidTopologyReanalysis(
            density,
            result.final_state,
            evidence,
            result.successful,
            evidence.solid_fraction <= self.maximum_solid_fraction,
        )


class _FixedIntegrationStateSolver(AbstractStateSolver):
    workflow: ThermofluidTopologyDesign

    def __init__(self, workflow, /):
        self.workflow = workflow

    @property
    def method_id(self) -> str:
        return "native-thermofluid-fixed-integration"

    def solve(self, problem, design, initial_state, /, *, args):
        del initial_state
        result = self.workflow.integrate(design, args=args)
        residual = problem.residual(result.final_state, design, args)
        status = jnp.where(
            result.successful,
            int(OptimizationStatus.SUCCESS),
            int(OptimizationStatus.BACKEND_FAILED),
        )
        acceptance = problem.state_evidence(
            result.final_state,
            design,
            residual,
            status,
            reference_norm=1.0,
            args=args,
        )
        count = int(round(self.workflow.final_time / self.workflow.step_size))
        return StateEquationResult(
            result.final_state,
            residual,
            status,
            OptimizationDiagnostics(
                iterations=count, residual_evaluations=1, counts_complete=False
            ),
            acceptance,
        )


__all__ = [
    "ThermofluidMaterial",
    "ThermofluidTopologyDesign",
    "ThermofluidTopologyEvidence",
    "ThermofluidTopologyReanalysis",
]
