#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..equations._unstructured_les import (
    PreparedUnstructuredLowMachLES,
    UnstructuredLowMachLESRateResult,
    UnstructuredLowMachLESState,
)
from ..linalg import LinearSolvePolicy
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ._unstructured_incompressible import (
    UnstructuredPressureProjectionPlan,
    UnstructuredPressureProjectionResult,
)


UNSTRUCTURED_LES_SUCCESS = 0
UNSTRUCTURED_LES_STEP_RESTRICTION = -1
UNSTRUCTURED_LES_PRESSURE_FAILURE = -2
UNSTRUCTURED_LES_CONSERVATION_FAILURE = -3
UNSTRUCTURED_LES_INADMISSIBLE_STATE = -4
UNSTRUCTURED_LES_ENERGY_FAILURE = -5


class UnstructuredLowMachLESStepInputs(StrictModule):
    """Thermodynamic and molecular-transport fields for one LES transition."""

    temperature: Array
    specific_heat_capacity_pressure: Array
    partial_specific_enthalpies: Array
    molecular_dynamic_viscosity: Array
    molecular_thermal_conductivity: Array
    molecular_scalar_diffusivities: Array

    def __init__(
        self,
        temperature: ArrayLike,
        specific_heat_capacity_pressure: ArrayLike,
        partial_specific_enthalpies: ArrayLike,
        molecular_dynamic_viscosity: ArrayLike,
        molecular_thermal_conductivity: ArrayLike,
        molecular_scalar_diffusivities: ArrayLike,
        /,
    ):
        self.temperature = _real_inexact(temperature, "temperature")
        self.specific_heat_capacity_pressure = _real_inexact(
            specific_heat_capacity_pressure, "specific_heat_capacity_pressure"
        )
        self.partial_specific_enthalpies = _real_inexact(
            partial_specific_enthalpies, "partial_specific_enthalpies"
        )
        self.molecular_dynamic_viscosity = _real_inexact(
            molecular_dynamic_viscosity, "molecular_dynamic_viscosity"
        )
        self.molecular_thermal_conductivity = _real_inexact(
            molecular_thermal_conductivity, "molecular_thermal_conductivity"
        )
        self.molecular_scalar_diffusivities = _real_inexact(
            molecular_scalar_diffusivities, "molecular_scalar_diffusivities"
        )


class UnstructuredLowMachLESRestartState(StrictModule):
    """Complete accepted-step state, pressure, flux, and continuation history."""

    conservative: UnstructuredLowMachLESState
    enthalpy_density: Array
    pressure: Array
    face_normal_velocity: Array
    mass_flux: Array
    pressure_increment: Array
    accepted_steps: Array

    def __init__(
        self,
        conservative: UnstructuredLowMachLESState,
        enthalpy_density: ArrayLike,
        pressure: ArrayLike,
        face_normal_velocity: ArrayLike,
        mass_flux: ArrayLike,
        pressure_increment: ArrayLike,
        accepted_steps: ArrayLike,
        /,
    ):
        if not isinstance(conservative, UnstructuredLowMachLESState):
            raise TypeError("conservative must be UnstructuredLowMachLESState.")
        enthalpy = _real_inexact(enthalpy_density, "enthalpy_density")
        pressure_ = _real_inexact(pressure, "pressure")
        face_velocity = _real_inexact(face_normal_velocity, "face_normal_velocity")
        mass_flux_ = _real_inexact(mass_flux, "mass_flux")
        increment = _real_inexact(pressure_increment, "pressure_increment")
        steps = jnp.asarray(accepted_steps)
        if steps.shape != () or not jnp.issubdtype(steps.dtype, jnp.integer):
            raise TypeError("accepted_steps must be one integer scalar array.")
        steps = eqx.error_if(
            steps,
            steps < 0,
            "Unstructured LES accepted_steps must be nonnegative.",
        )
        self.conservative = conservative
        self.enthalpy_density = enthalpy
        self.pressure = pressure_
        self.face_normal_velocity = face_velocity
        self.mass_flux = mass_flux_
        self.pressure_increment = increment
        self.accepted_steps = steps


class UnstructuredLowMachLESStepRestriction(StrictModule):
    """Auditable explicit advection, diffusion, and positivity step bounds."""

    advective_step: Array
    diffusive_step: Array
    source_step: Array
    positivity_step: Array
    maximum_step: Array
    maximum_advective_frequency: Array
    maximum_diffusive_frequency: Array
    finite: Array


class UnstructuredLowMachLESStepEvidence(StrictModule):
    """Constraint, conservation, common-flux, and energy evidence for one attempt."""

    divergence_before_norm: Array
    divergence_after_norm: Array
    pressure_residual_norm: Array
    pressure_rhs_norm: Array
    pressure_gauge_residual: Array
    pressure_compatibility_residual: Array
    mass_balance_residual: Array
    momentum_balance_residual: Array
    scalar_balance_residual: Array
    enthalpy_balance_residual: Array
    ksgs_balance_residual: Array | None
    resolved_kinetic_energy_change: Array
    sgs_kinetic_energy_change: Array | None
    modeled_sgs_dissipation: Array
    advective_kinetic_energy_rate: Array
    pressure_work_rate: Array
    molecular_viscous_work_rate: Array
    sgs_stress_work_rate: Array
    sgs_deviatoric_work_rate: Array
    ksgs_transport_rate: Array
    ksgs_source_rate: Array
    enthalpy_transport_rate: Array
    production_limit_thermalization_rate: Array
    modeled_transfer_residual: Array
    normalized_modeled_transfer_residual: Array
    normalized_positive_sgs_work: Array
    sgs_work_dissipative: Array
    modeled_transfer_balanced: Array
    temporal_energy_defect: Array
    resolved_sgs_energy_balance_residual: Array
    normalized_resolved_sgs_energy_balance: Array
    energy_balanced: Array
    shared_mass_flux: Array
    conservative: Array
    energy_finite: Array
    admissible: Array
    step_stable: Array
    pressure_converged: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class UnstructuredLowMachLESStepResult(StrictModule):
    """Detailed pressure-corrected attempt plus its atomic fixed-step result."""

    fixed_step: FixedStepResult
    predictor_rate: UnstructuredLowMachLESRateResult
    rate: UnstructuredLowMachLESRateResult
    pressure: UnstructuredPressureProjectionResult
    restriction: UnstructuredLowMachLESStepRestriction
    evidence: UnstructuredLowMachLESStepEvidence
    status: Array


class UnstructuredLowMachLESFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    """Pressure-corrected forward-Euler LES transition bound to one exact step."""

    dynamics: PreparedUnstructuredLowMachLES
    projection: UnstructuredPressureProjectionPlan
    maximum_courant_number: float = eqx.field(static=True)
    maximum_diffusion_number: float = eqx.field(static=True)
    maximum_source_fraction: float = eqx.field(static=True)
    _required_step_size: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedUnstructuredLowMachLES,
        step_size: ArrayLike,
        /,
        *,
        maximum_courant_number: float = 0.5,
        maximum_diffusion_number: float = 0.25,
        maximum_source_fraction: float = 0.25,
        pressure_tolerance: float = 1.0e-9,
        pressure_iterations: int = 200,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(dynamics, PreparedUnstructuredLowMachLES):
            raise TypeError("dynamics must be PreparedUnstructuredLowMachLES.")
        raw_step = np.asarray(step_size)
        if (
            np.iscomplexobj(raw_step)
            or raw_step.shape != ()
            or not np.isfinite(raw_step)
            or float(raw_step) <= 0.0
        ):
            raise ValueError("Unstructured LES step_size must be finite and positive.")
        courant = float(maximum_courant_number)
        diffusion = float(maximum_diffusion_number)
        source = float(maximum_source_fraction)
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (courant, diffusion, source)
        ):
            raise ValueError("Every unstructured LES step restriction must be positive.")
        step = float(raw_step)
        projection = UnstructuredPressureProjectionPlan(
            dynamics.operators,
            tolerance=pressure_tolerance,
            maximum_iterations=pressure_iterations,
            dtype=dynamics.operators.discretization.cell_volumes.dtype,
            linear_policy=linear_policy,
        )
        self.dynamics = dynamics
        self.projection = projection
        self.maximum_courant_number = courant
        self.maximum_diffusion_number = diffusion
        self.maximum_source_fraction = source
        self._required_step_size = step
        self.method_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-les-pressure-corrected-euler",
                "dynamics": dynamics.prepared_id,
                "projection": projection.plan_id,
                "step_size": step,
                "maximum_courant_number": courant,
                "maximum_diffusion_number": diffusion,
                "maximum_source_fraction": source,
                "transaction": "atomic-rollback-complete-restart",
            }
        )

    @property
    def required_step_size(self) -> float:
        return self._required_step_size

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def initialize(
        self,
        conservative: UnstructuredLowMachLESState,
        pressure: ArrayLike,
        args: UnstructuredLowMachLESStepInputs,
        /,
    ) -> UnstructuredLowMachLESRestartState:
        """Construct a restart-complete state after validating every initial field."""

        if not isinstance(conservative, UnstructuredLowMachLESState):
            raise TypeError("conservative must be UnstructuredLowMachLESState.")
        arguments = self._validate_inputs(args)
        density = jnp.asarray(conservative.density)
        pressure_ = self.dynamics.operators.gauge_project(
            self.dynamics.operators.validate_cell_scalar(pressure, "Pressure").astype(
                density.dtype
            )
        )
        inverse = jnp.asarray(self.required_step_size, dtype=density.dtype) / density
        rate = self._rate(conservative, pressure_, arguments, inverse)
        enthalpy = jnp.sum(
            conservative.scalar_densities
            * arguments.partial_specific_enthalpies.astype(density.dtype),
            axis=-1,
        )
        return UnstructuredLowMachLESRestartState(
            conservative,
            enthalpy,
            pressure_,
            rate.fluxes.face_normal_velocity,
            rate.fluxes.mass_flux,
            jnp.zeros_like(pressure_),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def step_restriction(
        self,
        time: ArrayLike,
        state: UnstructuredLowMachLESRestartState,
        args: UnstructuredLowMachLESStepInputs,
        /,
    ) -> UnstructuredLowMachLESStepRestriction:
        """Return the full explicit restriction evaluated at an accepted state."""

        del time
        self._validate_restart(state)
        arguments = self._validate_inputs(args)
        density = state.conservative.density
        step = jnp.asarray(self.required_step_size, dtype=density.dtype)
        inverse = step / density
        rate = self._rate(
            state.conservative,
            state.pressure,
            arguments,
            inverse,
            face_normal_velocity=state.face_normal_velocity,
        )
        return self._restriction(state, arguments, rate)

    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        state: UnstructuredLowMachLESRestartState,
        step_size: Array,
        args: Any,
        /,
    ) -> UnstructuredLowMachLESStepResult:
        """Attempt one pressure correction and atomically accept or retain all history."""

        del step_index, time
        self._validate_restart(state)
        arguments = self._validate_inputs(args)
        dtype = state.conservative.density.dtype
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        declared = jnp.asarray(self.required_step_size, dtype=dtype)
        step = eqx.error_if(
            step,
            ~(jnp.isfinite(step) & (step == declared)),
            "Unstructured LES step_size must exactly equal its prepared value.",
        )
        density = state.conservative.density
        velocity = state.conservative.momentum_density / density[:, None]
        inverse = step / density
        initial_rate = self._rate(
            state.conservative,
            state.pressure,
            arguments,
            inverse,
            face_normal_velocity=state.face_normal_velocity,
        )
        velocity_rate = (
            initial_rate.momentum_density_rate
            - velocity * initial_rate.density_rate[:, None]
        ) / density[:, None]
        predicted_velocity = velocity + step * velocity_rate
        boundary_velocity = jnp.zeros(
            (self.dynamics.operators.discretization.face_measures.size,), dtype=dtype
        )
        projected = self.projection.project(
            predicted_velocity,
            step,
            pressure=state.pressure,
            inverse_momentum_diagonal=inverse,
            boundary_normal_velocity=boundary_velocity,
        )
        rate = self._rate(
            state.conservative,
            projected.pressure,
            arguments,
            inverse,
            face_normal_velocity=projected.face_normal_velocity,
        )
        restriction = self._restriction(state, arguments, rate)
        candidate_conservative, candidate_enthalpy = self._candidate(state, rate, step)
        candidate = UnstructuredLowMachLESRestartState(
            candidate_conservative,
            candidate_enthalpy,
            projected.pressure.astype(dtype),
            rate.fluxes.face_normal_velocity,
            rate.fluxes.mass_flux,
            projected.pressure_increment.astype(dtype),
            state.accepted_steps + jnp.asarray(1, dtype=state.accepted_steps.dtype),
        )
        evidence = self._evidence(state, candidate, rate, projected, restriction, step)
        accepted = tree_where(evidence.successful, candidate, state)
        status = _step_status(evidence)
        residual = jnp.maximum(
            evidence.pressure_residual_norm,
            jnp.maximum(
                evidence.divergence_after_norm,
                jnp.max(jnp.abs(evidence.mass_balance_residual)),
            ),
        )
        fixed_step = FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=evidence.successful,
            residual=residual,
            iterations=projected.linear.diagnostics.iterations,
            work=projected.linear.diagnostics.matvec_count,
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.zeros((), dtype=dtype),
        )
        return UnstructuredLowMachLESStepResult(
            fixed_step,
            initial_rate,
            rate,
            projected,
            restriction,
            evidence,
            status,
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: UnstructuredLowMachLESRestartState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        return self.step_detailed(step_index, time, state, step_size, args).fixed_step

    def _validate_inputs(self, args: Any, /) -> UnstructuredLowMachLESStepInputs:
        if not isinstance(args, UnstructuredLowMachLESStepInputs):
            raise TypeError("args must be UnstructuredLowMachLESStepInputs.")
        return args

    def _validate_restart(self, state: UnstructuredLowMachLESRestartState, /) -> None:
        if not isinstance(state, UnstructuredLowMachLESRestartState):
            raise TypeError("state must be UnstructuredLowMachLESRestartState.")
        cells = self.dynamics.operators.discretization.cell_count
        faces = self.dynamics.operators.discretization.face_measures.size
        species = len(self.dynamics.plan.favre_model.fields.species_names)
        conservative = state.conservative
        if (
            conservative.density.shape != (cells,)
            or conservative.momentum_density.shape != (cells, 3)
            or conservative.scalar_densities.shape != (cells, species)
            or state.enthalpy_density.shape != (cells,)
            or state.pressure.shape != (cells,)
            or state.pressure_increment.shape != (cells,)
            or state.face_normal_velocity.shape != (faces,)
            or state.mass_flux.shape != (faces,)
        ):
            raise ValueError("Unstructured LES restart fields do not match the mesh.")
        if (conservative.ksgs is None) != (self.dynamics.plan.ksgs_plan is None):
            raise ValueError("Restart KSGS state must exactly match the prepared route.")

    def _rate(
        self,
        conservative: UnstructuredLowMachLESState,
        pressure: Array,
        args: UnstructuredLowMachLESStepInputs,
        inverse: Array,
        /,
        *,
        face_normal_velocity: Array | None = None,
    ) -> UnstructuredLowMachLESRateResult:
        return self.dynamics.semidiscrete_rate(
            conservative,
            pressure,
            args.temperature,
            args.specific_heat_capacity_pressure,
            args.partial_specific_enthalpies,
            args.molecular_dynamic_viscosity,
            args.molecular_thermal_conductivity,
            args.molecular_scalar_diffusivities,
            inverse,
            authoritative_face_normal_velocity=face_normal_velocity,
        )

    def _restriction(
        self,
        state: UnstructuredLowMachLESRestartState,
        args: UnstructuredLowMachLESStepInputs,
        rate: UnstructuredLowMachLESRateResult,
        /,
    ) -> UnstructuredLowMachLESStepRestriction:
        discretization = self.dynamics.operators.discretization
        density = state.conservative.density
        dtype = density.dtype
        volumes = discretization.cell_volumes.astype(dtype)
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        interior = self.dynamics.operators.interior_faces
        flux_magnitude = jnp.abs(rate.fluxes.mass_flux)
        throughput = jnp.zeros_like(density).at[owner].add(flux_magnitude)
        throughput = throughput.at[safe_neighbour].add(
            jnp.where(interior, flux_magnitude, 0.0)
        )
        advective_frequency = throughput / (density * volumes)
        maximum_advective_frequency = jnp.max(advective_frequency)
        advective_step = _frequency_step(
            maximum_advective_frequency,
            self.maximum_courant_number,
            dtype,
        )

        widths = jnp.min(self.dynamics.filter_scale.astype(dtype), axis=-1)
        molecular_nu = args.molecular_dynamic_viscosity.astype(dtype) / density
        sgs_nu = rate.kinematic_eddy_viscosity
        momentum_diffusivity = molecular_nu + sgs_nu
        thermal_diffusivity = (
            args.molecular_thermal_conductivity.astype(dtype)
            / (density * args.specific_heat_capacity_pressure.astype(dtype))
            + sgs_nu / self.dynamics.plan.favre_model.turbulent_prandtl_number
        )
        scalar_diffusivity = (
            args.molecular_scalar_diffusivities.astype(dtype)
            + sgs_nu[:, None] / self.dynamics.species_schmidt_numbers[None, :]
        )
        maximum_diffusivity = jnp.maximum(
            momentum_diffusivity,
            jnp.maximum(thermal_diffusivity, jnp.max(scalar_diffusivity, axis=-1)),
        )
        if rate.ksgs is not None:
            maximum_diffusivity = jnp.maximum(maximum_diffusivity, rate.ksgs.diffusivity)
        diffusive_frequency = (
            2.0 * discretization.cell_dimension * maximum_diffusivity / widths**2
        )
        maximum_diffusive_frequency = jnp.max(diffusive_frequency)
        diffusive_step = _frequency_step(
            maximum_diffusive_frequency,
            self.maximum_diffusion_number,
            dtype,
        )

        density_bound = _positive_update_bound(
            state.conservative.density, rate.density_rate
        )
        scalar_bound = _positive_update_bound(
            state.conservative.scalar_densities, rate.scalar_density_rate
        )
        positivity_step = jnp.minimum(density_bound, scalar_bound)
        source_step = jnp.asarray(jnp.inf, dtype=dtype)
        if rate.ksgs_density_rate is not None:
            if state.conservative.ksgs is None or rate.ksgs is None:
                raise ValueError("KSGS rate requires KSGS restart state.")
            ksgs_bound = _positive_update_bound(
                state.conservative.density * state.conservative.ksgs.kinetic_energy,
                rate.ksgs_density_rate,
            )
            positivity_step = jnp.minimum(positivity_step, ksgs_bound)
            local_loss = (
                rate.ksgs.contributions.dissipation
                + rate.ksgs.contributions.low_re_dissipation
            )
            source_step = jnp.min(
                jnp.where(
                    local_loss > 0.0,
                    state.conservative.ksgs.kinetic_energy / local_loss,
                    jnp.asarray(jnp.inf, dtype=dtype),
                )
            )
        positivity_step = self.maximum_source_fraction * positivity_step
        source_step = self.maximum_source_fraction * source_step
        maximum_step = jnp.minimum(
            jnp.minimum(advective_step, diffusive_step),
            jnp.minimum(source_step, positivity_step),
        )
        finite = (
            jnp.all(jnp.isfinite(advective_frequency))
            & jnp.all(jnp.isfinite(diffusive_frequency))
            & ~jnp.isnan(maximum_step)
            & (maximum_step > 0.0)
        )
        return UnstructuredLowMachLESStepRestriction(
            advective_step,
            diffusive_step,
            source_step,
            positivity_step,
            maximum_step,
            maximum_advective_frequency,
            maximum_diffusive_frequency,
            finite,
        )

    def _candidate(
        self,
        state: UnstructuredLowMachLESRestartState,
        rate: UnstructuredLowMachLESRateResult,
        step: Array,
        /,
    ) -> tuple[UnstructuredLowMachLESState, Array]:
        current = state.conservative
        density = current.density + step * rate.density_rate
        momentum = current.momentum_density + step * rate.momentum_density_rate
        scalars = current.scalar_densities + step * rate.scalar_density_rate
        enthalpy = state.enthalpy_density + step * rate.enthalpy_density_rate
        ksgs = current.ksgs
        if ksgs is not None:
            if rate.ksgs_density_rate is None:
                raise ValueError("KSGS restart requires a conservative KSGS rate.")
            ksgs_density = (
                current.density * ksgs.kinetic_energy + step * rate.ksgs_density_rate
            )
            kinetic_energy = ksgs_density / density
            ksgs = eqx.tree_at(lambda value: value.kinetic_energy, ksgs, kinetic_energy)
        return UnstructuredLowMachLESState(
            density,
            momentum,
            scalars,
            ksgs=ksgs,
        ), enthalpy

    def _evidence(
        self,
        current: UnstructuredLowMachLESRestartState,
        candidate: UnstructuredLowMachLESRestartState,
        rate: UnstructuredLowMachLESRateResult,
        pressure: UnstructuredPressureProjectionResult,
        restriction: UnstructuredLowMachLESStepRestriction,
        step: Array,
        /,
    ) -> UnstructuredLowMachLESStepEvidence:
        discretization = self.dynamics.operators.discretization
        volumes = discretization.cell_volumes.astype(step.dtype)
        boundary = ~self.dynamics.operators.interior_faces
        mass_balance = _transition_balance(
            candidate.conservative.density,
            current.conservative.density,
            rate.fluxes.mass_flux,
            volumes,
            boundary,
            step,
        )
        total_momentum_flux = (
            rate.fluxes.advective_momentum_flux
            + rate.fluxes.pressure_momentum_flux
            + rate.fluxes.molecular_momentum_flux
            + rate.fluxes.sgs_momentum_flux
        )
        momentum_balance = _transition_balance(
            candidate.conservative.momentum_density,
            current.conservative.momentum_density,
            total_momentum_flux,
            volumes,
            boundary,
            step,
        )
        total_scalar_flux = (
            rate.fluxes.advective_scalar_flux
            + rate.fluxes.molecular_scalar_flux
            + rate.fluxes.sgs_scalar_flux
        )
        scalar_balance = _transition_balance(
            candidate.conservative.scalar_densities,
            current.conservative.scalar_densities,
            total_scalar_flux,
            volumes,
            boundary,
            step,
        )
        total_enthalpy_flux = (
            rate.fluxes.advective_enthalpy_flux
            + rate.fluxes.molecular_enthalpy_flux
            + rate.fluxes.sgs_enthalpy_flux
        )
        enthalpy_balance = _transition_balance(
            candidate.enthalpy_density,
            current.enthalpy_density,
            total_enthalpy_flux,
            volumes,
            boundary,
            step,
        )
        production_limit_thermalization_rate = jnp.sum(
            volumes * rate.modeled_enthalpy_source_density
        )
        enthalpy_balance = enthalpy_balance - step * production_limit_thermalization_rate
        enthalpy_change = jnp.sum(
            volumes * (candidate.enthalpy_density - current.enthalpy_density)
        )
        enthalpy_transport_rate = jnp.sum(
            volumes * (rate.enthalpy_density_rate - rate.modeled_enthalpy_source_density)
        )
        ksgs_balance = None
        if current.conservative.ksgs is not None:
            if candidate.conservative.ksgs is None or rate.ksgs_density_rate is None:
                raise ValueError("KSGS transition evidence requires complete KSGS data.")
            previous_density = (
                current.conservative.density * current.conservative.ksgs.kinetic_energy
            )
            next_density = (
                candidate.conservative.density
                * candidate.conservative.ksgs.kinetic_energy
            )
            local_defect = next_density - previous_density - step * rate.ksgs_density_rate
            ksgs_balance = jnp.sum(volumes * local_defect)

        velocity_before = (
            current.conservative.momentum_density / current.conservative.density[:, None]
        )
        velocity_after = (
            candidate.conservative.momentum_density
            / candidate.conservative.density[:, None]
        )
        resolved_before = jnp.sum(
            volumes
            * 0.5
            * current.conservative.density
            * jnp.sum(velocity_before**2, axis=-1)
        )
        resolved_after = jnp.sum(
            volumes
            * 0.5
            * candidate.conservative.density
            * jnp.sum(velocity_after**2, axis=-1)
        )
        resolved_change = resolved_after - resolved_before
        sgs_change = None
        sgs_before = jnp.zeros((), dtype=step.dtype)
        if current.conservative.ksgs is not None:
            if candidate.conservative.ksgs is None:
                raise ValueError("Candidate KSGS energy is missing.")
            sgs_before = jnp.sum(
                volumes
                * current.conservative.density
                * current.conservative.ksgs.kinetic_energy
            )
            sgs_after = jnp.sum(
                volumes
                * candidate.conservative.density
                * candidate.conservative.ksgs.kinetic_energy
            )
            sgs_change = sgs_after - sgs_before

        zero_density_rate = jnp.zeros_like(rate.density_rate)
        advective_kinetic_energy_rate = _kinetic_energy_rate(
            velocity_before,
            _negative_divergence(
                rate.fluxes.advective_momentum_flux,
                discretization,
            ),
            rate.density_rate,
            volumes,
        )
        pressure_work_rate = _kinetic_energy_rate(
            velocity_before,
            _negative_divergence(
                rate.fluxes.pressure_momentum_flux,
                discretization,
            ),
            zero_density_rate,
            volumes,
        )
        molecular_viscous_work_rate = _kinetic_energy_rate(
            velocity_before,
            _negative_divergence(
                rate.fluxes.molecular_momentum_flux,
                discretization,
            ),
            zero_density_rate,
            volumes,
        )
        sgs_stress_work_rate = _kinetic_energy_rate(
            velocity_before,
            _negative_divergence(
                rate.fluxes.sgs_momentum_flux,
                discretization,
            ),
            zero_density_rate,
            volumes,
        )
        sgs_deviatoric_work_rate = _kinetic_energy_rate(
            velocity_before,
            _negative_divergence(
                rate.fluxes.sgs_deviatoric_momentum_flux,
                discretization,
            ),
            zero_density_rate,
            volumes,
        )
        ksgs_transport_rate = jnp.zeros((), dtype=step.dtype)
        ksgs_source_rate = jnp.zeros((), dtype=step.dtype)
        ksgs_production_rate = jnp.zeros((), dtype=step.dtype)
        if rate.ksgs_density_rate is not None:
            if (
                rate.fluxes.advective_ksgs_flux is None
                or rate.fluxes.diffusive_ksgs_flux is None
                or rate.ksgs is None
                or rate.ksgs_raw_production_density is None
                or rate.ksgs_production_density is None
            ):
                raise ValueError("KSGS energy evidence requires complete fluxes.")
            ksgs_transport_cell_rate = _negative_divergence(
                rate.fluxes.advective_ksgs_flux + rate.fluxes.diffusive_ksgs_flux,
                discretization,
            )
            ksgs_source_cell_rate = (
                rate.ksgs_production_density
                + current.conservative.density
                * (
                    -rate.ksgs.contributions.dissipation
                    + rate.ksgs.contributions.buoyancy
                    - rate.ksgs.contributions.low_re_dissipation
                )
            )
            ksgs_transport_rate = jnp.sum(volumes * ksgs_transport_cell_rate)
            ksgs_source_rate = jnp.sum(volumes * ksgs_source_cell_rate)
            ksgs_production_rate = jnp.sum(volumes * rate.ksgs_raw_production_density)
        resolved_rate = _kinetic_energy_rate(
            velocity_before,
            rate.momentum_density_rate,
            rate.density_rate,
            volumes,
        )
        direct_energy_rate = resolved_rate
        if rate.ksgs_density_rate is not None:
            direct_energy_rate = direct_energy_rate + jnp.sum(
                volumes * rate.ksgs_density_rate
            )
        direct_energy_rate = direct_energy_rate + jnp.sum(
            volumes * rate.enthalpy_density_rate
        )
        decomposed_energy_rate = (
            advective_kinetic_energy_rate
            + pressure_work_rate
            + molecular_viscous_work_rate
            + sgs_stress_work_rate
            + ksgs_transport_rate
            + ksgs_source_rate
            + enthalpy_transport_rate
            + production_limit_thermalization_rate
        )
        total_energy_change = (
            resolved_change
            + enthalpy_change
            + (jnp.zeros((), dtype=step.dtype) if sgs_change is None else sgs_change)
        )
        temporal_energy_defect = total_energy_change - step * direct_energy_rate
        energy_balance_residual = (
            total_energy_change - step * decomposed_energy_rate - temporal_energy_defect
        )
        energy_scale = jnp.maximum(
            jnp.abs(total_energy_change),
            jnp.abs(temporal_energy_defect),
        )
        for contribution in (
            advective_kinetic_energy_rate,
            pressure_work_rate,
            molecular_viscous_work_rate,
            sgs_stress_work_rate,
            ksgs_transport_rate,
            ksgs_source_rate,
            enthalpy_transport_rate,
            production_limit_thermalization_rate,
        ):
            energy_scale = jnp.maximum(
                energy_scale,
                jnp.abs(step * contribution),
            )
        reference_energy = (
            resolved_before + sgs_before + jnp.sum(volumes * current.enthalpy_density)
        )
        energy_scale = jnp.maximum(
            energy_scale,
            jnp.finfo(step.dtype).eps * jnp.maximum(jnp.abs(reference_energy), 1.0),
        )
        normalized_energy_balance = jnp.abs(energy_balance_residual) / energy_scale
        modeled_transfer_target = (
            ksgs_production_rate
            if rate.ksgs is not None
            else rate.evidence.modeled_sgs_dissipation
        )
        modeled_transfer_residual = sgs_deviatoric_work_rate + modeled_transfer_target
        modeled_transfer_scale = jnp.maximum(
            jnp.abs(sgs_deviatoric_work_rate),
            jnp.abs(modeled_transfer_target),
        )
        modeled_transfer_scale = jnp.maximum(
            modeled_transfer_scale,
            jnp.finfo(step.dtype).eps
            * jnp.maximum(jnp.abs(reference_energy) / step, 1.0),
        )
        normalized_modeled_transfer_residual = (
            jnp.abs(modeled_transfer_residual) / modeled_transfer_scale
        )
        positive_sgs_work = jnp.maximum(sgs_deviatoric_work_rate, 0.0)
        normalized_positive_sgs_work = positive_sgs_work / modeled_transfer_scale

        divergence_before_norm = _volume_norm(pressure.divergence_before, volumes)
        divergence_after_norm = _volume_norm(pressure.divergence_after, volumes)
        pressure_residual_norm = _volume_norm(pressure.pressure_residual, volumes)
        pressure_rhs_norm = _volume_norm(pressure.compatible_rhs, volumes)
        compatibility = jnp.abs(
            jnp.sum(volumes * pressure.divergence_before) / jnp.sum(volumes)
        )
        tolerance = jnp.asarray(
            self.dynamics.plan.conservation_tolerance, dtype=step.dtype
        )
        balance_values = (
            mass_balance,
            momentum_balance,
            scalar_balance,
            enthalpy_balance,
        )
        conservative = rate.evidence.conservative
        for value in balance_values:
            conservative = conservative & jnp.all(
                jnp.abs(value) <= tolerance * jnp.maximum(jnp.abs(value), 1.0)
            )
        if ksgs_balance is not None:
            conservative = conservative & (
                jnp.abs(ksgs_balance)
                <= tolerance * jnp.maximum(jnp.abs(ksgs_balance), 1.0)
            )
        scalar_sum = jnp.sum(candidate.conservative.scalar_densities, axis=-1)
        scalar_scale = jnp.maximum(jnp.abs(candidate.conservative.density), 1.0)
        admissible = (
            jnp.all(jnp.isfinite(candidate.conservative.density))
            & jnp.all(jnp.isfinite(candidate.conservative.momentum_density))
            & jnp.all(jnp.isfinite(candidate.conservative.scalar_densities))
            & jnp.all(jnp.isfinite(candidate.enthalpy_density))
            & jnp.all(jnp.isfinite(candidate.pressure))
            & jnp.all(jnp.isfinite(candidate.pressure_increment))
            & jnp.all(jnp.isfinite(candidate.face_normal_velocity))
            & jnp.all(jnp.isfinite(candidate.mass_flux))
            & jnp.all(candidate.conservative.density > 0.0)
            & jnp.all(candidate.conservative.scalar_densities >= 0.0)
            & jnp.all(
                jnp.abs(scalar_sum - candidate.conservative.density)
                <= 64.0
                * jnp.finfo(step.dtype).eps
                * scalar_scale
                * max(candidate.conservative.scalar_densities.shape[-1], 1)
            )
        )
        if candidate.conservative.ksgs is not None:
            admissible = (
                admissible
                & jnp.all(jnp.isfinite(candidate.conservative.ksgs.kinetic_energy))
                & jnp.all(candidate.conservative.ksgs.kinetic_energy >= 0.0)
            )
        energy_finite = rate.evidence.sgs_dissipative
        for value in (
            resolved_change,
            rate.evidence.modeled_sgs_dissipation,
            advective_kinetic_energy_rate,
            pressure_work_rate,
            molecular_viscous_work_rate,
            sgs_stress_work_rate,
            ksgs_transport_rate,
            ksgs_source_rate,
            sgs_deviatoric_work_rate,
            modeled_transfer_residual,
            temporal_energy_defect,
            energy_balance_residual,
            enthalpy_transport_rate,
            production_limit_thermalization_rate,
            normalized_energy_balance,
            normalized_modeled_transfer_residual,
            normalized_positive_sgs_work,
        ):
            energy_finite = energy_finite & jnp.isfinite(value)
        if sgs_change is not None:
            energy_finite = energy_finite & jnp.isfinite(sgs_change)
        energy_balance_tolerance = jnp.maximum(
            tolerance,
            256.0 * jnp.finfo(step.dtype).eps,
        )
        if rate.ksgs is None:
            sgs_work_dissipative = jnp.isfinite(normalized_positive_sgs_work) & (
                normalized_positive_sgs_work <= energy_balance_tolerance
            )
            modeled_transfer_balanced = sgs_work_dissipative
        else:
            sgs_work_dissipative = jnp.asarray(True)
            modeled_transfer_balanced = (
                jnp.isfinite(normalized_modeled_transfer_residual)
                & (normalized_modeled_transfer_residual <= energy_balance_tolerance)
                & jnp.all(rate.ksgs.evidence.production_nonnegative)
            )
        energy_balanced = (
            energy_finite
            & (normalized_energy_balance <= energy_balance_tolerance)
            & modeled_transfer_balanced
        )
        pressure_scale = jnp.maximum(pressure_rhs_norm, 1.0)
        pressure_converged = (
            pressure.converged
            & (pressure_residual_norm <= self.projection.tolerance * pressure_scale)
            & (
                divergence_after_norm
                <= self.projection.tolerance * jnp.maximum(divergence_before_norm, 1.0)
            )
            & (jnp.abs(pressure.gauge_defect) <= self.projection.tolerance)
            & (compatibility <= self.projection.tolerance)
        )
        step_stable = (
            restriction.finite
            & (step <= restriction.maximum_step)
            & jnp.isfinite(step)
            & (step > 0.0)
        )
        shared_mass_flux = (
            (rate.evidence.shared_momentum_mass_flux_residual <= tolerance)
            & (rate.evidence.shared_scalar_mass_flux_residual <= tolerance)
            & (rate.evidence.shared_enthalpy_mass_flux_residual <= tolerance)
        )
        if rate.evidence.shared_ksgs_mass_flux_residual is not None:
            shared_mass_flux = shared_mass_flux & (
                rate.evidence.shared_ksgs_mass_flux_residual <= tolerance
            )
        successful = (
            step_stable
            & pressure_converged
            & rate.evidence.successful
            & conservative
            & shared_mass_flux
            & energy_finite
            & energy_balanced
            & admissible
        )
        return UnstructuredLowMachLESStepEvidence(
            divergence_before_norm,
            divergence_after_norm,
            pressure_residual_norm,
            pressure_rhs_norm,
            pressure.gauge_defect,
            compatibility,
            mass_balance,
            momentum_balance,
            scalar_balance,
            enthalpy_balance,
            ksgs_balance,
            resolved_change,
            sgs_change,
            rate.evidence.modeled_sgs_dissipation,
            advective_kinetic_energy_rate,
            pressure_work_rate,
            molecular_viscous_work_rate,
            sgs_stress_work_rate,
            sgs_deviatoric_work_rate,
            ksgs_transport_rate,
            ksgs_source_rate,
            enthalpy_transport_rate,
            production_limit_thermalization_rate,
            modeled_transfer_residual,
            normalized_modeled_transfer_residual,
            normalized_positive_sgs_work,
            sgs_work_dissipative,
            modeled_transfer_balanced,
            temporal_energy_defect,
            energy_balance_residual,
            normalized_energy_balance,
            energy_balanced,
            shared_mass_flux,
            conservative,
            energy_finite,
            admissible,
            step_stable,
            pressure_converged,
            successful,
            self.dynamics.prepared_id,
        )


def _real_inexact(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.result_type(array, float))
    return array


def _step_status(evidence: UnstructuredLowMachLESStepEvidence, /) -> Array:
    return jnp.where(
        ~evidence.step_stable,
        jnp.asarray(UNSTRUCTURED_LES_STEP_RESTRICTION, dtype=jnp.int32),
        jnp.where(
            ~evidence.pressure_converged,
            jnp.asarray(UNSTRUCTURED_LES_PRESSURE_FAILURE, dtype=jnp.int32),
            jnp.where(
                ~evidence.conservative,
                jnp.asarray(UNSTRUCTURED_LES_CONSERVATION_FAILURE, dtype=jnp.int32),
                jnp.where(
                    ~evidence.energy_balanced,
                    jnp.asarray(UNSTRUCTURED_LES_ENERGY_FAILURE, dtype=jnp.int32),
                    jnp.where(
                        ~evidence.admissible,
                        jnp.asarray(
                            UNSTRUCTURED_LES_INADMISSIBLE_STATE,
                            dtype=jnp.int32,
                        ),
                        jnp.asarray(UNSTRUCTURED_LES_SUCCESS, dtype=jnp.int32),
                    ),
                ),
            ),
        ),
    )


def _frequency_step(frequency: Array, limit: float, dtype, /) -> Array:
    return jnp.where(
        frequency > 0.0,
        jnp.asarray(limit, dtype=dtype) / frequency,
        jnp.asarray(jnp.inf, dtype=dtype),
    )


def _positive_update_bound(value: Array, rate: Array, /) -> Array:
    return jnp.min(
        jnp.where(
            rate < 0.0,
            value / -rate,
            jnp.asarray(jnp.inf, dtype=value.dtype),
        )
    )


def _transition_balance(
    next_value: Array,
    previous_value: Array,
    face_flux: Array,
    volumes: Array,
    boundary: Array,
    step: Array,
    /,
) -> Array:
    trailing = next_value.shape[1:]
    volume_shape = (volumes.shape[0],) + (1,) * len(trailing)
    boundary_shape = (boundary.shape[0],) + (1,) * len(trailing)
    state_change = jnp.sum(
        volumes.reshape(volume_shape) * (next_value - previous_value), axis=0
    )
    boundary_flux = jnp.sum(
        jnp.where(boundary.reshape(boundary_shape), face_flux, 0.0), axis=0
    )
    return state_change + step * boundary_flux


def _negative_divergence(face_flux: Array, discretization, /) -> Array:
    owner = discretization.owner_cells
    neighbour = discretization.neighbour_cells
    interior = neighbour >= 0
    safe_neighbour = jnp.maximum(neighbour, 0)
    trailing = face_flux.shape[1:]
    net = jnp.zeros(
        (discretization.cell_count,) + trailing,
        dtype=face_flux.dtype,
    )
    net = net.at[owner].add(face_flux)
    mask = interior.reshape((interior.shape[0],) + (1,) * len(trailing))
    net = net.at[safe_neighbour].add(jnp.where(mask, -face_flux, 0.0))
    volumes = discretization.cell_volumes.astype(face_flux.dtype)
    return -net / volumes.reshape((discretization.cell_count,) + (1,) * len(trailing))


def _kinetic_energy_rate(
    velocity: Array,
    momentum_rate: Array,
    density_rate: Array,
    volumes: Array,
    /,
) -> Array:
    specific_kinetic_energy = 0.5 * jnp.sum(velocity**2, axis=-1)
    cell_rate = (
        jnp.sum(velocity * momentum_rate, axis=-1)
        - specific_kinetic_energy * density_rate
    )
    return jnp.sum(volumes * cell_rate)


def _volume_norm(value: Array, volumes: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(volumes * value**2))


__all__ = [
    "UNSTRUCTURED_LES_CONSERVATION_FAILURE",
    "UNSTRUCTURED_LES_ENERGY_FAILURE",
    "UNSTRUCTURED_LES_INADMISSIBLE_STATE",
    "UNSTRUCTURED_LES_PRESSURE_FAILURE",
    "UNSTRUCTURED_LES_STEP_RESTRICTION",
    "UNSTRUCTURED_LES_SUCCESS",
    "UnstructuredLowMachLESFixedStepMethod",
    "UnstructuredLowMachLESRestartState",
    "UnstructuredLowMachLESStepEvidence",
    "UnstructuredLowMachLESStepInputs",
    "UnstructuredLowMachLESStepRestriction",
    "UnstructuredLowMachLESStepResult",
]
