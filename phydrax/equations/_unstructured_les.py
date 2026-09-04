#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._unstructured_incompressible import (
    PreparedUnstructuredCollocatedOperators,
)
from ._favre_les import FavreLESInputs, FavreLESResult, PreparedFavreLESModel
from ._ksgs import KSGSInputs, KSGSResult, KSGSState, StaticKSGSPlan


_CLOSED_BOUNDARY_POLICY = "impermeable-zero-viscous-traction-adiabatic-zero-species-flux"
_RUNTIME_SCOPE = "single-device-fixed-conforming-3d-tetrahedral-low-mach"


class UnstructuredLowMachLESState(StrictModule):
    """Conservative cell state for density, momentum, species, and optional KSGS."""

    density: Array
    momentum_density: Array
    scalar_densities: Array
    ksgs: KSGSState | None

    def __init__(
        self,
        density: ArrayLike,
        momentum_density: ArrayLike,
        scalar_densities: ArrayLike,
        /,
        *,
        ksgs: KSGSState | None = None,
    ):
        if ksgs is not None and not isinstance(ksgs, KSGSState):
            raise TypeError("ksgs must be KSGSState or None.")
        density_ = _inexact(density)
        momentum_ = jnp.asarray(momentum_density)
        scalars_ = jnp.asarray(scalar_densities)
        if jnp.issubdtype(momentum_.dtype, jnp.complexfloating) or jnp.issubdtype(
            scalars_.dtype, jnp.complexfloating
        ):
            raise TypeError("Low-Mach conservative state fields must be real.")
        self.density = density_
        self.momentum_density = momentum_.astype(density_.dtype)
        self.scalar_densities = scalars_.astype(density_.dtype)
        self.ksgs = ksgs


class UnstructuredLowMachLESFluxLedger(StrictModule):
    """Integrated owner-oriented fluxes with independent physical/numerical IDs."""

    face_normal_velocity: Array
    volume_flux: Array
    unstabilized_mass_flux: Array
    pressure_stabilization_mass_flux: Array
    mass_flux: Array
    advective_momentum_flux: Array
    pressure_momentum_flux: Array
    molecular_momentum_flux: Array
    sgs_deviatoric_momentum_flux: Array
    sgs_isotropic_momentum_flux: Array
    sgs_momentum_flux: Array
    advective_scalar_flux: Array
    molecular_scalar_flux: Array
    sgs_scalar_flux: Array
    advective_enthalpy_flux: Array
    molecular_enthalpy_flux: Array
    sgs_enthalpy_flux: Array
    advective_ksgs_flux: Array | None
    diffusive_ksgs_flux: Array | None
    mass_flux_id: str = eqx.field(static=True)
    numerical_flux_id: str = eqx.field(static=True)
    limiter_id: str = eqx.field(static=True)
    pressure_stabilization_id: str = eqx.field(static=True)
    nonorthogonal_correction_id: str = eqx.field(static=True)
    sgs_transport_id: str = eqx.field(static=True)


class UnstructuredLowMachLESConservationEvidence(StrictModule):
    """Global conservation, common-flux, and modeled-energy proof for one rate."""

    mass_balance_residual: Array
    momentum_balance_residual: Array
    scalar_balance_residual: Array
    enthalpy_balance_residual: Array
    ksgs_transport_balance_residual: Array | None
    scalar_mass_closure_residual: Array
    shared_momentum_mass_flux_residual: Array
    shared_scalar_mass_flux_residual: Array
    shared_enthalpy_mass_flux_residual: Array
    shared_ksgs_mass_flux_residual: Array | None
    modeled_sgs_dissipation: Array
    production_limit_thermalization: Array
    modeled_energy_split_residual: Array
    sgs_dissipative: Array
    finite: Array
    conservative: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)
    resource_evidence_id: str = eqx.field(static=True)


class UnstructuredLowMachLESRateResult(StrictModule):
    """One conservative low-Mach Favre LES semidiscrete evaluation."""

    density_rate: Array
    momentum_density_rate: Array
    scalar_density_rate: Array
    enthalpy_density_rate: Array
    ksgs_density_rate: Array | None
    ksgs_specific_rate: Array | None
    kinematic_eddy_viscosity: Array
    dynamic_eddy_viscosity: Array
    ksgs_raw_production_density: Array | None
    ksgs_production_density: Array | None
    ksgs_production_limit_reduction_density: Array | None
    modeled_enthalpy_source_density: Array
    fluxes: UnstructuredLowMachLESFluxLedger
    favre: FavreLESResult
    ksgs: KSGSResult | None
    evidence: UnstructuredLowMachLESConservationEvidence
    prepared_id: str = eqx.field(static=True)


class UnstructuredLowMachLESPlan(StrictModule, NonTrainableState):
    """Bind Favre transport to the supported closed tetrahedral low-Mach route.

    The route deliberately excludes two-dimensional and polyhedral meshes,
    periodic/open boundaries, moving or coupled meshes, non-static KSGS models,
    and nonzero molecular bulk viscosity. Preparation exposes both the pure
    semidiscrete audit action and an exact-step, collocated pressure-corrected
    transaction; thermodynamic primitives remain explicit transition inputs.
    """

    favre_model: PreparedFavreLESModel
    ksgs_plan: StaticKSGSPlan | None
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        favre_model: PreparedFavreLESModel,
        /,
        *,
        ksgs_plan: StaticKSGSPlan | None = None,
        conservation_tolerance: float = 1.0e-10,
    ):
        if not isinstance(favre_model, PreparedFavreLESModel):
            raise TypeError("favre_model must be PreparedFavreLESModel.")
        if ksgs_plan is not None and not isinstance(ksgs_plan, StaticKSGSPlan):
            raise TypeError("The unstructured route supports only StaticKSGSPlan.")
        tolerance = float(conservation_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("conservation_tolerance must be finite and nonnegative.")
        resolved_filter = favre_model.provenance.resolved_filter
        filter_semantics = (
            resolved_filter.family,
            resolved_filter.topology,
            resolved_filter.boundary_class,
            resolved_filter.scale_rule,
            resolved_filter.commutation_status,
            resolved_filter.repeated_filter_semantics,
        )
        supported_semantics = (
            "implicit-grid-volume",
            "unstructured",
            "wall-bounded",
            "volume-equivalent",
            "unmodeled",
            "unmodeled",
        )
        if filter_semantics != supported_semantics:
            raise ValueError(
                "Unstructured low-Mach LES requires the wall-bounded, unstructured, "
                "implicit-grid-volume filter with volume-equivalent scale and "
                "explicitly unmodeled commutation and repeated filtering."
            )
        expected_trace_policy = (
            "provided-sgs-kinetic-energy" if ksgs_plan is not None else "neglected"
        )
        if favre_model.isotropic_trace_policy != expected_trace_policy:
            raise ValueError(
                "Favre isotropic-trace policy must exactly match optional KSGS transport."
            )
        if ksgs_plan is not None and (
            ksgs_plan.provenance.provenance_id != favre_model.provenance.provenance_id
        ):
            raise ValueError("Favre and KSGS transport require identical LES provenance.")
        self.favre_model = favre_model
        self.ksgs_plan = ksgs_plan
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-les-plan",
                "favre_model": favre_model.closure_id,
                "algebraic_model": favre_model.algebraic_model.prepared_id,
                "filter": resolved_filter.filter_id,
                "ksgs": None if ksgs_plan is None else ksgs_plan.plan_id,
                "eddy_viscosity_owner": (
                    "favre-algebraic" if ksgs_plan is None else "static-ksgs"
                ),
                "ksgs_production_discretization": (
                    None
                    if ksgs_plan is None
                    else "conservative-face-work-equal-cell-split"
                ),
                "ksgs_production_limit_disposition": (
                    None if ksgs_plan is None else "modeled-enthalpy-density-source"
                ),
                "boundary_policy": _CLOSED_BOUNDARY_POLICY,
                "numerical_flux": "piecewise-constant-upwind",
                "limiter": "none-piecewise-constant",
                "nonorthogonal_correction": "over-relaxed-deferred-tangential",
                "molecular_momentum_transport": "newtonian-stokes-zero-bulk",
                "molecular_scalar_transport": "fourier-fick-mass-corrected",
                "evolution": "collocated-pressure-corrected-fixed-step",
                "conservation_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        operators: PreparedUnstructuredCollocatedOperators,
        /,
    ) -> PreparedUnstructuredLowMachLES:
        return PreparedUnstructuredLowMachLES(self, operators)


class PreparedUnstructuredLowMachLES(StrictModule, NonTrainableState):
    """Prepared conservative Favre LES flux action on one exact tetrahedral mesh."""

    plan: UnstructuredLowMachLESPlan
    operators: PreparedUnstructuredCollocatedOperators
    filter_scale: Array
    species_schmidt_numbers: Array
    mesh_id: str = eqx.field(static=True)
    flux_geometry_id: str = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    mass_flux_id: str = eqx.field(static=True)
    numerical_flux_id: str = eqx.field(static=True)
    limiter_id: str = eqx.field(static=True)
    pressure_stabilization_id: str = eqx.field(static=True)
    nonorthogonal_correction_id: str = eqx.field(static=True)
    sgs_transport_id: str = eqx.field(static=True)
    boundary_policy: str = eqx.field(static=True)
    runtime_scope: str = eqx.field(static=True)
    resource_evidence_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: UnstructuredLowMachLESPlan,
        operators: PreparedUnstructuredCollocatedOperators,
        /,
    ):
        if not isinstance(plan, UnstructuredLowMachLESPlan):
            raise TypeError("plan must be UnstructuredLowMachLESPlan.")
        if not isinstance(operators, PreparedUnstructuredCollocatedOperators):
            raise TypeError("operators must be PreparedUnstructuredCollocatedOperators.")
        discretization = operators.discretization
        if discretization.cell_dimension != 3:
            raise ValueError("Unstructured low-Mach LES supports only three dimensions.")
        if discretization.tetrahedra.shape != (discretization.cell_count, 4):
            raise ValueError(
                "Unstructured low-Mach LES supports only conforming tetrahedral cells."
            )
        if plan.favre_model.provenance.discretization_id != discretization.prepared_id:
            raise ValueError("Favre provenance names a different discretization.")
        geometry_scale = discretization.directional_control_volume_widths()
        supplied_scale = plan.favre_model.filter_scale.directional_widths
        if supplied_scale.shape != geometry_scale.shape:
            raise ValueError(
                "Favre filter widths must match the prepared control-volume shape."
            )
        geometry_scale = eqx.error_if(
            geometry_scale,
            jnp.any(supplied_scale != geometry_scale),
            "Favre filter widths must exactly equal the prepared control-volume widths.",
        )
        resource_evidence_id = discretization.preparation.report_id
        mesh_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-les-mesh",
                "topology": discretization.topology_id,
                "geometry": discretization.geometry_id,
                "discretization": discretization.prepared_id,
                "boundary_patches": discretization.boundary_patch_names,
                "boundary_policy": _CLOSED_BOUNDARY_POLICY,
            }
        )
        flux_geometry_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-flux-geometry",
                "mesh": mesh_id,
                "operators": operators.prepared_id,
            }
        )
        numerical_flux_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-numerical-flux",
                "scheme": "piecewise-constant-upwind",
                "flux_geometry": flux_geometry_id,
            }
        )
        limiter_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-limiter",
                "policy": "none-piecewise-constant",
                "numerical_flux": numerical_flux_id,
            }
        )
        pressure_stabilization_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-pressure-stabilization",
                "scheme": "rhie-chow-two-point-minus-reconstructed",
                "operators": operators.prepared_id,
            }
        )
        nonorthogonal_correction_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-nonorthogonal-correction",
                "scheme": "over-relaxed-deferred-tangential",
                "operators": operators.prepared_id,
            }
        )
        sgs_transport_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-sgs-transport",
                "favre": plan.favre_model.closure_id,
                "algebraic_model": plan.favre_model.algebraic_model.prepared_id,
                "ksgs": None if plan.ksgs_plan is None else plan.ksgs_plan.plan_id,
                "eddy_viscosity_owner": (
                    "favre-algebraic" if plan.ksgs_plan is None else "static-ksgs"
                ),
                "ksgs_production_discretization": (
                    None
                    if plan.ksgs_plan is None
                    else "conservative-face-work-equal-cell-split"
                ),
                "ksgs_production_limit_disposition": (
                    None if plan.ksgs_plan is None else "modeled-enthalpy-density-source"
                ),
                "nonorthogonal_correction": nonorthogonal_correction_id,
            }
        )
        mass_flux_id = canonical_fingerprint(
            {
                "kind": "unstructured-low-mach-authoritative-mass-flux",
                "numerical_flux": numerical_flux_id,
                "limiter": limiter_id,
                "pressure_stabilization": pressure_stabilization_id,
                "boundary_policy": _CLOSED_BOUNDARY_POLICY,
            }
        )
        self.plan = plan
        self.operators = operators
        self.filter_scale = geometry_scale
        self.species_schmidt_numbers = jnp.asarray(
            tuple(
                value for _, value in plan.favre_model.species_turbulent_schmidt_numbers
            ),
            dtype=geometry_scale.dtype,
        )
        self.mesh_id = mesh_id
        self.flux_geometry_id = flux_geometry_id
        self.filter_id = plan.favre_model.provenance.resolved_filter.filter_id
        self.model_id = plan.favre_model.algebraic_model.prepared_id
        self.mass_flux_id = mass_flux_id
        self.numerical_flux_id = numerical_flux_id
        self.limiter_id = limiter_id
        self.pressure_stabilization_id = pressure_stabilization_id
        self.nonorthogonal_correction_id = nonorthogonal_correction_id
        self.sgs_transport_id = sgs_transport_id
        self.boundary_policy = _CLOSED_BOUNDARY_POLICY
        self.runtime_scope = _RUNTIME_SCOPE
        self.resource_evidence_id = resource_evidence_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-low-mach-les",
                "plan": plan.plan_id,
                "mesh": mesh_id,
                "flux_geometry": flux_geometry_id,
                "filter": self.filter_id,
                "model": self.model_id,
                "mass_flux": mass_flux_id,
                "numerical_flux": numerical_flux_id,
                "limiter": limiter_id,
                "pressure_stabilization": pressure_stabilization_id,
                "nonorthogonal_correction": nonorthogonal_correction_id,
                "sgs_transport": sgs_transport_id,
                "resource_evidence": resource_evidence_id,
                "runtime_scope": _RUNTIME_SCOPE,
            }
        )

    def prepare_fixed_step(
        self,
        step_size: ArrayLike,
        /,
        *,
        maximum_courant_number: float = 0.5,
        maximum_diffusion_number: float = 0.25,
        maximum_source_fraction: float = 0.25,
        pressure_tolerance: float = 1.0e-9,
        pressure_iterations: int = 200,
        linear_policy=None,
    ):
        """Bind the pressure-corrected transactional transition to one exact step."""

        from ..solver._unstructured_les import UnstructuredLowMachLESFixedStepMethod

        return UnstructuredLowMachLESFixedStepMethod(
            self,
            step_size,
            maximum_courant_number=maximum_courant_number,
            maximum_diffusion_number=maximum_diffusion_number,
            maximum_source_fraction=maximum_source_fraction,
            pressure_tolerance=pressure_tolerance,
            pressure_iterations=pressure_iterations,
            linear_policy=linear_policy,
        )

    def semidiscrete_rate(
        self,
        state: UnstructuredLowMachLESState,
        pressure: ArrayLike,
        temperature: ArrayLike,
        specific_heat_capacity_pressure: ArrayLike,
        partial_specific_enthalpies: ArrayLike,
        molecular_dynamic_viscosity: ArrayLike,
        molecular_thermal_conductivity: ArrayLike,
        molecular_scalar_diffusivities: ArrayLike,
        inverse_momentum_diagonal: ArrayLike,
        /,
        *,
        authoritative_face_normal_velocity: ArrayLike | None = None,
    ) -> UnstructuredLowMachLESRateResult:
        """Evaluate transport using either Rhie--Chow or a projected face flux."""

        if not isinstance(state, UnstructuredLowMachLESState):
            raise TypeError("state must be UnstructuredLowMachLESState.")
        discretization = self.operators.discretization
        cell_count = discretization.cell_count
        species_count = len(self.plan.favre_model.fields.species_names)
        density = jnp.asarray(state.density)
        momentum = jnp.asarray(state.momentum_density, dtype=density.dtype)
        scalar_densities = jnp.asarray(state.scalar_densities, dtype=density.dtype)
        if density.shape != (cell_count,):
            raise ValueError(f"density must have shape {(cell_count,)}.")
        if momentum.shape != (cell_count, 3):
            raise ValueError(f"momentum_density must have shape {(cell_count, 3)}.")
        if scalar_densities.shape != (cell_count, species_count):
            raise ValueError(
                "scalar_densities must match cell count and the Favre species contract."
            )
        scalar_sum_residual = jnp.abs(jnp.sum(scalar_densities, axis=-1) - density)
        scalar_tolerance = (
            64.0
            * jnp.finfo(density.dtype).eps
            * max(species_count, 1)
            * jnp.maximum(jnp.abs(density), 1.0)
        )
        admissible_state = (
            jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(momentum))
            & jnp.all(jnp.isfinite(scalar_densities))
            & jnp.all(density > 0.0)
            & jnp.all(scalar_densities >= 0.0)
            & jnp.all(scalar_sum_residual <= scalar_tolerance)
        )
        density = eqx.error_if(
            density,
            ~admissible_state,
            "Unstructured low-Mach LES requires finite positive density and finite, "
            "nonnegative scalar densities whose sum equals density; no flooring or "
            "composition repair is enabled.",
        )
        if (state.ksgs is None) != (self.plan.ksgs_plan is None):
            raise ValueError("State KSGS data must exactly match the prepared KSGS plan.")

        pressure_ = _cell_field(pressure, density, (), "pressure")
        temperature_ = _cell_field(temperature, density, (), "temperature")
        heat_capacity = _cell_field(
            specific_heat_capacity_pressure,
            density,
            (),
            "specific_heat_capacity_pressure",
        )
        enthalpies = _cell_field(
            partial_specific_enthalpies,
            density,
            (species_count,),
            "partial_specific_enthalpies",
        )
        dynamic_viscosity = _cell_field(
            molecular_dynamic_viscosity,
            density,
            (),
            "molecular_dynamic_viscosity",
        )
        thermal_conductivity = _cell_field(
            molecular_thermal_conductivity,
            density,
            (),
            "molecular_thermal_conductivity",
        )
        scalar_diffusivities = _cell_field(
            molecular_scalar_diffusivities,
            density,
            (species_count,),
            "molecular_scalar_diffusivities",
        )
        inverse_momentum = _cell_field(
            inverse_momentum_diagonal,
            density,
            (),
            "inverse_momentum_diagonal",
        )
        transport_admissible = (
            jnp.all(jnp.isfinite(pressure_))
            & jnp.all(jnp.isfinite(temperature_))
            & jnp.all(thermal_conductivity >= 0.0)
            & jnp.all(jnp.isfinite(heat_capacity))
            & jnp.all(jnp.isfinite(enthalpies))
            & jnp.all(jnp.isfinite(dynamic_viscosity))
            & jnp.all(jnp.isfinite(thermal_conductivity))
            & jnp.all(jnp.isfinite(scalar_diffusivities))
            & jnp.all(jnp.isfinite(inverse_momentum))
            & jnp.all(temperature_ > 0.0)
            & jnp.all(heat_capacity > 0.0)
            & jnp.all(dynamic_viscosity >= 0.0)
            & jnp.all(scalar_diffusivities >= 0.0)
            & jnp.all(inverse_momentum > 0.0)
        )
        dynamic_viscosity = eqx.error_if(
            dynamic_viscosity,
            ~transport_admissible,
            "Low-Mach thermodynamic and transport primitives must be finite; "
            "temperature, heat capacity, and inverse momentum must be positive, "
            "and molecular transport coefficients must be nonnegative.",
        )

        velocity = momentum / density[:, None]
        mass_fractions = scalar_densities / density[:, None]
        velocity_gradient = self.operators.cell_field_gradient(velocity, "Favre velocity")
        temperature_gradient = self.operators.cell_field_gradient(
            temperature_, "Temperature"
        )
        fraction_gradient = self.operators.cell_field_gradient(
            mass_fractions, "Mass fractions"
        )
        kinetic_energy = None if state.ksgs is None else state.ksgs.kinetic_energy
        kinetic_energy_gradient = (
            None
            if kinetic_energy is None
            else self.operators.cell_field_gradient(
                kinetic_energy,
                "KSGS kinetic energy",
            )
        )
        favre = self.plan.favre_model.evaluate(
            FavreLESInputs(
                density,
                temperature_,
                velocity,
                velocity_gradient,
                temperature_gradient,
                mass_fractions,
                fraction_gradient,
                heat_capacity,
                enthalpies,
                self.plan.favre_model.fields,
                specific_sgs_kinetic_energy=kinetic_energy,
                specific_sgs_kinetic_energy_gradient=kinetic_energy_gradient,
            )
        )
        molecular_kinematic_viscosity = dynamic_viscosity / density
        ksgs_transport = None
        selected_kinematic_eddy_viscosity = favre.kinematic_eddy_viscosity
        if self.plan.ksgs_plan is not None:
            if state.ksgs is None:
                raise ValueError("Prepared KSGS transport requires KSGS state.")
            ksgs_transport = self.plan.ksgs_plan.transport(
                state.ksgs,
                self.plan.favre_model.filter_scale,
                molecular_kinematic_viscosity,
            )
            selected_kinematic_eddy_viscosity = ksgs_transport.eddy_viscosity
        selected_dynamic_eddy_viscosity = density * selected_kinematic_eddy_viscosity

        interior = self.operators.interior_faces
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        area = discretization.face_measures.astype(density.dtype)
        normal = self.operators.unit_normals.astype(density.dtype)
        unstabilized_normal_velocity = jnp.where(
            interior,
            self.operators.interpolate_normal_velocity(velocity),
            0.0,
        )
        if authoritative_face_normal_velocity is None:
            face_normal_velocity = jnp.where(
                interior,
                self.operators.rhie_chow_face_velocity(
                    velocity,
                    pressure_,
                    inverse_momentum,
                ),
                0.0,
            )
        else:
            projected_face_velocity = self.operators.validate_face_scalar(
                authoritative_face_normal_velocity,
                "Authoritative face-normal velocity",
            ).astype(density.dtype)
            projected_face_velocity = eqx.error_if(
                projected_face_velocity,
                jnp.any(~jnp.isfinite(projected_face_velocity))
                | jnp.any(jnp.where(interior, False, projected_face_velocity != 0.0)),
                "Authoritative face-normal velocity must be finite and exactly zero "
                "on the closed boundary.",
            )
            face_normal_velocity = projected_face_velocity
        upwind = jnp.where(interior & (face_normal_velocity < 0.0), safe_neighbour, owner)
        unstabilized_upwind = jnp.where(
            interior & (unstabilized_normal_velocity < 0.0), safe_neighbour, owner
        )
        volume_flux = jnp.where(interior, face_normal_velocity * area, 0.0)
        mass_flux = jnp.where(
            interior,
            density[upwind] * face_normal_velocity * area,
            0.0,
        )
        unstabilized_mass_flux = jnp.where(
            interior,
            density[unstabilized_upwind] * unstabilized_normal_velocity * area,
            0.0,
        )
        pressure_stabilization_mass_flux = mass_flux - unstabilized_mass_flux

        advective_momentum_flux = mass_flux[:, None] * velocity[upwind]
        advective_scalar_flux = mass_flux[:, None] * mass_fractions[upwind]
        pressure_face = _face_average(pressure_, owner, safe_neighbour, interior)
        pressure_momentum_flux = pressure_face[:, None] * normal * area[:, None]

        face_velocity_gradient = self.operators.nonorthogonal_face_gradient(
            velocity, "Favre velocity"
        )
        symmetric_gradient = 0.5 * (
            face_velocity_gradient + jnp.swapaxes(face_velocity_gradient, -1, -2)
        )
        divergence = jnp.trace(symmetric_gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(3, dtype=density.dtype)
        deviatoric_strain = (
            symmetric_gradient - divergence[:, None, None] * identity / 3.0
        )
        molecular_viscosity_face = _face_average(
            dynamic_viscosity, owner, safe_neighbour, interior
        )
        sgs_viscosity_face = _face_average(
            selected_dynamic_eddy_viscosity, owner, safe_neighbour, interior
        )
        molecular_stress = (
            2.0 * molecular_viscosity_face[:, None, None] * deviatoric_strain
        )
        sgs_deviatoric_stress = (
            2.0 * sgs_viscosity_face[:, None, None] * deviatoric_strain
        )
        molecular_momentum_flux = (
            -ein.contract("fij,fj->fi", molecular_stress, normal, backend="jax")
            * area[:, None]
        )
        sgs_deviatoric_momentum_flux = (
            -ein.contract("fij,fj->fi", sgs_deviatoric_stress, normal, backend="jax")
            * area[:, None]
        )
        sgs_isotropic_momentum_flux = jnp.zeros_like(sgs_deviatoric_momentum_flux)
        if kinetic_energy is not None:
            isotropic_coefficient = _face_average(
                (2.0 / 3.0) * density * kinetic_energy,
                owner,
                safe_neighbour,
                interior,
            )
            sgs_isotropic_momentum_flux = (
                isotropic_coefficient[:, None] * normal * area[:, None]
            )
        boundary_vector = (~interior)[:, None]
        molecular_momentum_flux = jnp.where(boundary_vector, 0.0, molecular_momentum_flux)
        sgs_deviatoric_momentum_flux = jnp.where(
            boundary_vector, 0.0, sgs_deviatoric_momentum_flux
        )
        sgs_isotropic_momentum_flux = jnp.where(
            boundary_vector, 0.0, sgs_isotropic_momentum_flux
        )
        sgs_momentum_flux = sgs_deviatoric_momentum_flux + sgs_isotropic_momentum_flux
        ksgs_raw_production_density = None
        if state.ksgs is not None:
            velocity_jump = velocity[safe_neighbour] - velocity[owner]
            face_transfer = -jnp.sum(
                velocity_jump * sgs_deviatoric_momentum_flux,
                axis=-1,
            )
            face_transfer = jnp.where(interior, face_transfer, 0.0)
            cell_transfer = jnp.zeros_like(density)
            cell_transfer = cell_transfer.at[owner].add(0.5 * face_transfer)
            cell_transfer = cell_transfer.at[safe_neighbour].add(
                jnp.where(interior, 0.5 * face_transfer, 0.0)
            )
            ksgs_raw_production_density = (
                cell_transfer / discretization.cell_volumes.astype(density.dtype)
            )

        face_fraction_gradient = self.operators.nonorthogonal_face_gradient(
            mass_fractions, "Mass fractions"
        )
        normal_fraction_gradient = ein.contract(
            "fsj,fj->fs", face_fraction_gradient, normal, backend="jax"
        )
        molecular_density_diffusivity = _face_average(
            density[:, None] * scalar_diffusivities,
            owner,
            safe_neighbour,
            interior,
        )
        schmidt = self.species_schmidt_numbers.astype(density.dtype)
        sgs_density_diffusivity = _face_average(
            density[:, None]
            * selected_kinematic_eddy_viscosity[:, None]
            / schmidt[None, :],
            owner,
            safe_neighbour,
            interior,
        )
        raw_molecular_scalar_flux = (
            -molecular_density_diffusivity * normal_fraction_gradient * area[:, None]
        )
        raw_sgs_scalar_flux = (
            -sgs_density_diffusivity * normal_fraction_gradient * area[:, None]
        )
        face_mass_fractions = _face_average(
            mass_fractions, owner, safe_neighbour, interior
        )
        molecular_scalar_flux = raw_molecular_scalar_flux - face_mass_fractions * jnp.sum(
            raw_molecular_scalar_flux, axis=-1, keepdims=True
        )
        sgs_scalar_flux = raw_sgs_scalar_flux - face_mass_fractions * jnp.sum(
            raw_sgs_scalar_flux, axis=-1, keepdims=True
        )
        boundary_scalar = (~interior)[:, None]
        molecular_scalar_flux = jnp.where(boundary_scalar, 0.0, molecular_scalar_flux)
        sgs_scalar_flux = jnp.where(boundary_scalar, 0.0, sgs_scalar_flux)

        specific_enthalpy = ein.contract(
            "cs,cs->c", mass_fractions, enthalpies, backend="jax"
        )
        advective_enthalpy_flux = mass_flux * specific_enthalpy[upwind]
        face_temperature_gradient = self.operators.nonorthogonal_face_gradient(
            temperature_, "Temperature"
        )
        normal_temperature_gradient = jnp.sum(face_temperature_gradient * normal, axis=-1)
        molecular_conductivity_face = _face_average(
            thermal_conductivity, owner, safe_neighbour, interior
        )
        sgs_conductivity_face = _face_average(
            selected_dynamic_eddy_viscosity
            * heat_capacity
            / self.plan.favre_model.turbulent_prandtl_number,
            owner,
            safe_neighbour,
            interior,
        )
        partial_enthalpy_face = _face_average(enthalpies, owner, safe_neighbour, interior)
        molecular_enthalpy_flux = (
            -molecular_conductivity_face * normal_temperature_gradient * area
            + jnp.sum(partial_enthalpy_face * molecular_scalar_flux, axis=-1)
        )
        sgs_enthalpy_flux = (
            -sgs_conductivity_face * normal_temperature_gradient * area
            + jnp.sum(partial_enthalpy_face * sgs_scalar_flux, axis=-1)
        )
        molecular_enthalpy_flux = jnp.where(interior, molecular_enthalpy_flux, 0.0)
        sgs_enthalpy_flux = jnp.where(interior, sgs_enthalpy_flux, 0.0)

        density_rate = _negative_divergence(mass_flux, discretization)
        total_momentum_flux = (
            advective_momentum_flux
            + pressure_momentum_flux
            + molecular_momentum_flux
            + sgs_momentum_flux
        )
        momentum_rate = _negative_divergence(total_momentum_flux, discretization)
        total_scalar_flux = (
            advective_scalar_flux + molecular_scalar_flux + sgs_scalar_flux
        )
        scalar_rate = _negative_divergence(total_scalar_flux, discretization)
        total_enthalpy_flux = (
            advective_enthalpy_flux + molecular_enthalpy_flux + sgs_enthalpy_flux
        )
        enthalpy_rate = _negative_divergence(total_enthalpy_flux, discretization)
        modeled_enthalpy_source_density = jnp.zeros_like(density)

        advective_ksgs_flux = None
        diffusive_ksgs_flux = None
        ksgs_density_rate = None
        ksgs_specific_rate = None
        ksgs_result = None
        ksgs_transport_balance = None
        ksgs_production_density = None
        ksgs_production_limit_reduction_density = None
        ksgs_transport_scale = None
        shared_ksgs_residual = None
        if self.plan.ksgs_plan is not None:
            if state.ksgs is None or ksgs_transport is None:
                raise ValueError("Prepared KSGS transport requires KSGS state.")
            ksgs_gradient = self.operators.nonorthogonal_face_gradient(
                kinetic_energy, "KSGS kinetic energy"
            )
            normal_ksgs_gradient = jnp.sum(ksgs_gradient * normal, axis=-1)
            ksgs_density_diffusivity = _face_average(
                density * ksgs_transport.diffusivity,
                owner,
                safe_neighbour,
                interior,
            )
            advective_ksgs_flux = mass_flux * kinetic_energy[upwind]
            diffusive_ksgs_flux = -ksgs_density_diffusivity * normal_ksgs_gradient * area
            diffusive_ksgs_flux = jnp.where(interior, diffusive_ksgs_flux, 0.0)
            conservative_diffusion_rate = _negative_divergence(
                diffusive_ksgs_flux, discretization
            )
            specific_diffusion_rate = conservative_diffusion_rate / density
            ksgs_result = self.plan.ksgs_plan.evaluate(
                state.ksgs,
                KSGSInputs(
                    velocity_gradient,
                    self.plan.favre_model.filter_scale,
                    molecular_kinematic_viscosity,
                    specific_diffusion_rate,
                ),
            )
            production_ceiling_density = (
                density
                * self.plan.ksgs_plan.coefficients.production_limit
                * ksgs_result.contributions.dissipation
            )
            ksgs_production_density = jnp.minimum(
                ksgs_raw_production_density,
                production_ceiling_density,
            )
            ksgs_production_limit_reduction_density = (
                ksgs_raw_production_density - ksgs_production_density
            )
            raw_production = ksgs_raw_production_density / density
            production = ksgs_production_density / density
            production_limit_reduction = ksgs_production_limit_reduction_density / density
            authoritative_rhs = (
                production
                - ksgs_result.contributions.dissipation
                + ksgs_result.contributions.diffusion
                + ksgs_result.contributions.buoyancy
                - ksgs_result.contributions.low_re_dissipation
            )
            production_nonnegative = (raw_production >= 0.0) & (production >= 0.0)
            authoritative_finite = (
                ksgs_result.evidence.finite
                & jnp.isfinite(raw_production)
                & jnp.isfinite(production)
                & jnp.isfinite(production_limit_reduction)
                & jnp.isfinite(authoritative_rhs)
            )
            ksgs_result = eqx.tree_at(
                lambda result: (
                    result.contributions.raw_production,
                    result.contributions.production,
                    result.contributions.production_limit_reduction,
                    result.contributions.rhs,
                    result.evidence.production_limited,
                    result.evidence.production_nonnegative,
                    result.evidence.finite,
                ),
                ksgs_result,
                (
                    raw_production,
                    production,
                    production_limit_reduction,
                    authoritative_rhs,
                    raw_production > production,
                    production_nonnegative,
                    authoritative_finite,
                ),
            )
            local_ksgs_source = density * (authoritative_rhs - specific_diffusion_rate)
            total_ksgs_flux = advective_ksgs_flux + diffusive_ksgs_flux
            ksgs_density_rate = (
                _negative_divergence(total_ksgs_flux, discretization) + local_ksgs_source
            )
            ksgs_specific_rate = (
                ksgs_density_rate - kinetic_energy * density_rate
            ) / density
            ksgs_transport_balance = _global_balance(
                ksgs_density_rate - local_ksgs_source,
                total_ksgs_flux,
                discretization,
            )
            ksgs_transport_scale = _global_balance_scale(
                ksgs_density_rate - local_ksgs_source,
                total_ksgs_flux,
                discretization,
            )
            shared_ksgs_residual = jnp.max(
                jnp.abs(advective_ksgs_flux - mass_flux * kinetic_energy[upwind]),
                initial=jnp.asarray(0.0, dtype=density.dtype),
            )
        if ksgs_production_limit_reduction_density is not None:
            modeled_enthalpy_source_density = ksgs_production_limit_reduction_density
            enthalpy_rate = enthalpy_rate + modeled_enthalpy_source_density

        fluxes = UnstructuredLowMachLESFluxLedger(
            face_normal_velocity=face_normal_velocity,
            volume_flux=volume_flux,
            unstabilized_mass_flux=unstabilized_mass_flux,
            pressure_stabilization_mass_flux=pressure_stabilization_mass_flux,
            mass_flux=mass_flux,
            advective_momentum_flux=advective_momentum_flux,
            pressure_momentum_flux=pressure_momentum_flux,
            molecular_momentum_flux=molecular_momentum_flux,
            sgs_deviatoric_momentum_flux=sgs_deviatoric_momentum_flux,
            sgs_isotropic_momentum_flux=sgs_isotropic_momentum_flux,
            sgs_momentum_flux=sgs_momentum_flux,
            advective_scalar_flux=advective_scalar_flux,
            molecular_scalar_flux=molecular_scalar_flux,
            sgs_scalar_flux=sgs_scalar_flux,
            advective_enthalpy_flux=advective_enthalpy_flux,
            molecular_enthalpy_flux=molecular_enthalpy_flux,
            sgs_enthalpy_flux=sgs_enthalpy_flux,
            advective_ksgs_flux=advective_ksgs_flux,
            diffusive_ksgs_flux=diffusive_ksgs_flux,
            mass_flux_id=self.mass_flux_id,
            numerical_flux_id=self.numerical_flux_id,
            limiter_id=self.limiter_id,
            pressure_stabilization_id=self.pressure_stabilization_id,
            nonorthogonal_correction_id=self.nonorthogonal_correction_id,
            sgs_transport_id=self.sgs_transport_id,
        )
        mass_balance = _global_balance(density_rate, mass_flux, discretization)
        momentum_balance = _global_balance(
            momentum_rate, total_momentum_flux, discretization
        )
        scalar_balance = _global_balance(scalar_rate, total_scalar_flux, discretization)
        enthalpy_balance = _global_balance(
            enthalpy_rate - modeled_enthalpy_source_density,
            total_enthalpy_flux,
            discretization,
        )
        scalar_mass_closure = jnp.max(
            jnp.abs(jnp.sum(scalar_rate, axis=-1) - density_rate),
            initial=jnp.asarray(0.0, dtype=density.dtype),
        )
        shared_momentum_residual = jnp.max(
            jnp.abs(advective_momentum_flux - mass_flux[:, None] * velocity[upwind]),
            initial=jnp.asarray(0.0, dtype=density.dtype),
        )
        shared_scalar_residual = jnp.max(
            jnp.abs(advective_scalar_flux - mass_flux[:, None] * mass_fractions[upwind]),
            initial=jnp.asarray(0.0, dtype=density.dtype),
        )
        shared_enthalpy_residual = jnp.max(
            jnp.abs(advective_enthalpy_flux - mass_flux * specific_enthalpy[upwind]),
            initial=jnp.asarray(0.0, dtype=density.dtype),
        )
        cell_symmetric_gradient = 0.5 * (
            velocity_gradient + jnp.swapaxes(velocity_gradient, -1, -2)
        )
        cell_divergence = jnp.trace(cell_symmetric_gradient, axis1=-2, axis2=-1)
        cell_deviatoric_strain = (
            cell_symmetric_gradient - cell_divergence[:, None, None] * identity / 3.0
        )
        cell_strain_squared = ein.contract(
            "cij,cij->c",
            cell_deviatoric_strain,
            cell_deviatoric_strain,
            backend="jax",
        )
        modeled_sgs_dissipation = jnp.sum(
            discretization.cell_volumes.astype(density.dtype)
            * density
            * 2.0
            * selected_kinematic_eddy_viscosity
            * cell_strain_squared
        )
        production_limit_thermalization = jnp.sum(
            discretization.cell_volumes.astype(density.dtype)
            * modeled_enthalpy_source_density
        )
        modeled_energy_split_residual = jnp.asarray(0.0, dtype=density.dtype)
        if ksgs_raw_production_density is not None:
            modeled_energy_split_residual = jnp.sum(
                discretization.cell_volumes.astype(density.dtype)
                * (
                    ksgs_raw_production_density
                    - ksgs_production_density
                    - modeled_enthalpy_source_density
                )
            )
        tolerance = jnp.asarray(self.plan.conservation_tolerance, dtype=density.dtype)
        sgs_dissipative = modeled_sgs_dissipation >= -tolerance * jnp.maximum(
            jnp.abs(modeled_sgs_dissipation), 1.0
        )
        balance_fields = (
            (mass_balance, density_rate, mass_flux),
            (momentum_balance, momentum_rate, total_momentum_flux),
            (scalar_balance, scalar_rate, total_scalar_flux),
            (
                enthalpy_balance,
                enthalpy_rate - modeled_enthalpy_source_density,
                total_enthalpy_flux,
            ),
        )
        conservative = jnp.asarray(True)
        for balance, rate, flux in balance_fields:
            scale = _global_balance_scale(rate, flux, discretization)
            conservative = conservative & jnp.all(jnp.abs(balance) <= tolerance * scale)
        rate_scale = jnp.maximum(jnp.max(jnp.abs(density_rate)), 1.0)
        conservative = conservative & (scalar_mass_closure <= tolerance * rate_scale)
        conservative = (
            conservative
            & (shared_momentum_residual <= tolerance)
            & (shared_scalar_residual <= tolerance)
            & (shared_enthalpy_residual <= tolerance)
        )
        if ksgs_transport_balance is not None:
            conservative = conservative & jnp.all(
                jnp.abs(ksgs_transport_balance) <= tolerance * ksgs_transport_scale
            )
        if shared_ksgs_residual is not None:
            conservative = conservative & (shared_ksgs_residual <= tolerance)
        conservative = conservative & (
            jnp.abs(modeled_energy_split_residual)
            <= tolerance * jnp.maximum(jnp.abs(production_limit_thermalization), 1.0)
        )
        finite = jnp.asarray(True)
        finite_values = (
            density_rate,
            momentum_rate,
            scalar_rate,
            enthalpy_rate,
            face_normal_velocity,
            mass_flux,
            total_momentum_flux,
            total_scalar_flux,
            total_enthalpy_flux,
            modeled_sgs_dissipation,
            modeled_enthalpy_source_density,
            production_limit_thermalization,
            modeled_energy_split_residual,
        )
        for value in finite_values:
            finite = finite & jnp.all(jnp.isfinite(value))
        if ksgs_density_rate is not None:
            finite = finite & jnp.all(jnp.isfinite(ksgs_density_rate))
        successful = (
            finite
            & conservative
            & sgs_dissipative
            & jnp.all(favre.evidence.successful)
            & jnp.all(favre.input_evidence.successful)
        )
        if ksgs_result is not None:
            successful = successful & jnp.all(
                ksgs_result.evidence.finite & ksgs_result.evidence.production_nonnegative
            )
        evidence = UnstructuredLowMachLESConservationEvidence(
            mass_balance_residual=mass_balance,
            momentum_balance_residual=momentum_balance,
            scalar_balance_residual=scalar_balance,
            enthalpy_balance_residual=enthalpy_balance,
            ksgs_transport_balance_residual=ksgs_transport_balance,
            scalar_mass_closure_residual=scalar_mass_closure,
            shared_momentum_mass_flux_residual=shared_momentum_residual,
            shared_scalar_mass_flux_residual=shared_scalar_residual,
            shared_enthalpy_mass_flux_residual=shared_enthalpy_residual,
            shared_ksgs_mass_flux_residual=shared_ksgs_residual,
            modeled_sgs_dissipation=modeled_sgs_dissipation,
            production_limit_thermalization=production_limit_thermalization,
            modeled_energy_split_residual=modeled_energy_split_residual,
            sgs_dissipative=sgs_dissipative,
            finite=finite,
            conservative=conservative,
            successful=successful,
            prepared_id=self.prepared_id,
            resource_evidence_id=self.resource_evidence_id,
        )
        return UnstructuredLowMachLESRateResult(
            density_rate=density_rate,
            momentum_density_rate=momentum_rate,
            scalar_density_rate=scalar_rate,
            enthalpy_density_rate=enthalpy_rate,
            ksgs_density_rate=ksgs_density_rate,
            ksgs_specific_rate=ksgs_specific_rate,
            kinematic_eddy_viscosity=selected_kinematic_eddy_viscosity,
            dynamic_eddy_viscosity=selected_dynamic_eddy_viscosity,
            ksgs_raw_production_density=ksgs_raw_production_density,
            ksgs_production_density=ksgs_production_density,
            ksgs_production_limit_reduction_density=(
                ksgs_production_limit_reduction_density
            ),
            modeled_enthalpy_source_density=modeled_enthalpy_source_density,
            fluxes=fluxes,
            favre=favre,
            ksgs=ksgs_result,
            evidence=evidence,
            prepared_id=self.prepared_id,
        )


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError("Low-Mach density must be real.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.result_type(array, float))
    return array


def _cell_field(
    value: ArrayLike,
    density: Array,
    trailing_shape: tuple[int, ...],
    name: str,
    /,
) -> Array:
    raw = jnp.asarray(value)
    if jnp.issubdtype(raw.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    array = raw.astype(density.dtype)
    expected = (density.shape[0],) + trailing_shape
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}.")
    return array


def _face_average(
    value: Array,
    owner: Array,
    safe_neighbour: Array,
    interior: Array,
    /,
) -> Array:
    average = 0.5 * (value[owner] + value[safe_neighbour])
    mask = interior.reshape((interior.shape[0],) + (1,) * (value.ndim - 1))
    return jnp.where(mask, average, value[owner])


def _negative_divergence(flux: Array, discretization, /) -> Array:
    owner = discretization.owner_cells
    neighbour = discretization.neighbour_cells
    interior = neighbour >= 0
    safe_neighbour = jnp.maximum(neighbour, 0)
    trailing = flux.shape[1:]
    net = jnp.zeros((discretization.cell_count,) + trailing, dtype=flux.dtype)
    net = net.at[owner].add(flux)
    mask = interior.reshape((interior.shape[0],) + (1,) * len(trailing))
    net = net.at[safe_neighbour].add(jnp.where(mask, -flux, 0.0))
    volume_shape = (discretization.cell_count,) + (1,) * len(trailing)
    volumes = discretization.cell_volumes.astype(flux.dtype).reshape(volume_shape)
    return -net / volumes


def _global_balance(rate: Array, flux: Array, discretization, /) -> Array:
    trailing = rate.shape[1:]
    volume_shape = (discretization.cell_count,) + (1,) * len(trailing)
    volumes = discretization.cell_volumes.astype(rate.dtype).reshape(volume_shape)
    boundary = discretization.neighbour_cells < 0
    boundary_shape = (boundary.shape[0],) + (1,) * len(trailing)
    boundary_flux = jnp.where(boundary.reshape(boundary_shape), flux, 0.0)
    return jnp.sum(volumes * rate, axis=0) + jnp.sum(boundary_flux, axis=0)


def _global_balance_scale(rate: Array, flux: Array, discretization, /) -> Array:
    trailing = rate.shape[1:]
    volume_shape = (discretization.cell_count,) + (1,) * len(trailing)
    volumes = discretization.cell_volumes.astype(rate.dtype).reshape(volume_shape)
    boundary = discretization.neighbour_cells < 0
    boundary_shape = (boundary.shape[0],) + (1,) * len(trailing)
    boundary_flux = jnp.where(boundary.reshape(boundary_shape), flux, 0.0)
    return jnp.maximum(
        jnp.sum(jnp.abs(volumes * rate), axis=0)
        + jnp.sum(jnp.abs(boundary_flux), axis=0),
        1.0,
    )


__all__ = [
    "PreparedUnstructuredLowMachLES",
    "UnstructuredLowMachLESConservationEvidence",
    "UnstructuredLowMachLESFluxLedger",
    "UnstructuredLowMachLESPlan",
    "UnstructuredLowMachLESRateResult",
    "UnstructuredLowMachLESState",
]
