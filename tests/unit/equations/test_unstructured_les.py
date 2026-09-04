#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.equations._favre_les import (
    FavreLESFieldContract,
    PreparedFavreLESModel,
)
from phydrax.equations._ksgs import KSGSCoefficients, StaticKSGSPlan
from phydrax.equations._les_closures import (
    LESFilterScale,
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._unstructured_les import (
    UnstructuredLowMachLESPlan,
    UnstructuredLowMachLESState,
)


def _tetrahedral_grid(*, scale=1.0, skew=True):
    logical = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )
    if skew:
        x, y, z = logical.T
        logical = np.stack(
            (
                x + 0.13 * y + 0.04 * z,
                y + 0.09 * z + 0.03 * x,
                z + 0.07 * x,
            ),
            axis=-1,
        )
    tetrahedra = np.asarray(
        (
            (0, 1, 2, 3),
            (1, 2, 3, 4),
            (0, 2, 3, 5),
            (0, 1, 3, 6),
            (0, 1, 2, 7),
        ),
        dtype=np.int32,
    )
    return scale * logical, tetrahedra


def _operators(*, scale=1.0):
    vertices, tetrahedra = _tetrahedral_grid(scale=scale)
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        tetrahedra=tetrahedra,
    ).prepare()
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    return phx.discretization.PreparedUnstructuredCollocatedOperators(
        discretization, gradient
    )


def _filter():
    return ResolvedLESFilter(
        "tetrahedral-control-volume",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="unstructured",
        boundary_class="wall-bounded",
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )


def _prepared(*, coefficient=0.08, ksgs=False, scale=1.0):
    operators = _operators(scale=scale)
    discretization = operators.discretization
    resolved_filter = _filter()
    provenance = LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "variable-density-low-mach-tetrahedral-fv",
        source_kind="user",
        evidence_ids=(),
    )
    algebraic = SmagorinskyLESPlan(coefficient).prepare(provenance)
    fields = FavreLESFieldContract("binary-mixture", ("a", "b"))
    ksgs_plan = None
    if ksgs:
        ksgs_plan = StaticKSGSPlan(KSGSCoefficients(0.12, 1.0, 0.8, 1.0, 8.0), provenance)
    favre = PreparedFavreLESModel(
        algebraic,
        LESFilterScale(discretization.directional_control_volume_widths()),
        fields,
        0.9,
        (("a", 0.7), ("b", 0.8)),
        10.0,
        isotropic_trace_policy=("provided-sgs-kinetic-energy" if ksgs else "neglected"),
    )
    prepared = UnstructuredLowMachLESPlan(
        favre,
        ksgs_plan=ksgs_plan,
        conservation_tolerance=2.0e-6,
    ).prepare(operators)
    return prepared


def _case(prepared, *, constant_density=False):
    centers = prepared.operators.discretization.cell_centers
    density = (
        jnp.full((centers.shape[0],), 1.7)
        if constant_density
        else 1.2 + 0.15 * centers[:, 0] + 0.04 * centers[:, 2]
    )
    velocity = jnp.stack(
        (
            0.25 + 0.17 * centers[:, 0] - 0.08 * centers[:, 1],
            -0.11 + 0.06 * centers[:, 1] + 0.04 * centers[:, 2],
            0.09 - 0.05 * centers[:, 0] + 0.07 * centers[:, 2],
        ),
        axis=-1,
    )
    fraction_a = 0.42 + 0.06 * centers[:, 0] - 0.03 * centers[:, 2]
    fractions = jnp.stack((fraction_a, 1.0 - fraction_a), axis=-1)
    ksgs_state = None
    if prepared.plan.ksgs_plan is not None:
        ksgs_state = prepared.plan.ksgs_plan.initialize_state(
            0.025 + 0.006 * centers[:, 1]
        )
    state = UnstructuredLowMachLESState(
        density,
        density[:, None] * velocity,
        density[:, None] * fractions,
        ksgs=ksgs_state,
    )
    pressure = 2.0 + 0.2 * centers[:, 0] - 0.13 * centers[:, 1]
    temperature = 295.0 + 3.0 * centers[:, 1] + centers[:, 2]
    heat_capacity = 1000.0 + 2.0 * centers[:, 0]
    enthalpies = jnp.stack((1005.0 * temperature, 1120.0 * temperature), axis=-1)
    dynamic_viscosity = 0.012 + 0.002 * centers[:, 2]
    thermal_conductivity = 0.03 + 0.004 * centers[:, 0]
    scalar_diffusivities = jnp.stack(
        (
            0.008 + 0.001 * centers[:, 0],
            0.006 + 0.001 * centers[:, 1],
        ),
        axis=-1,
    )
    inverse_momentum = 0.08 + 0.01 * centers[:, 2]
    arguments = (
        pressure,
        temperature,
        heat_capacity,
        enthalpies,
        dynamic_viscosity,
        thermal_conductivity,
        scalar_diffusivities,
        inverse_momentum,
    )
    return state, arguments


def test_control_volume_scale_is_directional_volume_equivalent_and_scales_with_mesh():
    reference = _operators(scale=1.0).discretization
    dilated = _operators(scale=2.0).discretization
    widths = reference.directional_control_volume_widths()
    dilated_widths = dilated.directional_control_volume_widths()

    assert widths.shape == (reference.cell_count, 3)
    np.testing.assert_allclose(jnp.prod(widths, axis=-1), reference.cell_volumes)
    np.testing.assert_allclose(
        LESFilterScale(widths).equivalent_width,
        jnp.cbrt(reference.cell_volumes),
    )
    np.testing.assert_allclose(dilated_widths, 2.0 * widths, rtol=2e-12, atol=2e-12)
    assert jnp.max(jnp.ptp(widths, axis=-1)) > 0.0


def test_constant_density_shared_flux_zero_coefficient_and_separate_ledgers():
    prepared = _prepared(coefficient=0.0)
    state, arguments = _case(prepared, constant_density=True)
    result = prepared.semidiscrete_rate(state, *arguments)
    fluxes = result.fluxes
    interior = prepared.operators.interior_faces
    density = state.density[0]

    np.testing.assert_allclose(
        fluxes.mass_flux[interior],
        density * fluxes.volume_flux[interior],
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.density_rate,
        -density * prepared.operators.divergence(fluxes.face_normal_velocity),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(fluxes.sgs_momentum_flux, 0.0, atol=2e-13)
    np.testing.assert_allclose(fluxes.sgs_scalar_flux, 0.0, atol=2e-13)
    np.testing.assert_allclose(fluxes.sgs_enthalpy_flux, 0.0, atol=2e-13)
    assert fluxes.pressure_stabilization_id != fluxes.sgs_transport_id
    assert fluxes.numerical_flux_id != fluxes.limiter_id
    assert (
        len(
            {
                fluxes.mass_flux_id,
                fluxes.numerical_flux_id,
                fluxes.limiter_id,
                fluxes.pressure_stabilization_id,
                fluxes.nonorthogonal_correction_id,
                fluxes.sgs_transport_id,
            }
        )
        == 6
    )
    assert (
        prepared.filter_id
        == prepared.plan.favre_model.provenance.resolved_filter.filter_id
    )
    assert prepared.model_id == prepared.plan.favre_model.algebraic_model.prepared_id
    assert jnp.max(jnp.abs(fluxes.pressure_momentum_flux)) > 0.0
    assert jnp.max(jnp.abs(fluxes.pressure_stabilization_mass_flux)) > 0.0
    np.testing.assert_allclose(
        fluxes.mass_flux,
        fluxes.unstabilized_mass_flux + fluxes.pressure_stabilization_mass_flux,
    )
    np.testing.assert_allclose(result.evidence.shared_momentum_mass_flux_residual, 0.0)
    np.testing.assert_allclose(result.evidence.shared_scalar_mass_flux_residual, 0.0)
    np.testing.assert_allclose(result.evidence.shared_enthalpy_mass_flux_residual, 0.0)
    assert result.evidence.successful


def test_skew_nonorthogonal_manufactured_low_mach_path_is_conservative_and_dissipative():
    prepared = _prepared(coefficient=0.08)
    state, arguments = _case(prepared)
    result = prepared.semidiscrete_rate(state, *arguments)
    discretization = prepared.operators.discretization
    centers = discretization.cell_centers
    affine = jnp.stack(
        (
            0.2 + 0.3 * centers[:, 0] - 0.1 * centers[:, 2],
            -0.4 + 0.2 * centers[:, 1],
        ),
        axis=-1,
    )
    face_gradient = prepared.operators.nonorthogonal_face_gradient(
        affine, "manufactured affine field"
    )
    exact_gradient = jnp.asarray(((0.3, 0.0, -0.1), (0.0, 0.2, 0.0)))

    assert prepared.operators.report.maximum_nonorthogonality_degrees > 0.0
    np.testing.assert_allclose(
        face_gradient[prepared.operators.interior_faces],
        jnp.broadcast_to(
            exact_gradient,
            face_gradient[prepared.operators.interior_faces].shape,
        ),
        rtol=3e-10,
        atol=3e-10,
    )
    np.testing.assert_allclose(result.evidence.mass_balance_residual, 0.0, atol=2e-7)
    np.testing.assert_allclose(result.evidence.momentum_balance_residual, 0.0, atol=2e-7)
    np.testing.assert_allclose(result.evidence.scalar_balance_residual, 0.0, atol=2e-7)
    np.testing.assert_allclose(result.evidence.enthalpy_balance_residual, 0.0, atol=2e-6)
    np.testing.assert_allclose(
        result.evidence.scalar_mass_closure_residual, 0.0, atol=2e-6
    )
    assert jnp.max(jnp.abs(result.fluxes.sgs_momentum_flux)) > 0.0
    assert jnp.max(jnp.abs(result.fluxes.sgs_scalar_flux)) > 0.0
    assert jnp.max(jnp.abs(result.fluxes.sgs_enthalpy_flux)) > 0.0
    assert result.evidence.modeled_sgs_dissipation >= 0.0
    assert result.evidence.sgs_dissipative
    assert result.evidence.conservative
    assert result.evidence.successful
    assert result.evidence.resource_evidence_id == discretization.preparation.report_id


def test_static_ksgs_reuses_mass_flux_and_keeps_transport_conservative():
    prepared = _prepared(ksgs=True)
    state, arguments = _case(prepared)
    centers = prepared.operators.discretization.cell_centers
    velocity = centers @ jnp.diag(jnp.asarray((1.0, 1.0, -2.0))) + jnp.asarray(
        (0.1, -0.03, 0.02)
    )
    state = eqx.tree_at(
        lambda value: value.momentum_density,
        state,
        state.density[:, None] * velocity,
    )
    result = prepared.semidiscrete_rate(state, *arguments)
    fluxes = result.fluxes
    interior = prepared.operators.interior_faces
    neighbour = prepared.operators.discretization.neighbour_cells
    owner = prepared.operators.discretization.owner_cells
    upwind = jnp.where(
        interior & (fluxes.face_normal_velocity < 0.0),
        jnp.maximum(neighbour, 0),
        owner,
    )

    assert result.ksgs is not None
    assert result.ksgs_density_rate is not None
    assert fluxes.advective_ksgs_flux is not None
    np.testing.assert_allclose(
        fluxes.advective_ksgs_flux,
        fluxes.mass_flux * state.ksgs.kinetic_energy[upwind],
    )
    np.testing.assert_allclose(result.evidence.shared_ksgs_mass_flux_residual, 0.0)
    np.testing.assert_allclose(
        result.evidence.ksgs_transport_balance_residual, 0.0, atol=2e-7
    )
    assert result.evidence.conservative
    assert result.evidence.successful


def test_refuses_unsupported_geometry_filter_scale_and_nonpositive_transport():
    prepared = _prepared()
    state, arguments = _case(prepared)
    bad_density = UnstructuredLowMachLESState(
        state.density.at[0].set(0.0),
        state.momentum_density,
        state.scalar_densities,
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="finite positive density"
    ):
        output = prepared.semidiscrete_rate(bad_density, *arguments)
        jax.block_until_ready(output.density_rate)

    bad_scalars = UnstructuredLowMachLESState(
        state.density,
        state.momentum_density,
        state.scalar_densities.at[0, 0].set(-0.01),
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="nonnegative scalar densities"
    ):
        output = prepared.semidiscrete_rate(bad_scalars, *arguments)
        jax.block_until_ready(output.scalar_density_rate)

    invalid_transport = list(arguments)
    invalid_transport[4] = invalid_transport[4].at[0].set(-1.0)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="nonnegative"):
        output = prepared.semidiscrete_rate(state, *invalid_transport)
        jax.block_until_ready(output.momentum_density_rate)

    bad_favre = PreparedFavreLESModel(
        prepared.plan.favre_model.algebraic_model,
        LESFilterScale(1.01 * prepared.filter_scale),
        prepared.plan.favre_model.fields,
        prepared.plan.favre_model.turbulent_prandtl_number,
        prepared.plan.favre_model.species_turbulent_schmidt_numbers,
        prepared.plan.favre_model.kinematic_viscosity_upper_bound,
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="exactly equal"):
        bad_prepared = UnstructuredLowMachLESPlan(bad_favre).prepare(prepared.operators)
        jax.block_until_ready(bad_prepared.filter_scale)

    nx = ny = 3
    vertices = np.asarray(
        [(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    quadrilaterals = []
    for j in range(ny):
        for i in range(nx):
            lower = j * (nx + 1) + i
            quadrilaterals.append((lower, lower + 1, lower + nx + 2, lower + nx + 1))
    two_dimensional = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(quadrilaterals),
    ).prepare()
    two_dimensional_gradient = phx.discretization.CellPolynomialReconstructionPlan(
        1
    ).prepare(two_dimensional)
    two_dimensional_operators = (
        phx.discretization.PreparedUnstructuredCollocatedOperators(
            two_dimensional, two_dimensional_gradient
        )
    )
    with pytest.raises(ValueError, match="only three dimensions"):
        prepared.plan.prepare(two_dimensional_operators)


def test_semidiscrete_rate_is_jittable_and_has_finite_momentum_jvp():
    prepared = _prepared()
    state, arguments = _case(prepared)
    eager = prepared.semidiscrete_rate(state, *arguments)
    compiled = eqx.filter_jit(prepared.semidiscrete_rate)(state, *arguments)
    np.testing.assert_allclose(compiled.density_rate, eager.density_rate)
    np.testing.assert_allclose(
        compiled.momentum_density_rate, eager.momentum_density_rate
    )

    def momentum_rate(momentum):
        candidate = UnstructuredLowMachLESState(
            state.density,
            momentum,
            state.scalar_densities,
        )
        return prepared.semidiscrete_rate(candidate, *arguments).momentum_density_rate

    _, tangent = jax.jvp(
        momentum_rate,
        (state.momentum_density,),
        (jnp.full_like(state.momentum_density, 0.01),),
    )
    assert jnp.all(jnp.isfinite(tangent))
