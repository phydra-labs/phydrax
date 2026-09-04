#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume._mac_scalar import (
    MACScalarLayout,
    MACScalarProblem,
    MACScalarSGSField,
    MACScalarSGSPlan,
    MACScalarTransport,
    PreparedMACScalarTransport,
)
from phydrax.equations._ksgs import (
    BuoyancyKSGSPlan,
    DynamicKSGSPlan,
    KSGSCoefficients,
    LowReKSGSCoefficients,
    LowReKSGSPlan,
    StaticKSGSPlan,
)
from phydrax.equations._les_closures import (
    LESParameterProvenance,
    PreparedAlgebraicLESModel,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._mac_les import MACAlgebraicLESPlan
from phydrax.equations._mac_scalar_buoyancy import PreparedMACKSGS


def _discretization(*, periodic_vertical=False):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=periodic_vertical),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
    return phx.discretization.FiniteVolumePlan(grid, component_names=("ocean",)).prepare()


def _filter(discretization, *, name="mac-cell-volume"):
    return ResolvedLESFilter(
        name,
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class=(
            "periodic"
            if all(axis.periodic for axis in discretization.grid.structured_axes)
            else "wall-bounded"
        ),
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )


def _provenance(discretization, *, resolved_filter=None):
    return LESParameterProvenance(
        _filter(discretization) if resolved_filter is None else resolved_filter,
        discretization.prepared_id,
        "incompressible-unit-density",
        source_kind="user",
        evidence_ids=(),
    )


def _algebraic_les(discretization, coefficient=0.18):
    return MACAlgebraicLESPlan(
        SmagorinskyLESPlan(coefficient).prepare(_provenance(discretization))
    )


def _scalar_sgs(reference, *, temperature=0.7, salinity=1.4):
    return MACScalarSGSPlan(
        (
            MACScalarSGSField(
                reference.temperature_name,
                turbulent_prandtl_number=temperature,
            ),
            MACScalarSGSField(
                reference.salinity_name,
                turbulent_schmidt_number=salinity,
            ),
        )
    )


def _free_slip_boundaries(discretization):
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    return phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("z", "lower", "free-slip"),
            phx.discretization.MACBoundarySide("z", "upper", "free-slip"),
        ),
    )


def _no_slip_boundaries(discretization):
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    return phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("z", "lower", "no-slip"),
            phx.discretization.MACBoundarySide("z", "upper", "no-slip"),
        ),
    )


def _velocity(ocean):
    discretization = ocean.operators.discretization
    return (
        jnp.sin(2.0 * jnp.pi * discretization.face_centers[0][..., 2]),
        jnp.zeros(discretization.face_layouts[1].shape),
        jnp.zeros(discretization.face_layouts[2].shape),
    )


def _temperature_profile(ocean, *, stable):
    z = ocean.operators.discretization.cell_centers[..., 2]
    sign = 1.0 if stable else -1.0
    return ocean.plan.reference.reference_temperature + sign * z


def _salinity(ocean):
    return jnp.full(
        ocean.operators.discretization.cell_shape,
        ocean.plan.reference.reference_salinity,
    )


def _ocean_with_algebraic_les(*, coefficient=0.18, temperature_flux=None):
    discretization = _discretization()
    reference = phx.applications.ocean.LinearSeawaterReference()
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        reference,
        viscosity=1.0e-4,
        temperature_diffusivity=2.0e-5,
        salinity_diffusivity=1.0e-5,
        algebraic_les=_algebraic_les(discretization, coefficient),
        temperature_surface_flux=temperature_flux,
        scalar_sgs=_scalar_sgs(reference),
    ).prepare(discretization, boundaries=_free_slip_boundaries(discretization))
    return discretization, ocean


def _ksgs_coefficients():
    return KSGSCoefficients(0.12, 1.0, 1.3, 0.8, 10.0)


def _ocean_with_ksgs(plan_type=StaticKSGSPlan):
    discretization = _discretization()
    reference = phx.applications.ocean.LinearSeawaterReference()
    ksgs = plan_type(_ksgs_coefficients(), _provenance(discretization))
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        reference,
        viscosity=1.0e-4,
        temperature_diffusivity=2.0e-5,
        salinity_diffusivity=1.0e-5,
        scalar_sgs=_scalar_sgs(reference),
        ksgs=ksgs,
        ksgs_field_name="sgs_kinetic_energy",
    ).prepare(discretization, boundaries=_free_slip_boundaries(discretization))
    return discretization, ocean


def _dynamic_mac():
    discretization = _discretization(periodic_vertical=True)
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    resolved_filter = ResolvedLESFilter(
        "mac-dynamic-cell-volume",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="volume-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="unmodeled",
    )
    test_filter = ResolvedLESFilter(
        "mac-dynamic-binomial-test",
        family="explicit-filter",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="kernel-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="composed",
    )
    plan = DynamicKSGSPlan(
        _ksgs_coefficients(),
        _provenance(discretization, resolved_filter=resolved_filter),
        test_filter,
        2.0,
    )
    return discretization, PreparedMACKSGS(plan, momentum, "sgs_kinetic_energy")


def _dynamic_velocity(prepared):
    centers = prepared.momentum.operators.discretization.face_centers
    return (
        jnp.sin(2.0 * jnp.pi * centers[0][..., 1]),
        jnp.sin(2.0 * jnp.pi * centers[1][..., 2]),
        jnp.sin(2.0 * jnp.pi * centers[2][..., 0]),
    )


def test_ocean_les_requires_explicit_complete_named_scalar_declarations():
    discretization = _discretization()
    reference = phx.applications.ocean.LinearSeawaterReference()
    with pytest.raises(ValueError, match="explicit named scalar SGS"):
        phx.applications.ocean.CartesianBoussinesqOceanPlan(
            phx.applications.ocean.OceanAxisConvention(),
            reference,
            algebraic_les=_algebraic_les(discretization),
        )
    incomplete = MACScalarSGSPlan(
        (
            MACScalarSGSField(
                reference.temperature_name,
                turbulent_prandtl_number=0.7,
            ),
        )
    )
    with pytest.raises(ValueError, match="exactly match"):
        phx.applications.ocean.CartesianBoussinesqOceanPlan(
            phx.applications.ocean.OceanAxisConvention(),
            reference,
            algebraic_les=_algebraic_les(discretization),
            scalar_sgs=incomplete,
        )


def test_named_runtime_scalar_sgs_flux_and_ledgers_are_separated():
    discretization = _discretization(periodic_vertical=True)
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    layout = MACScalarLayout(operators, ("salinity", "temperature"))
    problem = MACScalarProblem(
        (
            MACScalarTransport("temperature", 0.2, advection="centered"),
            MACScalarTransport("salinity", 0.1, advection="centered"),
        )
    )
    transport = PreparedMACScalarTransport(
        problem,
        layout,
        phx.discretization.MACScalarBoundarySet(layout),
    )
    declaration = MACScalarSGSPlan(
        (
            MACScalarSGSField("temperature", turbulent_prandtl_number=0.5),
            MACScalarSGSField("salinity", no_sgs=True),
        )
    ).prepare(transport)
    eddy_viscosity = jnp.full(discretization.cell_shape, 0.15)
    sgs = declaration.diffusivities(eddy_viscosity)
    x = discretization.cell_centers[..., 0]
    fields = {
        "temperature": jnp.sin(2.0 * jnp.pi * x),
        "salinity": jnp.cos(2.0 * jnp.pi * x),
    }
    velocity = tuple(jnp.zeros(layout_.shape) for layout_ in discretization.face_layouts)

    result = transport.evaluate(0.0, fields, velocity, sgs_diffusivities=sgs)
    diagnostics = transport.diagnostics_from_fluxes(fields, result)

    np.testing.assert_allclose(sgs["temperature"], 0.3)
    np.testing.assert_allclose(sgs["salinity"], 0.0)
    np.testing.assert_allclose(
        result["temperature"].total_diffusivity,
        result["temperature"].molecular_diffusivity
        + result["temperature"].sgs_diffusivity,
    )
    for total, molecular, subgrid, boundary in zip(
        result["temperature"].diffusive_fluxes,
        result["temperature"].molecular_diffusive_fluxes,
        result["temperature"].sgs_diffusive_fluxes,
        result["temperature"].boundary_diffusive_fluxes,
        strict=True,
    ):
        np.testing.assert_allclose(total, molecular + subgrid + boundary)
    np.testing.assert_allclose(
        diagnostics.fields["temperature"].sgs_diffusive_content_rate,
        0.0,
        atol=2.0e-6,
    )
    assert diagnostics.fields["temperature"].sgs_diffusive_variance_rate < 0.0
    np.testing.assert_allclose(result["salinity"].sgs_diffusive_divergence, 0.0)
    assert bool(diagnostics.success)


def test_ocean_les_uses_one_momentum_stage_for_scalar_ratios_and_energy_ledgers(
    monkeypatch,
):
    _, ocean = _ocean_with_algebraic_les()
    evaluations = 0
    original_evaluate = PreparedAlgebraicLESModel.evaluate

    def counted_evaluate(model, inputs):
        nonlocal evaluations
        evaluations += 1
        return original_evaluate(model, inputs)

    assert ocean.prepared_algebraic_les is ocean.dynamics.base_dynamics.algebraic_les
    assert ocean.prepared_scalar_sgs is ocean.dynamics.scalar_sgs

    monkeypatch.setattr(PreparedAlgebraicLESModel, "evaluate", counted_evaluate)
    state = ocean.initial_state(
        _velocity(ocean),
        _temperature_profile(ocean, stable=True),
        _salinity(ocean),
    )

    stage = ocean.dynamics.stage(0.0, state)
    diagnostics = ocean.dynamics.diagnostics_from_stage(stage)
    eddy_viscosity = stage.momentum_components.les_stage.model_result.kinematic_viscosity

    assert stage.momentum_components.les_stage is not None
    np.testing.assert_allclose(
        stage.scalar_sgs_diffusivities[ocean.plan.reference.temperature_name],
        eddy_viscosity / 0.7,
    )
    np.testing.assert_allclose(
        stage.scalar_sgs_diffusivities[ocean.plan.reference.salinity_name],
        eddy_viscosity / 1.4,
    )
    assert diagnostics.sgs_dissipation >= 0.0
    assert diagnostics.buoyancy.potential_energy_mixing_available
    assert diagnostics.buoyancy.molecular_potential_energy_mixing > 0.0
    assert diagnostics.buoyancy.sgs_potential_energy_mixing >= 0.0
    assert bool(stage.success)
    assert bool(diagnostics.success)
    assert evaluations == 1

    restriction = ocean.dynamics.step_restriction(0.0, state)
    assert jnp.isfinite(restriction.scalars.diffusive["temperature"])
    assert not restriction.momentum.sgs_supported
    assert not bool(restriction.success)


def test_ocean_les_prescribed_surface_flux_is_total_and_not_duplicated():
    imposed = phx.discretization.MACScalarBoundaryCondition("flux", 3.0e-5)
    _, ocean = _ocean_with_algebraic_les(temperature_flux=imposed)
    state = ocean.initial_state(
        _velocity(ocean),
        _temperature_profile(ocean, stable=True),
        _salinity(ocean),
    )
    stage = ocean.dynamics.stage(0.0, state)
    result = stage.scalar_fluxes[ocean.plan.reference.temperature_name]

    np.testing.assert_allclose(
        jnp.take(result.diffusive_fluxes[2], -1, axis=2),
        -3.0e-5,
    )
    np.testing.assert_allclose(
        jnp.take(result.boundary_diffusive_fluxes[2], -1, axis=2),
        -3.0e-5,
    )
    np.testing.assert_allclose(
        jnp.take(result.molecular_diffusive_fluxes[2], -1, axis=2),
        0.0,
    )
    np.testing.assert_allclose(
        jnp.take(result.sgs_diffusive_fluxes[2], -1, axis=2),
        0.0,
    )


def test_stable_and_unstable_ocean_mixing_have_opposite_potential_energy_signs():
    _, ocean = _ocean_with_algebraic_les()
    stable = ocean.initial_state(
        _velocity(ocean),
        _temperature_profile(ocean, stable=True),
        _salinity(ocean),
    )
    unstable = ocean.initial_state(
        _velocity(ocean),
        _temperature_profile(ocean, stable=False),
        _salinity(ocean),
    )

    stable_stage = ocean.dynamics.stage(0.0, stable)
    unstable_stage = ocean.dynamics.stage(0.0, unstable)

    assert stable_stage.buoyancy.molecular_potential_energy_mixing > 0.0
    assert unstable_stage.buoyancy.molecular_potential_energy_mixing < 0.0
    assert ocean.dynamics._stratification_step(stable_stage.scalars) < jnp.inf
    assert jnp.isinf(ocean.dynamics._stratification_step(unstable_stage.scalars))


def test_zero_coefficient_les_has_no_les_parity():
    discretization, les_ocean = _ocean_with_algebraic_les(coefficient=0.0)
    reference = les_ocean.plan.reference
    baseline = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        reference,
        viscosity=1.0e-4,
        temperature_diffusivity=2.0e-5,
        salinity_diffusivity=1.0e-5,
    ).prepare(discretization, boundaries=_free_slip_boundaries(discretization))
    velocity = _velocity(les_ocean)
    temperature = _temperature_profile(les_ocean, stable=True)
    salinity = _salinity(les_ocean)
    les_state = les_ocean.initial_state(velocity, temperature, salinity)
    baseline_state = baseline.initial_state(velocity, temperature, salinity)

    les_stage = les_ocean.dynamics.stage(0.0, les_state)
    baseline_stage = baseline.dynamics.stage(0.0, baseline_state)

    for left, right in zip(
        les_stage.velocity_rate, baseline_stage.velocity_rate, strict=True
    ):
        np.testing.assert_allclose(left, right)
    for name in reference.field_names:
        np.testing.assert_allclose(
            les_stage.scalar_rates[name], baseline_stage.scalar_rates[name]
        )


def test_static_and_buoyant_ksgs_are_prognostic_positive_and_restart_complete(tmp_path):
    for plan_type in (StaticKSGSPlan, BuoyancyKSGSPlan):
        _, ocean = _ocean_with_ksgs(plan_type)
        initial_k = jnp.full(ocean.operators.discretization.cell_shape, 2.0e-4)
        state = ocean.initial_state(
            _velocity(ocean),
            _temperature_profile(ocean, stable=True),
            _salinity(ocean),
            sgs_kinetic_energy=initial_k,
        )
        stage = ocean.dynamics.stage(0.0, state)
        restriction = ocean.dynamics.step_restriction(0.0, state)

        assert stage.ksgs is not None
        np.testing.assert_allclose(stage.ksgs.state.kinetic_energy, initial_k)
        assert jnp.all(stage.ksgs.result.eddy_viscosity >= 0.0)
        np.testing.assert_allclose(
            stage.scalar_sgs_diffusivities["sgs_kinetic_energy"],
            ocean.plan.ksgs.coefficients.diffusion * stage.ksgs.result.eddy_viscosity,
        )
        assert jnp.all(jnp.isfinite(stage.ksgs.result.contributions.rhs))
        assert ocean.prepared_ksgs is ocean.dynamics.ksgs
        assert restriction.ksgs > 0.0
        assert jnp.isfinite(restriction.ksgs)
        assert ocean.state_view(state).sgs_kinetic_energy is not None
        assert bool(stage.success)
        if plan_type is BuoyancyKSGSPlan:
            assert jnp.all(stage.ksgs.result.contributions.buoyancy <= 0.0)
            unstable_state = ocean.initial_state(
                _velocity(ocean),
                _temperature_profile(ocean, stable=False),
                _salinity(ocean),
                sgs_kinetic_energy=initial_k,
            )
            unstable_stage = ocean.dynamics.stage(0.0, unstable_state)
            assert unstable_stage.ksgs is not None
            assert jnp.all(unstable_stage.ksgs.result.contributions.buoyancy >= 0.0)

        continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
            state,
            ksgs_state=stage.ksgs.result.state,
        )
        path = tmp_path / f"{plan_type.__name__}.npz"
        phx.applications.ocean.write_ocean_checkpoint(
            path, ocean, jnp.asarray(1.0), jnp.asarray(3), continuation
        )
        _, _, restored = phx.applications.ocean.read_ocean_checkpoint(
            path, ocean, continuation
        )
        assert all(
            np.array_equal(left, right)
            for left, right in zip(
                jax.tree.leaves(continuation),
                jax.tree.leaves(restored),
                strict=True,
            )
        )


def test_ocean_ksgs_step_rejects_a_negative_candidate_without_flooring():
    _, ocean = _ocean_with_ksgs(StaticKSGSPlan)
    velocity = tuple(
        jnp.zeros(layout.shape) for layout in ocean.operators.discretization.face_layouts
    )
    kinetic = jnp.full(ocean.operators.discretization.cell_shape, 2.0e-4)
    coordinates = ocean.initial_state(
        velocity,
        jnp.full(
            ocean.operators.discretization.cell_shape,
            ocean.plan.reference.reference_temperature,
        ),
        _salinity(ocean),
        sgs_kinetic_energy=kinetic,
    )
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        coordinates
    )
    result = phx.applications.ocean.OceanBoussinesqSSPRK33Method(ocean).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(1.0e8),
        None,
    )

    assert not bool(result.successful)
    np.testing.assert_array_equal(
        result.accepted_state.coordinates,
        continuation.coordinates,
    )


def test_dynamic_ksgs_uses_exact_periodic_uniform_filter_and_updates_history():
    discretization, prepared = _dynamic_mac()
    assert prepared.test_filter is not None
    assert prepared.test_filter.plan.test_filter.filter_id == (
        prepared.plan.test_filter.filter_id
    )
    assert prepared.test_filter.test_filter_ratio == (2.0, 2.0, 2.0)
    assert prepared.test_filter.boundary_support == "periodic-wrap-only"

    kinetic = jnp.full(discretization.cell_shape, 2.0e-4)
    viscosity = jnp.full(discretization.cell_shape, 1.0e-4)
    initial, transport = prepared.prepare_transport(kinetic, viscosity)
    velocity = _dynamic_velocity(prepared)
    boundary_stage = prepared.momentum.boundaries.homogeneous_stage()
    zeros = jnp.zeros(discretization.cell_shape)
    paused = prepared.evaluate(
        velocity,
        boundary_stage,
        initial,
        transport,
        zeros,
        viscosity,
        zeros,
        accept_update=False,
    )
    updated = prepared.evaluate(
        velocity,
        boundary_stage,
        initial,
        transport,
        zeros,
        viscosity,
        zeros,
        accept_update=True,
    )
    for left, right in zip(
        jax.tree.leaves(paused.result.state),
        jax.tree.leaves(initial),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    assert jnp.any(updated.result.state.dynamic_updates == 1)
    assert jnp.all(updated.result.state.eddy_viscosity_coefficient >= 0.0)
    assert jnp.any(
        updated.result.state.eddy_viscosity_coefficient
        != initial.eddy_viscosity_coefficient
    )
    assert bool(updated.success)


def test_dynamic_ksgs_rejected_update_retains_committed_history_exactly():
    discretization, prepared = _dynamic_mac()
    kinetic = jnp.full(discretization.cell_shape, 2.0e-4)
    viscosity = jnp.full(discretization.cell_shape, 1.0e-4)
    initial, transport = prepared.prepare_transport(kinetic, viscosity)
    zeros = jnp.zeros(discretization.cell_shape)
    accepted = prepared.evaluate(
        _dynamic_velocity(prepared),
        prepared.momentum.boundaries.homogeneous_stage(),
        initial,
        transport,
        zeros,
        viscosity,
        zeros,
        accept_update=True,
    )
    committed = accepted.result.state
    continued, next_transport = prepared.prepare_transport(
        kinetic,
        viscosity,
        continuation_state=committed,
    )
    rejected = prepared.evaluate(
        tuple(
            jnp.roll(component, 1, axis=0) for component in _dynamic_velocity(prepared)
        ),
        prepared.momentum.boundaries.homogeneous_stage(),
        continued,
        next_transport,
        zeros,
        viscosity,
        zeros,
        accept_update=False,
    )
    for left, right in zip(
        jax.tree.leaves(rejected.result.state),
        jax.tree.leaves(committed),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


def test_low_re_ksgs_resolves_wall_distance_damping_and_sqrt_k_gradient():
    discretization = _discretization()
    reference = phx.applications.ocean.LinearSeawaterReference()
    plan = LowReKSGSPlan(
        _ksgs_coefficients(),
        LowReKSGSCoefficients(0.01, 1.0),
        _provenance(discretization),
    )
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        reference,
        viscosity=1.0e-4,
        scalar_sgs=_scalar_sgs(reference),
        ksgs=plan,
        ksgs_field_name="sgs_kinetic_energy",
    ).prepare(discretization, boundaries=_no_slip_boundaries(discretization))
    z = ocean.operators.discretization.cell_centers[..., 2]
    kinetic = (0.02 + 0.01 * (z + 1.0)) ** 2
    state = ocean.initial_state(
        _velocity(ocean),
        _temperature_profile(ocean, stable=True),
        _salinity(ocean),
        sgs_kinetic_energy=kinetic,
    )
    stage = ocean.dynamics.stage(0.0, state)
    assert stage.ksgs is not None
    assert ocean.prepared_ksgs is not None
    wall_distance = ocean.prepared_ksgs.wall_distance
    assert wall_distance is not None
    assert jnp.all(wall_distance > 0.0)
    undamped = (
        plan.coefficients.eddy_viscosity
        * ocean.prepared_ksgs.filter_scale.equivalent_width
        * jnp.sqrt(kinetic)
    )
    assert jnp.all(stage.ksgs.result.eddy_viscosity < undamped)
    assert jnp.any(stage.ksgs.result.contributions.low_re_dissipation > 0.0)
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        state,
        ksgs_state=stage.ksgs.result.state,
    )
    diagnostic = phx.applications.ocean.ocean_diagnostic_view(ocean, 0.0, continuation)
    np.testing.assert_allclose(
        diagnostic.ksgs_wall_distance,
        wall_distance,
    )
    np.testing.assert_allclose(
        diagnostic.ksgs_low_re_dissipation,
        stage.ksgs.result.contributions.low_re_dissipation,
    )
    assert bool(stage.success)


def test_low_re_ksgs_refuses_free_slip_only_wall_distance():
    discretization = _discretization()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    momentum = phx.discretization.MACMomentumPlan(
        operators,
        boundaries=_free_slip_boundaries(discretization).prepare(),
    ).prepare()
    plan = LowReKSGSPlan(
        _ksgs_coefficients(),
        LowReKSGSCoefficients(1.0, 1.0),
        _provenance(discretization),
    )
    with pytest.raises(ValueError, match="resolved no-slip wall"):
        PreparedMACKSGS(plan, momentum, "sgs_kinetic_energy")
