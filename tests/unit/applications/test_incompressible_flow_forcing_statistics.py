import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import incompressible_flow as flow


def _periodic_space(count=6, dimension=3):
    names = ("x", "y", "z")[:dimension]
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in names),
        axis_names=names,
        field_name="velocity",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi) for _ in names)
    )


def _channel_space():
    return phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(4),
            phx.discretization.ChebyshevBasisPlan(8),
            phx.discretization.FourierBasisPlan(4),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )


def test_constant_power_forcing_is_admissible_exact_and_fail_closed():
    space = _periodic_space()
    projector = phx.discretization.PeriodicLerayProjector(space)
    basis = flow.SolenoidalHermitianFourierBasis(
        projector,
        maximum_wavenumber=1.1,
    )
    coefficients = jnp.linspace(0.5, 1.5, basis.coordinate_size)
    velocity = basis.evaluate(coefficients)
    np.testing.assert_allclose(
        jnp.sum(jnp.abs(velocity) ** 2),
        jnp.sum(coefficients**2),
        rtol=1e-12,
        atol=1e-12,
    )
    plan = flow.ConstantPowerFourierForcingPlan(
        projector,
        maximum_wavenumber=1.1,
        power_input=0.25,
        minimum_forced_energy=1.0e-8,
    )
    result = plan.evaluate(velocity)
    assert bool(result.successful)
    np.testing.assert_allclose(result.actual_power_density, 0.25, rtol=1e-12)
    np.testing.assert_allclose(projector.divergence(result.forcing), 0.0, atol=1e-12)
    assert float(result.forcing_reality_defect) < 1.0e-12
    assert not bool(jnp.any(result.forcing[~projector.admissibility_mask]))
    assert not bool(jnp.any(result.forcing[projector.wavenumber_squared == 0.0]))

    failed = plan.evaluate(1.0e-12 * velocity)
    assert not bool(failed.active)
    assert not bool(failed.successful)
    np.testing.assert_array_equal(failed.forcing, jnp.zeros_like(failed.forcing))

    _, tangent = jax.jvp(
        lambda amplitude: plan.evaluate(amplitude * velocity).forcing,
        (jnp.asarray(1.0),),
        (jnp.asarray(0.2),),
    )
    assert bool(jnp.all(jnp.isfinite(tangent)))
    forced_value, pullback = jax.vjp(lambda value: plan.evaluate(value).forcing, velocity)
    adjoint = pullback(jnp.ones_like(forced_value))[0]
    assert bool(jnp.all(jnp.isfinite(adjoint)))


def test_solenoidal_ou_subdivision_restart_and_stage_values_are_exact():
    space = _periodic_space(dimension=2)
    projector = phx.discretization.PeriodicLerayProjector(space)
    basis = flow.SolenoidalHermitianFourierBasis(
        projector,
        maximum_wavenumber=1.1,
    )
    plan = flow.SolenoidalOUForcingPlan(
        basis,
        correlation_time=0.7,
        rms_acceleration=0.2,
    )
    realization = phx.stochastic.OrnsteinUhlenbeckRealization(
        jax.random.key(17),
        (basis.coordinate_size,),
        support=(0.0, 1.0),
        tolerance=1.0e-6,
    )
    initial = plan.initialize(0.0, realization)
    whole = plan.advance(initial, 0.0, 0.4, realization)
    first = plan.advance(initial, 0.0, 0.2, realization)
    restarted = plan.advance(first.state, 0.2, 0.4, realization)
    np.testing.assert_allclose(
        whole.half_coefficients,
        first.end_coefficients,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        whole.end_coefficients,
        restarted.end_coefficients,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        whole.half_forcing,
        first.end_forcing,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(projector.divergence(whole.end_forcing), 0.0, atol=1e-12)
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space,
        component_shape=(2,),
    )
    assert float(coordinates.reality_defect(whole.end_forcing)) < 1.0e-12


def test_compiled_periodic_statistics_keep_equation_terms_separate():
    space = _periodic_space()
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.05),
        space,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
    )
    projector = dynamics.projector
    basis = flow.SolenoidalHermitianFourierBasis(
        projector,
        maximum_wavenumber=2.1,
    )
    velocity = basis.evaluate(jnp.linspace(0.2, 1.0, basis.coordinate_size))
    forcing_plan = flow.ConstantPowerFourierForcingPlan(
        projector,
        maximum_wavenumber=2.1,
        power_input=0.1,
        minimum_forced_energy=1.0e-10,
    )
    forcing = forcing_plan.evaluate(velocity)
    statistics_plan = flow.PeriodicModalTurbulenceStatisticsPlan(
        dynamics,
        jnp.linspace(0.0, 8.0, 9),
        tail_start_wavenumber=2.0,
    )
    stage = dynamics.stage(jnp.asarray(0.0), velocity)
    statistics = statistics_plan.evaluate(
        0.0,
        velocity,
        stage=stage,
        additive_forcing_rate=forcing.forcing,
    )
    native_energy = 0.5 * jnp.sum(jnp.abs(velocity) ** 2)
    native_molecular_dissipation = -jnp.real(
        jnp.sum(jnp.conj(velocity) * stage.rates.molecular_rate)
    )
    native_advective_transfer = jnp.real(
        jnp.sum(jnp.conj(velocity) * stage.rates.advective_rate)
    )

    assert bool(statistics.successful)
    assert not bool(statistics.sgs_available)
    np.testing.assert_allclose(statistics.energy_shells.integral.sum(), native_energy)
    np.testing.assert_allclose(
        statistics.molecular_dissipation_shells.integral.sum(),
        native_molecular_dissipation,
    )
    np.testing.assert_allclose(
        statistics.advective_transfer_shells.integral.sum(),
        native_advective_transfer,
        atol=np.finfo(float).eps,
    )
    np.testing.assert_array_equal(
        statistics.sgs_transfer_shells.integral,
        jnp.zeros_like(statistics.sgs_transfer_shells.integral),
    )
    np.testing.assert_allclose(
        statistics.forcing_injection_shells.integral.sum(),
        forcing.actual_total_power,
    )
    np.testing.assert_allclose(
        statistics.resolved_spectral_flux,
        -jnp.cumsum(statistics.advective_transfer_shells.integral),
    )
    np.testing.assert_allclose(statistics.mean_forcing_power, 0.1, rtol=1e-12)
    assert statistics.compilation_id == dynamics.compilation_id
    assert statistics.sgs_filter_id is None
    assert statistics.sgs_model_id is None
    assert statistics.sgs_prepared_action_id is None


def test_zero_coefficient_periodic_les_statistics_match_no_les_terms():
    space = _periodic_space(count=4)
    problem = phx.equations.IncompressibleFlowProblem(3, 0.05)
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    no_les = phx.equations.compile_periodic_incompressible_flow(
        problem,
        space,
        method,
    )
    resolved_filter = phx.equations.ResolvedLESFilter(
        "retained Fourier grid",
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        space.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    les_plan = phx.equations.PeriodicAlgebraicLESPlan(
        phx.equations.SmagorinskyLESPlan(0.0).prepare(provenance),
        phx.equations.PeriodicFourierGridFilterPlan(resolved_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(1.5)
        ),
    )
    zero_les = phx.equations.compile_periodic_incompressible_flow(
        problem,
        space,
        method,
        algebraic_les=les_plan,
    )
    basis = flow.SolenoidalHermitianFourierBasis(
        no_les.projector,
        maximum_wavenumber=1.1,
    )
    velocity = basis.evaluate(jnp.linspace(0.2, 0.8, basis.coordinate_size))
    plain_statistics = flow.PeriodicModalTurbulenceStatisticsPlan(
        no_les,
        jnp.linspace(0.0, 4.0, 5),
    ).evaluate(0.0, velocity)
    les_statistics = flow.PeriodicModalTurbulenceStatisticsPlan(
        zero_les,
        jnp.linspace(0.0, 4.0, 5),
    ).evaluate(0.0, velocity)

    np.testing.assert_allclose(
        les_statistics.molecular_dissipation_shells.integral,
        plain_statistics.molecular_dissipation_shells.integral,
    )
    np.testing.assert_allclose(
        les_statistics.advective_transfer_shells.integral,
        plain_statistics.advective_transfer_shells.integral,
    )
    np.testing.assert_allclose(les_statistics.sgs_transfer_shells.integral, 0.0)
    np.testing.assert_allclose(les_statistics.sgs_modeled_dissipation, 0.0)
    np.testing.assert_allclose(les_statistics.sgs_maximum_kinematic_viscosity, 0.0)
    assert bool(les_statistics.sgs_available)
    assert bool(les_statistics.sgs_stability_available)
    assert les_statistics.sgs_filter_id == resolved_filter.filter_id
    assert les_statistics.sgs_model_id == les_plan.prepared_model.model_id
    assert les_statistics.sgs_prepared_model_id == (les_plan.prepared_model.prepared_id)
    assert les_statistics.sgs_prepared_action_id == zero_les.algebraic_les.prepared_id
    assert les_statistics.sgs_regularization_id is not None
    assert not bool(les_statistics.sgs_regularization_available)


def test_channel_couette_and_poiseuille_keep_signed_walls_separate():
    space = _channel_space()
    plan = flow.SpectralChannelStatisticsPlan(
        space,
        density=2.0,
        kinematic_viscosity=0.25,
    )
    y = space.axes[1].nodes.reshape((1, -1, 1))
    zeros = jnp.zeros(space.physical_shape)

    couette_physical = jnp.stack(
        (jnp.broadcast_to(y, space.physical_shape), zeros, zeros),
        axis=-1,
    )
    couette = plan.evaluate(space.project(couette_physical))
    assert bool(couette.successful)
    np.testing.assert_allclose(couette.lower_wall_shear, 0.5, atol=1e-11)
    np.testing.assert_allclose(couette.upper_wall_shear, 0.5, atol=1e-11)
    np.testing.assert_allclose(couette.bulk_velocity, 0.0, atol=1e-11)

    poiseuille_u = jnp.broadcast_to(1.0 - y**2, space.physical_shape)
    poiseuille_physical = jnp.stack((poiseuille_u, zeros, zeros), axis=-1)
    poiseuille = plan.evaluate(space.project(poiseuille_physical))
    assert bool(poiseuille.successful)
    np.testing.assert_allclose(poiseuille.lower_wall_shear, 1.0, atol=1e-11)
    np.testing.assert_allclose(poiseuille.upper_wall_shear, -1.0, atol=1e-11)
    np.testing.assert_allclose(poiseuille.bulk_velocity, 2.0 / 3.0, atol=1e-11)
    np.testing.assert_allclose(
        poiseuille.lower_friction_velocity,
        poiseuille.upper_friction_velocity,
        atol=1e-12,
    )
    assert poiseuille.wall_shear_convention.endswith("in increasing y")
