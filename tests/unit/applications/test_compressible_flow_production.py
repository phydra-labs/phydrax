import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.compressible_flow._all_speed import (
    ShockAwareAllSpeedFluxPlan,
)
from phydrax.applications.compressible_flow._contracts import (
    AllSpeedCompressiblePolicy,
    CompressibleFlowCaseSpec,
    ShockResolvingPolicy,
)
from phydrax.applications.compressible_flow._production import (
    AdditiveIMEXCompressibleFixedStepAdapter,
    ExplicitCompressibleFixedStepAdapter,
    PreparedCompressibleProduction,
    SmoothCompressibleProductionPlan,
    StructuredFVCompressibleProductionPlan,
)
from phydrax.applications.compressible_flow._qualification import (
    ManufacturedViscousNSPlan,
)
from phydrax.discretization.finite_volume._riemann import HLLFluxPlan
from phydrax.equations import (
    ChemicalPhaseKind,
    ChemicalSpeciesSchema,
    HomogeneousHelmholtzPlan,
    HomogeneousMixtureEulerSystem,
    IdealGasReferenceHelmholtzTerm,
    PengRobinsonParameters,
    PengRobinsonResidualHelmholtzTerm,
    PolynomialSpeciesThermodynamicsPlan,
    UNIVERSAL_GAS_CONSTANT,
    ZeroResidualHelmholtzTerm,
)
from phydrax.equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
)
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.solver._balance_law_composition import AdditiveIMEXTableau
from phydrax.solver._conservation_temporal import (
    ConservationIMEXMethod,
    ImplicitConservationStageResult,
)
from phydrax.solver._phase_equilibrium import FixedTwoPhaseTPFlashPlan


def _schema(species_count=2):
    names = tuple(chr(ord("A") + index) for index in range(species_count))
    return ChemicalSpeciesSchema.from_unique_species(
        names,
        (ChemicalPhaseKind.GAS,) * species_count,
        jnp.linspace(0.020, 0.030, species_count),
        names,
        jnp.eye(species_count, dtype=jnp.int32),
        jnp.zeros((species_count,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )


def _ideal_model(species_count=2):
    schema = _schema(species_count)
    calorics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((species_count, 1), 2.5 * UNIVERSAL_GAS_CONSTANT),
        jnp.linspace(1.0e3, 2.0e3, species_count),
        reference_molar_entropy=jnp.linspace(100.0, 120.0, species_count),
        reference_temperature=300.0,
        minimum_temperature=120.0,
        maximum_temperature=1500.0,
    )
    ideal = IdealGasReferenceHelmholtzTerm(schema, calorics)
    return HomogeneousHelmholtzPlan(ideal, ZeroResidualHelmholtzTerm(schema))


def _peng_robinson_model():
    schema = ChemicalSpeciesSchema.from_unique_species(
        ("methane", "ethane"),
        (ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.016043, 0.03007)),
        ("C", "H"),
        jnp.asarray(((1, 2), (4, 6)), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
        provenance="public critical constants",
    )
    calorics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((2, 1), 2.5 * UNIVERSAL_GAS_CONSTANT),
        jnp.zeros((2,)),
        reference_molar_entropy=jnp.asarray((100.0, 110.0)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=1000.0,
    )
    parameters = PengRobinsonParameters(
        schema.catalog,
        jnp.asarray((190.564, 305.322)),
        jnp.asarray((4.5992e6, 4.8722e6)),
        jnp.asarray((0.01142, 0.0995)),
        jnp.zeros((2, 2)),
        provenance="public critical constants; zero binary interaction",
    )
    return HomogeneousHelmholtzPlan(
        IdealGasReferenceHelmholtzTerm(schema, calorics),
        PengRobinsonResidualHelmholtzTerm(schema, parameters),
        maximum_molar_density=5.0e4,
    )


def test_one_and_multi_species_ideal_and_pr_density_energy_round_trips():
    cases = (
        (_ideal_model(1), jnp.asarray((1.0, 0.2, 500.0))),
        (_ideal_model(2), jnp.asarray((0.35, 0.65, 0.2, 500.0))),
        (_peng_robinson_model(), jnp.asarray((0.064, 0.090, 0.2, 350.0))),
    )
    for model, primitive in cases:
        system = HomogeneousMixtureEulerSystem(model, 1)
        conserved = system.primitive_to_conserved(primitive)
        recovered = system.recover_thermodynamics(conserved)
        assert bool(recovered.successful)
        np.testing.assert_allclose(
            system.conserved_to_primitive(conserved), primitive, rtol=2.0e-5, atol=2.0e-5
        )
        direct = model.evaluate_density_temperature(
            primitive[: system.species_count], primitive[-1]
        )
        np.testing.assert_allclose(recovered.state.pressure, direct.pressure, rtol=1.0e-5)
        np.testing.assert_allclose(
            recovered.state.temperature, primitive[-1], rtol=1.0e-5
        )
        assert float(recovered.state.frozen_sound_speed_squared) > 0.0
        variables = system.entropy_variables(conserved)
        entropy_gradient = jax.grad(system.entropy)(conserved)
        np.testing.assert_allclose(variables, entropy_gradient, rtol=2.0e-4, atol=2.0e-4)


def test_entropy_requires_positive_chemical_evidence_and_flash_stays_solver_owned():
    model = _ideal_model(2)
    system = HomogeneousMixtureEulerSystem(model, 1)
    zero_species = system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 0.2, 500.0)))
    assert not bool(system.entropy_evidence(zero_species))
    flash = FixedTwoPhaseTPFlashPlan(_peng_robinson_model())
    with pytest.raises(ValueError, match="case specification"):
        CompressibleFlowCaseSpec(
            "unsupported-equilibrium-coupling",
            1,
            "euler",
            "structured-fv",
            flash,
        )


def test_canonical_ns_frozen_and_mass_closed_species_enthalpy_flux():
    model = _ideal_model(2)
    primitive = jnp.asarray((0.35, 0.65, 0.2, -0.1, 500.0))
    frozen = HomogeneousMixtureCompressibleNavierStokesSystem(
        model, ConstantTransport(0.02, 0.03), 2
    )
    state = frozen.primitive_to_conserved(primitive)
    gradient = (
        jnp.zeros((frozen.component_count, 2)).at[0, 0].set(0.03).at[1, 0].set(-0.02)
    )
    frozen_flux = frozen.viscous_flux(state, gradient)
    np.testing.assert_array_equal(frozen_flux[: frozen.species_count], 0.0)
    diffusive = HomogeneousMixtureCompressibleNavierStokesSystem(
        model,
        ConstantTransport(0.02, 0.03),
        2,
        species_diffusivities=(1.0e-5, 2.0e-5),
    )
    flux = diffusive.viscous_flux(state, gradient)
    np.testing.assert_allclose(jnp.sum(flux[:2], axis=0), 0.0, atol=1.0e-10)
    enthalpy_transport = jnp.sum(
        diffusive.partial_specific_enthalpies(state)[:, None] * flux[:2], axis=0
    )
    mechanical_and_fourier = diffusive.viscous_flux(state, jnp.zeros_like(gradient))[-1]
    np.testing.assert_allclose(
        flux[-1] - frozen.viscous_flux(state, gradient)[-1],
        enthalpy_transport,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert float(diffusive.maximum_diffusivity(state)) >= 2.0e-5
    assert bool(jnp.all(jnp.isfinite(mechanical_and_fourier)))


def test_low_mach_scaling_and_generic_hll_fallback_ledger_are_explicit():
    all_speed = AllSpeedCompressiblePolicy(reference_mach=1.0)
    mach = jnp.asarray((1.0e-3, 2.0e-3, 4.0e-3))
    np.testing.assert_allclose(all_speed.pressure_dissipation_scale(mach), mach)
    policy = ShockResolvingPolicy("teno", sensor_threshold=0.1, all_speed=all_speed)
    assert isinstance(policy.fallback_flux, HLLFluxPlan)
    assert policy.route_label.endswith("generic-hll")
    ledger = policy.ledger(
        jnp.asarray((0.01, 0.2, 0.01)), jnp.asarray((True, True, False))
    )
    np.testing.assert_array_equal(ledger.fallback_used, jnp.asarray((False, True, True)))
    assert int(ledger.fallback_count) == 2


def test_structured_fv_real_shock_aware_all_speed_and_ale_fallback():
    system = HomogeneousMixtureEulerSystem(_ideal_model(2), 1)
    low_dissipation = StructuredFVCompressibleProductionPlan(
        shock=ShockResolvingPolicy(
            all_speed=AllSpeedCompressiblePolicy(reference_mach=1.0)
        )
    )
    higher_dissipation = StructuredFVCompressibleProductionPlan(
        shock=ShockResolvingPolicy(
            all_speed=AllSpeedCompressiblePolicy(reference_mach=0.1)
        )
    )
    assert isinstance(low_dissipation.method.interface_solver, ShockAwareAllSpeedFluxPlan)
    left = system.primitive_to_conserved(jnp.asarray((0.35, 0.65, 1.0e-3, 500.0)))
    right = system.primitive_to_conserved(jnp.asarray((0.36, 0.65, 1.0e-3, 500.0)))
    low_flux = low_dissipation.method.interface_solver.face_flux(system, left, right, 0)
    high_flux = higher_dissipation.method.interface_solver.face_flux(
        system, left, right, 0
    )
    np.testing.assert_allclose(low_flux.max_speed, high_flux.max_speed)
    np.testing.assert_allclose(low_flux.max_speed, system.max_wave_speed(left, right, 0))
    assert not bool(jnp.all(low_flux.normal_flux == high_flux.normal_flux))
    shock = ShockAwareAllSpeedFluxPlan(ShockResolvingPolicy(sensor_threshold=1.0e-6))
    selected = shock.normal_ale_face_flux(
        system, left, right, jnp.asarray((1.0,)), jnp.asarray(0.1)
    )
    fallback = shock.policy.fallback_flux.normal_ale_face_flux(
        system, left, right, jnp.asarray((1.0,)), jnp.asarray(0.1)
    )
    assert bool(selected.fallback_activated)
    np.testing.assert_allclose(selected.normal_flux, fallback.normal_flux)
    assert "generic-hll" in low_dissipation.route_label


def test_smooth_route_refuses_absent_entropy_evidence_and_fv_never_claims_dns():
    with pytest.raises(TypeError, match="entropy evidence"):
        SmoothCompressibleProductionPlan(HLLFluxPlan(), HLLFluxPlan())
    model = _ideal_model(2)
    case = CompressibleFlowCaseSpec(
        "shock-fv", 2, "euler", "structured-fv", model, fidelity="dns-candidate"
    )
    fv = StructuredFVCompressibleProductionPlan(
        "structured", shock=ShockResolvingPolicy("mp5")
    )
    evidence = fv.qualification_evidence(case)
    assert not evidence.dns_claimed
    assert not evidence.signed
    assert not evidence.released
    assert fv.positivity.fallback_flux.flux_id == fv.shock.fallback_flux.flux_id


def test_explicit_and_additive_imex_adapters_restart_without_partition_loss():
    explicit = ExplicitCompressibleFixedStepAdapter(
        lambda time, state, args: -state, "linear-decay"
    )
    initial = jnp.asarray((1.0, 2.0))
    first = explicit.step(0, jnp.asarray(0.0), initial, jnp.asarray(0.1), None)
    assert bool(first.successful)
    prepared = PreparedCompressibleProduction(explicit, "test-route", "linear-decay")
    restart = prepared.checkpoint(first.accepted_state, 1, 0.1, "topology")
    restored, step, time = prepared.restore(restart, "topology")
    np.testing.assert_array_equal(restored, first.accepted_state)
    assert int(step) == 1
    assert float(time) == pytest.approx(0.1)
    uninterrupted = prepared.step(step, time, restored, 0.1)
    repeated = prepared.step(1, 0.1, first.accepted_state, 0.1)
    np.testing.assert_array_equal(uninterrupted.accepted_state, repeated.accepted_state)

    tableau = AdditiveIMEXTableau(
        jnp.asarray(((0.0,),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
    )

    def implicit_solver(provisional, time, coefficient, args):
        del time, args
        state = provisional / (1.0 + coefficient)
        return ImplicitConservationStageResult(
            state,
            jnp.asarray(True),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.zeros((), dtype=state.dtype),
        )

    imex_method = ConservationIMEXMethod(
        tableau,
        lambda time, state, args: 2.0 * state,
        lambda time, state, args: -state,
        implicit_solver,
        method_id="explicit-growth-implicit-decay",
    )
    imex = AdditiveIMEXCompressibleFixedStepAdapter(
        imex_method,
        explicit_operator_id="transport",
        implicit_operator_id="viscous",
    )
    result = imex.step(0, jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(0.1), None)
    assert bool(result.successful)
    assert int(result.iterations) == 1


def test_manufactured_canonical_mixture_navier_stokes_source_identity():
    system = HomogeneousMixtureCompressibleNavierStokesSystem(
        _ideal_model(2), ConstantTransport(0.02, 0.03), 1
    )

    def exact_state(time, point, args):
        del args
        phase = point[0] - time
        primitive = jnp.stack(
            (
                0.35 + 0.01 * jnp.sin(phase),
                0.65 + 0.02 * jnp.sin(phase),
                0.2 + 0.03 * jnp.cos(phase),
                500.0 + 2.0 * jnp.sin(phase),
            )
        )
        return system.primitive_to_conserved(primitive)

    evidence = ManufacturedViscousNSPlan(1, exact_state, "periodic-mms").evaluate(
        system, 0.2, jnp.linspace(0.0, 1.0, 5)[:, None]
    )
    assert bool(evidence.finite)
    np.testing.assert_allclose(evidence.identity_residual, 0.0, atol=1.0e-6)
    assert bool(jnp.all(system.admissible(evidence.state)))
