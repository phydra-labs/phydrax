import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.compressible_flow._all_speed import (
    AllSpeedHLLFluxPlan,
    ShockAwareAllSpeedFluxPlan,
)
from phydrax.applications.compressible_flow._contracts import (
    AllSpeedCompressiblePolicy,
    CompressibleFlowCaseSpec,
    ShockResolvingPolicy,
)
from phydrax.applications.compressible_flow._materials import (
    EOSConvexityCertificate,
    EOSDerivativeCertificate,
    ResearchRealGasMaterial,
    ThermallyPerfectGasMaterial,
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
from phydrax.applications.compressible_flow._system import MaterialEulerSystem
from phydrax.discretization.finite_volume._riemann import (
    EntropyConservativeEulerFluxPlan,
    EntropyStableEulerFluxPlan,
)
from phydrax.equations._hyperbolic_systems import (
    CompressibleNavierStokesSystem,
    EulerSystem,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.solver._balance_law_composition import AdditiveIMEXTableau
from phydrax.solver._conservation_temporal import (
    ConservationIMEXMethod,
    ImplicitConservationStageResult,
)


def test_thermally_perfect_inversion_and_jvp_and_real_gas_certificate_refusal():
    material = ThermallyPerfectGasMaterial(
        (3.5, 0.02),
        1.0,
        temperature_bounds=(0.5, 5.0),
    )
    temperature = jnp.asarray((0.75, 1.5, 3.0))
    energy = material.specific_internal_energy_from_temperature(temperature)
    recovered = material.temperature_from_specific_internal_energy(energy)
    np.testing.assert_allclose(recovered, temperature, rtol=1e-6, atol=1e-6)
    value, tangent = jax.jvp(
        material.temperature_from_specific_internal_energy,
        (energy,),
        (jnp.ones_like(energy),),
    )
    np.testing.assert_allclose(value, temperature, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        tangent,
        1.0 / (material.specific_heat_cp_from_temperature(temperature) - 1.0),
        rtol=1e-5,
    )
    system = MaterialEulerSystem(material, 1)
    primitive = jnp.stack(
        (jnp.ones_like(temperature), 0.1 * jnp.ones_like(temperature), temperature),
        axis=-1,
    )
    conserved = system.primitive_to_conserved(primitive)
    np.testing.assert_allclose(
        system.conserved_to_primitive(conserved), primitive, rtol=1e-6, atol=1e-6
    )
    _, flux_tangent = jax.jvp(
        lambda state: system.physical_flux(state, 0),
        (conserved,),
        (jnp.ones_like(conserved),),
    )
    assert bool(jnp.all(jnp.isfinite(flux_tangent)))
    case = CompressibleFlowCaseSpec(
        "thermally-perfect-fv",
        1,
        "navier_stokes",
        "structured-fv",
        material,
    )
    viscous_system = case.prepare_system(ConstantTransport(0.02, 0.03))
    viscous_flux = viscous_system.viscous_flux(
        conserved, jnp.zeros(conserved.shape + (1,))
    )
    assert bool(jnp.all(jnp.isfinite(viscous_flux)))

    providers = {
        "pressure_provider": lambda density, internal: 0.4 * density * internal,
        "energy_provider": lambda density, pressure: pressure / (0.4 * density),
        "temperature_provider": lambda density, pressure: pressure / density,
        "sound_speed_provider": lambda density, pressure: jnp.sqrt(
            1.4 * pressure / density
        ),
        "enthalpy_provider": lambda density, pressure: 3.5 * pressure / density,
        "heat_capacity_provider": lambda density, pressure: jnp.full_like(density, 3.5),
    }
    with pytest.raises(ValueError, match="derivative certificate"):
        ResearchRealGasMaterial("research-eos", **providers)
    derivative = EOSDerivativeCertificate(
        "research-eos",
        (0.5, 2.0),
        (0.5, 4.0),
        maximum_inverse_residual=1e-12,
        maximum_derivative_residual=1e-12,
        tolerance=1e-9,
    )
    convexity = EOSConvexityCertificate(
        "research-eos",
        minimum_sound_speed_squared=0.1,
        minimum_fundamental_derivative=0.2,
    )
    real_gas = ResearchRealGasMaterial(
        "research-eos",
        derivative_certificate=derivative,
        convexity_certificate=convexity,
        **providers,
    )
    assert bool(real_gas.admissible(jnp.asarray(1.0), jnp.asarray(1.0)))


def test_low_mach_scaling_and_shock_route_fallback_ledger_are_explicit():
    all_speed = AllSpeedCompressiblePolicy(reference_mach=1.0)
    mach = jnp.asarray((1e-3, 2e-3, 4e-3))
    np.testing.assert_allclose(all_speed.pressure_dissipation_scale(mach), mach)
    np.testing.assert_allclose(
        all_speed.pressure_dissipation_scale(mach[1:])
        / all_speed.pressure_dissipation_scale(mach[:-1]),
        2.0,
    )
    policy = ShockResolvingPolicy("teno", sensor_threshold=0.1, all_speed=all_speed)
    ledger = policy.ledger(
        jnp.asarray((0.01, 0.2, 0.01)),
        jnp.asarray((True, True, False)),
    )
    assert policy.route_label == "shock-resolving:teno:all-speed->robust-hll"
    np.testing.assert_array_equal(ledger.fallback_used, jnp.asarray((False, True, True)))
    assert int(ledger.fallback_count) == 2
    primary = jnp.ones((3, 2))
    fallback = -jnp.ones((3, 2))
    np.testing.assert_array_equal(
        policy.select_flux(primary, fallback, ledger),
        jnp.asarray(((1.0, 1.0), (-1.0, -1.0), (-1.0, -1.0))),
    )


def test_structured_fv_all_speed_policy_changes_the_primary_flux():
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
    assert (
        low_dissipation.method.interface_solver.policy.policy_id
        == low_dissipation.shock.policy_id
    )

    system = EulerSystem(1, material=IdealGasMaterial())
    primitive_left = jnp.asarray((1.0, 1.0e-3, 1.0))
    primitive_right = jnp.asarray((1.01, 1.0e-3, 1.0))
    left = system.primitive_to_conserved(primitive_left)
    right = system.primitive_to_conserved(primitive_right)
    low_flux = low_dissipation.method.interface_solver.face_flux(system, left, right, 0)
    higher_flux = higher_dissipation.method.interface_solver.face_flux(
        system, left, right, 0
    )

    assert float(low_flux.max_speed) < float(higher_flux.max_speed)
    assert not bool(jnp.all(low_flux.normal_flux == higher_flux.normal_flux))


def test_all_speed_zero_width_is_symmetric_and_shock_faces_use_fallback():
    system = EulerSystem(1, material=IdealGasMaterial())
    primary = AllSpeedHLLFluxPlan(
        AllSpeedCompressiblePolicy(reference_mach=1.0, minimum_mach=0.0)
    )
    left = system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 1.0)))
    right = system.primitive_to_conserved(jnp.asarray((1.2, 0.0, 2.0)))
    forward = primary.face_flux(system, left, right, 0)
    reverse = primary.face_flux(system, right, left, 0)
    np.testing.assert_allclose(forward.normal_flux, reverse.normal_flux)

    policy = ShockResolvingPolicy(sensor_threshold=0.01)
    shock_flux = ShockAwareAllSpeedFluxPlan(policy)
    selected = shock_flux.face_flux(system, left, right, 0)
    fallback = policy.fallback_flux.face_flux(system, left, right, 0)
    assert bool(selected.fallback_activated)
    np.testing.assert_allclose(selected.normal_flux, fallback.normal_flux)

    material_system = MaterialEulerSystem(IdealGasMaterial(), 1)
    material_left = material_system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 1.0)))
    material_right = material_system.primitive_to_conserved(jnp.asarray((1.2, 0.0, 2.0)))
    material_selected = shock_flux.face_flux(
        material_system, material_left, material_right, 0
    )
    material_fallback = shock_flux.generic_fallback.face_flux(
        material_system, material_left, material_right, 0
    )
    assert bool(material_selected.fallback_activated)
    np.testing.assert_allclose(
        material_selected.normal_flux, material_fallback.normal_flux
    )


def test_all_speed_ale_acoustic_scaling_uses_relative_mach():
    flux = AllSpeedHLLFluxPlan(
        AllSpeedCompressiblePolicy(reference_mach=1.0, minimum_mach=0.01)
    )
    system = MaterialEulerSystem(IdealGasMaterial(), 1)
    normal = jnp.asarray((1.0,))

    moving_left = system.primitive_to_conserved(jnp.asarray((1.0, 0.2, 1.0)))
    moving_right = system.primitive_to_conserved(jnp.asarray((1.01, 0.2, 1.0)))
    relative_left = system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 1.0)))
    relative_right = system.primitive_to_conserved(jnp.asarray((1.01, 0.0, 1.0)))
    moving = flux.normal_ale_face_flux(
        system,
        moving_left,
        moving_right,
        normal,
        jnp.asarray(0.2),
    )
    stationary = flux.normal_ale_face_flux(
        system,
        relative_left,
        relative_right,
        normal,
        jnp.asarray(0.0),
    )

    np.testing.assert_allclose(moving.max_speed, stationary.max_speed)


def test_smooth_and_fv_route_composition_are_exact_and_never_claim_dns():
    smooth = SmoothCompressibleProductionPlan(
        EntropyConservativeEulerFluxPlan(),
        EntropyStableEulerFluxPlan(),
    )
    assert smooth.method.viscous.formulation == "entropy_br1"
    assert "tensor-dgsem" in smooth.route_label
    case = CompressibleFlowCaseSpec(
        "smooth-periodic",
        2,
        "navier_stokes",
        "tensor-dgsem",
        IdealGasMaterial(),
        fidelity="dns-candidate",
    )
    evidence = smooth.qualification_evidence(case)
    assert not evidence.dns_claimed
    assert not evidence.signed
    assert not evidence.released
    assert not evidence.qualification_ready
    dependency = evidence.support_dependency("compressible-flow")
    assert dependency.support_tuple_id == evidence.support_tuple_id
    governed = evidence.bind_qualification_evidence(
        evidence_kind="scientific",
        build_id="test-build",
        environment_id="test-environment",
        backend="cpu",
        topology="single-process",
        precision="float32",
        reduction="deterministic",
        replay_id="test-replay",
        raw_artifact_ids=("raw-compressible-evidence",),
        reviewer_id="test-reviewer",
        issued_at=1,
        expires_at=2,
        reason="sampled compatibility evidence is absent",
    )
    assert governed.inconclusive
    assert evidence.support_tuple_id in governed.subject_ids

    fv = StructuredFVCompressibleProductionPlan(
        "structured",
        shock=ShockResolvingPolicy("mp5"),
    )
    assert fv.method.reconstruction.method == "mp5"
    assert fv.method.positivity is not None
    assert fv.positivity.fallback_flux.flux_id == fv.shock.fallback_flux.flux_id
    assert fv.route_label.startswith("shock:structured-fv")


def test_explicit_and_additive_imex_adapters_restart_without_partition_loss():
    explicit = ExplicitCompressibleFixedStepAdapter(
        lambda time, state, args: -state,
        "linear-decay",
    )
    initial = jnp.asarray((1.0, 2.0))
    first = explicit.step(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.1), None
    )
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
    assert imex.explicit_operator_id == "transport"
    assert imex.implicit_operator_id == "viscous"
    assert imex.partition_id
    assert int(result.iterations) == 1


def test_manufactured_viscous_navier_stokes_source_identity():
    system = CompressibleNavierStokesSystem(
        ConstantTransport(0.02, 0.03, bulk_viscosity=0.0),
        1,
    )

    def exact_state(time, point, args):
        del args
        phase = point[0] - time
        density = 1.0 + 0.05 * jnp.sin(phase)
        velocity = 0.2 + 0.03 * jnp.cos(phase)
        pressure = 1.0 + 0.04 * jnp.sin(phase)
        energy = pressure / 0.4 + 0.5 * density * velocity**2
        return jnp.stack((density, density * velocity, energy))

    evidence = ManufacturedViscousNSPlan(1, exact_state, "periodic-mms").evaluate(
        system,
        0.2,
        jnp.linspace(0.0, 1.0, 5)[:, None],
    )
    assert bool(evidence.finite)
    np.testing.assert_allclose(evidence.identity_residual, 0.0, atol=1e-7)
    assert bool(jnp.all(system.admissible(evidence.state)))
