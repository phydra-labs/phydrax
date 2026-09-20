import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.compact_objects._hawking import (
    evaluate_hawking_spectrum,
    evolve_kerr_evaporation,
    HawkingEvaporationTermination,
    HawkingScatteringData,
    HawkingSpectrumPlan,
    HawkingTailEvidence,
    KerrEvaporationPlan,
    KerrEvaporationState,
    QuantumFieldSpecies,
)


def _scale():
    return RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1)


def _tail(*, frequency=(0.0, 0.0, 0.0), modes=(0.0, 0.0, 0.0)):
    return HawkingTailEvidence(
        frequency,
        modes,
        qualified=True,
        derivative_valid=True,
        qualification_id=f"tails:{frequency}:{modes}",
    )


def _scattering(plan, graybody, slopes, tail=None):
    sources = tuple(
        tuple(
            f"radial:{mode}:{frequency}" for frequency in range(plan.frequency_capacity)
        )
        for mode in range(plan.mode_capacity)
    )
    flags = jnp.ones((plan.mode_capacity,), dtype="bool")
    return HawkingScatteringData(
        plan,
        graybody,
        slopes,
        finite=flags,
        converged=flags,
        physically_valid=flags,
        qualified=flags,
        derivative_valid=flags,
        tail_evidence=_tail() if tail is None else tail,
        source_ids=sources,
        qualification_id="wronskian-and-flux-balance",
    )


def _evaluate(
    plan,
    scattering,
    source_state,
    temperature,
    angular_velocity,
    horizon_source_id,
    *,
    horizon_qualified=True,
):
    return evaluate_hawking_spectrum(
        plan,
        source_state,
        scattering,
        temperature,
        angular_velocity,
        horizon_source_id=horizon_source_id,
        horizon_finite=True,
        horizon_converged=True,
        horizon_physically_valid=True,
        horizon_qualified=horizon_qualified,
        horizon_derivative_valid=True,
    )


def test_unruh_statistics_superradiance_and_large_exponents_are_finite():
    scale = _scale()
    scalar = QuantumFieldSpecies("massless-scalar", 0, "boson")
    spinor = QuantumFieldSpecies("massless-spinor", 0.5, "fermion")
    plan = HawkingSpectrumPlan(
        scale,
        (scalar, spinor),
        (0.5, 1.0, 1000.0),
        (scalar.species_id, spinor.species_id),
        (2.0, 0.5),
        (2.0, 0.5),
        mode_ids=("scalar:2:2", "spinor:1/2:1/2"),
    )
    scattering = _scattering(
        plan,
        ((-0.1, 0.0, 0.1), (0.25, 0.25, 0.25)),
        (0.2, 0.0),
    )
    source_state = KerrEvaporationState(1.0, 0.5, state_id="kerr:M=1:chi=0.5")

    result = eqx.filter_jit(_evaluate)(
        plan,
        scattering,
        source_state,
        0.2,
        0.5,
        "kerr:M=1:chi=0.5",
    )

    expected_superradiant = -0.1 / np.expm1(-2.5) / (2.0 * np.pi)
    expected_corotation = 0.2 * 0.2 / (2.0 * np.pi)
    expected_fermion = 0.25 / (np.exp(1.25) + 1.0) / (2.0 * np.pi)
    np.testing.assert_allclose(
        result.mode_number_flux_density[0, 0], expected_superradiant, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        result.mode_number_flux_density[0, 1], expected_corotation, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        result.mode_number_flux_density[1, 0], expected_fermion, rtol=1.0e-6
    )
    assert float(result.mode_number_flux_density[0, 0]) > 0.0
    assert float(result.mode_number_flux_density[0, 1]) > 0.0
    assert float(result.mode_number_flux_density[0, -1]) == 0.0
    assert bool(result.successful)
    assert bool(jnp.all(jnp.isfinite(result.mode_number_flux_density)))
    assert result.species_ids == (scalar.species_id, spinor.species_id)


def test_schwarzschild_mode_pair_cancels_angular_flux_and_tail_gates_convergence():
    scale = _scale()
    photon = QuantumFieldSpecies("photon", 1, "boson", multiplicity=2)
    plan = HawkingSpectrumPlan(
        scale,
        (photon,),
        (0.0, 0.2, 0.4),
        (photon.species_id, photon.species_id),
        (1, 1),
        (-1, 1),
        mode_ids=("photon:1:-1", "photon:1:1"),
    )
    graybody = ((0.0, 0.04, 0.08), (0.0, 0.04, 0.08))
    scattering = _scattering(plan, graybody, (0.2, 0.2))
    source_state = KerrEvaporationState(1.0, 0.0, state_id="schwarzschild:M=1")
    result = _evaluate(
        plan,
        scattering,
        source_state,
        0.1,
        0.0,
        "schwarzschild:M=1",
    )

    np.testing.assert_allclose(result.mode_number_flux[0], result.mode_number_flux[1])
    np.testing.assert_allclose(result.angular_momentum_flux, 0.0, atol=1.0e-12)
    assert float(result.energy_flux) > 0.0
    assert bool(result.successful)

    uncovered = _scattering(
        plan,
        graybody,
        (0.2, 0.2),
        tail=_tail(frequency=(1.0e-3, 0.0, 0.0)),
    )
    uncovered_result = _evaluate(
        plan,
        uncovered,
        source_state,
        0.1,
        0.0,
        "schwarzschild:M=1",
    )
    assert bool(uncovered_result.finite)
    assert bool(uncovered_result.physically_valid)
    assert bool(uncovered_result.qualified)
    assert not bool(uncovered_result.coverage_satisfied)
    assert not bool(uncovered_result.converged)

    unqualified_horizon = _evaluate(
        plan,
        scattering,
        source_state,
        0.1,
        0.0,
        "unqualified-schwarzschild-horizon",
        horizon_qualified=False,
    )
    assert bool(unqualified_horizon.finite)
    assert bool(unqualified_horizon.converged)
    assert bool(unqualified_horizon.physically_valid)
    assert not bool(unqualified_horizon.qualified)


def test_bounded_kerr_evaporation_coevolves_and_terminates_fail_closed():
    scale = _scale()
    scalar = QuantumFieldSpecies("scalar", 0, "boson")
    plan = HawkingSpectrumPlan(
        scale,
        (scalar,),
        (0.5, 1.0, 1.5),
        (scalar.species_id,),
        (2,),
        (2,),
        mode_ids=("scalar:2:2",),
    )
    scattering = _scattering(plan, ((-0.1, 0.0, 0.1),), (0.2,))
    initial = KerrEvaporationState(10.0, 20.0, state_id="initial-kerr")

    def spectrum_for_state(state):
        return _evaluate(
            plan,
            scattering,
            state,
            0.2,
            0.5,
            "kerr-flux-family",
        )

    spectrum = spectrum_for_state(initial)
    evolution_plan = KerrEvaporationPlan(
        scale,
        (0.0, 1.0e-3, 2.0e-3),
        semiclassical_mass_ratio=2.0,
        maximum_adiabatic_parameter=10.0,
        maximum_mass_fraction_per_step=0.5,
        maximum_spin_change_per_step=0.5,
    )
    evolved = evolve_kerr_evaporation(
        evolution_plan,
        initial,
        spectrum_for_state,
        spectrum_source_id="state-bound-qualified-spectrum",
    )

    assert evolved.masses.shape == (evolution_plan.step_capacity + 1,)
    assert evolved.angular_momenta.shape == (evolution_plan.step_capacity + 1,)
    assert int(evolved.completed_steps) == evolution_plan.step_capacity
    assert int(evolved.termination) == int(HawkingEvaporationTermination.CAPACITY_REACHED)
    assert float(evolved.final_state.mass) < float(initial.mass)
    assert float(evolved.final_state.angular_momentum) < float(initial.angular_momentum)
    assert bool(evolved.successful)

    stale = evolve_kerr_evaporation(
        evolution_plan,
        initial,
        lambda state: spectrum,
        spectrum_source_id="stale-fixed-spectrum",
    )
    assert int(stale.completed_steps) == 1
    assert int(stale.termination) == int(
        HawkingEvaporationTermination.STALE_SPECTRUM_BINDING
    )
    assert bool(stale.state_binding_satisfied[0])
    assert not bool(stale.state_binding_satisfied[1])
    assert not bool(stale.qualified)

    schwarzschild_plan = HawkingSpectrumPlan(
        scale,
        (scalar,),
        (0.0, 0.2, 0.4),
        (scalar.species_id, scalar.species_id),
        (1, 1),
        (-1, 1),
        mode_ids=("scalar:1:-1", "scalar:1:1"),
    )
    symmetric = _scattering(
        schwarzschild_plan,
        ((0.0, 0.04, 0.08), (0.0, 0.04, 0.08)),
        (0.2, 0.2),
    )
    near_boundary = KerrEvaporationState(2.1, 0.0, state_id="near-boundary")

    def schwarzschild_for_state(state):
        return _evaluate(
            schwarzschild_plan,
            symmetric,
            state,
            0.1,
            0.0,
            "schwarzschild-terminal-flux",
        )

    schwarzschild_spectrum = schwarzschild_for_state(near_boundary)
    crossing_time = 0.2 / float(schwarzschild_spectrum.energy_flux)
    endpoint_plan = KerrEvaporationPlan(
        scale,
        (0.0, crossing_time),
        semiclassical_mass_ratio=2.0,
        maximum_adiabatic_parameter=10.0,
        maximum_mass_fraction_per_step=10.0,
        maximum_spin_change_per_step=10.0,
    )

    stopped = evolve_kerr_evaporation(
        endpoint_plan,
        near_boundary,
        schwarzschild_for_state,
        spectrum_source_id="terminal-qualified-spectrum",
    )
    assert int(stopped.termination) == int(
        HawkingEvaporationTermination.SEMICLASSICAL_BOUNDARY
    )
    assert int(stopped.completed_steps) == 0
    np.testing.assert_allclose(stopped.final_state.mass, near_boundary.mass)
    assert float(stopped.final_state.mass) > float(endpoint_plan.planck_length)
    assert float(stopped.final_state.mass) > float(endpoint_plan.semiclassical_mass_floor)

    def uncovered_for_state(state):
        current = spectrum_for_state(state)
        return eqx.tree_at(
            lambda value: (value.coverage_satisfied, value.converged),
            current,
            (jnp.asarray(False), jnp.asarray(False)),
        )

    coverage_stopped = evolve_kerr_evaporation(
        evolution_plan,
        initial,
        uncovered_for_state,
        spectrum_source_id="uncovered-spectrum",
    )
    assert int(coverage_stopped.termination) == int(
        HawkingEvaporationTermination.INSUFFICIENT_COVERAGE
    )
    assert int(coverage_stopped.completed_steps) == 0

    restrictive_plan = KerrEvaporationPlan(
        scale,
        (0.0, 1.0e-3),
        semiclassical_mass_ratio=2.0,
        maximum_adiabatic_parameter=float(spectrum.energy_flux) / 2.0,
        maximum_mass_fraction_per_step=0.5,
        maximum_spin_change_per_step=0.5,
    )
    adiabatic_stopped = evolve_kerr_evaporation(
        restrictive_plan,
        initial,
        spectrum_for_state,
        spectrum_source_id="too-fast-spectrum",
    )
    assert int(adiabatic_stopped.termination) == int(
        HawkingEvaporationTermination.ADIABATICITY_LIMIT
    )
    assert int(adiabatic_stopped.completed_steps) == 0
