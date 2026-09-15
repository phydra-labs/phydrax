#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.nn.quantum.variable_sector import (
    BosonicJastrowAmplitude,
    FermionicDeterminantJastrowAmplitude,
)
from phydrax.operators.quantum._amplitude import LogAmplitude
from phydrax.operators.quantum.variable_sector import (
    ContactInteractionOperator,
    ContinuumKineticOperator,
    PairPotentialOperator,
    ParticleChangingLocalOperator,
    QuadraticExternalPotential,
    VariableParticleConfiguration,
    VariableSectorHamiltonian,
    VariableSectorMeasure,
    VariableSectorProposal,
    VariableSectorSpace,
)
from phydrax.solver._variable_sector_vmc import (
    evolve_variable_sector_tdvp,
    prepare_variable_sector_vmc,
    run_variable_sector_vmc,
    SectorTailEvidence,
    solve_stochastic_reconfiguration,
    VARIABLE_SECTOR_VMC_CUTOFF_TAIL_REFUSED,
    VariableSectorTDVPPlan,
    VariableSectorVMCPlan,
)


def _configuration(coordinates, active, species) -> VariableParticleConfiguration:
    return VariableParticleConfiguration(
        jnp.asarray(coordinates, dtype=float),
        jnp.asarray(active),
        jnp.asarray(species),
    )


def test_variable_sector_padding_invariance_and_inverse_factorial_measure() -> None:
    space = VariableSectorSpace(4, 1, 2)
    left = _configuration([[1.0], [2.0], [100.0], [-20.0]], [1, 1, 0, 0], [0, 0, 1, 1])
    right = _configuration([[1.0], [2.0], [-8.0], [50.0]], [1, 1, 0, 0], [0, 0, 0, 0])
    model = BosonicJastrowAmplitude(
        space,
        precisions=jnp.asarray([[1.0], [2.0]]),
        pair_cusp=jnp.asarray([[0.3, 0.2], [0.2, 0.1]]),
        pair_range=0.5,
    )
    assert np.allclose(model(left).log_abs, model(right).log_abs)
    measure = VariableSectorMeasure(space)
    assert np.allclose(measure.sector_factor(left), 1.0 / math.factorial(2))
    mixed = _configuration([[1.0], [2.0], [3.0], [0.0]], [1, 1, 1, 0], [0, 1, 1, 0])
    assert np.allclose(measure.sector_factor(mixed), 1.0 / math.factorial(2))


def test_variable_sector_birth_death_reverse_density_has_exact_combinatorics() -> None:
    space = VariableSectorSpace(3, 1, 1)
    proposal = VariableSectorProposal(
        space,
        move_weights=(1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        birth_scale=2.0,
    )
    vacuum = space.empty()
    one = _configuration([[0.25], [0.0], [0.0]], [1, 0, 0], [0, 0, 0])
    forward = proposal.log_prob(one, vacuum)
    reverse = proposal.log_prob(vacuum, one)
    expected_forward = -0.5 * ((0.25 / 2.0) ** 2 + math.log(2.0 * math.pi * 4.0))
    assert np.allclose(forward, expected_forward)
    assert np.allclose(reverse, math.log(0.5))
    measure = VariableSectorMeasure(space)
    log_ratio = (
        measure.log_sector_factor(one)
        - measure.log_sector_factor(vacuum)
        + reverse
        - forward
    )
    forward_flux = jnp.exp(
        measure.log_sector_factor(vacuum) + forward + jnp.minimum(log_ratio, 0.0)
    )
    reverse_flux = jnp.exp(
        measure.log_sector_factor(one) + reverse + jnp.minimum(-log_ratio, 0.0)
    )
    assert np.allclose(forward_flux, reverse_flux)
    assert np.isfinite(proposal.log_prob(proposal.sample(jr.key(4), vacuum), vacuum))


def test_variable_sector_pair_jacobian_and_exchange_are_reversible() -> None:
    space = VariableSectorSpace(4, 1, 2)
    proposal = VariableSectorProposal(
        space,
        move_weights=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        pair_species=jnp.asarray(((0, 1),)),
    )
    source = _configuration([[1.0], [2.0], [0.0], [0.0]], [1, 1, 0, 0], [0, 1, 0, 0])
    exchanged = _configuration([[1.0], [2.0], [0.0], [0.0]], [1, 1, 0, 0], [1, 0, 0, 0])
    assert np.allclose(
        proposal.log_prob(exchanged, source), proposal.log_prob(source, exchanged)
    )
    assert np.allclose(proposal.pair_log_abs_jacobian, math.log(2.0))
    pair = _configuration([[1.0], [2.0], [0.5], [-0.5]], [1, 1, 1, 1], [0, 1, 0, 1])
    assert np.isfinite(proposal.log_prob(pair, source))
    assert np.isfinite(proposal.log_prob(source, pair))


def test_bosonic_and_fermionic_exchange_symmetry() -> None:
    space = VariableSectorSpace(3, 1, 1)
    first = _configuration([[-0.7], [0.4], [0.0]], [1, 1, 0], [0, 0, 0])
    swapped = _configuration([[0.4], [-0.7], [0.0]], [1, 1, 0], [0, 0, 0])
    boson = BosonicJastrowAmplitude(space, pair_cusp=0.2, pair_range=0.4)
    boson_first, boson_swapped = boson(first), boson(swapped)
    assert np.allclose(boson_first.log_abs, boson_swapped.log_abs)
    assert np.allclose(boson_first.phase, boson_swapped.phase)
    fermion = FermionicDeterminantJastrowAmplitude(
        space,
        orbital_bias=jnp.asarray(((1.0, 0.0, 0.0),)),
        orbital_weights=jnp.asarray((((0.0,), (1.0,), (1.0,)),)),
        pair_cusp=0.1,
    )
    fermion_first, fermion_swapped = fermion(first), fermion(swapped)
    assert np.allclose(fermion_first.log_abs, fermion_swapped.log_abs)
    assert np.allclose(fermion_first.phase, -fermion_swapped.phase)


def test_jastrow_cusp_and_coordinate_derivatives_are_visible() -> None:
    space = VariableSectorSpace(2, 1, 1)
    model = BosonicJastrowAmplitude(space, pair_cusp=0.75, pair_range=0.4)
    derivative = jax.grad(lambda radius: jnp.real(model.pair_log_factor(radius, 0, 0)))(
        jnp.asarray(0.0)
    )
    assert np.allclose(derivative, 0.75)
    configuration = _configuration([[0.2], [0.0]], [1, 0], [0, 0])
    coordinate_gradient = jax.grad(
        lambda coordinate: (
            model(
                VariableParticleConfiguration(
                    coordinate.reshape((2, 1)),
                    configuration.active_mask,
                    configuration.species,
                )
            ).log_abs
        )
    )(configuration.coordinates.reshape((-1,)))
    assert np.allclose(coordinate_gradient, [-0.2, 0.0])


def test_free_and_quadratic_continuum_local_energy_references() -> None:
    space = VariableSectorSpace(2, 1, 1)
    configuration = _configuration([[0.31], [9.0]], [1, 0], [0, 0])
    wave_number = 1.7

    def plane_wave(value):
        phase = jnp.exp(
            1j * wave_number * jnp.sum(value.coordinates * value.active_mask[:, None])
        )
        return LogAmplitude(jnp.asarray(0.0), phase)

    kinetic = ContinuumKineticOperator(space, jnp.asarray((2.0,)))
    free = kinetic.local_value(plane_wave, configuration)
    assert free.valid
    assert np.allclose(free.value, wave_number**2 / 4.0, rtol=1e-5)
    gaussian = BosonicJastrowAmplitude(space, precisions=1.0)
    oscillator = VariableSectorHamiltonian(
        (kinetic, QuadraticExternalPotential(space, jnp.asarray((2.0,))))
    )
    quadratic = oscillator.local_value(gaussian, configuration)
    expected = 0.25 + 0.75 * 0.31**2
    assert quadratic.valid
    assert np.allclose(quadratic.value, expected, rtol=1e-5)


def test_pair_contact_and_particle_changing_local_operators_are_finite() -> None:
    space = VariableSectorSpace(3, 1, 1)
    configuration = _configuration([[-0.5], [0.5], [0.0]], [1, 1, 0], [0, 0, 0])
    model = BosonicJastrowAmplitude(
        space, sector_log_weights=jnp.zeros((1, 4), dtype=complex)
    )
    pair = PairPotentialOperator(space, jnp.asarray(((2.0,),)), softening=0.1)
    contact = ContactInteractionOperator(space, jnp.asarray(((0.3,),)), 0.2)
    changing = ParticleChangingLocalOperator(
        space,
        jnp.asarray((([0.0], [1.0]),)),
        jnp.asarray(((0.5, 0.5),)),
        jnp.asarray((0.1,)),
    )
    assert pair.local_value(model, configuration).valid
    assert contact.local_value(model, configuration).valid
    assert changing.local_value(model, configuration).valid


def test_stochastic_reconfiguration_solves_finite_reference_systems() -> None:
    derivatives = jnp.asarray(((-1.0, 0.0), (1.0, 0.0), (0.0, -2.0), (0.0, 2.0)))
    energies = jnp.asarray((-1.0, 1.0, -4.0, 4.0))
    result = solve_stochastic_reconfiguration(
        derivatives, energies, damping=0.5, parameter_mode="real"
    )
    expected = -np.linalg.solve(
        np.asarray(result.metric) + 0.5 * np.eye(2), np.asarray(result.force)
    )
    assert result.successful
    assert np.allclose(result.update, expected)
    assert result.residual_norm < 1e-5


def _tail_evidence(*, accepted: bool) -> SectorTailEvidence:
    status = 0 if accepted else VARIABLE_SECTOR_VMC_CUTOFF_TAIL_REFUSED
    return SectorTailEvidence(
        total_histogram=jnp.asarray((128, 0, 0)),
        species_histogram=jnp.asarray(((128, 0, 0),)),
        sample_count=jnp.asarray(128),
        cutoff_count=jnp.asarray(0 if accepted else 8),
        cutoff_probability=jnp.asarray(0.0 if accepted else 0.0625),
        cutoff_standard_error=jnp.asarray(0.0 if accepted else 0.02),
        cutoff_upper_bound=jnp.asarray(0.01 if accepted else 0.12),
        sufficient_samples=jnp.asarray(True),
        below_tolerance=jnp.asarray(accepted),
        status=jnp.asarray(status),
        capacity=2,
        tolerance=0.05,
        method="test-finite-reference",
    )


def test_real_and_imaginary_time_tdvp_evolution_and_tail_refusal() -> None:
    initial = jnp.asarray((1.0 + 0.0j,))
    real_plan = VariableSectorTDVPPlan(0.05, 4, evolution="real-time")
    real_result = evolve_variable_sector_tdvp(
        lambda parameters, time: -1j * parameters,
        initial,
        real_plan,
        _tail_evidence(accepted=True),
    )
    assert real_result.valid
    assert np.allclose(real_result.final_parameters, np.exp(-0.2j), atol=2e-6)
    imaginary_plan = VariableSectorTDVPPlan(0.05, 4, evolution="imaginary-time")
    imaginary_result = evolve_variable_sector_tdvp(
        lambda parameters, time: -parameters,
        initial,
        imaginary_plan,
        _tail_evidence(accepted=True),
    )
    assert imaginary_result.valid
    assert np.allclose(imaginary_result.final_parameters, np.exp(-0.2), atol=2e-6)
    refused = evolve_variable_sector_tdvp(
        lambda parameters, time: -parameters,
        initial,
        imaginary_plan,
        _tail_evidence(accepted=False),
    )
    assert not refused.valid
    assert refused.status == VARIABLE_SECTOR_VMC_CUTOFF_TAIL_REFUSED
    assert np.allclose(refused.final_parameters, initial)


def test_reversible_jump_vmc_preserves_chain_state_and_reports_tail() -> None:
    space = VariableSectorSpace(4, 1, 1)
    model = BosonicJastrowAmplitude(
        space,
        sector_log_weights=jnp.asarray(((0.0, -2.0, -5.0, -9.0, -14.0),)),
    )
    operator = VariableSectorHamiltonian(
        (
            ContinuumKineticOperator(space, jnp.asarray((1.0,))),
            QuadraticExternalPotential(space, jnp.asarray((1.0,))),
        )
    )
    proposal = VariableSectorProposal(space)
    measure = VariableSectorMeasure(space)
    plan = VariableSectorVMCPlan(
        chain_count=2,
        draw_count=4,
        warmup_steps=2,
        minimum_tail_samples=4,
        tail_probability_tolerance=0.99,
    )
    prepared = prepare_variable_sector_vmc(
        plan, model, operator, proposal, measure, (space.empty(), space.empty())
    )
    first = run_variable_sector_vmc(prepared)
    second = run_variable_sector_vmc(prepared, first.final_state)
    assert first.coordinates.shape == (2, 4, 4, 1)
    assert second.final_state.transition_index > first.final_state.transition_index
    assert second.tail_evidence.sample_count == 8
    assert jnp.sum(second.tail_evidence.total_histogram) == 8
