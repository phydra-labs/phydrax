#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.discretization.dlr import (
    DLRBasisPolicy,
    generate_dlr_basis,
    matsubara_frequencies,
    plan_dlr_basis,
    thermal_tau_kernel,
)
from phydrax.operators.quantum._analytic_continuation import (
    ContinuationStatus,
    maximum_entropy_continuation,
    pade_continuation,
    plan_pade_continuation,
    prepare_pade_continuation,
    plan_scalar_fermionic_maximum_entropy,
    prepare_scalar_fermionic_maximum_entropy,
    sparse_continuation,
    spectral_grid,
)
from phydrax.operators.quantum._thermal_green import (
    convolve_dlr,
    differentiate_dlr,
    dlr_from_poles,
    dlr_moments,
    dlr_to_imaginary_time,
    dlr_to_matsubara,
    dyson_solve,
    evaluate_dlr_matsubara,
    evaluate_fermionic_thermal_channel,
    evaluate_dlr_tau,
    evaluate_lehmann_matsubara,
    evaluate_matsubara_tail,
    extract_self_energy,
    fermionic_spectral_function,
    fermionic_thermal_sector_channel,
    imaginary_time_to_dlr,
    matsubara_to_dlr,
    MatsubaraGreenFunction,
    MatsubaraSelfEnergy,
    RetardedGreenFunction,
    thermal_lehmann_sum,
)


def _fermion_basis():
    return generate_dlr_basis(
        8.0,
        6.0,
        policy=DLRBasisPolicy(
            tolerance=2e-7,
            maximum_rank=36,
            candidate_count=80,
        ),
    )


def test_generated_dlr_records_and_meets_requested_tolerance():
    basis = _fermion_basis()

    assert basis.frequencies.shape == (36,)
    assert basis.active.shape == (36,)
    assert int(basis.rank) <= 36
    assert basis.evidence.requested_tolerance == pytest.approx(2e-7)
    assert float(basis.evidence.achieved_tolerance) <= 2e-7
    assert bool(basis.valid)
    assert basis.prepared_id == _fermion_basis().prepared_id

    with pytest.raises(ValueError, match="maximum_bytes"):
        plan_dlr_basis(
            8.0,
            6.0,
            policy=DLRBasisPolicy(
                maximum_rank=32,
                candidate_count=80,
                maximum_bytes=1024,
            ),
        )


def test_fermionic_and_bosonic_kernels_are_finite_at_their_boundaries():
    tau = jnp.asarray([0.0, 2.0, 4.0])
    frequency = jnp.asarray([-2.0, 0.0, 2.0])

    fermionic = thermal_tau_kernel(tau, frequency, beta=4.0, statistics="fermionic")
    bosonic = thermal_tau_kernel(tau, frequency, beta=4.0, statistics="bosonic")

    assert jnp.all(jnp.isfinite(fermionic))
    assert jnp.all(jnp.isfinite(bosonic))
    assert bosonic[1, 1] == pytest.approx(-0.25)


def test_single_and_multiple_poles_round_trip_through_tau_and_matsubara():
    basis = _fermion_basis()
    poles = jnp.asarray([-1.75, 0.4, 2.2])
    residues = jnp.asarray([0.2, 0.5, 0.3])
    green = dlr_from_poles(basis, poles, residues, tolerance=2e-6)
    tau = jnp.linspace(0.0, basis.beta, 31)
    labels = jnp.arange(-15, 16)
    exact_tau = (
        thermal_tau_kernel(tau, poles, beta=basis.beta, statistics="fermionic") @ residues
    )
    exact_iw = jnp.sum(
        residues[None, :]
        / (
            1j
            * matsubara_frequencies(labels, beta=basis.beta, statistics="fermionic")[
                :, None
            ]
            - poles[None, :]
        ),
        axis=1,
    )

    assert jnp.allclose(evaluate_dlr_tau(green, tau), exact_tau, rtol=2e-4, atol=2e-5)
    assert jnp.allclose(
        evaluate_dlr_matsubara(green, labels), exact_iw, rtol=2e-4, atol=2e-5
    )

    tau_samples = dlr_to_imaginary_time(green)
    iw_samples = dlr_to_matsubara(green)
    from_tau = imaginary_time_to_dlr(tau_samples, basis, tolerance=2e-6)
    from_iw = matsubara_to_dlr(iw_samples, basis, tolerance=2e-6)
    assert jnp.allclose(
        evaluate_dlr_tau(from_tau, tau), evaluate_dlr_tau(green, tau), atol=2e-5
    )
    assert jnp.allclose(
        evaluate_dlr_matsubara(from_iw, labels),
        evaluate_dlr_matsubara(green, labels),
        atol=2e-5,
    )

    single = dlr_from_poles(
        basis, jnp.asarray([0.75]), jnp.asarray([1.0]), tolerance=2e-6
    )
    derivative = differentiate_dlr(single)
    expected_derivative = -0.75 * evaluate_dlr_tau(single, tau)
    assert jnp.allclose(evaluate_dlr_tau(derivative, tau), expected_derivative, atol=2e-5)


def test_dlr_convolution_moments_and_tail_follow_frequency_algebra():
    basis = _fermion_basis()
    left = dlr_from_poles(basis, jnp.asarray([-0.6]), jnp.asarray([0.75]), tolerance=2e-6)
    right = dlr_from_poles(basis, jnp.asarray([1.1]), jnp.asarray([0.4]), tolerance=2e-6)
    convolution = convolve_dlr(left, right, tolerance=2e-6)
    labels = jnp.arange(-18, 19)
    expected = evaluate_dlr_matsubara(left, labels)
    expected = expected * evaluate_dlr_matsubara(right, labels)
    assert jnp.allclose(
        evaluate_dlr_matsubara(convolution, labels),
        expected,
        rtol=3e-4,
        atol=3e-5,
    )

    moments = dlr_moments(left, 3)
    assert jnp.allclose(
        moments.values,
        jnp.asarray([0.75, -0.45, 0.27]),
        rtol=3e-3,
        atol=2e-5,
    )
    high_labels = jnp.asarray([200, 400])
    tail = evaluate_matsubara_tail(moments, high_labels, beta=basis.beta)
    exact = evaluate_dlr_matsubara(left, high_labels)
    assert jnp.allclose(tail, exact, rtol=1e-7, atol=1e-9)


def test_scalar_dyson_solve_and_self_energy_extraction_close_the_identity():
    beta = 5.0
    labels = jnp.arange(-12, 12)
    frequency = matsubara_frequencies(labels, beta=beta, statistics="fermionic")
    g0_values = 1.0 / (1j * frequency - 0.35)
    sigma_values = jnp.full_like(g0_values, 0.2)
    g0 = MatsubaraGreenFunction(beta, labels, g0_values)
    sigma = MatsubaraSelfEnergy(beta, labels, sigma_values)

    solved = dyson_solve(g0, sigma)
    expected = 1.0 / (1.0 / g0_values - sigma_values)
    extracted = extract_self_energy(g0, solved.green)

    assert jnp.all(solved.evidence.valid)
    assert jnp.allclose(solved.green.values, expected)
    assert jnp.all(extracted.evidence.valid)
    assert jnp.allclose(
        extracted.self_energy.values, sigma_values, rtol=1e-6, atol=1e-7
    )



def test_retarded_spectral_physicality_and_sector_channels_keep_invariants_separate():
    retarded = RetardedGreenFunction(
        jnp.asarray([-1.0, 1.0]),
        -1j * jnp.pi * jnp.ones((2,)),
        broadening=0.05,
    )
    spectral = fermionic_spectral_function(
        retarded,
        jnp.asarray([0.5, 0.5]),
        expected_first_moment=0.0,
        moment_tolerance=1e-12,
    )
    assert bool(spectral.valid)
    assert spectral.physicality.causality_residual == pytest.approx(0.0)
    assert spectral.physicality.zeroth_moment_residual == pytest.approx(0.0)

    beta = 3.0
    log_partition = jnp.log1p(jnp.exp(-beta))
    channel = fermionic_thermal_sector_channel(
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        beta,
        log_partition,
        source_sector="N=1",
        target_sector="N=0",
    )
    z = jnp.asarray([0.2 + 0.1j])
    assert channel.evidence.spectral_sum == pytest.approx(1.0)
    assert jnp.allclose(evaluate_fermionic_thermal_channel(channel, z), 1.0 / (z - 1.0))


def test_scalar_maxent_profile_refuses_matrix_continuation():
    grid = spectral_grid(-2.0, 2.0, 16)
    plan = plan_scalar_fermionic_maximum_entropy(
        grid,
        alpha=1e-2,
        expected_first_moment=0.0,
        maximum_samples=8,
    )
    labels = jnp.arange(8)
    matrix_samples = MatsubaraGreenFunction(
        4.0, labels, jnp.ones((8, 1, 1), dtype=jnp.complex128)
    )
    with pytest.raises(ValueError, match="excludes matrix"):
        prepare_scalar_fermionic_maximum_entropy(plan, matrix_samples)

def test_hubbard_atom_lehmann_sum_has_two_poles_and_unit_spectral_weight():
    interaction = 4.0
    chemical_potential = interaction / 2.0
    energies = jnp.asarray(
        [
            0.0,
            -chemical_potential,
            -chemical_potential,
            interaction - 2 * chemical_potential,
        ]
    )
    annihilation_up = jnp.zeros((4, 4)).at[0, 1].set(1.0).at[2, 3].set(1.0)
    lehmann = thermal_lehmann_sum(energies, annihilation_up, 6.0)
    labels = jnp.arange(-20, 20)
    frequency = matsubara_frequencies(labels, beta=6.0, statistics="fermionic")
    expected = 0.5 / (1j * frequency - interaction / 2.0)
    expected = expected + 0.5 / (1j * frequency + interaction / 2.0)

    assert bool(lehmann.valid)
    assert lehmann.evidence.spectral_sum == pytest.approx(1.0)
    assert jnp.allclose(evaluate_lehmann_matsubara(lehmann, labels), expected)
    with pytest.raises(ValueError, match="maximum_states"):
        thermal_lehmann_sum(
            energies,
            annihilation_up,
            6.0,
            maximum_states=3,
        )


def test_pade_recovers_a_single_pole_and_reports_rank_failure():
    beta = 12.0
    labels = jnp.arange(14)
    frequency = matsubara_frequencies(labels, beta=beta, statistics="fermionic")
    pole = 0.65
    samples = MatsubaraGreenFunction(beta, labels, 1.0 / (1j * frequency - pole))
    omega = jnp.asarray([-1.0, 0.0, 1.0])

    result = pade_continuation(
        samples,
        omega,
        numerator_degree=0,
        denominator_degree=1,
        broadening=0.05,
    )
    assert bool(result.evidence.valid)
    assert jnp.allclose(result.values, 1.0 / (omega + 0.05j - pole), atol=2e-5)
    assert jnp.all(result.spectral_density >= 0.0)

    constant = MatsubaraGreenFunction(
        beta, labels, jnp.ones_like(frequency, dtype=complex)
    )
    failed = prepare_pade_continuation(
        plan_pade_continuation(labels.size, numerator_degree=3, denominator_degree=3),
        constant,
    )
    assert not bool(failed.evidence.valid)
    assert int(failed.evidence.status) == int(ContinuationStatus.RANK_DEFICIENT)
    with pytest.raises(ValueError, match="maximum_bytes"):
        plan_pade_continuation(labels.size, maximum_bytes=1)


def test_nonnegative_maxent_and_sparse_continuation_obey_sum_rules():
    beta = 10.0
    labels = jnp.arange(-24, 24)
    grid = spectral_grid(-4.0, 4.0, 64)
    center = 0.8
    width = 0.35
    density = jnp.exp(-0.5 * ((grid.frequencies - center) / width) ** 2)
    density = density / jnp.sum(grid.weights * density)
    mass = grid.weights * density
    frequency = matsubara_frequencies(labels, beta=beta, statistics="fermionic")
    values = jnp.sum(
        mass[None, :] / (1j * frequency[:, None] - grid.frequencies[None, :]),
        axis=1,
    )
    samples = MatsubaraGreenFunction(beta, labels, values)

    maxent = maximum_entropy_continuation(
        samples,
        grid,
        alpha=2e-3,
        noise=2e-3,
        maximum_iterations=1200,
        gradient_tolerance=5e-3,
        residual_tolerance=2.0,
    )
    sparse = sparse_continuation(
        samples,
        grid,
        l1_regularization=1e-4,
        l2_regularization=1e-5,
        nonnegative=True,
        sum_rule=1.0,
        noise=2e-3,
        maximum_iterations=1200,
        gradient_tolerance=2e-2,
        residual_tolerance=2.0,
    )

    for result in (maxent, sparse):
        assert jnp.min(result.density) >= 0.0
        assert jnp.sum(result.grid.weights * result.density) == pytest.approx(1.0)
        assert bool(result.evidence.valid)
        assert jnp.all(jnp.isfinite(result.uncertainty))
        assert result.evidence.regularization_value > 0.0
        assert result.evidence.uncertainty_scale >= 0.0
