#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


q = phx.operators.quantum


def test_transmon_charge_limit_and_external_phase_periodicity():
    basis = q.ChargeBasis(5)
    charging_rate = jnp.asarray(0.23)
    offset_charge = jnp.asarray(0.17)
    charge_limit = q.transmon_mode_problem(
        q.TransmonParameters(
            charging_rate,
            0.0,
            0.0,
            offset_charge=offset_charge,
        ),
        basis,
    )
    expected = 4.0 * charging_rate * (basis.charges - offset_charge) ** 2

    assert jnp.allclose(jnp.real(jnp.diag(charge_limit.hamiltonian)), expected)
    assert jnp.allclose(
        charge_limit.hamiltonian,
        jnp.diag(jnp.diag(charge_limit.hamiltonian)),
    )

    def spectrum(phase):
        problem = q.transmon_mode_problem(
            q.TransmonParameters(0.23, 8.0, 6.5, external_phase=phase),
            basis,
        )
        return q.prepare_mode_reduction(
            problem,
            policy=q.ModeReductionPolicy(4),
        ).energies

    assert jnp.allclose(spectrum(0.31), spectrum(0.31 + 2.0 * jnp.pi), atol=1e-10)
    assert jnp.allclose(
        spectrum(0.31),
        q.prepare_mode_reduction(
            q.transmon_mode_problem(
                q.TransmonParameters(0.23, 6.5, 8.0, external_phase=-0.31),
                basis,
            ),
            policy=q.ModeReductionPolicy(4),
        ).energies,
        atol=1e-10,
    )


def test_fluxonium_harmonic_limit_and_canonical_quadratures():
    charging_rate = 0.5
    inductive_rate = 2.0
    phase_scale = (8.0 * charging_rate / inductive_rate) ** 0.25
    basis = q.OscillatorBasis(12, phase_scale=phase_scale)
    problem = q.fluxonium_mode_problem(
        q.FluxoniumParameters(charging_rate, inductive_rate, 0.0),
        basis,
    )
    prepared = q.prepare_mode_reduction(
        problem,
        policy=q.ModeReductionPolicy(5),
    )
    expected_gap = jnp.sqrt(8.0 * charging_rate * inductive_rate)
    commutator = basis.phase @ basis.charge - basis.charge @ basis.phase

    assert bool(prepared.diagnostics.valid)
    assert jnp.allclose(jnp.diff(prepared.energies), expected_gap, atol=1e-10)
    assert jnp.allclose(commutator[:-1, :-1], 1j * jnp.eye(11), atol=1e-12)
    assert jnp.allclose(problem.hamiltonian, jnp.conj(problem.hamiltonian.T))


def test_harmonic_mode_uses_explicit_oscillator_scales():
    basis = q.OscillatorBasis(7, phase_scale=1.4)
    problem = q.harmonic_mode_problem(q.HarmonicModeParameters(2.3), basis)
    prepared = q.prepare_mode_reduction(
        problem,
        policy=q.ModeReductionPolicy(5),
    )

    assert jnp.allclose(jnp.diff(prepared.energies), 2.3)
    assert jnp.allclose(prepared.operator("number").matrix, jnp.diag(jnp.arange(5.0)))
    assert jnp.allclose(
        prepared.operator("lowering").matrix,
        basis.lowering[:5, :5],
    )


def test_circuit_mode_spectrum_gradients_are_finite_away_from_crossings():
    basis = q.ChargeBasis(6)
    nominal = q.TransmonParameters(0.25, 10.0, 9.0, external_phase=0.2)
    prepared = q.prepare_mode_reduction(
        q.transmon_mode_problem(nominal, basis),
        policy=q.ModeReductionPolicy(3),
    )

    derivative = jax.grad(
        lambda charging: q.refresh_mode_reduction(
            prepared,
            q.transmon_mode_problem(
                q.TransmonParameters(
                    charging,
                    10.0,
                    9.0,
                    external_phase=0.2,
                ),
                basis,
            ),
        ).energies[1]
    )(jnp.asarray(0.25))

    assert jnp.isfinite(derivative)


def test_circuit_mode_inputs_fail_closed():
    with pytest.raises(ValueError, match="positive"):
        q.ChargeBasis(0)
    with pytest.raises(ValueError, match="greater than one"):
        q.OscillatorBasis(1)
    with pytest.raises(ValueError, match="positive"):
        q.TransmonParameters(0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        q.FluxoniumParameters(0.2, 0.4, -1.0)
    with pytest.raises(TypeError, match="real scalar"):
        q.HarmonicModeParameters(1.0 + 1.0j)
