#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _rc_circuit(resistance=1.0):
    reference = phx.circuit.ElectricalWaveReference(50.0)
    source = phx.circuit.CircuitElement(
        phx.circuit.IndependentCurrentSourceLaw(1.0),
        element_id="source",
    )
    return phx.circuit.NodalCircuit(
        (
            phx.circuit.CircuitInstance(
                "resistor", phx.circuit.Resistor(resistance), ("n", "0")
            ),
            phx.circuit.CircuitInstance(
                "capacitor", phx.circuit.Capacitor(1.0), ("n", "0")
            ),
            phx.circuit.CircuitInstance("source", source, ("0", "n")),
        ),
        (phx.circuit.NodalPort("port", "n", "0", reference),),
        ground="0",
        circuit_id="harmonic-rc",
    )


def test_temporal_harmonic_plan_differentiates_constant_and_sinusoid():
    plan = phx.circuit.TemporalHarmonicPlan(2.0, 9, 2)
    constant = jnp.ones((9, 2))
    assert jnp.allclose(plan.derivative(constant), 0.0, atol=1e-12)

    waveform = jnp.stack((jnp.sin(2.0 * plan.times), jnp.cos(2.0 * plan.times)), axis=-1)
    expected = jnp.stack(
        (2.0 * jnp.cos(2.0 * plan.times), -2.0 * jnp.sin(2.0 * plan.times)), axis=-1
    )
    assert jnp.allclose(plan.derivative(waveform), expected, atol=1e-10)

    derivative_norm = lambda omega: jnp.linalg.norm(
        phx.circuit.TemporalHarmonicPlan(omega, 9, 1).derivative(
            jnp.sin(2.0 * jnp.pi * jnp.arange(9) / 9.0)[:, None]
        )
    )
    assert jnp.isfinite(jax.grad(derivative_norm)(jnp.asarray(1.0)))


def test_prepared_harmonic_balance_matches_convenience_solve():
    dae = phx.circuit.prepare_circuit_dae(_rc_circuit())
    waveform = jnp.ones((5, 1))
    plan = phx.circuit.plan_harmonic_balance(dae, 1.0, 5)
    prepared = phx.circuit.prepare_harmonic_balance(dae, waveform, 1.0, plan)

    result = eqx.filter_jit(phx.circuit.solve_prepared_harmonic_balance)(prepared)
    direct = phx.circuit.solve_harmonic_balance(dae, waveform, 1.0)

    assert result.diagnostics.residual_norm < 1e-9
    assert result.diagnostics.finite
    assert result.diagnostics.aliasing_tail_valid
    assert jnp.allclose(result.waveform, direct.waveform)
    assert jnp.allclose(result.coefficients, direct.coefficients)
    assert result.plan.plan_id == prepared.plan.plan_id
    assert result.prepared_id == prepared.prepared_id


def test_harmonic_refresh_preserves_structure_and_updates_numeric_provenance():
    dae = phx.circuit.prepare_circuit_dae(_rc_circuit())
    waveform = jnp.ones((5, 1))
    prepared = phx.circuit.prepare_harmonic_balance(dae, waveform, 1.0)
    refreshed = phx.circuit.refresh_harmonic_balance(prepared, dae, waveform, 2.0)
    result = phx.circuit.solve_prepared_harmonic_balance(refreshed)

    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.numeric_version == 1
    assert jnp.allclose(refreshed.plan.temporal.angular_frequency, 2.0)
    assert result.numeric_version == 1
    assert result.diagnostics.residual_norm < 1e-9

    with pytest.raises(ValueError, match="replan is required"):
        phx.circuit.refresh_harmonic_balance(prepared, dae, jnp.ones((7, 1)), 1.0)


def test_harmonic_resource_envelope_rejects_before_nonlinear_preparation():
    dae = phx.circuit.prepare_circuit_dae(_rc_circuit())
    with pytest.raises(MemoryError, match="maximum_samples"):
        phx.circuit.plan_harmonic_balance(
            dae,
            1.0,
            5,
            phx.circuit.HarmonicBalancePolicy(maximum_samples=4),
        )
    with pytest.raises(MemoryError, match="maximum_unknowns"):
        phx.circuit.plan_harmonic_balance(
            dae,
            1.0,
            5,
            phx.circuit.HarmonicBalancePolicy(maximum_unknowns=4),
        )
    with pytest.raises(MemoryError, match="maximum_waveform_bytes"):
        phx.circuit.plan_harmonic_balance(
            dae,
            1.0,
            5,
            phx.circuit.HarmonicBalancePolicy(maximum_waveform_bytes=8),
        )


def test_harmonic_preparation_rejects_wrong_waveform_shape_and_frequency():
    dae = phx.circuit.prepare_circuit_dae(_rc_circuit())
    with pytest.raises(ValueError, match="rank-two"):
        phx.circuit.prepare_harmonic_balance(dae, jnp.ones((5,)), 1.0)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="finite and positive"
    ):
        phx.circuit.plan_harmonic_balance(dae, -1.0, 5)
