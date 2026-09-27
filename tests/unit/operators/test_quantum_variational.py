from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx


def test_quantum_variational_scenario_1() -> None:
    current = phx.operators.LogAmplitude(800.0, 1.0j)
    proposed = phx.operators.LogAmplitude(799.5, -1.0 + 0.0j)
    ratio = phx.operators.amplitude_ratio(proposed, current)

    assert jnp.allclose(phx.operators.sampling_log_weight(current), 1600.0)
    assert ratio.valid
    assert jnp.allclose(ratio.value, 1.0j * jnp.exp(-0.5))
    nonzero = phx.operators.LogAmplitude(0.0, 1.0 + 0.0j)
    zero = phx.operators.LogAmplitude(-jnp.inf, 1.0 + 0.0j)

    proposed_zero = phx.operators.amplitude_ratio(zero, nonzero)
    current_zero = phx.operators.amplitude_ratio(nonzero, zero)

    assert zero.valid
    assert not zero.nonzero
    assert proposed_zero.valid
    assert proposed_zero.value == 0.0
    assert not current_zero.valid
    amplitude = phx.operators.LogAmplitude(0.0, 2.0 + 0.0j)

    assert not amplitude.valid
    assert jnp.isneginf(phx.operators.sampling_log_weight(amplitude))


def test_log_amplitude_phase_derivative_survives_real_parameters() -> None:
    def ratio_phase(theta: Any) -> Any:
        current = phx.operators.LogAmplitude(0.0, 1.0 + 0.0j)
        proposed = phx.operators.LogAmplitude(0.0, jnp.exp(1j * theta))
        return jnp.imag(phx.operators.amplitude_ratio(proposed, current).value)

    assert jnp.allclose(jax.grad(ratio_phase)(0.0), 1.0)
