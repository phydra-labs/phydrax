from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.operators.quantum._monopole_sphere import MonopoleKineticPolicy


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


def test_monopole_selected_hessian_matches_dense_and_gradient() -> None:
    rng = np.random.default_rng(717)
    coefficients = jnp.asarray(
        rng.normal(scale=0.07, size=(6, 6)) + 1j * rng.normal(scale=0.04, size=(6, 6))
    )
    point = jnp.asarray(rng.uniform(0.4, 1.1, size=(3, 2)))
    policy = MonopoleKineticPolicy()

    def logarithm(q: jax.Array) -> jax.Array:
        flat = q.reshape(-1)
        return flat @ coefficients @ flat + jnp.sum(jnp.sin(flat))

    def model(q: jax.Array) -> phx.operators.LogAmplitude:
        value = logarithm(q)
        return phx.operators.LogAmplitude(jnp.real(value), jnp.exp(1j * jnp.imag(value)))

    def dense(q: jax.Array) -> jax.Array:
        flat = q.reshape(-1)
        real_function = lambda x: jnp.real(logarithm(x.reshape(q.shape)))
        imag_function = lambda x: jnp.imag(logarithm(x.reshape(q.shape)))
        gradient = (
            jax.grad(real_function)(flat) + 1j * jax.grad(imag_function)(flat)
        ).reshape(q.shape)
        diagonal = (
            jnp.diag(jax.hessian(real_function)(flat))
            + 1j * jnp.diag(jax.hessian(imag_function)(flat))
        ).reshape(q.shape)
        sine, cosine = jnp.sin(q[:, 0]), jnp.cos(q[:, 0])
        laplacian = jnp.sum(
            diagonal[:, 0] + cosine / sine * gradient[:, 0] + diagonal[:, 1] / sine**2
        )
        square = jnp.sum(gradient[:, 0] ** 2 + gradient[:, 1] ** 2 / sine**2)
        magnetic = jnp.sum(
            (1.5 * cosine / sine) ** 2 + 3j * cosine / sine**2 * gradient[:, 1]
        )
        return (-laplacian - square + magnetic) / (2.0 * 1.7**2)

    actual, valid = policy.local_kinetic(model, point, 3, 1.7)
    assert valid
    np.testing.assert_allclose(actual, dense(point), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        jax.grad(lambda q: jnp.real(policy.local_kinetic(model, q, 3, 1.7)[0]))(point),
        jax.grad(lambda q: jnp.real(dense(q)))(point),
        atol=2e-11,
        rtol=2e-12,
    )
