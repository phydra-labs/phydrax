#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_feynman_kac_heat_solution_matches_analytic_and_pde_routes():
    kappa = 0.35
    wave_number = 1.1
    final_time = 0.7
    x0 = jnp.array([0.25])
    slicing = phx.discretization.TemporalMesh.uniform(0.0, final_time, 32, role="path")
    terminal = lambda x, t: jnp.cos(wave_number * x[0])
    drift = lambda x, t: jnp.zeros_like(x)

    estimate = phx.operators.feynman_kac_expectation(
        terminal,
        drift,
        jnp.sqrt(2.0 * kappa),
        x0,
        slicing=slicing,
        num_paths=32768,
        key=jr.key(0),
    )
    exact = jnp.cos(wave_number * x0[0]) * jnp.exp(-kappa * wave_number**2 * final_time)
    assert jnp.abs(estimate.value - exact) < 5.0 * estimate.standard_error

    space = phx.domain.HyperRectangle([-2.0], [2.0], label="x")
    time = phx.domain.TimeInterval(0.0, final_time)
    domain = space @ time

    @domain.Function("x", "t")
    def exact_field(x, t):
        return jnp.cos(wave_number * x[0]) * jnp.exp(
            -kappa * wave_number**2 * (final_time - t)
        )

    residual = phx.operators.dt(exact_field, var="t") + kappa * phx.operators.laplacian(
        exact_field,
        var="x",
    )
    assert jnp.abs(residual.func(x0, jnp.array(0.2))) < 1e-12


def test_feynman_kac_constant_killing_and_diffusion_gradient():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 0.6, 24, role="path")
    killing_rate = 0.3
    x0 = jnp.array([0.1])
    terminal = lambda x, t: 1.0
    drift = lambda x, t: jnp.zeros_like(x)
    killing = lambda x, t: killing_rate

    estimate = phx.operators.feynman_kac_expectation(
        terminal,
        drift,
        0.7,
        x0,
        slicing=slicing,
        num_paths=2048,
        killing=killing,
        key=jr.key(1),
    )
    assert jnp.allclose(
        estimate.value,
        jnp.exp(-killing_rate * slicing.duration),
        atol=2e-14,
        rtol=0.0,
    )
    assert jnp.allclose(estimate.effective_sample_size, estimate.num_paths)

    oscillatory_terminal = lambda x, t: jnp.cos(x[0])

    def value(diffusivity):
        return phx.operators.feynman_kac_expectation(
            oscillatory_terminal,
            drift,
            jnp.sqrt(2.0 * diffusivity),
            x0,
            slicing=slicing,
            num_paths=4096,
            key=jr.key(2),
        ).value

    gradient = jax.grad(value)(jnp.array(0.4))
    assert jnp.isfinite(gradient)
    assert gradient < 0.0


def test_ornstein_uhlenbeck_terminal_mean_matches_euler_reference():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 64, role="path")
    theta = 0.7
    sigma = 0.4
    x0 = jnp.array([1.2])
    estimate = phx.operators.feynman_kac_expectation(
        lambda x, t: x[0],
        lambda x, t: -theta * x,
        sigma,
        x0,
        slicing=slicing,
        num_paths=32768,
        key=jr.key(3),
    )
    expected = x0[0] * (1.0 - theta * slicing.dt) ** slicing.num_steps

    assert jnp.abs(estimate.value - expected) < 5.0 * estimate.standard_error
    assert jnp.allclose(estimate.effective_sample_size, estimate.num_paths)


def test_discrete_first_passage_converges_to_brownian_interval_survival():
    final_time = 0.5
    sigma = 0.5
    slicing = phx.discretization.TemporalMesh.uniform(0.0, final_time, 256, role="path")
    paths = phx.operators.sample_diffusion_paths(
        None,
        sigma,
        jnp.array([0.0]),
        slicing=slicing,
        num_paths=32768,
        key=jr.key(4),
    )
    estimate = phx.operators.survival_probability(
        paths,
        lambda x: jnp.abs(x[0]) < 1.0,
    )
    coarse_estimate = phx.operators.survival_probability(
        paths[..., ::32, :],
        lambda x: jnp.abs(x[0]) < 1.0,
    )

    series_index = jnp.arange(80)
    exact = (
        4.0
        / jnp.pi
        * jnp.sum(
            (-1.0) ** series_index
            / (2 * series_index + 1)
            * jnp.exp(
                -((2 * series_index + 1) ** 2) * jnp.pi**2 * sigma**2 * final_time / 8.0
            )
        )
    )

    # The tolerance includes both Bernoulli sampling error and positive slice bias:
    # crossings between stored nodes are intentionally not interpolated.
    assert jnp.abs(estimate.value - exact) < 0.02
    assert estimate.value <= coarse_estimate.value
    assert jnp.abs(estimate.value - exact) < jnp.abs(coarse_estimate.value - exact)
    assert estimate.standard_error < 0.002
