#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_path_discretization_uniform_grid_and_validation():
    slicing = phx.discretization.TemporalMesh.uniform(-1.0, 2.0, 6, role="path")

    assert slicing.num_nodes == 7
    assert jnp.allclose(slicing.dt, 0.5)
    assert jnp.allclose(slicing.times, jnp.linspace(-1.0, 2.0, 7))
    assert jnp.allclose(slicing.midpoints, jnp.linspace(-0.75, 1.75, 6))

    with pytest.raises(ValueError, match="intervals"):
        phx.discretization.TemporalMesh.uniform(0.0, 1.0, 0, role="path")
    with pytest.raises(TypeError, match="integer"):
        phx.discretization.TemporalMesh.uniform(0.0, 1.0, 2.5, role="path")
    with pytest.raises(ValueError, match="bounds"):
        phx.discretization.TemporalMesh.uniform(1.0, 1.0, 4, role="path")


def test_brownian_bridge_exact_endpoints_and_covariance():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 2, role="path")
    x0 = jnp.array([-0.2])
    x1 = jnp.array([0.4])
    paths = phx.operators.sample_brownian_bridge(
        x0,
        x1,
        slicing=slicing,
        num_paths=32768,
        diffusion=1.0,
        key=jr.key(0),
    )

    assert paths.shape == (32768, 3, 1)
    assert jnp.array_equal(paths[:, 0], jnp.broadcast_to(x0, (32768, 1)))
    assert jnp.array_equal(paths[:, -1], jnp.broadcast_to(x1, (32768, 1)))
    midpoint_fluctuation = paths[:, 1, 0] - 0.5 * (x0[0] + x1[0])
    assert jnp.abs(jnp.mean(midpoint_fluctuation)) < 1e-2
    assert jnp.allclose(jnp.var(midpoint_fluctuation), 0.25, atol=1e-2, rtol=0.0)


def test_discrete_euclidean_action_matches_midpoint_formula():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 2, role="path")
    paths = jnp.array([[[0.0], [1.0], [0.0]]])
    potential = lambda q, t: q[0] ** 2

    kinetic = phx.operators.kinetic_action(paths, slicing=slicing, mass=2.0)
    potential_value = phx.operators.potential_action(
        paths,
        potential,
        slicing=slicing,
    )
    action = phx.operators.discrete_euclidean_action(
        paths,
        potential,
        slicing=slicing,
        mass=2.0,
    )

    assert jnp.allclose(kinetic, jnp.array([4.0]))
    assert jnp.allclose(potential_value, jnp.array([0.25]))
    assert jnp.allclose(action, jnp.array([4.25]))


def test_euclidean_action_rejects_complex_potential():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 2, role="path")
    paths = jnp.zeros((3, 3, 1))

    with pytest.raises(TypeError, match="must be real"):
        phx.operators.potential_action(
            paths,
            lambda q, t: 1.0j * q[0],
            slicing=slicing,
        )


def test_euclidean_estimate_is_seeded_and_reports_diagnostics():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 16, role="path")
    potential = lambda q, t: 0.5 * q[0] ** 2

    estimate_a = phx.operators.euclidean_kernel(
        potential,
        jnp.array([0.0]),
        jnp.array([0.2]),
        slicing=slicing,
        num_paths=1024,
        chunk_size=128,
        key=jr.key(1),
    )
    estimate_b = phx.operators.euclidean_kernel(
        potential,
        jnp.array([0.0]),
        jnp.array([0.2]),
        slicing=slicing,
        num_paths=1024,
        chunk_size=1024,
        key=jr.key(1),
    )
    small_estimate = phx.operators.euclidean_kernel(
        potential,
        jnp.array([0.0]),
        jnp.array([0.2]),
        slicing=slicing,
        num_paths=256,
        chunk_size=128,
        key=jr.key(1),
    )

    assert jnp.allclose(estimate_a.value, estimate_b.value, atol=1e-14, rtol=0.0)
    assert jnp.allclose(
        estimate_a.standard_error,
        estimate_b.standard_error,
        atol=1e-14,
        rtol=0.0,
    )
    assert estimate_a.num_paths == 1024
    assert estimate_a.standard_error > 0.0
    assert estimate_a.standard_error < small_estimate.standard_error
    assert 0.0 < estimate_a.effective_sample_size <= 1024.0
    assert jnp.isfinite(estimate_a.log_mean_weight)


def test_euclidean_log_weight_reduction_stays_stable_at_large_scale():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 4, role="path")
    estimate = phx.operators.euclidean_kernel(
        lambda q, t: -500.0,
        jnp.array([0.0]),
        jnp.array([0.0]),
        slicing=slicing,
        num_paths=64,
        chunk_size=16,
        key=jr.key(9),
    )

    assert jnp.isfinite(estimate.value)
    assert jnp.allclose(estimate.log_mean_weight, 500.0)
    assert jnp.allclose(estimate.standard_error, 0.0, atol=1e-12)
    assert jnp.allclose(estimate.effective_sample_size, 64.0)


def test_euclidean_kernel_supports_jit_vmap_and_parameter_gradients():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 0.5, 8, role="path")
    x0 = jnp.array([0.0])

    def kernel(endpoint, stiffness):
        return phx.operators.euclidean_kernel(
            lambda q, t: 0.5 * stiffness * q[0] ** 2,
            x0,
            endpoint,
            slicing=slicing,
            num_paths=256,
            chunk_size=64,
            key=jr.key(2),
        ).value

    jitted = jax.jit(kernel)(jnp.array([0.2]), jnp.array(0.7))
    batched = jax.vmap(lambda endpoint: kernel(endpoint, jnp.array(0.7)))(
        jnp.array([[0.1], [0.2], [0.3]])
    )
    gradient = jax.grad(lambda stiffness: kernel(jnp.array([0.2]), stiffness))(
        jnp.array(0.7)
    )
    epsilon = 1e-4
    finite_difference = (
        kernel(jnp.array([0.2]), jnp.array(0.7 + epsilon))
        - kernel(jnp.array([0.2]), jnp.array(0.7 - epsilon))
    ) / (2.0 * epsilon)

    assert jnp.isfinite(jitted)
    assert batched.shape == (3,)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.isfinite(gradient)
    assert jnp.allclose(gradient, finite_difference, atol=1e-7, rtol=1e-5)


def test_diffusion_from_zero_noise_matches_euler_drift():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 4, role="path")
    noise = jnp.zeros((2, 4, 1))
    paths = phx.operators.diffusion_paths_from_noise(
        lambda x, t: -x,
        0.5,
        jnp.array([1.0]),
        noise,
        slicing=slicing,
    )

    expected = (1.0 - float(slicing.dt)) ** jnp.arange(5)
    assert paths.shape == (2, 5, 1)
    assert jnp.allclose(paths[..., 0], jnp.broadcast_to(expected, (2, 5)))


def test_first_exit_uses_discrete_crossing_and_survival_sentinel():
    slicing = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 2, role="path")
    paths = jnp.array(
        [
            [[0.0], [0.4], [1.1]],
            [[0.0], [0.2], [0.3]],
        ]
    )
    inside = lambda x: jnp.abs(x[0]) < 1.0

    index = phx.operators.first_exit_index(paths, inside)
    time = phx.operators.first_exit_time(paths, inside, slicing=slicing)
    survival = phx.operators.survival_probability(paths, inside)

    assert jnp.array_equal(index, jnp.array([2, -1]))
    assert jnp.allclose(time[0], 1.0)
    assert jnp.isinf(time[1])
    assert jnp.allclose(survival.value, 0.5)
    assert survival.num_paths == 2
