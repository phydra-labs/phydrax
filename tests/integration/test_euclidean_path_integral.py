#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


class _TrainableHarmonic(eqx.Module):
    stiffness: jax.Array

    def __call__(self, q, t):
        return 0.5 * self.stiffness * q[0] ** 2


def _harmonic_kernel(x0, x1, *, duration, mass, hbar, omega):
    sinh = jnp.sinh(omega * duration)
    cosh = jnp.cosh(omega * duration)
    normalization = jnp.sqrt(mass * omega / (2.0 * jnp.pi * hbar * sinh))
    exponent = (
        -mass * omega * ((x1**2 + x0**2) * cosh - 2.0 * x0 * x1) / (2.0 * hbar * sinh)
    )
    return normalization * jnp.exp(exponent)


def test_free_kernel_normalization_and_semigroup_composition():
    x0 = jnp.array([-0.3])
    x2 = jnp.array([0.4])
    duration_1 = 0.35
    duration_2 = 0.65

    estimate = phx.operators.euclidean_kernel(
        None,
        x0,
        x2,
        slicing=phx.operators.PathDiscretization(0.0, 1.0, num_steps=8),
        num_paths=32,
        key=jr.key(0),
    )
    exact = phx.operators.free_euclidean_kernel(x0, x2, duration=1.0)
    assert jnp.allclose(estimate.value, exact, atol=1e-14, rtol=0.0)
    assert estimate.standard_error == 0.0

    intermediate = jnp.linspace(-6.0, 6.0, 20001)
    left = phx.operators.free_euclidean_kernel(
        intermediate[:, None],
        x2,
        duration=duration_2,
    )
    right = phx.operators.free_euclidean_kernel(
        x0,
        intermediate[:, None],
        duration=duration_1,
    )
    composed = jnp.trapezoid(left * right, intermediate)
    assert jnp.allclose(composed, exact, atol=2e-8, rtol=0.0)


def test_harmonic_bridge_estimate_matches_analytic_kernel():
    duration = 1.0
    omega = 0.8
    x0 = jnp.array([0.0])
    x1 = jnp.array([0.3])
    slicing = phx.operators.PathDiscretization(0.0, duration, num_steps=48)

    estimate = phx.operators.euclidean_kernel(
        lambda q, t: 0.5 * omega**2 * q[0] ** 2,
        x0,
        x1,
        slicing=slicing,
        num_paths=16384,
        chunk_size=512,
        key=jr.key(1),
    )
    exact = _harmonic_kernel(
        x0[0],
        x1[0],
        duration=duration,
        mass=1.0,
        hbar=1.0,
        omega=omega,
    )
    coarse = phx.operators.euclidean_kernel(
        lambda q, t: 0.5 * omega**2 * q[0] ** 2,
        x0,
        x1,
        slicing=phx.operators.PathDiscretization(0.0, duration, num_steps=2),
        num_paths=4096,
        chunk_size=512,
        key=jr.key(1),
    )

    assert jnp.abs(estimate.value - exact) < 5.0 * estimate.standard_error + 7e-4
    assert estimate.effective_sample_size > 0.95 * estimate.num_paths
    assert jnp.abs(estimate.value - exact) < 0.2 * jnp.abs(coarse.value - exact)


def test_domain_potential_and_kernel_function_compose_with_sampled_fields():
    q = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
    time = phx.domain.TimeInterval(0.0, 0.5)
    potential_domain = q @ time

    @potential_domain.Function("q", "t")
    def potential(position, t):
        return 0.4 * position[0] ** 2 + 0.1 * t

    q0 = phx.domain.HyperRectangle([-0.5], [0.5], label="q0")
    q1 = phx.domain.HyperRectangle([-0.5], [0.5], label="q1")
    endpoint_domain = q0 @ q1
    slicing = phx.operators.PathDiscretization(0.0, 0.5, num_steps=12)
    kernel = phx.operators.euclidean_kernel_function(
        endpoint_domain,
        potential,
        start_var="q0",
        end_var="q1",
        slicing=slicing,
        num_paths=512,
        chunk_size=128,
        position_var="q",
        time_var="t",
    )
    batch = endpoint_domain.component().sample(
        phx.domain.PointSampling(6, layout=phx.domain.SampleLayout((("q0", "q1"),))),
        key=jr.key(2),
    )
    values = kernel(batch, key=jr.key(3))

    assert values.data.shape == (6,)
    assert jnp.all(jnp.isfinite(jnp.asarray(values.data)))
    assert jnp.all(jnp.asarray(values.data) > 0.0)


def test_kernel_function_preserves_trainable_domain_potential_gradients():
    q = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
    time = phx.domain.TimeInterval(0.0, 0.5)
    potential_domain = q @ time
    q0 = phx.domain.HyperRectangle([-0.5], [0.5], label="q0")
    q1 = phx.domain.HyperRectangle([-0.5], [0.5], label="q1")
    endpoint_domain = q0 @ q1
    slicing = phx.operators.PathDiscretization(0.0, 0.5, num_steps=8)

    def value(stiffness):
        potential = potential_domain.Function("q", "t")(_TrainableHarmonic(stiffness))
        kernel = phx.operators.euclidean_kernel_function(
            endpoint_domain,
            potential,
            start_var="q0",
            end_var="q1",
            slicing=slicing,
            num_paths=256,
            chunk_size=64,
        )
        return kernel.func(
            jnp.array([0.0]),
            jnp.array([0.2]),
            key=jr.key(4),
        )

    gradient = jax.grad(value)(jnp.array(0.8))
    assert jnp.isfinite(gradient)
    assert gradient < 0.0


def test_kernel_function_runs_through_operator_constraint_solver():
    q0 = phx.domain.HyperRectangle([-0.5], [0.5], label="q0")
    q1 = phx.domain.HyperRectangle([-0.5], [0.5], label="q1")
    endpoint_domain = q0 @ q1
    kernel = phx.operators.euclidean_kernel_function(
        endpoint_domain,
        None,
        start_var="q0",
        end_var="q1",
        slicing=phx.operators.PathDiscretization(0.0, 0.5, num_steps=4),
        num_paths=8,
    )
    condition = phx.conditions.Residual(
        "kernel",
        endpoint_domain.component(),
        lambda kernel_field: phx.operators.laplacian(
            kernel_field,
            var="q1",
        ),
    )
    constraint = phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(
            phx.integration.mean_over(condition.on),
            phx.integration.MonteCarloPlan(4),
        ),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"kernel": kernel},
        terms=[constraint],
    )

    loss = solver.loss(key=jr.key(5))
    assert jnp.isfinite(loss)
