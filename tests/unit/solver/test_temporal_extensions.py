#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_temporal_extensions_scenario_1() -> None:
    method = phx.solver.RKCMethod(10)
    initial = jnp.asarray([1.0])

    # ty: ignore[invalid-argument-type]
    result = method.step(lambda time, state, args: -100.0 * state, 0.0, initial, 0.01)

    assert jnp.all(jnp.isfinite(result))
    assert jnp.linalg.norm(result) < jnp.linalg.norm(initial)
    assert method.capabilities.order == 1
    times = jnp.linspace(0.0, 1.0, 21)
    result = phx.solver.AdamsBashforthMoultonMethod(4).integrate(
        lambda time, state, args: state,
        times,
        jnp.asarray([1.0]),
    )

    assert bool(result.finite)
    assert jnp.allclose(result.states[-1], jnp.exp(1.0), atol=2.0e-5)
    method = phx.solver.ExponentialRosenbrockEulerMethod()

    result = method.step(
        lambda time, state, args: -2.0 * state,
        jnp.asarray(0.0),
        jnp.asarray([1.0]),
        jnp.asarray(0.2),
    )

    assert jnp.allclose(result, jnp.exp(-0.4), atol=1.0e-12)


def test_temporal_extensions_scenario_2() -> None:
    method = phx.solver.RadauIIAIntegrator(3, residual_tolerance=1.0e-11)
    assert method.capabilities.equation_forms == ("explicit-ode",)

    state, defect, accepted = method.step(
        lambda time, value, args: -10.0 * value,
        jnp.asarray(0.0),
        jnp.asarray([1.0]),
        jnp.asarray(0.1),
    )

    assert bool(accepted)
    assert defect < 1.0e-11
    assert jnp.allclose(state, jnp.exp(-1.0), atol=5.0e-5)
    method = phx.solver.IMEXBDF2Integrator(residual_tolerance=1.0e-11)
    step = jnp.asarray(0.05)
    previous = jnp.asarray([1.0])
    current = jnp.exp(-step)[None]

    state, defect, accepted = method.step(
        lambda time, value, args: jnp.zeros_like(value),
        lambda time, value, args: -value,
        step,
        previous,
        current,
        step,
    )

    assert bool(accepted)
    assert defect < 1.0e-11
    assert jnp.allclose(state, jnp.exp(-2.0 * step), atol=2.0e-4)
    step = 0.25
    times = jnp.linspace(0.0, 1.0, 5)
    coarse = lambda time, value, args: value + step * value
    fine = lambda time, value, args: jnp.exp(step) * value

    result = phx.solver.parareal(
        coarse,
        fine,
        times,
        jnp.asarray([1.0]),
        4,
    )

    assert bool(result.finite)
    assert jnp.allclose(result.states[:, 0], jnp.exp(times), atol=1.0e-12)
