#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_lattice_boltzmann_compiler_matches_independent_periodic_trt_step():
    shape = (4, 3)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(shape[0], periodic=True),
            phx.discretization.UniformCellAxisSpec(shape[1], periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (4.0, 3.0))))
    velocity_set = phx.discretization.D2Q9()
    discretization = phx.discretization.LatticeBoltzmannPlan(grid, velocity_set).prepare()
    collision = phx.discretization.TRTCollisionPlan(3.0 / 16.0)
    method = phx.discretization.LatticeBoltzmannMethodPlan(
        collision, forcing=phx.discretization.GuoForcingPlan()
    )
    problem = phx.equations.LatticeBoltzmannProblem(
        "periodic-oracle",
        2,
        acceleration=lambda time, coordinates, parameters: parameters,
        acceleration_id="constant-oracle-force",
    )
    compiled = phx.equations.compile_lattice_boltzmann_problem(
        problem,
        discretization,
        method,
        phx.discretization.LatticeBoltzmannBoundaryPlan(),
        time_step=1.0,
    )
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(
        0.08, force_parameters=jnp.asarray((2e-5, -1e-5))
    )
    base = np.asarray(
        compiled.initialize_state(1.0, jnp.asarray((0.02, -0.01)), parameters)
    )
    perturbation = 1e-4 * np.sin(np.arange(base.size)).reshape(base.shape)
    populations = jnp.asarray(base + perturbation)
    result = compiled.dynamics.step_detailed(
        jnp.asarray(0),
        jnp.asarray(0.0),
        populations,
        jnp.asarray(1.0),
        parameters,
    )

    f = np.asarray(populations)
    c = np.asarray(velocity_set.velocities, dtype=float)
    w = np.asarray(velocity_set.weights)
    opposite = np.asarray(velocity_set.opposite)
    rho = np.sum(f, axis=-1)
    raw_momentum = np.einsum("...q,qd->...d", f, c)
    acceleration = np.asarray((2e-5, -1e-5))
    force = rho[..., None] * acceleration
    velocity = (raw_momentum + 0.5 * force) / rho[..., None]
    cu = np.einsum("...d,qd->...q", velocity, c)
    u2 = np.einsum("...d,...d->...", velocity, velocity)
    equilibrium = (
        w * rho[..., None] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2[..., None])
    )
    first = (c - velocity[..., None, :]) / (1.0 / 3.0)
    second = cu[..., :, None] * c / (1.0 / 3.0) ** 2
    source = w * np.einsum("...qd,...d->...q", first + second, force)
    even_rate = 1.0 / (0.5 + 0.08 / (1.0 / 3.0))
    odd_rate = 1.0 / (0.5 + collision.magic_parameter / (1.0 / even_rate - 0.5))
    even_f = 0.5 * (f + f[..., opposite])
    odd_f = 0.5 * (f - f[..., opposite])
    even_eq = 0.5 * (equilibrium + equilibrium[..., opposite])
    odd_eq = 0.5 * (equilibrium - equilibrium[..., opposite])
    even_source = 0.5 * (source + source[..., opposite])
    odd_source = 0.5 * (source - source[..., opposite])
    post = (
        f
        - even_rate * (even_f - even_eq)
        - odd_rate * (odd_f - odd_eq)
        + (1.0 - 0.5 * even_rate) * even_source
        + (1.0 - 0.5 * odd_rate) * odd_source
    )
    expected = np.stack(
        tuple(
            np.roll(post[..., direction], tuple(c[direction].astype(int)), axis=(0, 1))
            for direction in range(velocity_set.population_count)
        ),
        axis=-1,
    )

    assert result.successful
    np.testing.assert_allclose(result.accepted_state, expected, rtol=2e-12, atol=2e-12)
