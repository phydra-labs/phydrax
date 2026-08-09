#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import jax.numpy as jnp

import phydrax as phx


def test_stormer_verlet_bounds_long_horizon_harmonic_energy_error():
    vector_field = phx.solver.SeparableHamiltonianVectorField(
        lambda time, configuration, args: configuration,
        lambda time, momentum, args: momentum,
        1,
    )
    times = jnp.linspace(0.0, 100.0, 1001)
    solution = dfx.diffeqsolve(
        dfx.ODETerm(vector_field),
        phx.solver.StormerVerlet(1),
        t0=0.0,
        t1=100.0,
        dt0=0.1,
        y0=jnp.array([1.0, 0.0]),
        stepsize_controller=dfx.ConstantStepSize(),
        saveat=dfx.SaveAt(ts=times),
        max_steps=1001,
    )
    energies = 0.5 * jnp.sum(solution.ys**2, axis=-1)

    assert jnp.max(jnp.abs(energies - energies[0])) < 1.3e-3
    assert phx.solver.StormerVerlet(1).order(None) == 2
