#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _periodic_compiled(shape, bathymetry, reconstruction, *, source=None):
    dimension = len(shape)
    axis_names = tuple("xy"[:dimension])
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in shape
        ),
        axis_names=axis_names,
    ).prepare(jnp.stack((jnp.zeros(dimension), jnp.ones(dimension))))
    system = phx.equations.ShallowWaterSystem(dimension)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    method = phx.discretization.FiniteVolumeMethodPlan(
        reconstruction,
        phx.discretization.ShallowWaterHydrostaticHLLPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "shallow-water-workflow",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(axis_names),
        source=source,
        source_id=None if source is None else source.source_id,
    )
    return phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        bathymetry=bathymetry,
    )


def test_two_dimensional_dry_lake_is_stationary():
    x = (jnp.arange(10) + 0.5) / 10
    y = (jnp.arange(8) + 0.5) / 8
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    bed = 0.2 + 1.2 * jnp.exp(-80.0 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2))
    compiled = _periodic_compiled(
        bed.shape,
        bed,
        phx.discretization.MUSCLReconstruction(),
    )
    depth = jnp.maximum(1.0 - bed, 0.0)
    state = jnp.stack((depth, jnp.zeros_like(depth), jnp.zeros_like(depth)), axis=-1)

    residual = compiled(0.0, state)

    np.testing.assert_allclose(residual, 0.0, atol=5e-12)


def test_equilibrium_muscl_has_second_order_smooth_residual():
    def error(count):
        bed = jnp.zeros((count,))
        compiled = _periodic_compiled(
            (count,),
            bed,
            phx.discretization.MUSCLReconstruction(phx.discretization.UnlimitedLimiter()),
        )
        x = (jnp.arange(count) + 0.5) / count
        depth = 1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * x)
        velocity = 0.2
        state = jnp.stack((depth, velocity * depth), axis=-1)
        derivative = 0.2 * jnp.pi * jnp.cos(2.0 * jnp.pi * x)
        expected = jnp.stack(
            (
                -velocity * derivative,
                -(velocity**2 + 9.81 * depth) * derivative,
            ),
            axis=-1,
        )
        return jnp.sqrt(jnp.mean((compiled(0.0, state) - expected) ** 2))

    coarse = error(32)
    fine = error(64)

    assert fine < coarse / 3.0


def test_linearization_jvp_vjp_duality_away_from_dry_switches():
    count = 16
    bed = jnp.zeros((count,))
    compiled = _periodic_compiled((count,), bed, phx.discretization.MUSCLReconstruction())
    x = (jnp.arange(count) + 0.5) / count
    state = jnp.stack(
        (
            1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * x),
            0.1 + 0.02 * jnp.cos(2.0 * jnp.pi * x),
        ),
        axis=-1,
    )
    tangent = jnp.stack(
        (
            0.01 * jnp.cos(2.0 * jnp.pi * x),
            0.02 * jnp.sin(2.0 * jnp.pi * x),
        ),
        axis=-1,
    )
    cotangent = jnp.stack(
        (
            jnp.sin(4.0 * jnp.pi * x),
            jnp.cos(4.0 * jnp.pi * x),
        ),
        axis=-1,
    )

    _, jvp, vjp = compiled.linearize(0.0, state)
    forward = jvp(tangent)
    reverse = vjp(cotangent)[0]

    np.testing.assert_allclose(
        jnp.vdot(forward, cotangent),
        jnp.vdot(tangent, reverse),
        rtol=2e-10,
        atol=2e-10,
    )


def test_coriolis_runtime_preserves_mass_and_converges_inertial_rotation():
    shape = (6, 6)
    bed = jnp.zeros(shape)
    source = phx.equations.ShallowWaterCoriolisSource(0.5)
    compiled = _periodic_compiled(
        shape,
        bed,
        phx.discretization.PiecewiseConstantReconstruction(),
        source=source,
    )
    depth = jnp.ones(shape)
    state = jnp.stack((depth, jnp.ones(shape), jnp.zeros(shape)), axis=-1)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.25),
    )
    dt = 0.01
    runtime_state = runtime.initialize_state(state, 0.0, dt)
    for _ in range(10):
        result = runtime.advance(runtime_state)
        assert bool(result.accepted)
        runtime_state = result.runtime_state
    updated = runtime_state.cell_average()
    elapsed = runtime_state.time
    expected_u = jnp.cos(0.5 * elapsed)
    expected_v = -jnp.sin(0.5 * elapsed)

    np.testing.assert_allclose(updated[..., 0], 1.0, atol=2e-13)
    np.testing.assert_allclose(updated[..., 1], expected_u, atol=2e-7)
    np.testing.assert_allclose(updated[..., 2], expected_v, atol=2e-7)
