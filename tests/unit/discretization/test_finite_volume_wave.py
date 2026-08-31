#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

import phydrax as phx


def _periodic_grid(shape):
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in shape
        ),
        axis_names=tuple("xy"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def test_roe_wave_fluctuations_sum_to_roe_flux_jump():
    system = phx.equations.EulerSystem()
    left = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, 1.0]]))
    right = system.primitive_to_conserved(jnp.asarray([[0.8, -0.1, 0.7]]))
    decomposition = phx.discretization.RoeWavePropagationPlan().decompose(
        system, left, right, 0
    )
    _, right_matrix, eigenvalues = system.eigensystem(left, right, 0)
    left_matrix, _, _ = system.eigensystem(left, right, 0)
    expected = oe.contract(
        "...ij,...j->...i",
        right_matrix,
        eigenvalues * oe.contract("...ij,...j->...i", left_matrix, right - left),
    )

    np.testing.assert_allclose(
        decomposition.left_fluctuation + decomposition.right_fluctuation,
        expected,
        rtol=2e-11,
        atol=2e-11,
    )


def test_wave_family_limiter_preserves_wave_and_fluctuation_shapes():
    system = phx.equations.EulerSystem()
    primitive = jnp.asarray(
        [[1.0, 0.2, 1.0], [0.9, 0.1, 0.9], [0.8, -0.1, 0.8], [1.1, 0.0, 1.2]]
    )
    state = system.primitive_to_conserved(primitive)
    decomposition = phx.discretization.RoeWavePropagationPlan().decompose(
        system, state, jnp.roll(state, -1, axis=0), 0
    )
    limited = phx.discretization.WaveFamilyLimiterPlan("mc").limit(decomposition, 0)

    assert limited.waves.shape == decomposition.waves.shape
    assert limited.left_fluctuation.shape == state.shape
    assert jnp.all(jnp.isfinite(limited.waves))


def test_capacity_scales_hyperbolic_stable_step():
    grid = _periodic_grid((24,))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="capacity-transport",
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x",))
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR("capacity", "state", system, boundaries)
    unit = phx.equations.compile_conservation_problem(problem, discretization, method)
    doubled = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        capacity=2.0 * jnp.ones(discretization.cell_shape),
    )
    state = jnp.ones(discretization.state_shape)

    np.testing.assert_allclose(
        doubled.stable_step(state), 2.0 * unit.stable_step(state), rtol=1e-12
    )


def test_split_and_unsplit_steppers_preserve_constant_multidimensional_state():
    grid = _periodic_grid((8, 6))
    system = phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: (1.0 if axis == 0 else -0.3) * state,
        lambda left, right, axis, args: jnp.full(
            left.shape[:-1], 1.0 if axis == 0 else 0.3
        ),
        system_id="split-transport",
    )
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    problem = phx.equations.ConservationProblemIR(
        "split",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x", "y")),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    state = jnp.ones(discretization.state_shape)
    unsplit = phx.solver.UnsplitFiniteVolumeSSPRK3Plan(compiled.dynamics).advance(
        0.0, state, 0.01
    )
    split = phx.solver.DirectionalSplitFiniteVolumePlan(
        compiled.dynamics, splitting="strang"
    ).advance(0.0, state, 0.01)

    np.testing.assert_allclose(unsplit.state, state, atol=1e-13)
    np.testing.assert_allclose(split.state, state, atol=1e-13)


def test_transverse_solver_returns_finite_opposite_direction_splits():
    system = phx.equations.EulerSystem(2)
    left = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, 0.1, 1.0]]))
    right = system.primitive_to_conserved(jnp.asarray([[0.9, -0.1, 0.0, 0.9]]))
    fluctuation = right - left
    negative, positive = phx.discretization.TransverseWaveSolverPlan().split(
        system, left, right, fluctuation, 1
    )

    assert negative.shape == fluctuation.shape
    assert positive.shape == fluctuation.shape
    assert jnp.all(jnp.isfinite(negative))
    assert jnp.all(jnp.isfinite(positive))
