#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import pytest

import phydrax as phx


def _cell_grid(shape, *, periodic=None):
    periodic = (False,) * len(shape) if periodic is None else periodic
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def _sine_cell_averages(edges):
    widths = edges[1:] - edges[:-1]
    return (
        jnp.cos(2.0 * jnp.pi * edges[:-1])
        - jnp.cos(2.0 * jnp.pi * edges[1:])
    ) / (2.0 * jnp.pi * widths)


@pytest.mark.parametrize("method", ["weno_z", "teno", "mp5"])
def test_high_resolution_reconstruction_retains_fifth_order_smooth_accuracy(method):
    errors = []
    for cells in (32, 64, 128):
        edges = jnp.linspace(0.0, 1.0, cells + 1)
        values = _sine_cell_averages(edges)
        reconstruction = phx.discretization.HighResolutionReconstructionPlan(
            method
        )
        depth = reconstruction.radius
        ghosted = jnp.concatenate((values[-depth:], values, values[:depth]))
        left_ghosted, _ = reconstruction.reconstruct(ghosted)
        left = left_ghosted[depth : depth + cells]
        exact = jnp.sin(2.0 * jnp.pi * edges[1:])
        errors.append(float(jnp.sqrt(jnp.mean((left - exact) ** 2))))
    rate = np.log2(errors[-2] / errors[-1])
    assert rate > 4.5


def test_multidimensional_euler_roundtrip_and_directional_flux_shapes():
    system = phx.equations.EulerSystem(2)
    primitive = jnp.asarray([[1.0, 0.3, -0.1, 1.2], [0.7, -0.2, 0.4, 0.8]])
    state = system.primitive_to_conserved(primitive)

    np.testing.assert_allclose(system.conserved_to_primitive(state), primitive, rtol=1e-12)
    assert system.physical_flux(state, 0).shape == state.shape
    assert system.physical_flux(state, 1).shape == state.shape
    reflected = system.reflect_state(state, 1)
    np.testing.assert_allclose(reflected[..., 2], -state[..., 2])


def test_euler_roe_eigensystem_roundtrips_state_jump_in_two_dimensions():
    system = phx.equations.EulerSystem(2)
    left = system.primitive_to_conserved(jnp.asarray([[1.0, 0.3, 0.1, 1.0]]))
    right = system.primitive_to_conserved(jnp.asarray([[0.8, -0.1, 0.2, 0.7]]))
    left_matrix, right_matrix, eigenvalues = system.eigensystem(left, right, 0)
    jump = right - left
    recovered = oe.contract(
        "...ij,...j->...i",
        right_matrix,
        oe.contract("...ij,...j->...i", left_matrix, jump),
    )

    np.testing.assert_allclose(recovered, jump, rtol=2e-11, atol=2e-11)
    assert eigenvalues.shape == jump.shape


def test_entropy_flux_is_consistent_and_dissipative_variant_has_nonpositive_pairing():
    system = phx.equations.EulerSystem()
    left = system.primitive_to_conserved(
        jnp.asarray([[1.0, 0.4, 1.0], [0.7, -0.2, 0.8]])
    )
    right = system.primitive_to_conserved(
        jnp.asarray([[0.9, 0.1, 0.9], [1.1, 0.3, 1.2]])
    )
    central = phx.discretization.EntropyConservativeEulerFluxPlan()
    stable = phx.discretization.EntropyStableEulerFluxPlan()

    equal = central.face_flux(system, left, left, 0)
    np.testing.assert_allclose(equal.normal_flux, system.physical_flux(left, 0), rtol=1e-12)
    assert jnp.all(stable.entropy_dissipation(system, left, right) <= 2e-13)


def test_characteristic_weno_euler_step_preserves_positive_sod_state_and_mass():
    cells = 120
    grid = _cell_grid((cells,))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    reconstruction = phx.discretization.HighResolutionReconstructionPlan(
        "weno_z"
    )
    characteristic = phx.discretization.CharacteristicReconstructionPlan(
        reconstruction,
        phx.discretization.CharacteristicSystem(
            lambda left, right, args: system.eigensystem(left, right, 0, args),
            system_id=system.system_id,
        ),
    )
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,))
    method = phx.discretization.FiniteVolumeMethodPlan(
        characteristic,
        phx.discretization.HLLCFluxPlan(),
        positivity=phx.discretization.ConvexStateLimiterPlan(),
    )
    problem = phx.equations.ConservationProblemIR("sod", "state", system, boundaries)
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    x = grid.structured_axes[0].interval_centers
    primitive = jnp.stack(
        (
            jnp.where(x < 0.5, 1.0, 0.125),
            jnp.zeros_like(x),
            jnp.where(x < 0.5, 1.0, 0.1),
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    initial_mass = jnp.sum(discretization.cell_volumes * state[:, 0])
    stepper = phx.solver.UnsplitFiniteVolumeSSPRK3Plan(compiled.dynamics)
    time = jnp.asarray(0.0)
    for _ in range(10):
        dt = compiled.stable_step(state, cfl=0.25)
        result = stepper.advance(time, state, dt)
        state, time = result.state, result.time

    assert jnp.all(system.admissible(state))
    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes * state[:, 0]),
        initial_mass,
        atol=2e-10,
    )


def test_multispecies_and_mhd_fluxes_preserve_declared_components():
    multispecies = phx.equations.MultispeciesEulerSystem((1.4, 1.67))
    primitive = jnp.asarray([[0.6, 0.4, 0.2, 1.0]])
    multispecies_state = multispecies.primitive_to_conserved(primitive)
    assert multispecies.physical_flux(multispecies_state, 0).shape == multispecies_state.shape
    assert jnp.all(multispecies.admissible(multispecies_state))

    mhd = phx.equations.IdealMHDSystem()
    mhd_primitive = jnp.asarray([[1.0, 0.1, 0.0, 0.0, 1.0, 0.75, 0.1, 0.0]])
    mhd_state = mhd.primitive_to_conserved(mhd_primitive)
    flux = mhd.physical_flux(mhd_state, 0)
    np.testing.assert_allclose(flux[..., 5], 0.0, atol=0.0)
    assert jnp.all(mhd.admissible(mhd_state))


def test_unsplit_two_dimensional_scalar_residual_preserves_periodic_mass():
    grid = _cell_grid((18, 14), periodic=(True, True))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    velocity = (0.7, -0.2)
    system = phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: velocity[axis] * state,
        lambda left, right, axis, args: jnp.full(left.shape[:-1], abs(velocity[axis])),
        system_id="two-dimensional-transport",
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x", "y"))
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "two-dimensional-transport", "state", system, boundaries
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    x = grid.structured_axes[0].interval_centers
    y = grid.structured_axes[1].interval_centers
    state = (
        jnp.sin(2.0 * jnp.pi * x)[:, None]
        + 0.3 * jnp.cos(2.0 * jnp.pi * y)[None, :]
    )[..., None]

    residual = compiled(0.0, state)
    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes[..., None] * residual), 0.0, atol=2e-11
    )


def test_nonuniform_weno_prepares_ghost_geometry_for_bounded_faces():
    edges = jnp.asarray([0.0, 0.08, 0.2, 0.38, 0.62, 0.82, 1.0])
    widths = edges[1:] - edges[:-1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    axis = phx.discretization.AxisDiscretization(
        nodes=centers,
        quad_weights=widths,
        basis="uniform",
        periodic=False,
        primary_entity="interval",
        bounds=jnp.asarray([0.0, 1.0]),
    )
    grid = phx.discretization.PreparedTensorGrid(
        (axis,), axis_names=("x",)
    )
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="bounded-nonuniform-advection",
    )
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    problem = phx.equations.ConservationProblemIR(
        "bounded-nonuniform-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.NonuniformWENOReconstructionPlan(edges),
        phx.discretization.RusanovFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(
        problem, discretization, method
    )
    state = jnp.ones(discretization.state_shape)
    fluxes, _ = compiled.face_fluxes(0.0, state)

    assert fluxes[0].shape == (7, 1)
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=1e-12)
