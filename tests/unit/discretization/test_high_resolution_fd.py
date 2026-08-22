#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _sine_cell_averages(edges):
    widths = edges[1:] - edges[:-1]
    return (jnp.cos(2.0 * jnp.pi * edges[:-1]) - jnp.cos(2.0 * jnp.pi * edges[1:])) / (
        2.0 * jnp.pi * widths
    )


@pytest.mark.parametrize("method", ["weno_z", "teno", "mp5"])
def test_uniform_high_resolution_methods_recover_fifth_order_smooth_faces(method):
    errors = []
    for cells in (32, 64, 128):
        edges = jnp.linspace(0.0, 1.0, cells + 1)
        values = _sine_cell_averages(edges)
        reconstruction = phx.discretization.HighResolutionReconstructionPlan(method)

        left, right = reconstruction.reconstruct(values)
        exact = jnp.sin(2.0 * jnp.pi * edges[1:])
        errors.append(
            float(
                jnp.maximum(
                    jnp.sqrt(jnp.mean((left - exact) ** 2)),
                    jnp.sqrt(jnp.mean((right - exact) ** 2)),
                )
            )
        )

    rates = [np.log(errors[index] / errors[index + 1]) / np.log(2.0) for index in range(2)]
    assert rates[-1] > 4.5


def test_nonuniform_weno_z_is_high_order_and_constant_preserving():
    errors = []
    for cells in (24, 48, 96):
        uniform = jnp.linspace(0.0, 1.0, cells + 1)
        edges = uniform**1.15
        values = _sine_cell_averages(edges)
        reconstruction = phx.discretization.NonuniformWENOReconstructionPlan(
            edges,
            method="weno_z",
        )

        left, right = reconstruction.reconstruct(values)
        exact = jnp.sin(2.0 * jnp.pi * edges[1:])
        errors.append(float(jnp.sqrt(jnp.mean((left - exact) ** 2))))
        constant_left, constant_right = reconstruction.reconstruct(jnp.ones((cells,)))

        np.testing.assert_allclose(constant_left, 1.0, rtol=0.0, atol=5e-12)
        np.testing.assert_allclose(constant_right, 1.0, rtol=0.0, atol=5e-12)

    rate = np.log(errors[-2] / errors[-1]) / np.log(2.0)
    assert rate > 4.0


def test_mp5_and_teno_do_not_create_new_scalar_extrema_at_jump():
    values = jnp.concatenate((jnp.ones((32,)), jnp.zeros((32,))))
    for method in ("teno", "mp5"):
        left, right = phx.discretization.HighResolutionReconstructionPlan(method).reconstruct(
            values
        )
        assert jnp.min(left) >= -2e-12
        assert jnp.max(left) <= 1.0 + 2e-12
        assert jnp.min(right) >= -2e-12
        assert jnp.max(right) <= 1.0 + 2e-12


def test_euler_roe_characteristic_projection_roundtrips_face_basis():
    system = phx.discretization.Euler1DSystem()
    primitive = jnp.asarray(
        [[1.0, 0.3, 1.0], [0.8, -0.1, 0.7], [1.2, 0.05, 1.4]]
    )
    state = system.conservative(primitive)
    left = jnp.tile(state, (3, 1))
    right = jnp.roll(left, -1, axis=0)

    left_matrix, right_matrix, eigenvalues = system.eigensystem(left, right)

    identity = jnp.einsum("nij,njk->nik", left_matrix, right_matrix)
    np.testing.assert_allclose(
        identity,
        jnp.broadcast_to(jnp.eye(3), identity.shape),
        rtol=2e-12,
        atol=2e-12,
    )
    assert eigenvalues.shape == left.shape


def test_positivity_limiter_restores_admissible_face_state_without_changing_average():
    system = phx.discretization.Euler1DSystem()
    limiter = phx.discretization.PositivityLimiterPlan(gamma=system.gamma)
    average = system.conservative(jnp.asarray([[1.0, 0.0, 1.0]]))
    invalid_face = jnp.asarray([[-0.2, 3.0, 0.1]])

    limited = limiter.limit(average, invalid_face)

    assert bool(jnp.all(limiter.admissible(limited)))
    direction = invalid_face - average
    ratio = (limited - average) / direction
    np.testing.assert_allclose(ratio[..., 0], ratio[..., 1], rtol=2e-8, atol=2e-8)


def test_rusanov_euler_flux_has_nonpositive_entropy_dissipation():
    system = phx.discretization.Euler1DSystem()
    flux = phx.discretization.EntropyStableEulerFlux(system)
    left = system.conservative(
        jnp.asarray([[1.0, 0.4, 1.0], [0.7, -0.2, 0.8]])
    )
    right = system.conservative(
        jnp.asarray([[0.8, -0.1, 0.6], [1.1, 0.3, 1.3]])
    )

    dissipation = flux.entropy_dissipation(left, right)

    assert jnp.all(dissipation <= 2e-13)


def _sod_state(system, cells):
    x = (jnp.arange(cells) + 0.5) / cells
    primitive = jnp.stack(
        (
            jnp.where(x < 0.5, 1.0, 0.125),
            jnp.zeros_like(x),
            jnp.where(x < 0.5, 1.0, 0.1),
        ),
        axis=-1,
    )
    return system.conservative(primitive)


def test_characteristic_weno_z_evolves_sod_problem_with_positive_conservative_state():
    cells = 160
    system = phx.discretization.Euler1DSystem()
    dynamics = phx.discretization.Euler1DDynamics(
        system,
        phx.discretization.HighResolutionReconstructionPlan(
            "weno_z",
            boundary="outflow",
        ),
        1.0 / cells,
    )
    state = _sod_state(system, cells)
    initial_mass = jnp.sum(state[:, 0]) / cells
    time = jnp.asarray(0.0)

    for _ in range(25):
        dt = 0.35 * dynamics.stable_step(state)
        state = dynamics.ssprk3_step(time, state, dt)
        time = time + dt

    primitive = system.primitive(state)
    assert jnp.min(primitive[:, 0]) > 0.0
    assert jnp.min(primitive[:, 2]) > 0.0
    assert jnp.max(primitive[:, 0]) < 1.1
    assert jnp.min(primitive[:, 0]) > 0.11
    np.testing.assert_allclose(
        jnp.sum(state[:, 0]) / cells,
        initial_mass,
        rtol=0.0,
        atol=2e-10,
    )
