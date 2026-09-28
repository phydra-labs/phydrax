#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx


D = phx.discretization


def _momentum(
    counts: tuple[int, int],
    upper: tuple[float, float],
    sides: Sequence[D.MACBoundarySide],
    /,
    *,
    periodic_x: bool,
) -> D.PreparedMACMomentumOperators:
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(counts[0], periodic=periodic_x),
            D.UniformCellAxisSpec(counts[1]),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [upper[0], upper[1]]]))
    operators = D.MACOperatorPlan(D.FiniteVolumePlan(grid).prepare()).prepare()
    boundaries = D.MACBoundaryPlan(operators, sides).prepare()
    return D.MACMomentumPlan(operators, boundaries=boundaries).prepare()


def _couette_error(
    cells: int, /
) -> tuple[float, phx.solver.MACVariationalViscosityResult]:
    speed, lower_mu, upper_mu = 1.0, 1.0, 0.25
    momentum = _momentum(
        (4, cells),
        (1.0, 1.0),
        (
            D.MACBoundarySide("y", "lower", "no-slip"),
            D.MACBoundarySide(
                "y",
                "upper",
                "no-slip",
                provider=D.MACBoundaryProvider(jnp.asarray([speed, 0.0])),
            ),
        ),
        periodic_x=True,
    )
    discretization = momentum.operators.discretization
    centers = discretization.grid.structured_axes[1].interval_centers
    lower = jnp.broadcast_to(centers < 0.5, discretization.cell_shape)
    face_y = jnp.linspace(0.0, 1.0, cells + 1)
    density = (
        jnp.where(lower, 1.0, 0.5),
        jnp.broadcast_to(jnp.where(face_y < 0.5, 1.0, 0.5), (4, cells + 1)),
    )
    velocity = tuple(jnp.zeros(layout.shape) for layout in discretization.face_layouts)
    plan = phx.solver.MACVariationalViscosityPlan(
        momentum, tolerance=1.0e-10, maximum_iterations=1000
    )
    # One implicit step with a step far beyond the viscous time scale reaches
    # the steady profile to within 1e-9.
    result = plan.solve(
        velocity,
        density,
        jnp.where(lower, lower_mu, upper_mu),
        1.0e9,
        momentum.boundaries.evaluate(0.0),
    )
    stress = speed / (0.5 / lower_mu + 0.5 / upper_mu)
    exact = jnp.where(
        centers < 0.5,
        stress * centers / lower_mu,
        stress * 0.5 / lower_mu + stress * (centers - 0.5) / upper_mu,
    )
    return float(jnp.max(jnp.abs(result.velocity[0] - exact[None, :]))), result


def test_two_layer_couette_converges_first_order_to_the_piecewise_linear_profile() -> (
    None
):
    errors = []
    for cells in (8, 16, 32):
        error, result = _couette_error(cells)
        errors.append(error)
        assert result.successful
        assert jnp.max(jnp.abs(result.velocity[1])) < 1.0e-12
        assert result.dissipation > 0.0
        # At steady state the moving wall's power is dissipated exactly.
        np.testing.assert_allclose(result.wall_power, result.dissipation, rtol=1.0e-6)
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(orders > 0.85)
    assert errors[-1] < 1.0e-2


def _hysing_like() -> tuple[
    D.PreparedMACMomentumOperators,
    tuple[Array, Array],
    tuple[Array, Array],
    Array,
]:
    momentum = _momentum(
        (10, 20),
        (1.0, 2.0),
        (
            D.MACBoundarySide("x", "lower", "free-slip"),
            D.MACBoundarySide("x", "upper", "free-slip"),
            D.MACBoundarySide("y", "lower", "no-slip"),
            D.MACBoundarySide("y", "upper", "no-slip"),
        ),
        periodic_x=False,
    )
    discretization = momentum.operators.discretization
    x = discretization.grid.structured_axes[0].interval_centers
    y = discretization.grid.structured_axes[1].interval_centers
    bubble = (x[:, None] - 0.5) ** 2 + (y[None, :] - 0.5) ** 2 < 0.25**2
    cell_density = jnp.where(bubble, 100.0, 1000.0)
    faces = []
    for axis in range(2):
        moved = jnp.moveaxis(cell_density, axis, 0)
        harmonic = 2.0 / (1.0 / moved[:-1] + 1.0 / moved[1:])
        faces.append(
            jnp.moveaxis(
                jnp.concatenate((moved[:1], harmonic, moved[-1:]), axis=0), 0, axis
            )
        )
    face_x = jnp.linspace(0.0, 1.0, 11)
    face_y = jnp.linspace(0.0, 2.0, 21)
    velocity = (
        1.0e-3 * jnp.sin(jnp.pi * face_x)[:, None] * jnp.sin(0.5 * jnp.pi * y)[None, :],
        3.0e-4 * jnp.cos(3.0 * x)[:, None] * jnp.sin(0.5 * jnp.pi * face_y)[None, :],
    )
    return momentum, (faces[0], faces[1]), velocity, jnp.where(bubble, 1.0, 10.0)


def test_refreshed_high_contrast_wall_stage_matches_a_fresh_plan() -> None:
    momentum, density, velocity, viscosity = _hysing_like()
    stage = momentum.boundaries.evaluate(0.0)
    plan = phx.solver.MACVariationalViscosityPlan(momentum)
    first = plan.solve(velocity, density, viscosity, 5.0e-3, stage)
    refreshed_density = tuple(0.5 * value for value in density)
    refreshed = plan.solve(velocity, refreshed_density, 2.0 * viscosity, 1.0e-2, stage)
    fresh = phx.solver.MACVariationalViscosityPlan(momentum).solve(
        velocity, refreshed_density, 2.0 * viscosity, 1.0e-2, stage
    )

    for result in (first, refreshed, fresh):
        assert result.successful
        assert result.linear_status == int(phx.linalg.LinearSolveStatus.SUCCESS)
        assert result.dissipation > 0.0
        assert result.energy_after < result.energy_before
    for left, right in zip(refreshed.velocity, fresh.velocity, strict=True):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1.0e-15)
    assert not all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(first.velocity, refreshed.velocity, strict=True)
    )


def test_zero_viscosity_returns_the_stage_enforced_input() -> None:
    momentum, density, velocity, viscosity = _hysing_like()
    stage = momentum.boundaries.evaluate(0.0)
    result = phx.solver.MACVariationalViscosityPlan(momentum).solve(
        velocity, density, jnp.zeros_like(viscosity), 5.0e-3, stage
    )

    assert result.successful
    for solved, expected in zip(
        result.velocity, momentum.boundaries.enforce(velocity, stage), strict=True
    ):
        np.testing.assert_allclose(solved, expected, rtol=1.0e-12, atol=0.0)
    assert result.dissipation == 0.0
    np.testing.assert_allclose(result.energy_after, result.energy_before, rtol=1.0e-12)
