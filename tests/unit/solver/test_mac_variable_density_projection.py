#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


STEP = 0.01


def _bubble_case() -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    """Density-1 disk in density-1000 fluid; periodic x, walls in y."""
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(12, periodic=True),
            phx.discretization.UniformCellAxisSpec(20),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 2.0))))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    x = (jnp.arange(12, dtype=jnp.float64) + 0.5) / 12.0
    y = 2.0 * (jnp.arange(20, dtype=jnp.float64) + 0.5) / 20.0
    inside = (x[:, None] - 0.5) ** 2 + (y[None, :] - 0.6) ** 2 < 0.25**2
    density = jnp.where(inside, 1.0, 1000.0)
    face_density = operators.interpolate_inverse_momentum(density)
    y_faces = 2.0 * jnp.arange(21, dtype=jnp.float64) / 20.0
    velocity_x = jnp.broadcast_to(jnp.sin(jnp.pi * y)[None, :], (12, 20))
    velocity_y = jnp.full((12, 21), -0.5) + 0.2 * jnp.cos(2.0 * jnp.pi * x)[:, None]
    velocity_y = velocity_y * (y_faces > 0.0) * (y_faces < 2.0)
    momentum = (face_density[0] * velocity_x, face_density[1] * velocity_y)
    inverse = tuple(1.0 / value for value in face_density)
    return operators, momentum, inverse


def _volume_norm(operators: Any, value: Any) -> float:
    volumes = operators.discretization.cell_volumes
    return float(jnp.sqrt(jnp.sum(volumes * value**2)))


def test_projection_accepts_exactly_when_the_native_pressure_solve_succeeds() -> None:
    operators, momentum, inverse = _bubble_case()
    eps = float(jnp.finfo(operators.pressure_space.dtype).eps)
    rounding = 10.0 * eps * operators.pressure_space.size
    for tolerance in (1.0e-6, 1.0e-10):
        reference = phx.solver.MACVariableDensityProjectionPlan(
            operators, tolerance=tolerance, maximum_iterations=400
        ).project(momentum, inverse, STEP)
        needed = int(reference.linear.diagnostics.iterations)
        outcomes = []
        for iterations in (needed - 2, needed - 1, needed, needed + 5):
            result = phx.solver.MACVariableDensityProjectionPlan(
                operators, tolerance=tolerance, maximum_iterations=iterations
            ).project(momentum, inverse, STEP)
            native = bool(result.linear.successful)
            assert bool(result.successful) == native
            outcomes.append(native)
            if native:
                threshold = tolerance * (
                    float(result.divergence_scale)
                    + _volume_norm(operators, result.compatible_rhs)
                )
                assert float(result.linear.diagnostics.residual_norm) <= threshold
                assert float(result.divergence_norm) <= (
                    threshold + rounding * float(result.divergence_scale)
                )
            else:
                for projected, original in zip(result.momentum, momentum, strict=True):
                    np.testing.assert_array_equal(projected, original)
                np.testing.assert_array_equal(
                    result.divergence_after, result.divergence_before
                )
                np.testing.assert_array_equal(result.pressure_increment, 0.0)
        assert outcomes[-1] and not outcomes[0]


def test_projection_acceptance_is_invariant_to_the_flow_scale() -> None:
    operators, momentum, inverse = _bubble_case()
    plan = phx.solver.MACVariableDensityProjectionPlan(
        operators, tolerance=1.0e-9, maximum_iterations=400
    )
    base = plan.project(momentum, inverse, STEP)
    assert bool(base.successful)
    for scale in (2.0**-20, 2.0**20):
        scaled = plan.project(tuple(scale * value for value in momentum), inverse, STEP)
        assert bool(scaled.successful)
        assert int(scaled.linear.diagnostics.iterations) == int(
            base.linear.diagnostics.iterations
        )
        for scaled_velocity, velocity in zip(scaled.velocity, base.velocity, strict=True):
            np.testing.assert_allclose(
                scaled_velocity, scale * velocity, rtol=1.0e-12, atol=0.0
            )


def test_incompatible_divergence_target_is_refused_after_a_successful_solve() -> None:
    operators, momentum, inverse = _bubble_case()
    plan = phx.solver.MACVariableDensityProjectionPlan(
        operators, tolerance=1.0e-9, maximum_iterations=400
    )
    target = jnp.full(operators.discretization.cell_shape, 0.1)
    result = plan.project(momentum, inverse, STEP, target_divergence=target)

    assert bool(result.linear.successful)
    assert not bool(result.successful)
    np.testing.assert_allclose(
        float(result.compatibility_defect), 0.1 * np.sqrt(2.0), rtol=1.0e-12
    )
    for projected, original in zip(result.momentum, momentum, strict=True):
        np.testing.assert_array_equal(projected, original)
