#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import jax.numpy as jnp
import optax
import pytest

import phydrax as phx


def test_diffrax_characteristics_recover_constant_velocity_foot():
    terminal = jnp.asarray([[0.8], [0.2]])
    result = phx.solver.trace_characteristics(
        lambda _time, points, _args: jnp.ones_like(points),
        terminal,
        0.0,
        0.5,
        solver=dfx.Tsit5(),
        rtol=1e-8,
        atol=1e-10,
    )

    assert bool(result.successful)
    assert jnp.allclose(result.foot_points, terminal - 0.5, rtol=1e-6, atol=1e-7)


def test_characteristic_projection_advances_fixed_field_over_time_grid():
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(jnp.asarray(2.0))
    solver = phx.solver.FunctionalSolver(functions={"u": field}, terms=())
    problem = phx.solver.CharacteristicProjectionProblem(
        "u",
        domain.component(),
        phx.domain.PointSampling(
            8,
            layout=phx.domain.SampleLayout((("x",),)),
            design="uniform",
        ),
        lambda _time, points, _args: jnp.ones_like(points),
        problem_id="constant-characteristic-projection",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.1, 0.2]),
        time_id="characteristic-grid",
    )
    result = phx.solver.solve_characteristic_projection(
        solver,
        problem,
        grid,
        inner_num_iter=1,
        optim=optax.sgd(1e-2),
        characteristic_solver=dfx.Euler(),
        characteristic_dt0=0.01,
        log_every=0,
    )

    assert bool(result.successful)
    assert result.completed_steps == 2
    assert len(result.fields) == 3
    assert len(result.traces) == 2
    assert jnp.all(result.projection_losses < 1e-12)
    assert len(result.solver.terms) == 0


def test_characteristic_trace_and_projection_validate_shapes_and_labels():
    with pytest.raises(ValueError, match="increasing"):
        phx.solver.trace_characteristics(
            lambda _time, points, _args: points,
            jnp.ones((2, 1)),
            1.0,
            0.0,
        )
    with pytest.raises(ValueError, match="preserve point shape"):
        phx.solver.trace_characteristics(
            lambda _time, _points, _args: jnp.ones((2, 2)),
            jnp.ones((2, 1)),
            0.0,
            1.0,
            solver=dfx.Euler(),
            dt0=0.1,
        )

    domain = phx.domain.Interval1d(0.0, 1.0)
    with pytest.raises(ValueError, match="coordinate_label"):
        phx.solver.CharacteristicProjectionProblem(
            "u",
            domain.component(),
            phx.domain.PointSampling(4),
            lambda _time, points, _args: points,
            coordinate_label="y",
        )
