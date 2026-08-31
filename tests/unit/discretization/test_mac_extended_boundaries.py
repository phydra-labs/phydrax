#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _operators(*, periodic_x=True, count=6):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic_x),
            phx.discretization.UniformCellAxisSpec(count),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    return finite_volume, phx.discretization.MACOperatorPlan(finite_volume).prepare()


def test_time_dependent_wall_provider_enforces_value_and_rate():
    _, operators = _operators()

    def controlled_wall(time, _coordinates, amplitude):
        value = jnp.asarray([amplitude * jnp.sin(time), 0.0])
        rate = jnp.asarray([amplitude * jnp.cos(time), 0.0])
        return value, rate

    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("y", "lower", "free-slip"),
            phx.discretization.MACBoundarySide(
                "y",
                "upper",
                "no-slip",
                provider=phx.discretization.MACBoundaryProvider(
                    jnp.zeros(2),
                    function=controlled_wall,
                    provider_id="oscillatory-lid",
                ),
            ),
        ),
    ).prepare()
    stage = boundaries.evaluate(jnp.asarray(0.3), jnp.asarray(0.7))
    zero_velocity = tuple(
        jnp.zeros(layout.shape) for layout in operators.discretization.face_layouts
    )
    enforced = boundaries.enforce(zero_velocity, stage)
    rate = boundaries.enforce_rate(zero_velocity, stage)

    assert stage.successful
    np.testing.assert_allclose(stage.values[1][0], 0.7 * jnp.sin(0.3), atol=1e-12)
    np.testing.assert_allclose(stage.rates[1][0], 0.7 * jnp.cos(0.3), atol=1e-12)
    assert boundaries.defect(enforced, stage) < 1e-12
    assert jnp.max(jnp.abs(rate[1])) == 0.0
    gradient = jax.grad(
        lambda amplitude: jnp.sum(boundaries.evaluate(0.3, amplitude).values[1])
    )(jnp.asarray(0.7))
    assert jnp.isfinite(gradient)


def test_pressure_outlet_removes_pressure_gauge_and_accepts_inflow_flux():
    finite_volume, operators = _operators(periodic_x=False)
    zero = phx.discretization.MACBoundaryProvider(jnp.zeros(2))
    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide(
                "x",
                "lower",
                "normal-flux-inflow",
                provider=phx.discretization.MACBoundaryProvider(0.1),
            ),
            phx.discretization.MACBoundarySide(
                "x",
                "upper",
                "pressure-outlet",
                provider=phx.discretization.MACBoundaryProvider(0.0),
            ),
            phx.discretization.MACBoundarySide("y", "lower", "no-slip", provider=zero),
            phx.discretization.MACBoundarySide("y", "upper", "no-slip", provider=zero),
        ),
    )
    prepared = boundaries.prepare()
    stage = prepared.evaluate(0.0)
    projection = phx.solver.MACPressureProjectionPlan(
        operators,
        boundaries=prepared,
        solve_method="iterative",
        tolerance=1e-8,
    )
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    velocity = prepared.enforce(velocity, stage)
    result = projection.project(velocity, 1.0, boundary_stage=stage)

    assert stage.successful
    assert projection.gauge_kind == "none"
    assert projection.compatibility_kind == "unprojected"
    assert result.closure.mass_defect >= 0.0
    assert result.closure.successful
