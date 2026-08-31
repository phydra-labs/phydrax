#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _periodic_core(count=6):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=1e-10
    )
    return finite_volume, operators, momentum, projection


def _taylor_green(discretization):
    x_faces = discretization.face_centers[0]
    y_faces = discretization.face_centers[1]
    return (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]),
    )


def test_named_scalar_buoyancy_compiler_closes_content_and_exchange_ledgers():
    finite_volume, operators, momentum, projection = _periodic_core()
    scalar_problem = phx.discretization.MACScalarProblem(
        (
            phx.discretization.MACScalarTransport(
                "temperature", 0.01, advection="centered"
            ),
            phx.discretization.MACScalarTransport("tracer", 0.0, advection="upwind"),
        )
    )
    transport = scalar_problem.prepare(operators)
    buoyancy = phx.equations.MACBuoyancyLaw(
        jnp.asarray([0.0, -1.0]),
        {"temperature": 0.2},
        references={"temperature": 0.0},
    )
    compiled = phx.equations.compile_mac_scalar_buoyancy(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        momentum,
        projection,
        scalar_problem,
        transport,
        buoyancy,
    )
    cells = finite_volume.cell_centers
    scalars = {
        "temperature": jnp.sin(cells[..., 0]) * jnp.sin(cells[..., 1]),
        "tracer": jnp.cos(cells[..., 0]),
    }
    state = compiled.project_state(_taylor_green(finite_volume), scalars)
    rate = compiled(0.0, state, None)
    diagnostics = compiled.diagnostics(0.0, state)

    assert rate.shape == compiled.state_shape
    assert diagnostics.projection_converged
    assert diagnostics.scalars.success
    assert diagnostics.buoyancy.success
    assert jnp.isfinite(diagnostics.buoyancy.exchange_defect)
    gradient = jax.grad(lambda value: jnp.sum(compiled(0.0, state, value)))(
        jnp.asarray(0.0)
    )
    assert jnp.isfinite(gradient)


def test_variable_density_constant_state_reduces_to_divergence_free_mac_flow():
    finite_volume, operators, momentum, _ = _periodic_core()
    variable = phx.discretization.MACVariableDensityPlan(momentum).prepare()
    projection = phx.solver.MACVariableDensityProjectionPlan(
        operators, tolerance=1e-9, maximum_iterations=200
    )
    compiled = phx.equations.compile_mac_variable_density_flow(
        phx.equations.MACVariableDensityFlowProblem(2, 0.01),
        variable,
        projection,
    )
    density = jnp.ones(finite_volume.cell_shape)
    momentum_state = _taylor_green(finite_volume)
    state = compiled.project_coordinates(compiled.pack_state(density, momentum_state))
    diagnostics = compiled.diagnostics(0.0, state)
    physical = compiled.physical_state(0.0, state)

    assert variable.report.passed
    assert diagnostics.positive
    assert diagnostics.projection_converged
    assert diagnostics.divergence_norm < 1e-7
    assert jnp.min(physical.density) > 0.0
    assert jnp.all(jnp.isfinite(compiled(0.0, state, None)))
