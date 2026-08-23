#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _cell_grid(points, dimension=1, *, periodic=False):
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(points, periodic=periodic)
            for _ in range(dimension)
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))


def test_red_black_and_line_smoothers_contract_anisotropic_diffusion_residual():
    grid = _cell_grid(24, 2)
    coefficient = jnp.broadcast_to(jnp.asarray((50.0, 1.0)), grid.shape + (2,))
    boundaries = {axis: ("dirichlet", "dirichlet") for axis in grid.axis_names}
    diffusion = phx.discretization.ConservativeDiffusionPlan(
        grid,
        boundaries=boundaries,
    ).prepare(coefficient)
    x = grid.axes[0].nodes[:, None]
    y = grid.axes[1].nodes[None, :]
    exact = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    ratios = {}
    for smoother, line_axis in (
        ("jacobi", None),
        ("red_black", None),
        ("line", "x"),
    ):
        multigrid = phx.discretization.StructuredMultigridPlan(
            diffusion,
            smoother=smoother,
            line_axis=line_axis,
            minimum_coarse_points=4,
        ).prepare()
        operator = multigrid.level_operators[0]
        rhs = operator.mv(exact)
        correction = multigrid.apply(rhs)
        ratios[smoother] = float(
            jnp.linalg.norm(rhs - operator.mv(correction)) / jnp.linalg.norm(rhs)
        )

    assert ratios["red_black"] < 0.5
    assert ratios["line"] < ratios["jacobi"]


def test_multi_axis_collective_schedule_exposes_mesh_permutations_and_corner_routes():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(8, endpoint=False, periodic=True),
            phx.discretization.UniformAxisSpec(8, endpoint=False, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    fd = phx.discretization.FiniteDifferencePlan(
        grid,
        (
            phx.discretization.DerivativeRequest("dx", grid, "x"),
            phx.discretization.DerivativeRequest("dy", grid, "y"),
        ),
    ).prepare()
    schedule = phx.discretization.DistributedHaloSchedule(
        grid.shape,
        (1, 1),
        fd.halo_plan,
        periodic_axes=(True, True),
    )

    assert schedule.permutation(0, 1) == ((0, 0),)
    assert schedule.permutation(1, -1) == ((0, 0),)
    assert any(route.codimension == 2 for route in schedule.exchanges)



def test_variable_density_projection_and_extended_compatible_systems():
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid(3, 3))
    velocity = jnp.sin(jnp.arange(bridge.cochain.cell_counts[1], dtype=float))
    density = 1.0 + 0.2 * jnp.cos(jnp.arange(bridge.cochain.cell_counts[0], dtype=float))
    projection = phx.solver.CompatibleVariableDensityProjection(bridge)

    projected = projection.project(velocity, density)

    assert jnp.linalg.norm(projected.divergence_after) < 1e-9

    state_size = bridge.cochain.cell_counts[0]
    displacement = jnp.sin(jnp.arange(state_size, dtype=float) / 5.0)
    rate = jnp.cos(jnp.arange(state_size, dtype=float) / 7.0)
    scalar = jnp.sin(jnp.arange(state_size, dtype=float) / 9.0)
    poro = phx.solver.CompatiblePoroelasticDynamics(bridge)
    thermo = phx.solver.CompatibleThermoelasticDynamics(bridge)
    poro_drift = poro.drift(
        phx.solver.CompatiblePoroelasticState(displacement, rate, scalar)
    )
    thermo_drift = thermo.drift(
        phx.solver.CompatibleThermoelasticState(displacement, rate, scalar)
    )
    assert jnp.all(jnp.isfinite(poro_drift.pressure))
    assert jnp.all(jnp.isfinite(thermo_drift.temperature))


def test_compatible_mhd_induction_preserves_discrete_magnetic_divergence():
    bridge = phx.discretization.StructuredCochainBridge(_cell_grid(3, 3))
    electric = jnp.sin(jnp.arange(bridge.cochain.cell_counts[1], dtype=float))
    magnetic = bridge.exterior_derivative(1, electric)
    dynamics = phx.solver.CompatibleIdealMHDInductionDynamics(
        bridge,
        lambda time, magnetic_state, args: electric,
    )
    state = dynamics.pack(magnetic)

    stepped = dynamics.step(0.0, state, 0.01)

    np.testing.assert_allclose(dynamics.magnetic_constraint(stepped), 0.0, atol=2e-12)


def test_precision_and_resource_preflight_enforce_memory_budget():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(128, endpoint=False, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    precision = phx.discretization.FDExecutionPrecisionPolicy(
        coefficient_dtype=jnp.float32,
        field_dtype=jnp.float32,
        accumulation_dtype=jnp.float64,
        certification_dtype=jnp.float64,
    )
    fd = phx.discretization.periodic_finite_difference(
        grid,
        accuracy_order=4,
        precision=precision,
    )
    lowered = phx.discretization.lower_stencil_operator(fd.operator("d_x_1"))
    plan = phx.discretization.FDExecutionPreflightPlan(
        grid,
        field_count=3,
        halo_widths=((2, 2),),
        operators=(lowered,),
        temporary_fields=4,
        checkpoint_copies=2,
        precision=precision,
        memory_budget_bytes=1_000_000,
    )

    estimate = plan.estimate()

    assert estimate.fits_budget
    assert (
        estimate.stencil_metadata_bytes == lowered.execution.report.lowered_metadata_bytes
    )
    assert estimate.total_bytes < 1_000_000

    with pytest.raises(ValueError, match="exceeding"):
        phx.discretization.FDExecutionPreflightPlan(
            grid,
            field_count=3,
            precision=precision,
            memory_budget_bytes=100,
        ).estimate()
