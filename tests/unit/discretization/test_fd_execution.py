#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _periodic_fd(points):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                points,
                endpoint=False,
                periodic=True,
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    return phx.discretization.periodic_finite_difference(grid, accuracy_order=4)


def test_periodic_interior_lowering_removes_per_row_metadata_and_matches_operator():
    discretization = _periodic_fd(1024)
    reference = discretization.operator("d_x_1")
    execution = phx.discretization.lower_stencil_operator(reference)
    x = discretization.grid.axes[0].nodes
    state = jnp.sin(6.0 * jnp.pi * x) + 0.2 * jnp.cos(10.0 * jnp.pi * x)

    result = eqx.filter_jit(execution.mv)(state)

    np.testing.assert_allclose(result, reference.mv(state), rtol=2e-12, atol=2e-12)
    assert execution.execution.report.interior_rows == 1024
    assert execution.execution.report.closure_rows == 0
    assert execution.execution.report.lowered_metadata_bytes < 0.01 * (
        execution.execution.report.canonical_metadata_bytes
    )


def test_bounded_execution_keeps_only_closures_and_preserves_transpose():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(257),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    request = phx.discretization.DerivativeRequest(
        "dx",
        grid,
        "x",
        accuracy_order=4,
    )
    reference = phx.discretization.FiniteDifferencePlan(
        grid,
        (request,),
    ).prepare().operator("dx")
    execution = phx.discretization.lower_stencil_operator(reference)
    source = jnp.sin(2.0 * jnp.pi * grid.axes[0].nodes)
    target = jnp.cos(3.0 * jnp.pi * grid.axes[0].nodes)

    left = jnp.vdot(target, execution.mv(source))
    right = jnp.vdot(execution.transpose_mv(target), source)

    np.testing.assert_allclose(execution.mv(source), reference.mv(source), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-12)
    assert execution.execution.report.closure_rows < 12
    assert execution.execution.report.interior_rows > 240


def test_fused_pipeline_reuses_identical_source_operator_application():
    discretization = _periodic_fd(128)
    program = phx.discretization.StencilProgramPlan(
        discretization,
        ("u", "v"),
        (
            phx.discretization.StencilAssignment("u", "v", "d_x_1", scale=1.0),
            phx.discretization.StencilAssignment("u", "v", "d_x_1", scale=2.0),
        ),
    ).prepare()
    x = discretization.grid.axes[0].nodes
    state = {
        "u": jnp.zeros((128,)),
        "v": jnp.sin(2.0 * jnp.pi * x),
    }

    result = eqx.filter_jit(program)(state)
    expected = 3.0 * discretization.operator("d_x_1").mv(state["v"])

    np.testing.assert_allclose(result["u"], expected, rtol=2e-12, atol=2e-12)
    assert program.report.assignment_count == 2
    assert program.report.unique_application_count == 1
    assert program.report.reused_application_count == 1
    assert program.report.lowered_metadata_bytes < program.report.canonical_metadata_bytes


def _two_dimensional_halo_plan(periodic):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                4,
                endpoint=not periodic,
                periodic=periodic,
            ),
            phx.discretization.UniformAxisSpec(
                5,
                endpoint=not periodic,
                periodic=periodic,
            ),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    requests = (
        phx.discretization.DerivativeRequest("dx", grid, "x"),
        phx.discretization.DerivativeRequest("dy", grid, "y"),
    )
    return phx.discretization.FiniteDifferencePlan(grid, requests).prepare().halo_plan


def test_distributed_schedule_fills_faces_edges_and_corners_with_periodicity():
    schedule = phx.discretization.DistributedHaloSchedule(
        (4, 5),
        (1, 1),
        _two_dimensional_halo_plan(True),
        periodic_axes=(True, True),
    )
    block = jnp.arange(20.0).reshape((1, 1, 4, 5))

    exchanged = schedule.exchange_reference(block)

    assert exchanged.shape == (1, 1, 6, 7)
    assert any(value.codimension == 2 for value in schedule.exchanges)
    np.testing.assert_allclose(exchanged[0, 0, 0, 0], block[0, 0, -1, -1])
    np.testing.assert_allclose(exchanged[0, 0, -1, -1], block[0, 0, 0, 0])
    assert schedule.shard(block[0, 0]).sharding == schedule.sharding


def test_distributed_physical_boundary_slots_are_zero_and_interior_is_explicit():
    schedule = phx.discretization.DistributedHaloSchedule(
        (4, 5),
        (1, 1),
        _two_dimensional_halo_plan(False),
        periodic_axes=(False, False),
    )
    block = jnp.ones((1, 1, 4, 5))

    exchanged = schedule.exchange_reference(block)

    np.testing.assert_allclose(exchanged[0, 0, 0], 0.0)
    np.testing.assert_allclose(exchanged[0, 0, -1], 0.0)
    assert schedule.interior_slices() == (slice(2, 2), slice(2, 3))


def test_compact_metadata_is_independent_of_periodic_interior_size():
    small = phx.discretization.lower_stencil_operator(
        _periodic_fd(128).operator("d_x_2")
    )
    large = phx.discretization.lower_stencil_operator(
        _periodic_fd(1024).operator("d_x_2")
    )

    assert (
        small.execution.report.lowered_metadata_bytes
        == large.execution.report.lowered_metadata_bytes
    )
    assert (
        large.execution.report.canonical_metadata_bytes
        == 8 * small.execution.report.canonical_metadata_bytes
    )
