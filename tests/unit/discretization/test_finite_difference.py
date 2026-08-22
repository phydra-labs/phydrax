#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import jax.numpy as jnp

import phydrax as phx


def _bounded_grid(points=17):
    plan = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(points),),
        axis_names=("x",),
    )
    return plan.prepare(jnp.asarray([[0.0], [1.0]]))


def _periodic_grid(points=32):
    plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                points,
                periodic=True,
                endpoint=False,
            ),
        ),
        axis_names=("x",),
    )
    return plan.prepare(jnp.asarray([[0.0], [1.0]]))


def test_fornberg_coefficients_recover_classical_centered_stencils():
    first = phx.discretization.StencilCoefficientPlan(
        [-1.0, 0.0, 1.0],
        0.0,
        1,
        2,
    )
    second = phx.discretization.StencilCoefficientPlan(
        [-1.0, 0.0, 1.0],
        0.0,
        2,
        2,
    )

    assert jnp.allclose(first.weights, jnp.asarray([-0.5, 0.0, 0.5]))
    assert jnp.allclose(second.weights, jnp.asarray([1.0, -2.0, 1.0]))
    assert jnp.max(jnp.abs(first.moment_residuals[:3])) < 1e-12
    assert jnp.max(jnp.abs(second.moment_residuals[:4])) < 1e-12


def test_tensor_grid_plan_prepares_support_without_calculus():
    grid = _bounded_grid()

    assert isinstance(grid, phx.discretization.PreparedTensorGrid)
    assert grid.shape == (17,)
    assert grid.centered_location.offsets == (Fraction(0, 1),)
    assert not hasattr(grid, "partial_derivative")


def test_bounded_finite_difference_uses_explicit_one_sided_closures():
    grid = _bounded_grid()
    request = phx.discretization.DerivativeRequest(
        "dx",
        grid,
        "x",
        derivative_order=1,
        accuracy_order=2,
    )
    prepared = phx.discretization.FiniteDifferencePlan(grid, (request,)).prepare()
    nodes = grid.axes[0].nodes
    values = nodes**2

    derivative = prepared.operator("dx").mv(values)

    assert prepared.stencil("dx").kind == "one_sided"
    assert jnp.allclose(derivative, 2.0 * nodes, atol=2e-5)
    assert prepared.aggregate_footprint.lower == (2,)
    assert prepared.aggregate_footprint.upper == (2,)
    assert prepared.halo_plan.lower_widths == (2,)
    assert prepared.halo_plan.upper_widths == (2,)
    assert prepared.halo_plan.physical_boundaries[0].realization == "closure"


def test_direct_second_derivative_is_polynomially_exact():
    grid = _bounded_grid()
    request = phx.discretization.DerivativeRequest(
        "dxx",
        grid,
        "x",
        derivative_order=2,
        accuracy_order=2,
    )
    prepared = phx.discretization.FiniteDifferencePlan(grid, (request,)).prepare()
    nodes = grid.axes[0].nodes

    derivative = prepared.operator("dxx").mv(nodes**3)
    stencil = prepared.stencil("dxx").stencil

    assert jnp.allclose(derivative, 6.0 * nodes, atol=3e-4)
    assert all(
        plan.derivative_order == 2
        for plan in prepared.stencil("dxx").stencil.coefficient_plans
    )
    assert jnp.array_equal(
        jnp.count_nonzero(stencil.valid, axis=1)[jnp.asarray([0, 1, -1])],
        jnp.asarray([4, 3, 4]),
    )
    assert jnp.any(~jnp.isfinite(jnp.where(stencil.valid, 0.0, stencil.weights)))
    assert stencil.row_reports[0].kind == "lower_closure"
    assert stencil.row_reports[-1].kind == "upper_closure"
    assert jnp.all(jnp.isfinite(derivative))


def test_periodic_stencil_and_coordinate_transpose_satisfy_dot_product_identity():
    grid = _periodic_grid()
    request = phx.discretization.DerivativeRequest(
        "dx",
        grid,
        "x",
        derivative_order=1,
        accuracy_order=4,
    )
    prepared = phx.discretization.FiniteDifferencePlan(grid, (request,)).prepare()
    operator = prepared.operator("dx")
    nodes = grid.axes[0].nodes
    left = jnp.sin(2.0 * jnp.pi * nodes)
    right = jnp.cos(4.0 * jnp.pi * nodes)

    lhs = jnp.vdot(right, operator.mv(left))
    rhs = jnp.vdot(operator.transpose_mv(right), left)

    assert prepared.stencil("dx").kind == "periodic"
    assert jnp.allclose(lhs, rhs, atol=2e-5)
    assert (
        jnp.max(jnp.abs(operator.mv(left) - 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * nodes)))
        < 2e-3
    )


def test_periodic_fd_laplacian_exposes_transform_diagonal_direct_solve():
    grid = _periodic_grid()
    request = phx.discretization.DerivativeRequest(
        "dxx",
        grid,
        "x",
        derivative_order=2,
        accuracy_order=2,
    )
    prepared = phx.discretization.FiniteDifferencePlan(grid, (request,)).prepare()
    representation = prepared.transform_diagonalization("dxx")
    direct = phx.linalg.TransformDiagonalSolvePlan(
        representation,
        compatibility="error",
        gauge="zero_mean",
    ).prepare()
    nodes = grid.axes[0].nodes
    right_hand_side = jnp.sin(2.0 * jnp.pi * nodes)

    result = direct.solve(right_hand_side)
    expected = -right_hand_side / (4.0 * jnp.pi**2)

    assert result.converged
    assert jnp.allclose(jnp.real(result.value), expected, rtol=5e-3, atol=5e-4)
    assert jnp.max(jnp.abs(jnp.imag(result.value))) < 1e-6


def test_center_to_face_request_has_distinct_source_and_target_spaces():
    grid = _bounded_grid()
    face_location = grid.location(((1, 2),))
    request = phx.discretization.DerivativeRequest(
        "center_to_face",
        grid,
        "x",
        derivative_order=1,
        accuracy_order=2,
        target_location=face_location,
    )
    prepared = phx.discretization.FiniteDifferencePlan(grid, (request,)).prepare()
    operator = prepared.operator("center_to_face")
    nodes = grid.axes[0].nodes
    targets = grid.layout_at(face_location).coordinates_by_axis[0]

    derivative = operator.mv(nodes**2)

    assert operator.source.space_id != operator.target.space_id
    assert derivative.shape == (16,)
    assert jnp.allclose(derivative, 2.0 * targets, atol=2e-5)


def test_patch_kernels_support_vectorized_regions_without_view_index_matrices():
    plan = phx.discretization.PatchKernelPlan(
        (3,),
        (
            lambda patch, args: jnp.sum(patch),
            lambda patch, args: jnp.max(patch),
        ),
    )
    prepared = plan.prepare((5,))
    indices = jnp.asarray([0, 1, 0], dtype=jnp.int32)

    result = prepared(jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]), kernel_indices=indices)

    assert jnp.allclose(result, jnp.asarray([6.0, 4.0, 12.0]))


def test_ordered_patch_kernel_exposes_causal_scan_semantics():
    sweep = phx.discretization.OrderedPatchKernelPlan(
        3,
        lambda patch, args: jnp.sum(patch),
    )

    result = sweep(jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert jnp.allclose(result, jnp.asarray([3.0, 8.0, 15.0, 24.0, 29.0]))


def test_stencil_program_compiles_named_fields_to_packed_dynamics():
    grid = _periodic_grid()
    request = phx.discretization.DerivativeRequest(
        "dx",
        grid,
        "x",
        derivative_order=1,
        accuracy_order=4,
    )
    discretization = phx.discretization.FiniteDifferencePlan(
        grid,
        (request,),
        field_name="u",
    ).prepare()
    program = phx.discretization.StencilProgramPlan(
        discretization,
        ("u",),
        (phx.discretization.StencilAssignment("u", "u", "dx"),),
    ).prepare()
    compiled = phx.equations.compile_stencil_dynamics(program)
    values = jnp.sin(2.0 * jnp.pi * grid.axes[0].nodes)
    state = compiled.layout.pack({"u": values})

    derivative = compiled.drift(jnp.asarray(0.0), state, None)

    assert derivative.shape == compiled.layout.state_shape
    assert (
        jnp.max(
            jnp.abs(
                derivative[..., 0]
                - 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * grid.axes[0].nodes)
            )
        )
        < 2e-3
    )
    assert (
        compiled.discretization_bundle.record(discretization.key).artifact_id
        == discretization.prepared_id
    )
