#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _bounded_grid(points=33):
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(points),),
        axis_names=("xi",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _periodic_grid(points=64):
    return phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                points,
                periodic=True,
                endpoint=False,
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def test_diagonal_norm_sbp_identity_and_boundary_order():
    sbp = phx.discretization.SBPDerivativePlan(_bounded_grid(), "xi").prepare()

    residual = sbp.identity_residual()

    assert jnp.max(jnp.abs(residual)) < 1e-12
    assert sbp.operator.stencil_set.closure_accuracy_order == 1
    assert sbp.operator.stencil_set.interior_accuracy_order == 2


def test_periodic_compact_derivative_is_fourth_order_accurate():
    grid = _periodic_grid()
    derivative = phx.discretization.CompactFirstDerivative(grid)
    nodes = grid.axes[0].nodes
    values = jnp.sin(2.0 * jnp.pi * nodes)

    result = derivative.mv(values)

    assert jnp.max(jnp.abs(result - 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * nodes))) < 2e-4
    assert jnp.allclose(
        jnp.vdot(values, derivative.mv(values)),
        0.0,
        atol=2e-5,
    )


def test_mapped_derivative_preserves_free_stream_and_physical_polynomial():
    grid = _bounded_grid(65)
    mapped = phx.discretization.MappedTensorGridPlan(
        grid,
        lambda reference: jnp.asarray(
            (
                reference[0]
                + 0.05
                * jnp.sin(2.0 * jnp.pi * reference[0])
                / (2.0 * jnp.pi),
            )
        ),
        sbp_order=4,
    ).prepare()
    coordinates = mapped.physical_coordinates[..., 0]

    free_stream = mapped.gradient(jnp.ones(grid.shape))
    polynomial = mapped.gradient(coordinates**2)[..., 0]

    assert jnp.max(jnp.abs(free_stream)) < 1e-12
    assert jnp.max(jnp.abs(polynomial - 2.0 * coordinates)) < 3e-2


def test_mapped_metric_evaluation_is_differentiable_at_fixed_topology():
    grid = _bounded_grid(17)

    def total_jacobian(amplitude):
        _, _, _, jacobian = phx.discretization.evaluate_mapped_metrics(
            grid,
            lambda reference: jnp.asarray(
                (
                    reference[0]
                    + amplitude * jnp.sin(jnp.pi * reference[0]),
                )
            ),
            sbp_order=2,
        )
        return jnp.sum(jacobian)

    value, tangent = jax.jvp(
        total_jacobian,
        (jnp.asarray(0.05),),
        (jnp.asarray(0.01),),
    )
    assert jnp.isfinite(value)
    assert jnp.isfinite(tangent)


def test_distributed_partition_uses_named_sharding_and_explicit_halo_exchange():
    grid = _periodic_grid(8)
    request = phx.discretization.DerivativeRequest("dx", grid, "x")
    finite_difference = phx.discretization.FiniteDifferencePlan(
        grid,
        (request,),
    ).prepare()
    partition = phx.discretization.DistributedStencilPartition(
        (8,),
        0,
        finite_difference.halo_plan,
        periodic=True,
    )
    values = jnp.arange(8.0)
    block = jnp.arange(8.0).reshape((1, 8))

    sharded = partition.shard(values)
    exchanged = partition.exchange_block_halos_1d(block)

    assert sharded.sharding == partition.sharding
    assert exchanged.shape == (1, 10)
    assert jnp.allclose(exchanged[0], jnp.asarray([7.0, 0, 1, 2, 3, 4, 5, 6, 7, 0]))
