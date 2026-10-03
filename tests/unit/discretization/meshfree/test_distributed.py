"""Distributed meshfree operators on forced CPU devices.

Run in a dedicated process with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``. Forced CPU devices
prove functional parity with the single-device operator only; accelerator and
multi-host performance require real hardware runs. References are the prepared
single-device ``MeshfreeOperator`` and independent NumPy dense assemblies.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._execution_runtime import ExecutionGroup, ExecutionRuntime
from phydrax.discretization.meshfree import (
    distributed_inner,
    distributed_norm,
    distributed_sum,
    DistributedMeshfreeOperator,
    LocalStencilPolicy,
    MeshfreeFunctional,
    MeshfreeNeighborhoodPlan,
    MeshfreeOperator,
    prepare_local_stencils,
)
from phydrax.discretization.spatial import (
    DistributedNeighborQueryPlan,
    DistributedOwnershipPlan,
    DistributedPointLayout,
    MortonAddressPlan,
)
from phydrax.linalg import ArraySpace, DiagonalPairing
from phydrax.sparse import SparseCoordinateOperator


def _owner_group() -> ExecutionGroup:
    devices = jax.devices()
    if len(devices) < 4 or len(devices) % 4:
        pytest.skip(
            "Run with XLA_FLAGS=--xla_force_host_platform_device_count=4 to "
            "exercise four owners."
        )
    return ExecutionRuntime.current().child_groups(len(devices) // 4)[0]


def _cloud() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(21)
    points = np.concatenate(
        (rng.uniform(0.0, 0.5, (40, 2)), rng.uniform(0.0, 1.0, (40, 2))), axis=0
    )
    # Uneven ownership by angular sector about the domain center.
    angle = np.arctan2(points[:, 1] - 0.5, points[:, 0] - 0.5)
    owners = np.digitize(angle, (-2.0, 0.3, 1.2)).astype(np.int32)
    return points, owners


def _layout(group: ExecutionGroup) -> tuple[np.ndarray, DistributedPointLayout]:
    points, owners = _cloud()
    loads = np.bincount(owners, minlength=4)
    assert loads.max() >= 2 * loads.min()
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10)
    plan = DistributedOwnershipPlan(address, group, int(loads.max()) + 2)
    return points, DistributedPointLayout.from_global(plan, points, owners)


def _laplacian(points: np.ndarray) -> MeshfreeOperator:
    cloud = jnp.asarray(points)
    neighborhood = MeshfreeNeighborhoodPlan(cloud, 12).prepare()
    stencils = prepare_local_stencils(
        neighborhood,
        cloud,
        cloud,
        (MeshfreeFunctional(((2, 0), (0, 2)), (1.0, 1.0), name="laplacian"),),
        LocalStencilPolicy(polynomial_degree=2),
    )
    return MeshfreeOperator(stencils)


def test_distributed_operator_equals_single_device_operator() -> None:
    points, layout = _layout(_owner_group())
    operator = _laplacian(points)
    distributed = DistributedMeshfreeOperator.bind(
        operator.operator, layout, layout, halo_capacity=layout.plan.local_capacity
    )
    assert bool(distributed.evidence.halo.successful)
    assert int(distributed.evidence.halo.halo_columns) > 0
    rng = np.random.default_rng(1)
    values = rng.normal(size=(points.shape[0],))
    np.testing.assert_allclose(
        layout.collect(distributed.apply(layout.distribute(values))),
        operator.apply(values),
        rtol=0,
        atol=1e-10,
    )
    cotangent = rng.normal(size=(points.shape[0], 2))
    np.testing.assert_allclose(
        layout.collect(distributed.transpose_apply(layout.distribute(cotangent))),
        operator.transpose_apply(cotangent),
        rtol=0,
        atol=1e-10,
    )


def test_distributed_duality_adjoint_and_global_conservation() -> None:
    points, layout = _layout(_owner_group())
    count = points.shape[0]
    reference = _laplacian(points).operator
    rng = np.random.default_rng(2)
    source_weights = rng.uniform(0.5, 2.0, count)
    target_weights = rng.uniform(0.5, 2.0, count)
    weighted = SparseCoordinateOperator(
        reference.relation,
        reference.coefficients,
        source=ArraySpace((count,), pairing=DiagonalPairing(jnp.asarray(source_weights))),
        target=ArraySpace((count,), pairing=DiagonalPairing(jnp.asarray(target_weights))),
    )
    distributed = DistributedMeshfreeOperator.bind(
        weighted, layout, layout, halo_capacity=layout.plan.local_capacity
    )
    x = layout.distribute(rng.normal(size=count))
    y = layout.distribute(rng.normal(size=count))
    ax = distributed.apply(x)
    np.testing.assert_allclose(
        distributed_inner(layout, ax, y),
        distributed_inner(layout, x, distributed.transpose_apply(y)),
        rtol=1e-12,
    )
    adjoint = distributed.adjoint_apply(y)
    np.testing.assert_allclose(
        layout.collect(adjoint), weighted.adjoint_mv(layout.collect(y)), atol=1e-10
    )
    np.testing.assert_allclose(
        distributed_inner(layout, ax, y, weights=layout.distribute(target_weights)),
        distributed_inner(layout, x, adjoint, weights=layout.distribute(source_weights)),
        rtol=1e-12,
    )
    coefficients = np.asarray(reference.coefficients)[
        np.asarray(reference.relation.valid)
    ]
    np.testing.assert_allclose(
        distributed_sum(
            layout, distributed.transpose_apply(layout.distribute(np.ones(count)))
        ),
        coefficients.sum(),
        # Laplacian rows sum to zero, so the total is pure cancellation: the
        # tolerance scales with the magnitudes that are being cancelled.
        rtol=0,
        atol=1e-13 * np.abs(coefficients).sum(),
    )


def test_neighbor_rows_bind_without_a_global_operator() -> None:
    points, layout = _layout(_owner_group())
    count = points.shape[0]
    k = 6
    rows = DistributedNeighborQueryPlan(layout.plan, layout.plan, k).query(
        layout, layout, exclude_self=True
    )
    assert bool(rows.evidence.successful)
    coefficients = 1.0 / (1.0 + 100.0 * rows.distance_squared)
    operator = DistributedMeshfreeOperator.from_neighbor_rows(
        rows,
        coefficients,
        layout,
        layout,
        halo_capacity=layout.plan.local_capacity,
        operator_id="inverse-distance-smoother",
    )
    relative = points[:, None, :] - points[None, :, :]
    distance = np.sum(relative * relative, axis=-1)
    np.fill_diagonal(distance, np.inf)
    order = np.lexsort((np.broadcast_to(np.arange(count), distance.shape), distance))
    dense = np.zeros((count, count))
    for target in range(count):
        for source in order[target, :k]:
            dense[target, source] = 1.0 / (1.0 + 100.0 * distance[target, source])
    values = np.random.default_rng(3).normal(size=(count, 3))
    np.testing.assert_allclose(
        layout.collect(operator.apply(layout.distribute(values))),
        dense @ values,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        layout.collect(operator.transpose_apply(layout.distribute(values))),
        dense.T @ values,
        atol=1e-12,
    )


def test_incomplete_halo_is_refused_at_binding() -> None:
    points, layout = _layout(_owner_group())
    with pytest.raises(ValueError, match="halo is incomplete"):
        DistributedMeshfreeOperator.bind(
            _laplacian(points).operator, layout, layout, halo_capacity=1
        )


def test_global_reductions_count_owned_rows_exactly_once() -> None:
    points, layout = _layout(_owner_group())
    rng = np.random.default_rng(4)
    values = rng.normal(size=(points.shape[0], 2))
    weights = rng.uniform(0.1, 1.0, points.shape[0])
    blocked = layout.distribute(values)
    blocked_weights = layout.distribute(weights)
    np.testing.assert_allclose(
        distributed_sum(layout, blocked, weights=blocked_weights),
        (weights[:, None] * values).sum(axis=0),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        distributed_norm(layout, blocked, weights=blocked_weights),
        np.sqrt(np.sum(weights[:, None] * values * values)),
        rtol=1e-13,
    )
