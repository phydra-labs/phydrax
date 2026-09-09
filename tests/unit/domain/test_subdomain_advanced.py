#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def test_multidimensional_nonuniform_cover_has_exact_faces_and_pou_calculus():
    domain = phx.domain.HyperRectangle([0.0, -1.0], [1.0, 1.0])
    axes = (
        phx.domain.AxisPartition([0.0, 0.25, 1.0], overlap_fraction=0.1),
        phx.domain.AxisPartition([-1.0, 0.0, 1.0], overlap_fraction=0.2),
    )
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        axis_partitions=axes,
    )

    assert len(cover.patches) == 4
    assert len(cover.pairings) == 4
    assert cover.maximum_overlap == 4
    first = cover.patches[0].domain
    np.testing.assert_allclose(first.lower, [0.0, -1.0])
    np.testing.assert_allclose(first.upper, [0.275, 0.2])

    local = {
        patch.patch_id: patch.domain.Function("x")(lambda x: x[0] ** 2 + 2.0 * x[1])
        for patch in cover.patches
    }
    family = phx.domain.LocalFieldFamily("u", cover, local)
    field = phx.domain.partition_of_unity_field(family)
    points = jnp.asarray([[0.1, -0.7], [0.24, -0.05], [0.6, 0.8]])
    batch = domain.component().points({"x": points})

    np.testing.assert_allclose(
        field(batch).data,
        points[:, 0] ** 2 + 2.0 * points[:, 1],
        atol=1.0e-12,
    )
    pairing = cover.pairings[0]
    interface_batch = pairing.component.sample(
        phx.domain.PointSampling(8),
        key=jr.key(1),
    )
    evidence = pairing.audit(
        interface_batch,
        cover.patch(pairing.left_patch_id),
        cover.patch(pairing.right_patch_id),
    )
    assert evidence.verified
    np.testing.assert_allclose(
        pairing.trace(family.field(pairing.left_patch_id), side="left")(
            interface_batch
        ).data,
        pairing.trace(family.field(pairing.right_patch_id), side="right")(
            interface_batch
        ).data,
    )


def test_periodic_axis_builds_explicit_periodic_pairing():
    domain = phx.domain.TimeInterval(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "t",
        3,
        periodic=True,
    )

    assert [pairing.topology for pairing in cover.pairings] == [
        "shared-interface",
        "shared-interface",
        "periodic-interface",
    ]
    periodic = cover.pairings[-1]
    batch = periodic.component.sample(phx.domain.PointSampling(4), key=jr.key(2))
    left = cover.patch(periodic.left_patch_id).domain.Function("t")(lambda t: t)
    right = cover.patch(periodic.right_patch_id).domain.Function("t")(lambda t: t)
    np.testing.assert_allclose(periodic.trace(left, side="left")(batch).data, 1.0)
    np.testing.assert_allclose(periodic.trace(right, side="right")(batch).data, 0.0)

    wrapped_axis = phx.domain.AxisPartition(
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        overlap_fraction=0.2,
        periodic=True,
    )
    wrapped = phx.domain.cartesian_subdomain_cover(
        domain,
        "t",
        axis_partitions=(wrapped_axis,),
    )
    family = phx.domain.LocalFieldFamily(
        "periodic",
        wrapped,
        {
            patch.patch_id: patch.domain.Function("t")(
                lambda t: jnp.sin(2.0 * jnp.pi * t)
            )
            for patch in wrapped.patches
        },
    )
    field = phx.domain.partition_of_unity_field(family)
    np.testing.assert_allclose(
        field.func(jnp.asarray(0.99)),
        jnp.sin(2.0 * jnp.pi * 0.99),
    )
    assert wrapped.patches[0].domain.start < 0.0
    assert wrapped.patches[-1].domain.end > 1.0


def test_ownership_and_prepared_routes_cover_every_point_without_truncation():
    domain = phx.domain.HyperRectangle([0.0, 0.0], [1.0, 1.0])
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        (2, 2),
        overlap_fraction=0.2,
    )
    ownership = phx.domain.cover_integration_ownership(cover)
    routing = phx.domain.prepare_field_routing(cover, ownership=ownership)
    points = domain.component().points(
        {"x": jnp.asarray([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])}
    )

    assert ownership.audit(points).verified
    routing.validate(points)
    active = routing.active_indices(points)
    assert tuple(active[0]) == (0, -1, -1, -1)
    assert set(np.asarray(active[1])) == {0, 1, 2, 3}
    assert len(routing.colors) == 4


def test_adaptive_refinement_transaction_preserves_the_ambient_field():
    domain = phx.domain.Interval1d(0.0, 1.0)
    axis = phx.domain.AxisPartition([0.0, 0.5, 1.0], overlap_fraction=0.2)
    source_cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        axis_partitions=(axis,),
    )
    source = phx.domain.LocalFieldFamily(
        "u",
        source_cover,
        {
            patch.patch_id: patch.domain.Function("x")(lambda x: x[0] ** 2)
            for patch in source_cover.patches
        },
    )
    refined_axis = phx.solver.refine_axis_partition(axis, 0)
    candidate = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        axis_partitions=(refined_axis,),
    )
    points = domain.component().points({"x": jnp.linspace(0.0, 1.0, 65)[:, None]})

    transaction = phx.solver.prepare_adaptive_topology_transaction(
        source,
        candidate,
        points,
    )

    assert transaction.evidence.accepted
    assert transaction.evidence.maximum_transfer_error <= 1.0e-12
    assert len(transaction.commit().fields) == 3


def test_partition_adapter_and_trainable_coarsening_preserve_topology_contracts():
    domain = phx.domain.Interval1d(0.0, 1.0)
    axis = phx.domain.AxisPartition([0.0, 0.2, 0.7, 1.0], overlap_fraction=0.1)
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        axis_partitions=(axis,),
    )
    partition = phx.discretization.CellPartition([0, 1, 2], 3)
    trainable = phx.solver.TrainableAxisPartition(
        axis,
        minimum_fraction=0.05,
    )

    assert phx.domain.validate_cell_partition_cover(partition, cover).verified
    assert trainable.evidence().verified
    np.testing.assert_allclose(trainable.materialize().boundaries, axis.boundaries)
    assert jnp.all(
        jnp.isfinite(
            jax.grad(lambda logits: jnp.var(0.05 + 0.85 * jax.nn.softmax(logits)))(
                trainable.width_logits
            )
        )
    )
    assert phx.solver.coarsen_axis_partition(axis, 1).boundaries == (
        0.0,
        0.7,
        1.0,
    )
