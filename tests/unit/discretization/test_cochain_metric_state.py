#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _complex():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    signature = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 1),)
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0,), (4,)),),
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(
        hierarchy
    ).initial_topology()
    return (
        phx.discretization.VariablePatchEntityComplexPlan(
            phx.discretization.BlockHierarchyCapacityPlan(
                ((8, 8),),
                (8,),
            )
        )
        .prepare(topology)[0]
        .complex
    )


def test_cochain_metric_plan_separates_static_topology_from_runtime_metrics():
    complex_ = _complex()
    topology = phx.discretization.PreparedCochainTopology(complex_)
    plan = phx.discretization.CochainMetricPlan(
        topology,
        geometry_family_id="identity-map",
        geometry_layout_id="level-zero",
        coordinate_shapes=((8, 1), (8, 1)),
    )
    stars = (jnp.ones((8,)), jnp.ones((8,)))
    measures = (jnp.ones((8,)), jnp.ones((8,)))
    coordinates = (
        jnp.concatenate((jnp.arange(5.0)[:, None], jnp.zeros((3, 1))), axis=0),
        jnp.concatenate((jnp.arange(4.0)[:, None] + 0.5, jnp.zeros((4, 1))), axis=0),
    )

    first = plan.prepare(
        stars,
        primal_measures=measures,
        coordinates=coordinates,
        revision=1,
    )
    second = plan.prepare(
        (2.0 * stars[0], 3.0 * stars[1]),
        primal_measures=measures,
        coordinates=coordinates,
        revision=2,
    )

    assert bool(first.valid)
    assert first.metric_layout_id == second.metric_layout_id
    assert first.host_snapshot().topology.topology_id == complex_.topology_id


def test_cochain_metric_runtime_validity_is_traceable_without_content_fingerprints():
    complex_ = _complex()
    topology = phx.discretization.PreparedCochainTopology(complex_)
    plan = phx.discretization.CochainMetricPlan(
        topology,
        geometry_family_id="ale-map",
        geometry_layout_id="fixed-connectivity",
    )
    measures = (jnp.ones((8,)), jnp.ones((8,)))

    valid = jax.jit(
        lambda scale: (
            plan.state(
                (scale * jnp.ones((8,)), jnp.ones((8,))),
                primal_measures=measures,
                revision=scale.astype(jnp.int32),
            ).valid
        )
    )

    assert bool(valid(jnp.asarray(1.0)))
    assert not bool(valid(jnp.asarray(-1.0)))


def test_cochain_metric_runtime_ignores_inactive_nan_padding_but_requires_coordinates():
    complex_ = _complex()
    topology = phx.discretization.PreparedCochainTopology(complex_)
    plan = phx.discretization.CochainMetricPlan(
        topology,
        geometry_family_id="mapped",
        geometry_layout_id="fixed",
        coordinate_shapes=((8, 1), (8, 1)),
    )
    masks = tuple(entity.active_mask for entity in complex_.entity_sets)
    stars = tuple(jnp.where(mask, 1.0, jnp.nan) for mask in masks)
    measures = tuple(jnp.where(mask, 1.0, jnp.nan) for mask in masks)
    coordinates = (
        jnp.where(
            masks[0][:, None],
            jnp.arange(8.0)[:, None],
            jnp.nan,
        ),
        jnp.where(
            masks[1][:, None],
            jnp.arange(8.0)[:, None],
            jnp.nan,
        ),
    )

    state = plan.state(
        stars,
        primal_measures=measures,
        coordinates=coordinates,
    )

    assert bool(state.valid)
    with pytest.raises(ValueError, match="coordinates"):
        plan.state(
            stars,
            primal_measures=measures,
            coordinates=(None, None),
        )
