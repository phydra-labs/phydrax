#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.units import METER


@pytest.fixture
def surface_preparation() -> tuple[
    phx.geometry.CompiledGeometry, phx.geometry.ImplicitSurfacePlan
]:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.75, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(9) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))
    plan = phx.geometry.discover_implicit_surface(
        geometry,
        grid,
        policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            maximum_intersection_pairs=500_000,
        ),
        source_id="sphere-surface",
    )
    return geometry, plan


def test_surface_restore_preserves_data_and_source_binding(
    surface_preparation: tuple[
        phx.geometry.CompiledGeometry, phx.geometry.ImplicitSurfacePlan
    ],
) -> None:
    geometry, plan = surface_preparation
    restored = phx.geometry.ImplicitSurfacePlan.from_stored_fields(
        plan, geometry=geometry
    )

    assert restored is not plan
    assert restored.kernel is geometry.kernel
    assert restored.schema is geometry.schema
    assert restored.tolerance is geometry.tolerance
    assert restored.projection is plan.projection
    assert restored.policy is plan.policy
    assert (restored.source_id, restored.topology_id, restored.plan_id) == (
        plan.source_id,
        plan.topology_id,
        plan.plan_id,
    )
    original_leaves, original_structure = jax.tree_util.tree_flatten(plan)
    restored_leaves, restored_structure = jax.tree_util.tree_flatten(restored)
    assert restored_structure == original_structure
    for original, retained in zip(original_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(retained, original)


def test_surface_restore_revalidates_retained_regularization(
    surface_preparation: tuple[
        phx.geometry.CompiledGeometry, phx.geometry.ImplicitSurfacePlan
    ],
) -> None:
    geometry, plan = surface_preparation
    invalid = eqx.tree_at(
        lambda value: value.qef_regularization,
        plan,
        jnp.zeros_like(plan.qef_regularization),
    )
    with pytest.raises(ValueError, match="qef_regularization"):
        phx.geometry.ImplicitSurfacePlan.from_stored_fields(invalid, geometry=geometry)
