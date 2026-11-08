from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.implicit import AnalyticImplicitProfile
from phydrax.meshing.providers._native_implicit import _PreparedFixedImplicitRoute
from phydrax.meshing.providers._native_sources import NativeImplicitSource
from tools.meshing_qualification.scenarios.native_design import _sphere_source


@pytest.fixture(scope="module")
def fixed_case() -> tuple[Any, ...]:
    return _sphere_source(2)


def test_direct_execution_rejects_replaced_prepared_source(
    fixed_case: tuple[Any, ...],
) -> None:
    geometry, grid, source, plan, _, _ = fixed_case
    profile = AnalyticImplicitProfile(
        geometry,
        plan.coordinate_contract,
        source_id=source.source_id,
        source_revision="foreign-revision",
    )
    replacement = NativeImplicitSource(
        geometry,
        source.source_id,
        "foreign-revision",
        grid=grid,
        authoring_profile=profile,
    )
    edited = eqx.tree_at(lambda value: value.prepared.source, plan, replacement)
    with pytest.raises(ValueError, match="plan|source|binding|revision"):
        edited.execute()


def test_direct_execution_rejects_edited_discovery_witnesses(
    fixed_case: tuple[Any, ...],
) -> None:
    _, _, _, plan, _, _ = fixed_case
    recipe = plan.prepared.discovery_recipe
    old = recipe[-1][-1]
    changed = jnp.zeros_like(old)
    assert not np.array_equal(np.asarray(old), np.asarray(changed))
    edited = eqx.tree_at(
        lambda value: value.prepared.discovery_recipe[-1][-1], plan, changed
    )
    with pytest.raises(ValueError, match="discovery|authority|witness|current"):
        edited.execute()


def test_fixed_preparation_does_not_publish_supplied_volume(
    fixed_case: tuple[Any, ...],
) -> None:
    _, _, _, plan, _, _ = fixed_case
    prepared = plan.prepared
    actual = prepared.require_current()
    domain = np.asarray(actual.volume.domain)
    counterfeit = eqx.tree_at(lambda value: value.volume.domain, actual, domain * 1.1)
    rebound = _PreparedFixedImplicitRoute(
        prepared.source,
        prepared.specification,
        prepared.coordinate_contract,
        counterfeit,
        prepared.surface_plan,
        origin_geometry=prepared.origin_geometry,
        origin_grid=prepared.origin_grid,
        revision_bindings=prepared.revision_bindings,
    )
    assert not hasattr(rebound, "discovery")
    assert np.array_equal(rebound.require_current().volume.domain, domain)
    edited = eqx.tree_at(
        lambda value: value.prepared.discovery_recipe[1], plan, domain * 1.1
    )
    with pytest.raises(ValueError, match="discovery|authority|current"):
        edited.execute()


def test_chained_rebind_rejects_reused_revision_for_new_state(
    fixed_case: tuple[Any, ...],
) -> None:
    geometry, grid, source, first, _, _ = fixed_case
    _, _, _, second, _, _ = _sphere_source(2, radius=0.755, fixed_plan=first)
    index = geometry.schema.index(phx.geometry.ParameterId("design-sphere", "radius"))
    third_geometry = geometry.with_state(
        geometry.state.replace_at(index, jnp.asarray(0.76))
    )
    reused = NativeImplicitSource(
        third_geometry, source.source_id, source.source_revision, grid=grid
    )
    with pytest.raises(ValueError, match="revision"):
        phx.meshing.NativeMeshingProvider(second.options).plan(
            reused,
            first.specification,
            coordinate_contract=second.coordinate_contract,
            fixed_plan=second,
        )
    _, _, returned, _, _, _ = _sphere_source(2, radius=0.75, fixed_plan=second)
    assert returned.source_revision == source.source_revision
