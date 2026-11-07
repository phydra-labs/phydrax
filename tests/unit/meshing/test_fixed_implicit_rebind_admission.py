from typing import Any

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.meshing._contracts import SurfaceMeshingSpec
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers._native_implicit import (
    _PreparedFixedImplicitRoute,
    _resolve_discovery,
)
from phydrax.meshing.providers._native_sources import NativeImplicitSource
from tools.meshing_qualification.scenarios.native_design import _sphere_source


@pytest.fixture(scope="module")
def fixed_case() -> tuple[Any, ...]:
    return _sphere_source(2)


def test_rebind_refuses_changed_state_under_same_revision(fixed_case: tuple[Any, ...]) -> None:
    geometry, grid, source, plan, _, _ = fixed_case
    index = geometry.schema.index(phx.geometry.ParameterId("design-sphere", "radius"))
    changed = geometry.with_state(geometry.state.replace_at(index, jnp.asarray(0.755)))
    stale = NativeImplicitSource(changed, source.source_id, source.source_revision, grid=grid)
    with pytest.raises(ValueError, match="revision"):
        phx.meshing.NativeMeshingProvider(plan.options).plan(
            stale, plan.specification, coordinate_contract=plan.coordinate_contract, fixed_plan=plan
        )


def test_rebind_refuses_different_static_kernel_contract(fixed_case: tuple[Any, ...]) -> None:
    geometry, grid, source, plan, _, _ = fixed_case
    original = geometry.kernel
    kernel = type(original)(
        original.center, original.radius, dimension=original.dimension,
        source_id="foreign-static-kernel",
    )
    changed = phx.geometry.CompiledGeometry(kernel, geometry.state, tolerance=geometry.tolerance)
    counterfeit = NativeImplicitSource(changed, source.source_id, "counterfeit-revision", grid=grid)
    scope = phx.meshing.MeshingScope(
        counterfeit.source_id, counterfeit.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY, 2,
        counterfeit.scope_entity_set_id(2), counterfeit.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        plan.specification.target, scope,
        size_controls=(UniformSizeControl(scope, 0.35, strength=SizeControlStrength.SOFT),),
    )
    with pytest.raises(ValueError, match="kernel"):
        phx.meshing.NativeMeshingProvider(plan.options).plan(
            counterfeit, specification, coordinate_contract=plan.coordinate_contract, fixed_plan=plan
        )


def test_rebind_refuses_tree_edited_fixed_plan(fixed_case: tuple[Any, ...]) -> None:
    _, _, source, plan, _, _ = fixed_case
    changed = eqx.tree_at(
        lambda value: value.prepared.surface_plan.qef_regularization,
        plan,
        plan.prepared.surface_plan.qef_regularization * 1.01,
    )
    assert changed.prepared.surface_plan.plan_id == plan.prepared.surface_plan.plan_id
    with pytest.raises(ValueError, match="plan|preparation|current"):
        phx.meshing.NativeMeshingProvider(plan.options).plan(
            source, plan.specification, coordinate_contract=plan.coordinate_contract, fixed_plan=changed
        )


def test_soft_size_is_not_a_hard_fidelity_tolerance(fixed_case: tuple[Any, ...]) -> None:
    _, _, source, plan, _, _ = fixed_case
    old = plan.specification
    control = UniformSizeControl(old.scope, 0.2, strength=SizeControlStrength.SOFT)
    specification = SurfaceMeshingSpec(old.target, old.scope, size_controls=(control,))
    discovery = _resolve_discovery(source, specification, None)
    prepared = _PreparedFixedImplicitRoute(
        source, specification, plan.coordinate_contract,
        discovery, plan.prepared.surface_plan,
    )
    construction = prepared.execute()
    tolerance = construction.certification.fidelity_tolerance
    assert tolerance is not None
    assert tolerance != pytest.approx(control.target_size)
    assert tolerance > control.target_size
