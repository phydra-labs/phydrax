from collections.abc import Sequence

import equinox as eqx
import jax
import numpy as np
import pytest

import phydrax as phx
from tests._support.cell_meshes import unit_square


def _scope(
    ids: Sequence[int] | np.ndarray, *, revision: str = "r1"
) -> phx.meshing.MeshingScope:
    return phx.meshing.MeshingScope(
        "mesh",
        revision,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        "faces",
        np.asarray(ids, dtype=np.int64),
    )


def test_scope_set_algebra_requires_exact_binding() -> None:
    first = _scope([1, 2, 3])
    second = _scope([3, 4])

    np.testing.assert_array_equal(first.union(second).entity_ids, [1, 2, 3, 4])
    np.testing.assert_array_equal(first.intersection(second).entity_ids, [3])
    np.testing.assert_array_equal(first.difference(second).entity_ids, [1, 2])
    with pytest.raises(ValueError, match=r"MeshingScope\.set_operation"):
        first.union(_scope([5], revision="r2"))
    with pytest.raises(ValueError, match="empty"):
        first.difference(first)


def test_scope_selection_preserves_scientific_entity_ids() -> None:
    entities = phx.discretization.EntitySet(
        "faces",
        2,
        np.asarray((90, -1, 12, 47), dtype=np.int64),
        active_mask=np.asarray((True, False, True, True)),
    )
    selection = phx.discretization.EntitySelection(
        entities, np.asarray((True, False, True, False))
    )

    scope = phx.meshing.MeshingScope.from_selection("mesh", "r1", entities, selection)
    expected = phx.meshing.MeshingScope(
        "mesh",
        "r1",
        phx.meshing.MeshingEntityKind.MESH,
        2,
        entities.entity_set_id,
        np.asarray((12, 90), dtype=np.int64),
    )

    np.testing.assert_array_equal(scope.entity_ids, (12, 90))
    assert scope.scope_id == expected.scope_id
    with pytest.raises(ValueError, match="at least one entity"):
        phx.meshing.MeshingScope.from_selection(
            "mesh", "r1", entities, selection.difference(selection)
        )


def test_scope_selection_refuses_foreign_entity_set() -> None:
    entities = phx.discretization.EntitySet(
        "faces", 2, np.asarray((10, 20), dtype=np.int64)
    )
    foreign = phx.discretization.EntitySet(
        "faces", 2, np.asarray((30, 40), dtype=np.int64)
    )
    selection = phx.discretization.EntitySelection(foreign, np.asarray((True, False)))

    with pytest.raises(ValueError, match=r"MeshingScope\.selection"):
        phx.meshing.MeshingScope.from_selection("mesh", "r1", entities, selection)


@pytest.mark.parametrize(
    ("mask", "active_mask"),
    (((True,), (True,)), ((True, False), (True, False))),
    ids=("wrong-capacity", "different-active-mask"),
)
def test_scope_selection_requires_capacity_and_active_mask(
    mask: tuple[bool, ...], active_mask: tuple[bool, ...]
) -> None:
    entities = phx.discretization.EntitySet(
        "faces", 2, np.asarray((10, 20), dtype=np.int64)
    )
    selection = phx.discretization.EntitySelection(
        entities.entity_set_id, np.asarray(mask), active_mask=np.asarray(active_mask)
    )
    with pytest.raises(ValueError, match=r"MeshingScope\.selection"):
        phx.meshing.MeshingScope.from_selection("mesh", "r1", entities, selection)


def test_zone_membership_requires_disjoint_scopes() -> None:
    first = _scope([1, 2])
    second = _scope([2, 3])
    zones = (
        phx.meshing.MeshZone("wall", phx.meshing.MeshZoneRole.BOUNDARY, first),
        phx.meshing.MeshZone("inlet", phx.meshing.MeshZoneRole.BOUNDARY, second),
    )
    with pytest.raises(ValueError, match="disjoint"):
        phx.meshing.validate_mesh_zones(zones)


def test_label_membership_permits_overlapping_scopes() -> None:
    first = _scope([1, 2])
    second = _scope([2, 3])
    labels = (
        phx.meshing.MeshLabel("hot", first),
        phx.meshing.MeshLabel("observed", second),
    )
    assert phx.meshing.validate_mesh_labels(labels) == labels


def test_region_zone_identity_includes_material() -> None:
    scope = _scope([1, 2])
    zone = phx.meshing.MeshZone(
        "fluid",
        phx.meshing.MeshZoneRole.REGION,
        scope,
        material_id="water",
        region_role=phx.meshing.RegionRole.FLUID,
    )
    changed = phx.meshing.MeshZone(
        "fluid",
        phx.meshing.MeshZoneRole.REGION,
        scope,
        material_id="oil",
        region_role=phx.meshing.RegionRole.FLUID,
    )
    assert zone.zone_id != changed.zone_id


def test_patch_adjacency_is_canonical() -> None:
    scope = _scope([1, 2])
    patch = phx.meshing.MeshPatch(
        "interface",
        scope,
        adjacent_zone_ids=("zone-b", "zone-a"),
    )
    canonical = phx.meshing.MeshPatch(
        "interface",
        scope,
        adjacent_zone_ids=("zone-a", "zone-b"),
    )

    assert patch.adjacent_zone_ids == ("zone-a", "zone-b")
    assert patch.patch_id == canonical.patch_id


def test_region_zone_requires_paired_material_metadata() -> None:
    scope = _scope([1, 2])
    with pytest.raises(ValueError, match="supplied together"):
        phx.meshing.MeshZone(
            "invalid",
            phx.meshing.MeshZoneRole.REGION,
            scope,
            material_id="water",
        )


def test_nonregion_zone_refuses_region_metadata() -> None:
    scope = _scope([1, 2])
    with pytest.raises(ValueError, match=r"MeshZone\.material_id"):
        phx.meshing.MeshZone(
            "invalid",
            phx.meshing.MeshZoneRole.MATERIAL,
            scope,
            material_id="water",
            region_role=phx.meshing.RegionRole.FLUID,
        )


@pytest.mark.parametrize("capacity", (1, 3, 17), ids=("single", "small", "irregular"))
def test_scope_host_validation_preserves_device_leaves(capacity: int) -> None:
    identifiers = np.arange(capacity, dtype=np.int64)[::-1]
    host_scope = _scope(identifiers)
    device_scope = phx.meshing.MeshingScope(
        "mesh",
        "r1",
        phx.meshing.MeshingEntityKind.MESH,
        2,
        "faces",
        jax.device_put(identifiers),
    )

    assert isinstance(host_scope.global_entity_ids, jax.Array)
    assert isinstance(device_scope.global_entity_ids, jax.Array)
    assert isinstance(device_scope.entity_owner, jax.Array)
    assert device_scope.entity_ids is device_scope.global_entity_ids
    assert host_scope.scope_id == device_scope.scope_id
    np.testing.assert_array_equal(device_scope.entity_ids, identifiers[::-1])
    np.testing.assert_array_equal(device_scope.entity_owner, np.zeros(capacity, np.int32))


@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.uint64))
def test_canonical_scope_members_remain_independently_owned(dtype: type) -> None:
    from phydrax.meshing._scope import _host_member_ids

    identifiers = np.asarray((0, 3, 11), dtype=dtype)
    canonical = _host_member_ids(identifiers)
    assert canonical.dtype == np.int64
    assert not np.shares_memory(canonical, identifiers)
    identifiers[0] = 2
    np.testing.assert_array_equal(canonical, (0, 3, 11))
    np.testing.assert_array_equal(
        _host_member_ids(np.asarray((11, 3, 0), dtype=dtype)), canonical
    )


@pytest.mark.parametrize("identifiers", ((0, 0, 3), (3, 0, 0), (-1, 0, 3)))
def test_scope_member_order_fast_path_preserves_refusals(
    identifiers: tuple[int, ...],
) -> None:
    from phydrax.meshing._scope import _host_member_ids

    with pytest.raises(ValueError, match="unique and nonnegative"):
        _host_member_ids(np.asarray(identifiers, dtype=np.int64))


def test_scope_admits_host_banks_in_one_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfers: list[tuple[np.ndarray, ...]] = []
    device_put = jax.device_put

    def admit(banks: tuple[np.ndarray, ...]) -> tuple[jax.Array, ...]:
        transfers.append(banks)
        return device_put(banks)

    monkeypatch.setattr(jax, "device_put", admit)
    scope = _scope([3, 1, 2])

    assert len(transfers) == 1
    assert len(transfers[0]) == 2
    assert scope.entity_ids is scope.global_entity_ids
    np.testing.assert_array_equal(scope.entity_ids, (1, 2, 3))
    np.testing.assert_array_equal(scope.entity_owner, (0, 0, 0))


def test_local_scope_reconstruction_preserves_membership_and_identity() -> None:
    mesh = unit_square(cell_global_ids=np.asarray((17, 29), dtype=np.int64))
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        mesh.entity_set(2).entity_set_id,
        np.asarray((29,), dtype=np.int64),
        local_mesh=mesh,
    )
    from tools.compile_counter import CompileCounter

    assert scope.local_entity_universe is not None
    copied = eqx.tree_at(
        lambda value: value.local_entity_universe,
        scope,
        scope.local_entity_universe.copy(),
    )
    changed = eqx.tree_at(
        lambda value: value.local_entity_universe,
        scope,
        jax.device_put(np.asarray((29, 17), dtype=np.int64)),
    )

    with CompileCounter() as compiler:
        assert scope.matches_local_mesh(mesh)
        assert copied.matches_local_mesh(mesh)
        assert not changed.matches_local_mesh(mesh)
    assert compiler.count == 0
    restored = phx.meshing.MeshingScope.from_stored_fields(scope, local_mesh=mesh)
    assert restored.scope_id == scope.scope_id
    assert restored.logical_entity_set_id == scope.logical_entity_set_id
    np.testing.assert_array_equal(restored.entity_ids, (29,))
    np.testing.assert_array_equal(restored.entity_owner, scope.entity_owner)
    scope.require_same_local_placement(restored)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(scope)


def test_local_scope_reconstruction_refuses_stale_mesh_revision() -> None:
    mesh = unit_square()
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        mesh.entity_set(2).entity_set_id,
        np.asarray((0,), dtype=np.int64),
        local_mesh=mesh,
    )
    moved = mesh.with_coordinates(mesh.coordinates, numeric_version="stale")
    assert not scope.matches_local_mesh(moved)
    with pytest.raises(ValueError, match=r"MeshingScope\.local_mesh"):
        phx.meshing.MeshingScope.from_stored_fields(scope, local_mesh=moved)


def test_scope_host_unique_ids_do_not_compile_eager_primitives() -> None:
    from phydrax.meshing._scope import unique_ids
    from tools.compile_counter import CompileCounter

    identifiers = jax.device_put(np.asarray((3, 1, 3, 2), dtype=np.int64))
    with CompileCounter() as compiler:
        result = unique_ids(identifiers)
    np.testing.assert_array_equal(result, (1, 2, 3))
    assert compiler.count == 0


def test_scope_membership_reuses_the_same_capacity_kernel() -> None:
    from phydrax.meshing._scope import contains_ids
    from tools.compile_counter import CompileCounter

    jax.clear_caches()
    inventory = jax.device_put(np.asarray((1, 3, 7), dtype=np.int64))
    first = jax.device_put(np.asarray((1, 2), dtype=np.int64))
    second = jax.device_put(np.asarray((3, 7), dtype=np.int64))
    with CompileCounter() as compiler:
        np.testing.assert_array_equal(contains_ids(inventory, first), (True, False))
        np.testing.assert_array_equal(contains_ids(inventory, second), (True, True))
    assert compiler.count == 1
    assert compiler.events[0].kernel_name == "jit(contains_ids)"


def test_local_scope_resolution_and_overlap_do_not_compile() -> None:
    from phydrax.meshing._scope import overlapping_ids, resolve_mesh_scope
    from tools.compile_counter import CompileCounter

    mesh = unit_square(cell_global_ids=np.asarray((17, 29), dtype=np.int64))
    entities = mesh.entity_set(2)
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        entities.entity_set_id,
        np.asarray((29,), dtype=np.int64),
    )
    with CompileCounter() as compiler:
        selected = resolve_mesh_scope(mesh, scope)
        assert overlapping_ids(entities.entity_ids, scope.entity_ids)
    np.testing.assert_array_equal(selected.mask, (False, True))
    assert compiler.count == 0
