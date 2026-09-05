import numpy as np
import pytest

import phydrax as phx


def _scope(ids, *, revision="r1"):
    return phx.meshing.MeshingScope(
        "mesh",
        revision,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        "faces",
        np.asarray(ids, dtype=np.int64),
    )


def test_scope_set_algebra_is_exact_and_revision_bound():
    first = _scope([1, 2, 3])
    second = _scope([3, 4])

    np.testing.assert_array_equal(first.union(second).entity_ids, [1, 2, 3, 4])
    np.testing.assert_array_equal(first.intersection(second).entity_ids, [3])
    np.testing.assert_array_equal(first.difference(second).entity_ids, [1, 2])
    with pytest.raises(ValueError, match="one exact binding"):
        first.union(_scope([5], revision="r2"))
    with pytest.raises(ValueError, match="empty"):
        first.difference(first)


def test_scope_from_selection_resolves_persistent_ids_not_storage_rows():
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


def test_scope_from_selection_rejects_foreign_entity_set():
    entities = phx.discretization.EntitySet(
        "faces", 2, np.asarray((10, 20), dtype=np.int64)
    )
    foreign = phx.discretization.EntitySet(
        "faces", 2, np.asarray((30, 40), dtype=np.int64)
    )
    selection = phx.discretization.EntitySelection(foreign, np.asarray((True, False)))

    with pytest.raises(ValueError, match="supplied entity set"):
        phx.meshing.MeshingScope.from_selection("mesh", "r1", entities, selection)


@pytest.mark.parametrize(
    ("mask", "active_mask"),
    (
        ((True,), (True,)),
        ((True, False), (True, False)),
    ),
)
def test_scope_from_selection_rejects_incompatible_selection_masks(mask, active_mask):
    entities = phx.discretization.EntitySet(
        "faces", 2, np.asarray((10, 20), dtype=np.int64)
    )
    selection = phx.discretization.EntitySelection(
        entities.entity_set_id, np.asarray(mask), active_mask=np.asarray(active_mask)
    )

    with pytest.raises(ValueError, match="capacity and active mask"):
        phx.meshing.MeshingScope.from_selection("mesh", "r1", entities, selection)


def test_zones_are_exclusive_while_labels_may_overlap():
    first = _scope([1, 2])
    second = _scope([2, 3])
    zones = (
        phx.meshing.MeshZone("wall", phx.meshing.MeshZoneRole.BOUNDARY, first),
        phx.meshing.MeshZone("inlet", phx.meshing.MeshZoneRole.BOUNDARY, second),
    )
    with pytest.raises(ValueError, match="disjoint"):
        phx.meshing.validate_mesh_zones(zones)

    labels = (
        phx.meshing.MeshLabel("hot", first),
        phx.meshing.MeshLabel("observed", second),
    )
    assert phx.meshing.validate_mesh_labels(labels) == labels
