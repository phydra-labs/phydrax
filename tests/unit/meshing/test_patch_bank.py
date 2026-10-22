#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Canonical column banks for publication patches and their scopes."""

from dataclasses import fields
from pathlib import Path
from typing import NoReturn

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._model._structure import (
    _canonical_recipe_payload,
    _restore_dataclass,
    model_structure_recipe,
)
from phydrax.lifecycle import read_meshing_source_closure, write_meshing_source_closure
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from phydrax.meshing._organization_records import MeshPatch, MeshPatchBank
from phydrax.meshing import _scope as scope_owner
from phydrax.meshing._scope import (
    mesh_entity_scope,
    MeshingEntityKind,
    MeshingScope,
    MeshingScopeBank,
)


def _mesh(cells: int) -> phx.discretization.CellMesh:
    points = [[float(index), 0.0] for index in range(cells + 1)]
    points += [[float(index), 1.0] for index in range(cells + 1)]
    triangles = []
    for index in range(cells):
        lower, upper = index, index + cells + 1
        triangles += [[lower, lower + 1, upper + 1], [lower, upper + 1, upper]]
    return phx.discretization.CellMesh(
        np.asarray(points),
        (
            phx.discretization.CellBlock(
                "cell", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )


def _patches(mesh: phx.discretization.CellMesh) -> tuple[MeshPatch, ...]:
    edges = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    local = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        mesh.entity_set(1).entity_set_id,
        edges[:3],
        local_mesh=mesh,
    )
    return (
        *(
            MeshPatch(f"facet:{index}", mesh_entity_scope(mesh, 1, edges[index : index + 1]))
            for index in range(edges.size)
        ),
        MeshPatch(
            "sided",
            mesh_entity_scope(mesh, 1, edges[:2]),
            connected=False,
            adjacent_zone_ids=("zone-b", "zone-a"),
            source_adjacent_region_ids=("region", None),
        ),
        MeshPatch("placed", local),
    )


def _same_rows(first: tuple[MeshPatch, ...], second: tuple[MeshPatch, ...]) -> None:
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert left.patch_id == right.patch_id
        assert left.scope.scope_id == right.scope.scope_id
        assert left.scope.logical_entity_set_id == right.scope.logical_entity_set_id
        assert (left.name, left.connected, left.adjacent_zone_ids) == (
            right.name,
            right.connected,
            right.adjacent_zone_ids,
        )
        assert left.source_adjacent_region_ids == right.source_adjacent_region_ids
        for name in ("global_entity_ids", "entity_ids", "entity_owner"):
            np.testing.assert_array_equal(
                getattr(left.scope, name), getattr(right.scope, name)
            )
        assert (left.scope.entity_ids is left.scope.global_entity_ids) == (
            right.scope.entity_ids is right.scope.global_entity_ids
        )


def test_patch_bank_rows_reproduce_patch_and_scope_identities() -> None:
    patches = _patches(_mesh(4))
    bank = MeshPatchBank(patches)
    assert len(bank) == len(patches)
    assert bank.names == tuple(patch.name for patch in patches)
    assert bank.index("sided") == len(patches) - 2
    _same_rows(tuple(bank), patches)
    _same_rows(bank[-2:], patches[-2:])
    assert bank[0].patch_id == patches[0].patch_id
    assert bank.scopes.explicit_scope_ids == ()
    with pytest.raises(KeyError, match=r"MeshPatchBank\.name"):
        bank.index("absent")
    with pytest.raises(IndexError, match=r"MeshPatchBank\.index"):
        bank.patch(len(patches))



def test_scope_bank_traversal_prepares_only_local_metadata_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scopes = tuple(patch.scope for patch in _patches(_mesh(4)))
    bank = MeshingScopeBank(scopes)
    indices: list[tuple[dict[int, MeshingScope], dict[int, str]]] = []
    materialize = MeshingScopeBank._scope

    def row(
        self: MeshingScopeBank,
        index: int,
        projected: dict[int, MeshingScope],
        explicit: dict[int, str],
        /,
    ) -> MeshingScope:
        indices.append((projected, explicit))
        return materialize(self, index, projected, explicit)

    monkeypatch.setattr(MeshingScopeBank, "_scope", row)
    for actual, expected in zip(bank, scopes, strict=True):
        assert actual.scope_id == expected.scope_id
        np.testing.assert_array_equal(actual.entity_ids, expected.entity_ids)
    assert len(indices) == len(scopes)
    assert all(
        projected is indices[0][0] and explicit is indices[0][1]
        for projected, explicit in indices
    )
    first_indices = indices[0]
    indices.clear()
    tuple(bank)
    assert indices[0][0] is not first_indices[0]
    assert indices[0][1] is not first_indices[1]


def test_scope_bank_shares_copied_universes_without_changing_rows() -> None:
    scope = _patches(_mesh(3))[-1].scope
    placement = (
        scope.local_entity_universe,
        scope.local_owner_universe,
        scope.global_entity_universe,
    )
    copied = eqx.tree_at(
        lambda value: (
            value.local_entity_universe,
            value.local_owner_universe,
            value.global_entity_universe,
        ),
        scope,
        tuple(None if value is None else value.copy() for value in placement),
        is_leaf=lambda value: value is None,
    )
    assert copied.local_entity_universe is not scope.local_entity_universe
    bank = MeshingScopeBank((scope, copied))
    assert len(bank.bindings) == len(bank.universes) == 1
    np.testing.assert_array_equal(bank.binding_codes, [0, 0])
    for actual, expected in zip(bank.universes[0], placement, strict=True):
        assert actual is expected
    for row in bank:
        assert row.scope_id == scope.scope_id
        for name in (
            "global_entity_ids", "entity_ids", "entity_owner",
            "local_entity_universe", "local_owner_universe", "global_entity_universe",
        ):
            np.testing.assert_array_equal(getattr(row, name), getattr(scope, name))


def test_scope_bank_digest_collision_does_not_erase_different_universes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _patches(_mesh(3))[-1].scope
    owners = scope.local_owner_universe
    assert owners is not None
    # An adversarial changed numerical owner must remain visible to subsequent
    # scope admission even when metadata and candidate digests match.
    changed = eqx.tree_at(
        lambda value: value.local_owner_universe, scope, owners.at[0].add(1)
    )
    digest = scope_owner.logical_array_value_collection_digest

    def collide(arrays, **kwargs):
        if set(arrays).issubset({"0", "1", "2"}):
            return "placement-collision"
        return digest(arrays, **kwargs)

    monkeypatch.setattr(scope_owner, "logical_array_value_collection_digest", collide)
    bank = MeshingScopeBank((scope, changed))
    assert len(bank.bindings) == len(bank.universes) == 2
    np.testing.assert_array_equal(bank.binding_codes, [0, 1])
    np.testing.assert_array_equal(bank.scope(0).local_owner_universe, owners)
    np.testing.assert_array_equal(
        bank.scope(1).local_owner_universe, changed.local_owner_universe
    )



def test_patch_bank_recipe_grows_by_column_values_not_records() -> None:
    register_meshing_source_artifacts()
    small = _canonical_recipe_payload(
        model_structure_recipe(MeshPatchBank(_patches(_mesh(4))))
    )
    large = _canonical_recipe_payload(
        model_structure_recipe(MeshPatchBank(_patches(_mesh(64))))
    )
    patches = _patches(_mesh(64))
    records = _canonical_recipe_payload(model_structure_recipe(patches))
    # Only names and array shape digits grow with the patch count.
    per_patch = (len(large) - len(small)) / (len(patches) - len(_patches(_mesh(4))))
    assert per_patch < 24
    assert len(large) * 10 < len(records)


def test_scope_bank_refuses_misaligned_columns() -> None:
    bank = MeshingScopeBank(tuple(patch.scope for patch in _patches(_mesh(3))))
    forged = eqx.tree_at(
        lambda value: value.owner_offsets,
        bank,
        np.asarray(bank.owner_offsets[:-1]),
    )
    with pytest.raises(ValueError, match=r"MeshingScopeBank\.columns"):
        tuple(forged)
    unbound = eqx.tree_at(
        lambda value: value.binding_codes,
        bank,
        np.full(bank.binding_codes.shape, -1, dtype=np.int32),
    )
    with pytest.raises(ValueError, match=r"MeshingScopeBank\.columns"):
        tuple(unbound)

@pytest.mark.parametrize("empty", (False, True))
def test_scope_bank_restores_inventory_columns_without_device_rows(
    monkeypatch: pytest.MonkeyPatch, empty: bool,
) -> None:
    mesh = _mesh(4)
    scopes = () if empty else tuple(patch.scope for patch in _patches(mesh)[:-1])
    bank = MeshingScopeBank(scopes)

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("plain inventory restoration must not materialize rows")

    with monkeypatch.context() as patch:
        patch.setattr(MeshingScopeBank, "_scope", forbidden)
        patch.setattr(scope_owner.jax, "device_put", forbidden)
        restored = MeshingScopeBank.from_stored_fields(
            bank, reconstruct_scope=forbidden
        )
    assert restored.bindings == bank.bindings
    assert restored.members is bank.members
    assert not restored.members.flags.writeable
    for actual, expected in zip(restored, scopes, strict=True):
        assert actual.scope_id == expected.scope_id
        np.testing.assert_array_equal(actual.entity_ids, expected.entity_ids)
        np.testing.assert_array_equal(actual.entity_owner, expected.entity_owner)


@pytest.mark.parametrize(
    "fault",
    (
        "negative", "duplicate", "unordered", "owners", "unshared", "binding",
        "identity", "dimension-kind", "container", "placement",
    ),
)
def test_scope_bank_column_restore_refuses_noncanonical_inventory(fault: str) -> None:
    mesh = _mesh(3)
    identifiers = np.asarray(mesh.entity_set(1).entity_ids)
    scope = mesh_entity_scope(mesh, 1, identifiers[:3])
    bank = MeshingScopeBank((scope,))
    if fault in ("negative", "duplicate", "unordered"):
        members = np.array(bank.members)
        if fault == "negative":
            members[0] = -1
        elif fault == "duplicate":
            members[1] = members[0]
        else:
            members[[0, 1]] = members[[1, 0]]
        forged = eqx.tree_at(lambda value: value.members, bank, members)
    elif fault == "owners":
        forged = eqx.tree_at(
            lambda value: value.owners, bank, np.ones_like(bank.owners)
        )
    elif fault == "unshared":
        forged = eqx.tree_at(
            lambda value: value.shares_members,
            bank,
            np.zeros_like(bank.shares_members),
        )
    else:
        register_meshing_source_artifacts()
        stored = {member.name: getattr(bank, member.name) for member in fields(bank)}
        if fault == "binding":
            binding = bank.bindings[0]
            stored["bindings"] = ((binding[0] + " ", *binding[1:]),)
        elif fault == "dimension-kind":
            binding = bank.bindings[0]
            stored["bindings"] = ((*binding[:3], float(binding[3]), *binding[4:]),)
        elif fault == "container":
            stored["projected"] = []
        elif fault == "placement":
            stored["universes"] = ((),)
        else:
            stored["explicit_scope_ids"] = ((0, "foreign-scope"),)
        # Stage legacy persisted fields through the registered archive boundary;
        # the owning constructor below must authenticate the corrupt metadata.
        forged = _restore_dataclass(
            MeshingScopeBank, list(range(len(stored))), list(stored.values())
        )
    with pytest.raises(ValueError, match=r"MeshingScopeBank\."):
        MeshingScopeBank.from_stored_fields(forged)


@pytest.mark.parametrize("include_vertices", (False, True))
def test_scope_bank_column_restore_retains_local_constructor_context(
    include_vertices: bool,
) -> None:
    mesh = _mesh(3)
    scopes = tuple(
        MeshingScope.from_stored_fields(patch.scope, local_mesh=mesh)
        for patch in _patches(mesh)
    )
    if include_vertices:
        scopes += (
            MeshingScope.from_stored_fields(
                mesh_entity_scope(mesh, 0, np.asarray(mesh.entity_set(0).entity_ids)[:2]),
                local_mesh=mesh,
            ),
        )
    bank = MeshingScopeBank(scopes)
    visited: list[np.ndarray] = []

    def reconstruct(scope: MeshingScope) -> MeshingScope:
        visited.append(np.asarray(scope.global_entity_ids))
        return MeshingScope.from_stored_fields(scope, local_mesh=mesh)

    restored = MeshingScopeBank.from_stored_fields(
        bank, reconstruct_scope=reconstruct
    )
    assert len(visited) == len(bank.bindings) == (2 if include_vertices else 1)
    np.testing.assert_array_equal(
        visited[0],
        np.unique(
            np.concatenate(
                [
                    np.asarray(scope.global_entity_ids)
                    for scope in scopes if scope.entity_dimension == 1
                ]
            )
        ),
    )
    assert len(restored.universes) == len(bank.universes)
    for actual, expected in zip(restored, scopes, strict=True):
        assert actual.scope_id == expected.scope_id
        np.testing.assert_array_equal(
            actual.local_entity_universe, expected.local_entity_universe
        )

@pytest.mark.parametrize("fault", ("local-member", "row-owner", "universe-owner"))
def test_scope_bank_column_restore_authenticates_all_placed_columns(fault: str) -> None:
    mesh = _mesh(3)
    scopes = tuple(
        MeshingScope.from_stored_fields(patch.scope, local_mesh=mesh)
        for patch in _patches(mesh)
    )
    bank = MeshingScopeBank(scopes)
    if fault == "local-member":
        values = np.array(bank.local_members)
        values[-1] = values[-1] + 1
        forged = eqx.tree_at(lambda value: value.local_members, bank, values)
    elif fault == "row-owner":
        values = np.array(bank.owners)
        values[-1] = values[-1] + 1
        forged = eqx.tree_at(lambda value: value.owners, bank, values)
    else:
        owners = bank.universes[0][1]
        assert owners is not None
        forged = eqx.tree_at(
            lambda value: value.universes[0][1], bank, owners.at[-1].add(1)
        )
    with pytest.raises(ValueError, match=r"MeshingScopeBank\."):
        MeshingScopeBank.from_stored_fields(
            forged,
            reconstruct_scope=lambda scope: MeshingScope.from_stored_fields(
                scope, local_mesh=mesh
            ),
        )



def test_scope_bank_inventory_archive_roundtrip_uses_column_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = MeshingScopeBank(
        tuple(patch.scope for patch in _patches(_mesh(3))[:-1])
    )
    receipt = write_meshing_source_closure(tmp_path / "scope-bank.zip", bank)

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("inventory archive admission must not materialize rows")

    with monkeypatch.context() as patch:
        patch.setattr(MeshingScopeBank, "_scope", forbidden)
        restored = read_meshing_source_closure(
            receipt.path, expected_content_id=receipt.content_id
        )
    assert type(restored) is MeshingScopeBank
    assert restored.bindings == bank.bindings
    np.testing.assert_array_equal(restored.members, bank.members)



def test_patch_bank_refuses_forged_dictionary_codes() -> None:
    bank = MeshPatchBank(_patches(_mesh(3)))
    forged = eqx.tree_at(
        lambda value: value.adjacency_codes,
        bank,
        np.full(bank.adjacency_codes.shape, len(bank.adjacency), dtype=np.int32),
    )
    with pytest.raises(ValueError, match=r"MeshPatchBank\.columns"):
        tuple(forged)


def test_patch_bank_archive_roundtrip_revalidates_rows(tmp_path: Path) -> None:
    mesh = _mesh(3)
    bank = MeshPatchBank(_patches(mesh))
    limits = ArrayArchiveLimits(max_members=4096, max_manifest_nesting=128)
    receipt = write_meshing_source_closure(tmp_path / "patch-bank.zip", (mesh, bank), limits=limits)
    restored_mesh, restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id, limits=limits
    )
    assert type(restored) is MeshPatchBank
    assert restored_mesh.mesh_id == mesh.mesh_id
    _same_rows(tuple(restored), tuple(bank))
