from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phydrax._array_archive import (
    array_payload_digest,
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.lifecycle._composition_rebind import Composition, CompositionEntry
from phydrax.meshing._model_codec import (
    read_model_archive,
    read_model_artifact,
    restore_model_archive_value,
    write_model_archive,
    write_model_artifact,
)


def _mesh() -> CellMesh:
    return CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))),
        np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32),
    )


def _entry(mesh: CellMesh, entry_id: str, /) -> CompositionEntry:
    return CompositionEntry(
        mesh,
        entry_id=entry_id,
        role="topology",
        owner_id="phydrax.discretization",
        structure_id=mesh.topology_id,
        revision_id=mesh.mesh_id,
        semantics_id="shared-archive-mesh",
    )


def _composition(*meshes: CellMesh) -> Composition:
    return Composition(
        tuple(_entry(mesh, f"mesh/{index}") for index, mesh in enumerate(meshes)),
        boundary_id="shared-archive",
    )


def _nodes(node: Any, kind: str, /) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            if current.get("kind") == kind:
                found.append(current)
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return found


def _first_mesh(node: Any, /) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if node.get("kind") == "dataclass" and str(node.get("type", "")).endswith(
            ":CellMesh"
        ):
            return node
        children = list(node.values())
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _first_mesh(child)
        if found is not None:
            return found
    return None


def _archive(path: Path, /) -> tuple[dict[str, Any], dict[str, str]]:
    manifest, arrays = read_array_archive(path, limits=None)
    return manifest, {name: array_payload_digest(value) for name, value in arrays.items()}


def _read(path: Path, /) -> Composition:
    return read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=None,
    )


def test_pack_and_restore_reuse_one_validated_descriptor_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._model import _structure
    from phydrax.meshing._model_codec import _register_model_archive_types

    _register_model_archive_types()
    value = _composition(_mesh())
    recipe = _structure.model_structure_recipe(value)
    original = _structure._descriptor_tree
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_structure, "_descriptor_tree", counted)
    with pytest.raises(ValueError, match="prefix"):
        _structure.model_recipe_array_inventory(recipe, prefix="")
    assert calls == 0
    arrays = _structure.pack_model_array_tree(value, recipe, prefix="model/leaves")
    assert calls == 1
    restored = _structure.model_from_array_recipe(
        recipe, arrays, prefix="model/leaves", validate=False
    )
    assert calls == 2
    assert restored.entries[0].value.mesh_id == value.entries[0].value.mesh_id


def test_new_model_recipe_is_not_copied_for_sharing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.meshing import _model_codec

    def forbidden(value: Any) -> Any:
        raise AssertionError("Fresh model recipes must not be expanded again.")

    monkeypatch.setattr(_model_codec, "_unaliased_recipe", forbidden)
    path = write_model_artifact(tmp_path / "fresh.phx", _composition(_mesh()), limits=None)
    assert _read(path).entries[0].value.mesh_id == _mesh().mesh_id


def test_equal_subtrees_are_archived_once_independent_of_aliasing(
    tmp_path: Path,
) -> None:
    mesh = _mesh()
    single = write_model_artifact(
        tmp_path / "single.phx", _composition(mesh), limits=None
    )
    aliased = write_model_artifact(
        tmp_path / "aliased.phx", _composition(mesh, mesh), limits=None
    )
    distinct = write_model_artifact(
        tmp_path / "distinct.phx", _composition(mesh, _mesh()), limits=None
    )

    single_manifest, single_arrays = _archive(single)
    aliased_manifest, aliased_arrays = _archive(aliased)
    assert (aliased_manifest, aliased_arrays) == _archive(distinct)
    # The repeated mesh adds one reference and no array members.
    assert (
        len(_nodes(aliased_manifest, "shared"))
        == len(_nodes(single_manifest, "shared")) + 1
    )
    assert aliased_arrays == single_arrays

    restored = _read(distinct)
    assert [entry.value.mesh_id for entry in restored.entries] == [mesh.mesh_id] * 2
    np.testing.assert_array_equal(
        np.asarray(restored.entries[1].value.coordinates), np.asarray(mesh.coordinates)
    )


@pytest.mark.parametrize("tamper", ("definition", "duplicate", "unresolved"))
def test_shared_archive_nodes_refuse_changed_or_noncanonical_definitions(
    tmp_path: Path, tamper: str
) -> None:
    mesh = _mesh()
    source = write_model_artifact(
        tmp_path / "source.phx", _composition(mesh, _mesh()), limits=None
    )
    manifest, arrays = read_array_archive(source, limits=None)
    manifest.pop("arrays")
    recipe = manifest["model_recipe"]
    definition = _first_mesh(recipe)
    assert definition is not None
    reference = recipe["fields"]["entries"]["items"][1]["fields"]["value"]
    assert reference["kind"] == "shared"
    if tamper == "definition":
        definition["fields"]["numeric_version"]["value"] = "changed"
    elif tamper == "duplicate":
        reference.clear()
        reference.update(copy.deepcopy(definition))
    else:
        reference["key"] = "0" * 64
    changed = tmp_path / "changed"
    changed.mkdir()
    write_array_archive(
        changed / "source.phx", manifest=manifest, arrays=arrays, limits=None
    )
    with pytest.raises(
        ArrayArchiveCorruptionError, match="incompatible with its registered recipe"
    ) as refused:
        _read(changed / "source.phx")
    assert "Shared model recipe" in str(refused.value.__cause__)


def test_model_archive_readers_see_only_expanded_recipes(tmp_path: Path) -> None:
    mesh = _mesh()
    source = write_model_artifact(
        tmp_path / "source.phx", _composition(mesh, _mesh()), limits=None
    )
    manifest, arrays = read_model_archive(source, limits=None)
    assert set(manifest) == {"kind", "model_recipe"}
    assert not _nodes(manifest["model_recipe"], "shared")
    entries = manifest["model_recipe"]["fields"]["entries"]["items"]
    # Equal definitions expand to one node, so editing it edits every referrer.
    assert entries[0]["fields"]["value"] is entries[1]["fields"]["value"]

    rewritten = write_model_archive(
        tmp_path / "rewritten.phx", manifest=manifest, arrays=arrays, limits=None
    )
    assert _archive(rewritten) == _archive(source)

    # One changed occurrence is stored as its own content, not dropped.
    occurrence = max(arrays)
    changed_arrays = dict(arrays)
    changed_arrays[occurrence] = np.asarray(arrays[occurrence]).copy()
    changed_arrays[occurrence].flat[0] += 1
    changed = write_model_archive(
        tmp_path / "changed.phx", manifest=manifest, arrays=changed_arrays, limits=None
    )
    changed_manifest, restored_arrays = read_model_archive(changed, limits=None)
    changed_entries = changed_manifest["model_recipe"]["fields"]["entries"]["items"]
    assert (
        changed_entries[0]["fields"]["value"] is not changed_entries[1]["fields"]["value"]
    )
    assert set(restored_arrays) == set(arrays)
    for name, value in restored_arrays.items():
        np.testing.assert_array_equal(value, changed_arrays[name])
