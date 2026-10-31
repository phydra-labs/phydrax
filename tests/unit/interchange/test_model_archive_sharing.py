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
    read_model_artifact,
    restore_model_archive_value,
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


def test_equal_subtrees_are_archived_once_independent_of_aliasing(
    tmp_path: Path,
) -> None:
    mesh = _mesh()
    single = write_model_artifact(tmp_path / "single.phx", _composition(mesh), limits=None)
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
