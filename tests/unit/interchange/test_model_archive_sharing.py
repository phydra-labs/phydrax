from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
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
    _same_model_content,
    MODEL_ARCHIVE_LIMITS,
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


def test_content_comparison_preserves_large_admitted_payloads(tmp_path: Path) -> None:
    extent = MODEL_ARCHIVE_LIMITS.max_axis_length + 1
    payload = np.zeros(extent, dtype=np.uint8)
    limits = dataclasses.replace(
        MODEL_ARCHIVE_LIMITS,
        max_axis_length=extent,
        max_array_elements=extent,
    )
    path = tmp_path / "large-payload.phx"
    value = Composition(
        (
            CompositionEntry(
                payload,
                entry_id="payload",
                role="model-state",
                owner_id="phydrax.meshing",
                structure_id="large-byte-column",
                revision_id="original",
                semantics_id="coverage-witness-bytes",
            ),
        ),
        boundary_id="large-payload",
    )
    write_model_artifact(path, value, limits=limits)
    restored = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=limits,
    )
    assert _same_model_content((payload,), (restored.value("payload"),))
    payload[-1] = 1
    assert not _same_model_content((payload,), (restored.value("payload"),))


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


def test_model_array_recipe_releases_restored_strict_owner() -> None:
    import weakref

    from phydrax._model._structure import (
        model_from_array_recipe,
        model_structure_recipe,
        pack_model_array_tree,
    )
    from phydrax.geometry._mesh_certificates import _same_tree_content
    from phydrax.meshing._model_codec import _register_model_archive_types

    _register_model_archive_types()
    mesh = _mesh()
    recipe = model_structure_recipe(mesh)
    arrays = pack_model_array_tree(mesh, recipe, prefix="model")
    restored = model_from_array_recipe(recipe, arrays, prefix="model")
    assert _same_tree_content(restored, mesh)
    assert restored.mesh_id == mesh.mesh_id
    assert restored.topology_id == mesh.topology_id
    with pytest.raises((AttributeError, TypeError)):
        restored.mesh_id = "invalid"
    owner = weakref.ref(restored)
    del restored
    assert owner() is None


def test_row_restore_preserves_aliases_without_retaining_owners() -> None:
    import weakref

    from phydrax._model._structure import restore_row_record_recipes

    mesh = _mesh()
    owner = weakref.ref(mesh)
    restored = restore_row_record_recipes((mesh, mesh))
    assert restored[0] is mesh and restored[1] is mesh
    del mesh, restored
    assert owner() is None


def test_selected_archive_fields_retain_shared_payloads_and_refuse_wrong_root(
    tmp_path: Path,
) -> None:
    from phydrax.meshing._model_codec import read_model_archive_fields

    mesh = _mesh()
    path = write_model_artifact(
        tmp_path / "fields.phx", _composition(mesh, mesh), limits=None
    )
    selected = read_model_archive_fields(
        path, expected_type=Composition, fields=("entries", "boundary_id"), limits=None
    )
    assert selected["boundary_id"] == "shared-archive"
    assert [entry.value.mesh_id for entry in selected["entries"]] == [mesh.mesh_id] * 2
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_archive_fields(
            path, expected_type=CellMesh, fields=("entries",), limits=None
        )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_archive_fields(
            path, expected_type=Composition, fields=("absent",), limits=None
        )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_archive_fields(
            path, expected_type=Composition, fields=("boundary_id.value",), limits=None
        )
    entry_path = write_model_artifact(
        tmp_path / "entry.phx", _entry(mesh, "mesh"), limits=None
    )
    nested = read_model_archive_fields(
        entry_path,
        expected_type=CompositionEntry,
        fields=("value.coordinates",),
        limits=None,
    )
    np.testing.assert_array_equal(nested["value.coordinates"], mesh.coordinates)


def test_new_model_recipe_is_not_copied_for_sharing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.meshing import _model_codec

    def forbidden(value: Any) -> Any:
        raise AssertionError("Fresh model recipes must not be expanded again.")

    monkeypatch.setattr(_model_codec, "_unaliased_recipe", forbidden)
    path = write_model_artifact(
        tmp_path / "fresh.phx", _composition(_mesh()), limits=None
    )
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


def test_static_sequence_sharing_is_deterministic_and_authenticated() -> None:
    from phydrax._model._structure import (
        expand_shared_model_archive_recipe,
        model_structure_recipe,
        share_model_archive_recipe,
    )

    sequence = ("source-occurrence-" + "a" * 80, "target-occurrence-" + "b" * 80)
    aliased = model_structure_recipe((sequence, sequence))
    distinct = model_structure_recipe((sequence, tuple(list(sequence))))
    shared, members = share_model_archive_recipe(aliased, {}, prefix="leaves")
    assert (shared, members) == share_model_archive_recipe(distinct, {}, prefix="leaves")
    assert shared["items"][1]["kind"] == "shared"
    expanded, arrays = expand_shared_model_archive_recipe(
        shared, members, prefix="leaves"
    )
    assert expanded == aliased
    assert arrays == {}

    # Pre-existing writers emitted equal array-free sequences as definitions.
    legacy = copy.deepcopy(aliased)
    expanded, _ = expand_shared_model_archive_recipe(legacy, {}, prefix="leaves")
    assert expanded == aliased

    tampered = copy.deepcopy(shared)
    tampered["items"][0]["items"][0]["value"] = "forged"
    with pytest.raises(ValueError, match="no earlier equal definition"):
        expand_shared_model_archive_recipe(tampered, {}, prefix="leaves")


def test_cad_graph_columns_preserve_inventory_and_reject_tampering() -> None:
    from phydrax._model._structure import (
        model_archive_projection,
        model_from_array_recipe,
        model_structure_recipe,
        pack_model_array_tree,
        restore_row_record_recipes,
    )
    from phydrax.geometry._cad_revision import (
        AssociationCoverageEvidence,
        AssociationGraph,
        CADOccurrence,
        CADRevision,
        OccurrenceCorrespondence,
        OccurrenceCorrespondenceTransaction,
    )
    from phydrax.meshing._model_codec import _register_model_archive_types

    _register_model_archive_types()
    source = CADRevision(
        "source",
        "body",
        (CADOccurrence("source", "s", "a", "face", ("s",)),),
        "source-proof",
    )
    target = CADRevision(
        "target",
        "body",
        (CADOccurrence("target", "t", "b", "face", ("t",)),),
        "target-proof",
    )
    transaction = OccurrenceCorrespondenceTransaction(
        "transaction",
        "source",
        "target",
        (OccurrenceCorrespondence("s", "t", "proof", -1),),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(True, True, "certificate", "provider"),
    )
    graph = AssociationGraph(source, target, transaction)
    projected = model_archive_projection(graph)
    recipe = model_structure_recipe(projected)
    arrays = pack_model_array_tree(projected, recipe, prefix="leaves")
    assert arrays
    raw = model_from_array_recipe(
        recipe, arrays, prefix="leaves", validate=False, restore_rows=False
    )
    assert restore_row_record_recipes(raw) == graph
    with pytest.raises(ValueError, match="identity changed"):
        restore_row_record_recipes(raw._replace(graph_id="forged"))

    import weakref

    owner = _mesh()
    owner_ref = weakref.ref(owner)

    def refuse_after_owner(value: CellMesh, /) -> None:
        try:
            restore_row_record_recipes((value, raw._replace(graph_id="forged")))
        except ValueError as error:
            assert "identity changed" in str(error)
        else:
            pytest.fail("Forged graph was accepted.")

    refuse_after_owner(owner)
    del owner
    assert owner_ref() is None


@pytest.mark.parametrize("tamper", ("definition", "duplicate", "unresolved", "cycle"))
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
    elif tamper == "unresolved":
        reference["key"] = "0" * 64
    else:
        definition["fields"]["coordinates"] = dict(reference)
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


def test_shared_archive_raw_manifest_depth_is_bounded(tmp_path: Path) -> None:
    from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS

    path = write_model_artifact(
        tmp_path / "depth.phx", _composition(_mesh()), limits=None
    )
    limits = dataclasses.replace(MODEL_ARCHIVE_LIMITS, max_manifest_nesting=3)
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=limits,
        )


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


def test_pack_charges_only_proven_immutable_owners() -> None:
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax._model._structure import (
        _pack_owned_model_array_tree,
        model_structure_recipe,
    )

    limits = ArrayArchiveLimits(max_total_array_elements=8, max_manifest_nesting=64)
    shared = jnp.arange(8)
    value = (shared, shared)
    recipe = model_structure_recipe(value)
    packed = _pack_owned_model_array_tree(value, recipe, prefix="leaves", limits=limits)
    assert packed["leaves/000000"] is packed["leaves/000001"]

    distinct = (jnp.arange(8), jnp.arange(8))
    with pytest.raises(ValueError, match="total element limit"):
        _pack_owned_model_array_tree(
            distinct, model_structure_recipe(distinct), prefix="leaves", limits=limits
        )
    mutable = np.arange(8)
    with pytest.raises(ValueError, match="total element limit"):
        _pack_owned_model_array_tree(
            (mutable, mutable),
            model_structure_recipe((mutable, mutable)),
            prefix="leaves",
            limits=limits,
        )
    immutable = np.frombuffer(mutable.tobytes(), dtype=mutable.dtype)
    packed = _pack_owned_model_array_tree(
        (immutable, immutable),
        model_structure_recipe((immutable, immutable)),
        prefix="leaves",
        limits=limits,
    )
    assert packed["leaves/000000"] is packed["leaves/000001"]


def test_model_immutable_shared_admission_preserves_wire_and_records(
    tmp_path: Path,
) -> None:
    from phydrax._model._structure import model_archive_root_projection
    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
    )

    mesh = _mesh()
    value = _composition(mesh, mesh)
    projected = model_archive_root_projection(model_archive_root_projection(value))
    leaves = jax.tree_util.tree_leaves(projected)
    seen: set[int] = set()
    physical = logical = 0
    for leaf in leaves:
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            continue
        count = int(leaf.size)
        logical += count
        backing: object = leaf
        while isinstance(backing, np.ndarray) and not backing.flags.writeable:
            backing = backing.base
        if isinstance(leaf, jax.Array) or isinstance(backing, bytes):
            if id(leaf) in seen:
                continue
            seen.add(id(leaf))
        physical += count
    assert 0 < physical < logical
    limits = dataclasses.replace(MODEL_ARCHIVE_LIMITS, max_total_array_elements=physical)
    bounded = write_model_artifact(tmp_path / "bounded.phx", value, limits=limits)
    baseline = write_model_artifact(tmp_path / "baseline.phx", value, limits=None)
    assert bounded.read_bytes() == baseline.read_bytes()
    assert model_archive_content_id(bounded, limits=limits) == model_archive_content_id(
        baseline, limits=None
    )
    restored = read_model_artifact(
        bounded,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=limits,
    )
    assert restored.entries[0].value is not restored.entries[1].value
    assert [entry.value.mesh_id for entry in restored.entries] == [mesh.mesh_id] * 2
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_archive(bounded, limits=limits)

    distinct = write_model_artifact(
        tmp_path / "distinct-records.phx", _composition(mesh, _mesh()), limits=None
    )
    decoded = _read(distinct)
    assert decoded.entries[0].value is not decoded.entries[1].value


def test_shared_payloads_bound_bytes_once_per_stored_owner() -> None:
    """The aggregate byte limit counts each stored owner once (plan section 10).

    Restoring the torus implicit volume (281 MB stored, 2.25 GB logical) left
    202 MB of distinct array buffers: owner constructors reuse shared immutable
    arrays, so restored array memory follows stored, not logical, bytes.
    Expanded per-occurrence APIs still charge every occurrence.
    """
    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax._model._structure import (
        _expand_shared_immutable_model_archive_recipe,
        _pack_owned_model_array_tree,
        expand_shared_model_archive_recipe,
        model_structure_recipe,
        pack_model_array_tree,
        share_model_archive_recipe,
    )

    limits = ArrayArchiveLimits(
        max_total_array_elements=1024, max_aggregate_bytes=64, max_manifest_nesting=64
    )

    def payload(start: int, /) -> np.ndarray:
        return np.frombuffer(bytes(range(start, start + 32)), dtype=np.uint8)

    def shared_recipe(value: Any, /) -> tuple[Any, Any, Any]:
        recipe = model_structure_recipe(value)
        arrays = pack_model_array_tree(value, recipe, prefix="leaves", limits=None)
        return recipe, *share_model_archive_recipe(recipe, arrays, prefix="leaves")

    # Three occurrences of one 32-byte owner: 96 logical bytes, 32 stored.
    one = payload(0)
    value = ((one,), (one,), (one,))
    recipe, shared, members = shared_recipe(value)
    assert _nodes(shared, "shared") and len(members) == 1
    packed = _pack_owned_model_array_tree(value, recipe, prefix="leaves", limits=limits)
    assert len(packed) == 3
    expanded, arrays = _expand_shared_immutable_model_archive_recipe(
        shared, members, "leaves", limits
    )
    assert expanded == recipe and len(arrays) == 3
    with pytest.raises(ValueError, match="total array-byte limit"):
        expand_shared_model_archive_recipe(
            shared, members, prefix="leaves", limits=limits
        )

    # Three distinct 32-byte owners store 96 bytes and are refused.
    distinct = ((payload(0),), (payload(32),), (payload(64),))
    distinct_recipe, distinct_shared, distinct_members = shared_recipe(distinct)
    assert len(distinct_members) == 3
    with pytest.raises(ValueError, match="total array-byte limit"):
        _pack_owned_model_array_tree(
            distinct, distinct_recipe, prefix="leaves", limits=limits
        )
    with pytest.raises(ValueError, match="total array-byte limit"):
        _expand_shared_immutable_model_archive_recipe(
            distinct_shared, distinct_members, "leaves", limits
        )


def _array_sharing_value() -> tuple[Any, ...]:
    # Equal int64 content under different parents (distinct objects), plus the
    # same eight bytes as another dtype and shape, which must stay distinct.
    return (
        (np.arange(6, dtype=np.int64), "left"),
        (np.arange(6, dtype=np.int64), "right"),
        (np.zeros(8, dtype=np.uint8), np.zeros(1, dtype=np.int64)),
    )


def _shared_array_recipe(
    value: Any, /
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    from phydrax._model._structure import (
        model_structure_recipe,
        pack_model_array_tree,
        share_model_archive_recipe,
    )

    recipe = model_structure_recipe(value)
    arrays = pack_model_array_tree(value, recipe, prefix="leaves", limits=None)
    shared, members = share_model_archive_recipe(recipe, arrays, prefix="leaves")
    return recipe, arrays, shared, members


def test_equal_arrays_under_different_parents_are_archived_once() -> None:
    from phydrax._model._structure import expand_shared_model_archive_recipe

    recipe, arrays, shared, members = _shared_array_recipe(_array_sharing_value())
    assert len(arrays) == 4
    assert list(members) == ["leaves/000000", "leaves/000001", "leaves/000002"]
    assert shared["items"][1]["items"][0]["kind"] == "shared"
    assert shared["items"][2]["items"][1]["kind"] == "array"

    # Archives depend only on content, not on whether the equal arrays alias.
    left = np.arange(6, dtype=np.int64)
    aliased = ((left, "left"), (left, "right"), _array_sharing_value()[2])
    _, _, aliased_shared, aliased_members = _shared_array_recipe(aliased)
    assert aliased_shared == shared
    assert list(aliased_members) == list(members)
    for name, value in aliased_members.items():
        assert value.dtype == members[name].dtype
        np.testing.assert_array_equal(value, members[name])

    expanded, restored = expand_shared_model_archive_recipe(
        shared, members, prefix="leaves", limits=None
    )
    assert expanded == recipe
    assert expanded["items"][0]["items"][0] is expanded["items"][1]["items"][0]
    assert set(restored) == set(arrays)
    for name, value in arrays.items():
        assert restored[name].dtype == value.dtype
        np.testing.assert_array_equal(restored[name], value)


@pytest.mark.parametrize("tamper", ("duplicate", "unresolved", "forward", "changed"))
def test_shared_array_references_refuse_noncanonical_or_changed_members(
    tamper: str,
) -> None:
    from phydrax._model._structure import expand_shared_model_archive_recipe

    _, _, shared, members = _shared_array_recipe(_array_sharing_value())
    shared = copy.deepcopy(shared)
    payloads = list(members.values())
    definition = shared["items"][0]["items"][0]
    reference = shared["items"][1]["items"][0]
    if tamper == "duplicate":
        shared["items"][1]["items"][0] = copy.deepcopy(definition)
        payloads.insert(1, payloads[0].copy())
        match = "repeats an equal array"
    elif tamper == "unresolved":
        reference["key"] = "0" * 64
        match = "no earlier equal definition"
    elif tamper == "forward":
        shared["items"][0]["items"][0] = reference
        shared["items"][1]["items"][0] = definition
        match = "no earlier equal definition"
    else:
        payloads[0] = payloads[0] + 1
        match = "no earlier equal definition"
    tampered = {f"leaves/{index:06d}": value for index, value in enumerate(payloads)}
    with pytest.raises(ValueError, match=match):
        expand_shared_model_archive_recipe(shared, tampered, prefix="leaves", limits=None)


def test_shared_array_digest_collision_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._model import _structure

    digest = _structure._RecipeContent.digest

    def colliding(self: Any, node: Any, /) -> None:
        digest(self, node)
        if node["kind"] == "array":
            self.keys[id(node)] = "0" * 64

    monkeypatch.setattr(_structure._RecipeContent, "digest", colliding)
    value = (
        (np.arange(3, dtype=np.int64), "left"),
        (np.ones(3, dtype=np.int64), "right"),
    )
    with pytest.raises(RuntimeError, match="content keys collided"):
        _shared_array_recipe(value)


def test_model_archive_stores_equal_arrays_of_different_meshes_once(
    tmp_path: Path,
) -> None:
    mesh = _mesh()
    # Same coordinates, different cells: the meshes differ, the coordinates don't.
    other = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))),
        np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int32),
    )
    assert other.mesh_id != mesh.mesh_id
    path = write_model_artifact(
        tmp_path / "meshes.phx", _composition(mesh, other), limits=None
    )
    _, digests = _archive(path)
    assert len(set(digests.values())) == len(digests)
    manifest, arrays = read_model_archive(path, limits=None)
    entries = manifest["model_recipe"]["fields"]["entries"]["items"]
    first, second = (entry["fields"]["value"] for entry in entries)
    assert first is not second
    assert first["fields"]["coordinates"] is second["fields"]["coordinates"]
    restored = _read(path)
    assert [entry.value.mesh_id for entry in restored.entries] == [
        mesh.mesh_id,
        other.mesh_id,
    ]
    np.testing.assert_array_equal(
        np.asarray(restored.entries[1].value.coordinates), np.asarray(other.coordinates)
    )
    assert len(arrays) > len(digests)
