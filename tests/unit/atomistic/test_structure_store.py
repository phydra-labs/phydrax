#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phydrax._fingerprint import canonical_json
from phydrax._publication import publish_resource_set
from phydrax._resource_set import ResourceSetLimits
from phydrax.atomistic import _structure_store as structure_store
from phydrax.atomistic._structure_store import (
    AtomisticStoreCommit,
    AtomisticStoreCompression,
    AtomisticStoreError,
    AtomisticStoreField,
    AtomisticStoreIndex,
    AtomisticStoreLayout,
    AtomisticStoreLevel,
    AtomisticStoreLimits,
    AtomisticStoreShard,
    AtomisticStructureBatch,
    AtomisticStructureRecord,
    AtomisticStructureSource,
    AtomisticStructureStoreSchema,
    AtomisticStructureStoreWriter,
    open_atomistic_structure_store,
)
from phydrax.axes import Axis, AxisKey
from phydrax.units import (
    ANGSTROM,
    derived_unit,
    ELECTRONVOLT,
    KELVIN,
    ONE,
    UnitDefinition,
)


CARTESIAN = Axis(AxisKey("atomistic", "cartesian"), 3, ("x", "y", "z"))
ENDPOINT = Axis(AxisKey("atomistic", "edge-endpoint"), 2, ("sender", "receiver"))
FORCE = derived_unit("eV/angstrom", ((ELECTRONVOLT, 1), (ANGSTROM, -1)))
CURVATURE = derived_unit("eV/angstrom^2", ((ELECTRONVOLT, 1), (ANGSTROM, -2)))
COMMIT_ONE = "00000000000000000001.json"


def _levels(*, edge_parent: str = "structure") -> tuple[AtomisticStoreLevel, ...]:
    return (
        AtomisticStoreLevel("structure", "structure"),
        AtomisticStoreLevel("atom", "atom", parent="structure"),
        AtomisticStoreLevel("molecule", "group", parent="structure"),
        AtomisticStoreLevel("residue", "segment", parent="molecule"),
        AtomisticStoreLevel(
            "edge",
            "edge",
            parent=edge_parent,
            derivation_id="minimum-image-radius-5-angstrom",
        ),
        AtomisticStoreLevel(
            "hessian_block",
            "product",
            factors=("atom", "atom"),
            maximum_rows_per_structure=16,
        ),
    )


def _fields(
    *, temperature_unit: UnitDefinition = KELVIN
) -> tuple[AtomisticStoreField, ...]:
    return (
        AtomisticStoreField(
            "cell",
            "structure",
            "float64",
            kind="continuous",
            role="geometry",
            convention="row-lattice-vectors",
            axes=(CARTESIAN, CARTESIAN),
            unit=ANGSTROM,
            extensivity="intensive",
        ),
        AtomisticStoreField(
            "charge",
            "structure",
            "int8",
            kind="categorical",
            role="condition",
            convention="total-charge-elementary",
        ),
        AtomisticStoreField(
            "temperature",
            "structure",
            "float32",
            kind="continuous",
            role="condition",
            convention="thermostat-target",
            unit=temperature_unit,
            extensivity="intensive",
        ),
        AtomisticStoreField(
            "energy",
            "structure",
            "float32",
            kind="continuous",
            role="label",
            convention="total-energy",
            unit=ELECTRONVOLT,
            extensivity="extensive",
            availability="optional",
        ),
        AtomisticStoreField(
            "polarizability",
            "structure",
            "complex128",
            kind="continuous",
            role="label",
            convention="dimensionless-test-response",
            axes=(CARTESIAN, CARTESIAN),
            unit=ONE,
            extensivity="extensive",
            availability="optional",
        ),
        AtomisticStoreField(
            "positions",
            "atom",
            "float64",
            kind="continuous",
            role="geometry",
            convention="cartesian",
            axes=(CARTESIAN,),
            unit=ANGSTROM,
            extensivity="intensive",
        ),
        AtomisticStoreField(
            "species",
            "atom",
            "int16",
            kind="categorical",
            role="identity",
            convention="atomic-number",
        ),
        AtomisticStoreField(
            "molecule",
            "atom",
            "int64",
            kind="index",
            role="identity",
            convention="molecule-membership",
            index=AtomisticStoreIndex("molecule", sentinel=-1),
        ),
        AtomisticStoreField(
            "forces",
            "atom",
            "float64",
            kind="continuous",
            role="label",
            convention="cartesian",
            axes=(CARTESIAN,),
            unit=FORCE,
            extensivity="intensive",
            availability="optional",
        ),
        AtomisticStoreField(
            "endpoints",
            "edge",
            "int32",
            kind="index",
            role="derived",
            convention="sender-receiver",
            axes=(ENDPOINT,),
            index=AtomisticStoreIndex("atom"),
        ),
        AtomisticStoreField(
            "image",
            "edge",
            "int32",
            kind="integer",
            role="geometry",
            convention="lattice-image-shift",
            axes=(CARTESIAN,),
        ),
        AtomisticStoreField(
            "residue_name",
            "residue",
            "int32",
            kind="categorical",
            role="identity",
            convention="test-residue-code",
        ),
        AtomisticStoreField(
            "hessian",
            "hessian_block",
            "float64",
            kind="continuous",
            role="label",
            convention="cartesian-atom-pair-block",
            axes=(CARTESIAN, CARTESIAN),
            unit=CURVATURE,
            extensivity="intensive",
            availability="optional",
        ),
    )


def _schema() -> AtomisticStructureStoreSchema:
    return AtomisticStructureStoreSchema(_levels(), _fields())


def _record_inputs(
    index: int,
    atoms: int,
    edges: int,
    residues: Sequence[int],
    *,
    energy: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(index)
    molecules = len(residues)
    forces_available = np.arange(atoms) % 2 == 0
    forces = rng.normal(size=(atoms, 3))
    forces[~forces_available] = 0.0
    values: dict[str, Any] = {
        "cell": rng.normal(size=(1, 3, 3)),
        "charge": np.asarray([index - 1], dtype=np.int8),
        "temperature": np.asarray([300.25 + index], dtype=np.float32),
        "polarizability": rng.normal(size=(1, 3, 3)) + 1j * rng.normal(size=(1, 3, 3)),
        "positions": rng.normal(size=(atoms, 3)),
        "species": (np.arange(atoms, dtype=np.int16) + 1) * 6,
        "molecule": (
            np.full(atoms, -1, dtype=np.int64)
            if molecules == 0
            else np.arange(atoms, dtype=np.int64) % molecules
        ),
        "forces": forces,
        "endpoints": rng.integers(0, atoms, size=(edges, 2), dtype=np.int32),
        "image": rng.integers(-1, 2, size=(edges, 3), dtype=np.int32),
        "residue_name": np.arange(sum(residues), dtype=np.int32) + 100,
        "hessian": rng.normal(size=(atoms * atoms, 3, 3)),
    }
    if energy:
        values["energy"] = np.asarray([-1.25 * (index + 1)], dtype=np.float32)
    return {
        "record_id": f"structure-{index}",
        "lineage_id": f"trajectory-{index % 2}",
        "counts": {
            "atom": [atoms],
            "edge": [edges],
            "molecule": [molecules],
            "residue": np.asarray(residues, dtype=np.int64),
        },
        "stable_ids": np.arange(atoms, dtype=np.int64) * 7 + 1000 * index,
        "values": values,
        "available": {"forces": forces_available},
    }


def _records(
    schema: AtomisticStructureStoreSchema,
) -> tuple[AtomisticStructureRecord, ...]:
    return (
        AtomisticStructureRecord(schema, **_record_inputs(0, 2, 2, (2,))),
        AtomisticStructureRecord(schema, **_record_inputs(1, 3, 4, (1, 2))),
        AtomisticStructureRecord(schema, **_record_inputs(2, 1, 0, (), energy=False)),
    )


def _source() -> AtomisticStructureSource:
    schema = _schema()
    return AtomisticStructureSource(schema, _records(schema))


def _limits(**overrides: int) -> AtomisticStoreLimits:
    values = {
        "max_commit_bytes": 1 << 20,
        "max_member_bytes": 1 << 20,
        "max_shard_bytes": 1 << 24,
        "max_shard_members": 8192,
        "max_chunk_bytes": 1 << 16,
        "max_index_bytes": 1 << 20,
        "max_decoded_bytes": 1 << 22,
        "max_read_bytes": 1 << 22,
        "max_records": 1000,
    }
    values.update(overrides)
    return AtomisticStoreLimits(**values)


def _writer(
    root: Path,
    *,
    records_per_shard: int = 2,
    target_chunk_bytes: int = 64,
    compression: AtomisticStoreCompression = "none",
    compression_level: int | None = None,
    limits: AtomisticStoreLimits | None = None,
) -> AtomisticStructureStoreWriter:
    return AtomisticStructureStoreWriter(
        root,
        layout=AtomisticStoreLayout(
            records_per_shard=records_per_shard,
            target_chunk_bytes=target_chunk_bytes,
            compression=compression,
            compression_level=compression_level,
        ),
        limits=_limits() if limits is None else limits,
    )


def _assert_records_equal(
    left: AtomisticStructureBatch, right: AtomisticStructureBatch
) -> None:
    assert left.record_ids == right.record_ids
    assert left.lineage_ids == right.lineage_ids
    assert left.record_digests == right.record_digests
    assert left.fields == right.fields
    assert left.index_basis == right.index_basis
    np.testing.assert_array_equal(left.stable_ids, right.stable_ids)
    assert set(left.parent_counts) == set(right.parent_counts)
    for level, counts in left.parent_counts.items():
        np.testing.assert_array_equal(counts, right.parent_counts[level])
    for level, counts in left.structure_counts.items():
        np.testing.assert_array_equal(counts, right.structure_counts[level])
    assert tuple(left.values) == tuple(right.values)
    for name, values in left.values.items():
        assert values.dtype == right.values[name].dtype
        assert values.shape == right.values[name].shape
        assert values.tobytes() == right.values[name].tobytes()
    assert tuple(left.available) == tuple(right.available)
    for name, mask in left.available.items():
        np.testing.assert_array_equal(mask, right.available[name])


def _assert_batches_equal(
    left: AtomisticStructureBatch, right: AtomisticStructureBatch
) -> None:
    np.testing.assert_array_equal(left.ordinals, right.ordinals)
    assert left.source_id == right.source_id
    _assert_records_equal(left, right)


@pytest.fixture
def zarr_runtime() -> None:
    pytest.importorskip("zarr")


def test_schema_declares_explicit_level_and_field_semantics() -> None:
    assert _schema().schema_id == _schema().schema_id
    assert (
        AtomisticStructureStoreSchema(_levels(), _fields(temperature_unit=ONE)).schema_id
        != _schema().schema_id
    )
    schema = _schema()
    assert schema.counted_levels == ("atom", "molecule", "residue", "edge")
    assert schema.level("edge").derivation_id == "minimum-image-radius-5-angstrom"
    with pytest.raises(ValueError, match="floating or complex"):
        AtomisticStoreField(
            "x",
            "atom",
            "int64",
            kind="continuous",
            role="label",
            convention="c",
            unit=ONE,
            extensivity="intensive",
        )
    with pytest.raises(ValueError, match="explicit unit"):
        AtomisticStoreField(
            "x",
            "atom",
            "float64",
            kind="continuous",
            role="label",
            convention="c",
            extensivity="intensive",
        )
    with pytest.raises(ValueError, match="index semantics"):
        AtomisticStoreField(
            "x", "atom", "int64", kind="index", role="derived", convention="c"
        )
    with pytest.raises(ValueError, match="index semantics"):
        AtomisticStoreField(
            "x",
            "atom",
            "int64",
            kind="categorical",
            role="identity",
            convention="c",
            index=AtomisticStoreIndex("atom"),
        )
    with pytest.raises(ValueError, match="direct children"):
        AtomisticStructureStoreSchema(_levels(edge_parent="molecule"), ())
    with pytest.raises(ValueError, match="undeclared"):
        AtomisticStructureStoreSchema(
            (
                AtomisticStoreLevel("structure", "structure"),
                AtomisticStoreLevel("atom", "atom", parent="structure"),
                AtomisticStoreLevel(
                    "block",
                    "product",
                    factors=("atom", "edge"),
                    maximum_rows_per_structure=4,
                ),
            ),
            (),
        )
    with pytest.raises(ValueError, match="Only edge levels"):
        AtomisticStoreLevel(
            "molecule", "group", parent="structure", derivation_id="cutoff"
        )


def test_integral_fields_are_indices_only_when_declared() -> None:
    schema = _schema()
    categorical = _record_inputs(0, 2, 2, (2,))
    categorical["values"]["species"] = np.asarray([99, 7000], dtype=np.int16)
    AtomisticStructureRecord(schema, **categorical)
    index = _record_inputs(0, 2, 2, (2,))
    index["values"]["molecule"] = np.asarray([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="molecule"):
        AtomisticStructureRecord(schema, **index)
    edges = _record_inputs(0, 2, 2, (2,))
    edges["values"]["endpoints"] = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="endpoints"):
        AtomisticStructureRecord(schema, **edges)


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (
            lambda inputs: inputs["values"].update(energy=np.asarray([1.0])),
            TypeError,
            "float32",
        ),
        (
            lambda inputs: inputs.update(stable_ids=np.asarray([5, 5], dtype=np.int64)),
            ValueError,
            "unique",
        ),
        (
            lambda inputs: inputs["values"]["forces"].__setitem__(1, 1.0),
            ValueError,
            "canonical zero",
        ),
        (
            lambda inputs: inputs["counts"].update(residue=np.asarray([1, 1])),
            ValueError,
            "parent row",
        ),
        (
            lambda inputs: inputs["values"].update(unknown=np.zeros(1)),
            ValueError,
            "Unexpected",
        ),
        (
            lambda inputs: inputs["values"].pop("positions"),
            ValueError,
            "missing",
        ),
        (
            lambda inputs: inputs["available"].update(cell=np.ones(1, dtype=np.bool_)),
            ValueError,
            "optional",
        ),
        (lambda inputs: inputs.update(record_id="has space"), ValueError, "printable"),
    ],
)
def test_record_refuses_inexact_or_inconsistent_input(
    mutate: Callable[[dict[str, Any]], object], error: type[Exception], match: str
) -> None:
    inputs = _record_inputs(0, 2, 2, (2,))
    mutate(inputs)
    with pytest.raises(error, match=match):
        AtomisticStructureRecord(_schema(), **inputs)


def test_product_level_rows_are_bounded_per_structure() -> None:
    with pytest.raises(ValueError, match="row bound"):
        AtomisticStructureRecord(_schema(), **_record_inputs(3, 5, 0, ()))


def test_reads_preserve_order_duplicates_projection_and_exact_dtypes() -> None:
    schema = _schema()
    records = _records(schema)
    source = AtomisticStructureSource(schema, records)
    batch = source.read([2, 0, 2], fields=("forces", "energy"))
    assert batch.record_ids == ("structure-2", "structure-0", "structure-2")
    assert batch.lineage_ids == ("trajectory-0", "trajectory-0", "trajectory-0")
    assert batch.fields == ("forces", "energy")
    assert tuple(batch.values) == ("forces", "energy")
    np.testing.assert_array_equal(batch.ordinals, [2, 0, 2])
    order = (records[2], records[0], records[2])
    expected_forces = np.concatenate([record.values["forces"] for record in order])
    assert batch.values["forces"].dtype == np.float64
    assert batch.values["forces"].tobytes() == expected_forces.tobytes()
    assert batch.values["energy"].dtype == np.float32
    np.testing.assert_array_equal(
        batch.values["energy"], np.asarray([0.0, -1.25, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(batch.available["energy"], [False, True, False])
    np.testing.assert_array_equal(
        batch.stable_ids, np.concatenate([record.stable_ids for record in order])
    )
    np.testing.assert_array_equal(batch.structure_counts["atom"], [1, 2, 1])
    np.testing.assert_array_equal(batch.offsets("atom"), [0, 1, 3, 4])
    assert "positions" not in batch.values
    with pytest.raises(ValueError):
        batch.values["forces"][0, 0] = 1.0
    full = source.read([1])
    assert full.values["polarizability"].dtype == np.complex128
    assert full.values["charge"].dtype == np.int8
    assert (
        full.values["temperature"].tobytes() == records[1].values["temperature"].tobytes()
    )
    with pytest.raises(ValueError, match="repeat"):
        source.read([0], fields=("energy", "energy"))
    with pytest.raises(ValueError, match="ordinal"):
        source.read([3])
    with pytest.raises(ValueError, match="non-empty"):
        source.read([])
    with pytest.raises(TypeError, match="integers"):
        source.read(np.asarray([0.0]))


def test_index_rebasing_is_explicit_and_keeps_sentinels() -> None:
    schema = _schema()
    records = _records(schema)
    source = AtomisticStructureSource(schema, records)
    fields = ("endpoints", "molecule")
    local = source.read([1, 0], fields=fields)
    rebased = source.read([1, 0], fields=fields, index_basis="batch-local")
    np.testing.assert_array_equal(
        local.values["endpoints"],
        np.concatenate([records[1].values["endpoints"], records[0].values["endpoints"]]),
    )
    np.testing.assert_array_equal(
        rebased.values["endpoints"],
        np.concatenate(
            [records[1].values["endpoints"], records[0].values["endpoints"] + 3]
        ),
    )
    assert rebased.values["endpoints"].dtype == np.int32
    np.testing.assert_array_equal(
        rebased.values["molecule"],
        np.concatenate(
            [records[1].values["molecule"], records[0].values["molecule"] + 2]
        ),
    )
    sentinel = source.read([0, 2], fields=("molecule",), index_basis="batch-local")
    np.testing.assert_array_equal(sentinel.values["molecule"], [0, 0, -1])
    repeated = source.read([0, 0], fields=("endpoints",), index_basis="batch-local")
    np.testing.assert_array_equal(
        repeated.values["endpoints"],
        np.concatenate(
            [records[0].values["endpoints"], records[0].values["endpoints"] + 2]
        ),
    )


def test_heterogeneous_levels_keep_explicit_cardinality() -> None:
    source = _source()
    np.testing.assert_array_equal(source.level_counts("hessian_block"), [4, 9, 1])
    np.testing.assert_array_equal(source.level_counts("residue"), [2, 3, 0])
    np.testing.assert_array_equal(source.level_counts("edge"), [2, 4, 0])
    batch = source.read([1])
    np.testing.assert_array_equal(batch.parent_counts["residue"], [1, 2])
    np.testing.assert_array_equal(batch.parent_counts["molecule"], [2])
    np.testing.assert_array_equal(batch.structure_counts["residue"], [3])
    assert batch.values["residue_name"].shape == (3,)
    assert batch.values["hessian"].shape == (9, 3, 3)
    with pytest.raises(ValueError, match="Unknown store level"):
        source.level_counts("bond")


def test_source_identity_binds_content_and_order() -> None:
    schema = _schema()
    records = _records(schema)
    assert _source().source_id == AtomisticStructureSource(schema, records).source_id
    assert (
        AtomisticStructureSource(schema, records[::-1]).source_id
        != AtomisticStructureSource(schema, records).source_id
    )
    with pytest.raises(ValueError, match="unique"):
        AtomisticStructureSource(schema, (records[0], records[0]))
    other = AtomisticStructureStoreSchema(_levels(), _fields(temperature_unit=ONE))
    with pytest.raises(ValueError, match="schema identity"):
        AtomisticStructureSource(other, records)


@pytest.mark.parametrize(("compression", "level"), [("none", None), ("zstd", 3)])
def test_store_round_trip_is_bitwise_and_layout_independent(
    zarr_runtime: None,
    tmp_path: Path,
    compression: AtomisticStoreCompression,
    level: int | None,
) -> None:
    source = _source()
    root = tmp_path / "store"
    writer = _writer(root, compression=compression, compression_level=level)
    created = writer.create(source.schema, name="fixture-store")
    appended = writer.append(created, source)
    assert appended.sequence == 1 and len(appended.shards) == 2
    assert appended.lineage.parent_revision_id == created.revision.revision_id
    opened = open_atomistic_structure_store(root, limits=_limits())
    assert opened.commit.commit_id == appended.commit_id
    assert opened.source_id == source.source_id == appended.content_id
    assert opened.record_ids == source.record_ids
    for level_name in source.schema.level_names:
        np.testing.assert_array_equal(
            opened.level_counts(level_name), source.level_counts(level_name)
        )
    for ordinals in ([0, 1, 2], [2, 0, 2, 1]):
        for basis in ("structure-local", "batch-local"):
            _assert_batches_equal(
                opened.read(ordinals, index_basis=basis),
                source.read(ordinals, index_basis=basis),
            )
    _assert_batches_equal(
        opened.read([1, 1, 0], fields=("hessian", "residue_name", "energy")),
        source.read([1, 1, 0], fields=("hessian", "residue_name", "energy")),
    )


def test_interrupted_commit_publication_keeps_prior_revision(
    zarr_runtime: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source()
    root = tmp_path / "store"
    writer = _writer(root)
    created = writer.create(source.schema, name="fixture-store")
    publish = structure_store.publish_bytes

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise OSError("interrupted before the commit link")

    monkeypatch.setattr(structure_store, "publish_bytes", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        writer.append(created, source)
    monkeypatch.setattr(structure_store, "publish_bytes", publish)
    prior = open_atomistic_structure_store(root, limits=_limits())
    assert prior.commit.commit_id == created.commit_id
    assert prior.record_count == 0
    assert sorted(os.listdir(root / "commits")) == ["00000000000000000000.json"]
    appended = writer.append(created, source)
    assert appended.sequence == 1
    reopened = open_atomistic_structure_store(root, limits=_limits())
    _assert_batches_equal(reopened.read([0, 1, 2]), source.read([0, 1, 2]))


def test_interrupted_shard_publication_publishes_no_revision(
    zarr_runtime: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source()
    root = tmp_path / "store"
    writer = _writer(root)
    created = writer.create(source.schema, name="fixture-store")
    publish = structure_store.publish_resource_set
    calls: list[object] = []

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError("interrupted shard publication")
        return publish(*args, **kwargs)

    monkeypatch.setattr(structure_store, "publish_resource_set", flaky)
    with pytest.raises(OSError, match="interrupted"):
        writer.append(created, source)
    monkeypatch.setattr(structure_store, "publish_resource_set", publish)
    assert open_atomistic_structure_store(root, limits=_limits()).commit.sequence == 0
    appended = writer.append(created, source)
    assert appended.shards[0].shard_id == Path(str(calls[0])).name
    reopened = open_atomistic_structure_store(root, limits=_limits())
    _assert_batches_equal(reopened.read([2, 1, 0]), source.read([2, 1, 0]))


def test_readers_keep_complete_prior_revisions_after_append(
    zarr_runtime: None, tmp_path: Path
) -> None:
    schema = _schema()
    records = _records(schema)
    first = AtomisticStructureSource(schema, records[:2])
    second = AtomisticStructureSource(schema, records[2:])
    root = tmp_path / "store"
    writer = _writer(root)
    created = writer.create(schema, name="fixture-store")
    revision = writer.append(created, first)
    reader = open_atomistic_structure_store(root, limits=_limits())
    head = writer.append(revision, second)
    _assert_batches_equal(reader.read([1, 0]), first.read([1, 0]))
    pinned = open_atomistic_structure_store(root, limits=_limits(), sequence=1)
    assert pinned.commit.commit_id == revision.commit_id
    assert pinned.source_id == first.source_id
    latest = open_atomistic_structure_store(root, limits=_limits())
    assert latest.commit.commit_id == head.commit_id
    assert latest.source_id == AtomisticStructureSource(schema, records).source_id


def test_revision_and_identity_refusal(zarr_runtime: None, tmp_path: Path) -> None:
    schema = _schema()
    records = _records(schema)
    first = AtomisticStructureSource(schema, records[:2])
    second = AtomisticStructureSource(schema, records[2:])
    root = tmp_path / "store"
    writer = _writer(root)
    created = writer.create(schema, name="fixture-store")
    revision = writer.append(created, first)
    with pytest.raises(AtomisticStoreError) as stale:
        writer.append(created, second)
    assert stale.value.reason == "conflict"
    with pytest.raises(AtomisticStoreError) as duplicate:
        writer.append(revision, first)
    assert duplicate.value.reason == "identity"
    other_schema = AtomisticStructureStoreSchema(_levels(), _fields(temperature_unit=ONE))
    other = AtomisticStructureSource(
        other_schema,
        (AtomisticStructureRecord(other_schema, **_record_inputs(5, 1, 0, ())),),
    )
    with pytest.raises(AtomisticStoreError) as foreign:
        writer.append(revision, other)
    assert foreign.value.reason == "identity"
    with pytest.raises(AtomisticStoreError) as recreate:
        writer.create(schema, name="fixture-store")
    assert recreate.value.reason == "conflict"
    with pytest.raises(AtomisticStoreError) as missing:
        open_atomistic_structure_store(root, limits=_limits(), sequence=5)
    assert missing.value.reason == "identity"
    (root / "commits" / f".00000000000000000002.json.{'0' * 32}.tmp").write_bytes(b"{")
    assert open_atomistic_structure_store(root, limits=_limits()).commit.sequence == 1
    (root / "commits" / "unexpected.json").write_bytes(b"{}")
    with pytest.raises(AtomisticStoreError) as stray:
        open_atomistic_structure_store(root, limits=_limits())
    assert stray.value.reason == "malformed"


def test_retract_and_compact_publish_lineage_revisions(
    zarr_runtime: None, tmp_path: Path
) -> None:
    schema = _schema()
    records = _records(schema)
    source = AtomisticStructureSource(schema, records)
    root = tmp_path / "store"
    writer = _writer(root)
    created = writer.create(schema, name="fixture-store")
    appended = writer.append(created, source)
    retracted = writer.retract(
        appended, ["structure-1"], reason="duplicate reference label"
    )
    assert retracted.operation == "retract" and retracted.record_count == 2
    assert retracted.lineage.parent_revision_id == appended.revision.revision_id
    assert retracted.retractions[0].record_ids == ("structure-1",)
    remaining = AtomisticStructureSource(schema, (records[0], records[2]))
    head = open_atomistic_structure_store(root, limits=_limits())
    assert head.record_ids == ("structure-0", "structure-2")
    assert head.source_id == remaining.source_id
    _assert_batches_equal(head.read([1, 0]), remaining.read([1, 0]))
    prior = open_atomistic_structure_store(root, limits=_limits(), sequence=1)
    assert prior.record_count == 3
    with pytest.raises(AtomisticStoreError) as reappend:
        writer.append(retracted, AtomisticStructureSource(schema, (records[1],)))
    assert reappend.value.reason == "identity"
    with pytest.raises(AtomisticStoreError) as unknown:
        writer.retract(retracted, ["structure-1"], reason="again")
    assert unknown.value.reason == "identity"
    compactor = _writer(
        root, records_per_shard=1, compression="zstd", compression_level=5
    )
    compacted = compactor.compact(retracted)
    assert compacted.operation == "compact"
    assert compacted.content_id == retracted.content_id
    assert compacted.revision.revision_id != retracted.revision.revision_id
    assert compacted.lineage.parent_lineage_id == retracted.lineage.lineage_id
    assert compacted.retractions == retracted.retractions
    assert len(compacted.shards) == 2
    assert all(not shard.retracted for shard in compacted.shards)
    assert not {shard.shard_id for shard in compacted.shards} & {
        shard.shard_id for shard in retracted.shards
    }
    reopened = open_atomistic_structure_store(root, limits=_limits())
    assert reopened.commit.commit_id == compacted.commit_id
    for basis in ("structure-local", "batch-local"):
        _assert_batches_equal(
            reopened.read([1, 0, 1], index_basis=basis),
            remaining.read([1, 0, 1], index_basis=basis),
        )


def test_tampered_store_content_is_refused(zarr_runtime: None, tmp_path: Path) -> None:
    source = _source()
    root = tmp_path / "store"
    writer = _writer(root)
    created = writer.create(source.schema, name="fixture-store")
    appended = writer.append(created, source)
    shard = appended.shards[0]
    chunk = next(
        path for path, _, _ in shard.members if path.startswith("value.positions/c/")
    )
    member = root / "shards" / shard.shard_id / chunk
    data = bytearray(member.read_bytes())
    data[0] ^= 1
    member.write_bytes(bytes(data))
    opened = open_atomistic_structure_store(root, limits=_limits())
    _assert_batches_equal(
        opened.read([0], fields=("energy",)), source.read([0], fields=("energy",))
    )
    with pytest.raises(AtomisticStoreError) as changed:
        opened.read([0], fields=("positions",))
    assert changed.value.reason == "inconsistent"
    commit = root / "commits" / COMMIT_ONE
    payload = json.loads(commit.read_bytes())
    payload["record_count"] = 99
    commit.write_bytes(canonical_json(payload).encode("ascii"))
    with pytest.raises(AtomisticStoreError) as forged:
        open_atomistic_structure_store(root, limits=_limits())
    assert forged.value.reason == "malformed"


def test_zarr_metadata_is_validated_before_decoding(
    zarr_runtime: None, tmp_path: Path
) -> None:
    source = _source()
    root = tmp_path / "store"
    writer = _writer(root, records_per_shard=3)
    created = writer.create(source.schema, name="fixture-store")
    appended = writer.append(created, source)
    shard = appended.shards[0]
    target = "value.positions/zarr.json"
    directory = root / "shards" / shard.shard_id
    metadata = json.loads((directory / target).read_bytes())
    metadata["shape"][0] = 10**12
    forged_bytes = json.dumps(metadata).encode("utf-8")
    members = tuple(
        (path, len(forged_bytes), hashlib.sha256(forged_bytes).hexdigest())
        if path == target
        else (path, size, digest)
        for path, size, digest in shard.members
    )
    forged_id = structure_store._aggregate_digest(members)
    (directory / target).write_bytes(forged_bytes)
    directory.rename(root / "shards" / forged_id)
    forged_shard = AtomisticStoreShard(
        forged_id,
        record_count=shard.record_count,
        level_rows=dict(shard.level_rows),
        record_id_bytes=shard.record_id_bytes,
        lineage_id_bytes=shard.lineage_id_bytes,
        target_chunk_bytes=shard.target_chunk_bytes,
        compression=shard.compression,
        compression_level=shard.compression_level,
        members=members,
    )
    forged = AtomisticStoreCommit(
        appended.schema,
        store_name=appended.store_name,
        sequence=1,
        operation="append",
        shards=(forged_shard,),
        retractions=(),
        content_id=appended.content_id,
        parent_commit_id=created.commit_id,
        parent_revision_id=created.revision.revision_id,
        parent_lineage_id=created.lineage.lineage_id,
    )
    (root / "commits" / COMMIT_ONE).write_bytes(
        canonical_json(structure_store._commit_payload(forged)).encode("ascii")
    )
    opened = open_atomistic_structure_store(root, limits=_limits())
    with pytest.raises(AtomisticStoreError) as malformed:
        opened.read([0], fields=("positions",))
    assert malformed.value.reason == "malformed"


def test_resource_limits_refuse_before_allocation(
    zarr_runtime: None, tmp_path: Path
) -> None:
    source = _source()
    root = tmp_path / "store"
    writer = _writer(root)
    created = writer.create(source.schema, name="fixture-store")
    with pytest.raises(AtomisticStoreError) as member:
        _writer(root, limits=_limits(max_member_bytes=16)).append(created, source)
    assert member.value.reason == "limit"
    assert open_atomistic_structure_store(root, limits=_limits()).commit.sequence == 0
    writer.append(created, source)
    small_read = open_atomistic_structure_store(root, limits=_limits(max_read_bytes=64))
    with pytest.raises(AtomisticStoreError) as read:
        small_read.read([0, 1, 2])
    assert read.value.reason == "limit"
    for overrides in (
        {"max_records": 2},
        {"max_commit_bytes": 128},
        {"max_index_bytes": 16},
        {"max_chunk_bytes": 8},
        {"max_shard_members": 4},
    ):
        with pytest.raises(AtomisticStoreError) as opened:
            open_atomistic_structure_store(root, limits=_limits(**overrides))
        assert opened.value.reason == "limit"


def test_limits_are_exact_int64_bounds() -> None:
    with pytest.raises(ValueError, match="max_decoded_bytes"):
        _limits(max_chunk_bytes=1 << 20, max_decoded_bytes=1 << 10)
    with pytest.raises(ValueError, match="int64"):
        _limits(max_read_bytes=1 << 63)
    with pytest.raises(ValueError, match="int64"):
        AtomisticStoreLevel(
            "pairs",
            "product",
            factors=("atom", "atom"),
            maximum_rows_per_structure=1 << 63,
        )


def _empty_shards(root: Path) -> bool:
    shards = root / "shards"
    return not shards.exists() or not any(shards.iterdir())


def test_decoded_chunk_and_read_decode_limits(zarr_runtime: None, tmp_path: Path) -> None:
    source = _source()
    root = tmp_path / "store"
    chunk = 1 << 12
    strict = _writer(
        root,
        target_chunk_bytes=chunk,
        compression="zstd",
        compression_level=3,
        limits=_limits(max_chunk_bytes=chunk // 2),
    )
    created = strict.create(source.schema, name="fixture-store")
    with pytest.raises(AtomisticStoreError, match="chunk limit") as encoded:
        strict.append(created, source)
    assert encoded.value.reason == "limit"
    assert _empty_shards(root)
    writer = _writer(
        root, target_chunk_bytes=chunk, compression="zstd", compression_level=3
    )
    appended = writer.append(created, source)
    # Compressed members are tiny; the padded decoded chunks are what is refused.
    assert max(size for shard in appended.shards for _, size, _ in shard.members) < chunk
    with pytest.raises(AtomisticStoreError, match="chunk limit") as decoded:
        open_atomistic_structure_store(root, limits=_limits(max_chunk_bytes=chunk // 2))
    assert decoded.value.reason == "limit"
    bounded = open_atomistic_structure_store(
        root, limits=_limits(max_chunk_bytes=chunk, max_decoded_bytes=2 * chunk)
    )
    # Records 0 and 1 share a shard whose stable_id and charge arrays each decode
    # exactly one padded chunk; repeated records do not decode a chunk twice.
    _assert_batches_equal(
        bounded.read([1, 0, 1], fields=("charge",)),
        source.read([1, 0, 1], fields=("charge",)),
    )
    with pytest.raises(AtomisticStoreError, match="decode limit") as touched:
        bounded.read([0], fields=("charge", "positions"))
    assert touched.value.reason == "limit"


def _forged_commit(
    root: Path,
    parent: AtomisticStoreCommit,
    shards: tuple[AtomisticStoreShard, ...],
    content_id: str,
) -> None:
    forged = AtomisticStoreCommit(
        parent.schema,
        store_name=parent.store_name,
        sequence=parent.sequence + 1,
        operation="append",
        shards=shards,
        retractions=(),
        content_id=content_id,
        parent_commit_id=parent.commit_id,
        parent_revision_id=parent.revision.revision_id,
        parent_lineage_id=parent.lineage.lineage_id,
    )
    path = root / "commits" / structure_store._commit_name(forged.sequence)
    path.write_bytes(
        canonical_json(structure_store._commit_payload(forged)).encode("ascii")
    )


def test_shard_cardinality_is_admitted_before_member_enumeration(
    zarr_runtime: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source()
    root = tmp_path / "store"
    writer = _writer(root, records_per_shard=3)
    created = writer.create(source.schema, name="fixture-store")
    shard = writer.append(created, source).shards[0]
    forged = AtomisticStoreShard(
        shard.shard_id,
        record_count=shard.record_count,
        level_rows={**dict(shard.level_rows), "atom": 10**15},
        record_id_bytes=shard.record_id_bytes,
        lineage_id_bytes=shard.lineage_id_bytes,
        target_chunk_bytes=shard.target_chunk_bytes,
        compression=shard.compression,
        compression_level=shard.compression_level,
        members=shard.members,
    )
    _forged_commit(root, created, (forged,), source.source_id)

    def enumerated(*args: object) -> object:
        raise AssertionError("member paths were enumerated before admission")

    monkeypatch.setattr(structure_store, "_member_paths", enumerated)
    with pytest.raises(AtomisticStoreError, match="directory entries") as entries:
        open_atomistic_structure_store(root, limits=_limits())
    assert entries.value.reason == "limit"
    with pytest.raises(AtomisticStoreError, match="canonical Zarr members") as listed:
        open_atomistic_structure_store(
            root, limits=_limits(max_shard_members=(1 << 63) - 1)
        )
    assert listed.value.reason == "malformed"
    with pytest.raises(ValueError, match="int64"):
        AtomisticStoreShard(
            shard.shard_id,
            record_count=shard.record_count,
            level_rows={**dict(shard.level_rows), "atom": 1 << 63},
            record_id_bytes=shard.record_id_bytes,
            lineage_id_bytes=shard.lineage_id_bytes,
            target_chunk_bytes=shard.target_chunk_bytes,
            compression=shard.compression,
            compression_level=shard.compression_level,
            members=shard.members,
        )


def test_wrapping_stored_counts_are_refused(zarr_runtime: None, tmp_path: Path) -> None:
    schema = _schema()
    source = AtomisticStructureSource(
        schema,
        tuple(
            AtomisticStructureRecord(schema, **_record_inputs(index, 2, 0, (1,)))
            for index in range(4)
        ),
    )
    root = tmp_path / "store"
    writer = _writer(root)
    created = writer.create(schema, name="fixture-store")
    batch = source.read(np.arange(4, dtype=np.int64))
    arrays = dict(
        structure_store._shard_arrays(
            schema, structure_store._columns_from_batch(schema, batch)
        )
    )
    # Four 2**62 counts wrap int64 offsets back to the committed zero edge rows.
    arrays["count.edge"] = np.full(4, 1 << 62, dtype=np.int64)
    level_rows = {
        level: int(source.level_counts(level).sum()) for level in schema.level_names
    }
    assert level_rows["edge"] == 0
    specs = structure_store._array_specs(
        schema,
        record_count=4,
        level_rows=level_rows,
        record_id_bytes=arrays["record_id.utf8"].size,
        lineage_id_bytes=arrays["lineage_id.utf8"].size,
        target_chunk_bytes=64,
    )
    members = structure_store._encode_shard(schema.schema_id, arrays, specs, "none", None)
    inventory = tuple(
        sorted(
            (path, len(data), hashlib.sha256(data).hexdigest())
            for path, data in members.items()
        )
    )
    shard = AtomisticStoreShard(
        structure_store._aggregate_digest(inventory),
        record_count=4,
        level_rows=level_rows,
        record_id_bytes=arrays["record_id.utf8"].size,
        lineage_id_bytes=arrays["lineage_id.utf8"].size,
        target_chunk_bytes=64,
        compression="none",
        compression_level=None,
        members=inventory,
    )
    publish_resource_set(
        root / "shards" / shard.shard_id,
        members,
        limits=ResourceSetLimits(
            max_total_bytes=1 << 24,
            max_member_bytes=1 << 20,
            max_members=8192,
            max_depth=16,
        ),
    )
    _forged_commit(root, created, (shard,), source.source_id)
    with pytest.raises(AtomisticStoreError, match="overflow int64") as wrapped:
        open_atomistic_structure_store(root, limits=_limits())
    assert wrapped.value.reason == "malformed"


def test_repeated_read_accounting_refuses_int64_overflow() -> None:
    schema = _schema()
    level_counts = {
        level: np.asarray([1 << 61], dtype=np.int64) for level in schema.level_names
    }
    with pytest.raises(AtomisticStoreError, match="int64") as accounting:
        structure_store._read_bytes(
            schema, level_counts, np.zeros(4, dtype=np.int64), ("positions",)
        )
    assert accounting.value.reason == "limit"


def test_writer_never_publishes_a_head_outside_its_limits(
    zarr_runtime: None, tmp_path: Path
) -> None:
    schema = _schema()
    records = _records(schema)
    first = AtomisticStructureSource(schema, records[:2])
    second = AtomisticStructureSource(schema, records[2:])
    root = tmp_path / "store"
    counted = _writer(root, limits=_limits(max_records=2))
    created = counted.create(schema, name="fixture-store")
    head = counted.append(created, first)
    sealed = sorted(os.listdir(root / "shards"))
    with pytest.raises(AtomisticStoreError, match="record limit") as records_limit:
        counted.append(head, second)
    assert records_limit.value.reason == "limit"
    assert sorted(os.listdir(root / "shards")) == sealed
    assert sorted(os.listdir(root / "commits")) == [
        "00000000000000000000.json",
        COMMIT_ONE,
    ]
    index_bytes = sum(
        structure_store._admit_specs(
            structure_store._shard_specs(schema, shard), _limits()
        )[1]
        for shard in head.shards
    )
    indexed = _writer(root, limits=_limits(max_index_bytes=index_bytes))
    with pytest.raises(AtomisticStoreError, match="index limit") as index_limit:
        indexed.append(head, second)
    assert index_limit.value.reason == "limit"
    reopened = open_atomistic_structure_store(
        root, limits=_limits(max_index_bytes=index_bytes)
    )
    assert reopened.commit.commit_id == head.commit_id
    _assert_batches_equal(reopened.read([1, 0]), first.read([1, 0]))
    retracted = indexed.retract(head, ["structure-0"], reason="recover under limits")
    assert (
        open_atomistic_structure_store(
            root, limits=_limits(max_index_bytes=index_bytes)
        ).commit.commit_id
        == retracted.commit_id
    )
