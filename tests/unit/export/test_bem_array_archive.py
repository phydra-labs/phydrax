import io
import json
import zipfile

import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax.export._array_archive import (
    BEMArchiveDescriptor,
    BEMArchiveLimits,
    BEMPlanArchiveRecord,
    BEMResultArchiveRecord,
    read_bem_array_archive,
    write_bem_array_archive,
)


def _descriptor():
    return BEMArchiveDescriptor(
        ambient_dimension=3,
        pde="laplace",
        geometry="closed-oriented-triangular-surface",
        formulation="dp0-galerkin-single-layer-strong",
        provider="blocked-direct-dp0-galerkin-3d",
        precision="float64",
        resource_evidence=("face-count=4", "resident-bytes=1024"),
        error_evidence=("prepared-pair-quadrature-estimates",),
        non_goals=(
            "continuum-discretization-certification",
            "3d-fmm-or-h-matrix-acceleration",
        ),
    )


def test_plan_and_result_records_round_trip_without_pickle_or_implicit_claims(tmp_path):
    plan = BEMPlanArchiveRecord(
        "plan-tetrahedron",
        _descriptor(),
        {"pair_classes": [4, 12, 0, 0, 0], "accuracy_supported": True},
        {
            "pair/keys": np.asarray([0, 1, 4, 5], dtype=np.int64),
            "pair/values": np.asarray([0.2, 0.1, 0.1, 0.3], dtype=np.float64),
        },
    )
    plan_path = tmp_path / "plan.pba"
    write_bem_array_archive(plan_path, plan)
    restored_plan = read_bem_array_archive(plan_path)

    assert isinstance(restored_plan, BEMPlanArchiveRecord)
    assert restored_plan.plan_id == plan.plan_id
    assert restored_plan.descriptor == plan.descriptor
    assert restored_plan.metadata == plan.metadata
    for name in plan.arrays:
        assert np.array_equal(restored_plan.arrays[name], plan.arrays[name])
        assert not restored_plan.arrays[name].flags.writeable

    result = BEMResultArchiveRecord(
        "result-three-columns",
        plan.plan_id,
        _descriptor(),
        {"transpose": False, "rhs_count": 3},
        {
            "values": np.asarray(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, np.nan, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                ]
            ),
            "column_status": np.asarray([0, 1, 0], dtype=np.int32),
        },
    )
    result_path = tmp_path / "result.pba"
    write_bem_array_archive(result_path, result)
    restored_result = read_bem_array_archive(result_path)

    assert isinstance(restored_result, BEMResultArchiveRecord)
    assert restored_result.result_id == result.result_id
    assert restored_result.plan_id == plan.plan_id
    assert np.array_equal(
        restored_result.arrays["values"], result.arrays["values"], equal_nan=True
    )
    assert np.array_equal(
        restored_result.arrays["column_status"], result.arrays["column_status"]
    )
    assert not restored_result.descriptor.continuum_certified

    with zipfile.ZipFile(result_path, mode="r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert "schema" not in manifest
        assert "version" not in manifest
        assert "timestamp" not in manifest
        for record in manifest["arrays"].values():
            payload = archive.read(record["member"])
            loaded = np.load(io.BytesIO(payload), allow_pickle=False)
            assert loaded.dtype.kind in "biufc"


def test_archive_checksum_corruption_and_read_size_limits_fail_closed(tmp_path):
    record = BEMResultArchiveRecord(
        "result-corruption",
        "plan-corruption",
        _descriptor(),
        {"rhs_count": 2},
        {"values": np.arange(8, dtype=np.float64).reshape((4, 2))},
    )
    path = tmp_path / "corrupt.pba"
    write_bem_array_archive(path, record)

    with zipfile.ZipFile(path, mode="r") as archive:
        members = [
            (info.filename, archive.read(info.filename)) for info in archive.infolist()
        ]
    rewritten = []
    for name, payload in members:
        if name.startswith("arrays/"):
            damaged = bytearray(payload)
            damaged[-1] ^= 0x01
            payload = bytes(damaged)
        rewritten.append((name, payload))
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in rewritten:
            archive.writestr(name, payload)

    with pytest.raises(ArrayArchiveCorruptionError, match="checksum failed"):
        read_bem_array_archive(path)

    clean = tmp_path / "oversized.pba"
    write_bem_array_archive(clean, record)
    with pytest.raises(ArrayArchiveCorruptionError, match="max_file_bytes"):
        read_bem_array_archive(clean, limits=BEMArchiveLimits(max_file_bytes=1))


def test_archive_write_limits_reject_before_emitting_output(tmp_path):
    record = BEMPlanArchiveRecord(
        "plan-too-large",
        _descriptor(),
        {"face_count": 32},
        {"pair/values": np.arange(32, dtype=np.float64)},
    )
    path = tmp_path / "rejected.pba"
    with pytest.raises(ValueError, match="max_array_bytes"):
        write_bem_array_archive(
            path,
            record,
            limits=BEMArchiveLimits(max_array_bytes=64),
        )
    assert not path.exists()


def test_records_reject_object_nonfinite_and_continuum_claims():
    with pytest.raises(TypeError, match="numeric or boolean"):
        BEMPlanArchiveRecord(
            "bad-object",
            _descriptor(),
            {},
            {"values": np.asarray([object()], dtype=object)},
        )
    with pytest.raises(ValueError, match="finite"):
        BEMPlanArchiveRecord(
            "bad-finite",
            _descriptor(),
            {},
            {"values": np.asarray([np.inf])},
        )
    with pytest.raises(ValueError, match="cannot claim continuum"):
        BEMArchiveDescriptor(
            ambient_dimension=3,
            pde="laplace",
            geometry="triangular-surface",
            formulation="dp0-galerkin",
            provider="blocked-direct",
            precision="float64",
            resource_evidence=("bounded",),
            error_evidence=("quadrature-only",),
            non_goals=("cad",),
            continuum_certified=True,
        )
