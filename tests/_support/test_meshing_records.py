# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellMesh, ExactPlcCellGeometrySource
from phydrax.geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from tests._support.distributed_meshing.cold_lifecycle_identities import (
    array_record,
    ArrayRecord,
    coordinate_record,
    image_source_identity,
    mesh_record,
    plc_source_record,
    record_array,
)
from tests._support.distributed_meshing.cold_lifecycle_images import (
    categorical_extraction,
)
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
    _require_instance,
    _require_present,
)
from tests._support.distributed_meshing.volume_generation import (
    _mixed_refinement_of_nondyadic_plc,
)


def test_array_record_preserves_int64_values_above_float_precision() -> None:
    value = 2**54 + 1
    array = np.asarray([value, -value], dtype="<i8")
    assert array_record(array) == {
        "dtype": "<i8",
        "shape": [2],
        "bytes": (
            value.to_bytes(8, "little", signed=True)
            + (-value).to_bytes(8, "little", signed=True)
        ).hex(),
    }


def test_record_array_decodes_independent_bytes_and_empty_rank() -> None:
    expected: ArrayRecord = {
        "dtype": "<i8",
        "shape": [1, 2],
        "bytes": "0100000000004000ffffffffffffbfff",
    }
    decoded = record_array(expected)
    assert decoded.dtype == np.dtype("<i8")
    np.testing.assert_array_equal(decoded, [[2**54 + 1, -(2**54 + 1)]])
    empty = record_array({"dtype": "<i8", "shape": [0, 3], "bytes": ""})
    assert empty.shape == (0, 3)
    assert empty.dtype == np.dtype("<i8")


def test_coordinate_record_preserves_rationals_and_row_order() -> None:
    assert coordinate_record(
        ((Fraction(1, 3), Fraction(-2, 7)), (Fraction(5, 11), Fraction(0)))
    ) == [[[1, 3], [-2, 7]], [[5, 11], [0, 1]]]


def test_mesh_record_retains_connectivity_and_stable_identity_banks() -> None:
    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
        vertex_global_ids=np.asarray((2**54 + 1, 9, 17), dtype=np.int64),
        cell_global_ids=np.asarray((91,), dtype=np.int64),
    )
    record = mesh_record(mesh)
    assert set(record) == {"coordinates", "vertex_global_ids", "blocks"}
    np.testing.assert_array_equal(record_array(record["coordinates"]), mesh.coordinates)
    np.testing.assert_array_equal(
        record_array(record["vertex_global_ids"]), [2**54 + 1, 9, 17]
    )
    assert len(record["blocks"]) == 1
    assert set(record["blocks"][0]) == {"global_ids", "vertices"}
    np.testing.assert_array_equal(record_array(record["blocks"][0]["global_ids"]), [91])
    np.testing.assert_array_equal(
        record_array(record["blocks"][0]["vertices"]), [[0, 1, 2]]
    )


@pytest.fixture(scope="module")
def exact_source() -> ExactPlcCellGeometrySource:
    generated, _ = _mixed_refinement_of_nondyadic_plc(
        phx.meshing.MeshingLimits(maximum_vertices=100, maximum_cells=1000)
    )
    source = generated.geometry.exact_source
    assert isinstance(source, ExactPlcCellGeometrySource)
    return source


def test_plc_source_record_retains_all_ten_scientific_banks(
    exact_source: ExactPlcCellGeometrySource,
) -> None:
    record = plc_source_record(exact_source)
    expected = {
        "source_points": exact_source.source_points,
        "source_triangles": exact_source.source_triangles,
        "source_segments": exact_source.source_segments,
        "source_triangle_ids": exact_source.source_triangle_ids,
        "source_triangle_bounds": exact_source.source_triangle_bounds,
        "source_segment_ids": exact_source.source_segment_ids,
        "source_segment_bounds": exact_source.source_segment_bounds,
        "vertex_strata": exact_source.vertex_strata,
        "vertex_rows": exact_source.vertex_rows,
        "vertex_parameters": exact_source.vertex_parameters,
    }
    assert set(record) == set(expected)
    for name, array in expected.items():
        decoded = record_array(record[name])
        assert decoded.dtype == np.asarray(array).dtype, name
        assert decoded.shape == np.asarray(array).shape, name
        assert decoded.tobytes() == np.asarray(array).tobytes(), name


def test_image_identity_retains_source_and_alias_contracts() -> None:
    source = LabelFieldVolumeBinding(
        categorical_extraction(), phx.SpatialCoordinateContract.si()
    )
    record = image_source_identity(source)
    assert set(record) == {
        "type",
        "source",
        "revision",
        "interpretation",
        "coordinates",
        "interfaces",
        "binding",
        "extraction",
        "domain",
        "lineage_alias",
        "evidence_alias",
    }
    assert record["source"] == source.source_id
    assert record["revision"] == source.source_revision
    assert record["binding"] == source.binding_id
    assert record["lineage_alias"] is True
    assert record["evidence_alias"] is True


def test_distributed_array_observations_retain_dtype_rank_and_large_ids() -> None:
    checks = _new_checks(("large_ids", "empty", "unused"))
    assert checks == {"large_ids": [], "empty": [], "unused": []}
    large = np.asarray((2**54 + 1,), dtype=np.int64)
    _record_array(checks, "large_ids", large, large)
    assert checks["large_ids"] == [
        [
            {"values": [2**54 + 1], "shape": [1], "dtype": np.dtype(np.int64).str},
            {"values": [2**54 + 1], "shape": [1], "dtype": np.dtype(np.int64).str},
        ]
    ]
    empty = np.empty((0, 3), dtype=np.int32)
    _record_array(checks, "empty", empty, empty)
    assert checks["empty"] == [
        [
            {"values": [], "shape": [0, 3], "dtype": np.dtype(np.int32).str},
            {"values": [], "shape": [0, 3], "dtype": np.dtype(np.int32).str},
        ]
    ]
    assert checks["unused"] == []


def test_required_witness_presence_retains_identity_and_refuses_absence() -> None:
    witness = Fraction(1, 3)
    assert _require_present(witness) is witness
    with pytest.raises(AssertionError, match="Scenario prerequisite"):
        _require_present(None)


def test_required_witness_type_does_not_infer_identity_from_equal_values() -> None:
    witness = Fraction(1, 3)
    assert _require_instance(witness, Fraction) is witness
    with pytest.raises(AssertionError, match="Scenario prerequisite"):
        _require_instance(1 / 3, Fraction)
