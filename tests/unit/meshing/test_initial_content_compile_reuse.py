# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Initial source joins preserve retained bytes and reuse dynamic numerical work."""

import jax
import numpy as np
import pytest

from phydrax._fingerprint import (
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from phydrax.meshing._initial_certification import (
    _collective_initial_content,
    _initial_content_values,
    _InitialOwnerRows,
    _placement,
    _SURFACE_CHECKS,
)
from tools.compile_counter import CompileCounter


def _inputs() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    # Two owners, duplicate seam vertices, and unused slots on both owners.
    raw = {
        "vertex_ids": np.asarray((0, 1, 2, -1, 1, 2, 3, -1), dtype=np.int64),
        "vertex_owners": np.asarray((0, 0, 0, -1, 0, 0, 1, -1), dtype=np.int32),
        "coordinates": np.asarray(
            (
                (-0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0),
                (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0),
            ),
            dtype=np.float64,
        ),
        "cell_ids": np.asarray((10, -1, 20, -1), dtype=np.int64),
        "corners": np.asarray(
            ((0, 1, 2), (-1, -1, -1), (1, 3, 2), (-1, -1, -1)),
            dtype=np.int64,
        ),
        "curve_edges": np.full((12, 2), -1, dtype=np.int64),
        "curve_values": np.full((12,), -1, dtype=np.int64),
        "curve_along": np.full((12, 2), -1, dtype=np.int64),
        "curve_uses": np.ones((4,), dtype=np.int32),
        "work_checks": np.eye(2, dtype=np.bool_),
        "checks": np.ones((2, len(_SURFACE_CHECKS)), dtype=np.bool_),
        "global_counts": np.asarray((4, 2), dtype=np.int64),
        "source_basis": np.arange(32, dtype=np.uint8),
    }
    curve_rows = np.asarray((0, 1, 6, 7), dtype=np.int64)
    raw["curve_edges"][curve_rows] = ((0, 1), (0, 2), (1, 3), (2, 3))
    raw["curve_values"][curve_rows] = (0, 1, 2, 3)
    raw["curve_along"][curve_rows] = ((1, 0), (0, 2), (3, 1), (2, 3))
    vertices = {
        "source_dimensions": np.asarray((0, 1, 1, -1, 1, 1, 2, -1), dtype=np.int64),
        "source_parameters": np.asarray(
            (
                (np.nan, -0.0), (0.25, np.nan), (0.5, np.nan), (0.0, 0.0),
                (0.25, np.nan), (0.5, np.nan), (0.75, np.nan), (0.0, 0.0),
            ),
            dtype=np.float64,
        ),
    }
    cells = {"patches": np.asarray((4, -1, 5, -1), dtype=np.int64)}
    curves = {"deviations": np.zeros((12,), dtype=np.float64)}
    curves["deviations"][curve_rows] = (0.125, -0.0, 0.375, 0.5)
    return raw, vertices, cells, curves


def _expected() -> tuple[
    dict[str, np.ndarray],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
]:
    vertex_ids = np.asarray((0, 1, 2, 3, -1, -1, -1, -1), dtype=np.int64)
    cell_ids = np.asarray((10, 20, -1, -1), dtype=np.int64)
    edge_ids = np.asarray((0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1), dtype=np.int64)
    edges = np.full((12, 2), -1, dtype=np.int64)
    edges[:5] = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
    corners = np.full((4, 3), -1, dtype=np.int64)
    corners[:2] = ((0, 1, 2), (1, 3, 2))
    coordinates = np.zeros((8, 2), dtype=np.float64)
    coordinates[:4] = ((-0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    parameters = np.zeros((8, 2), dtype=np.float64)
    parameters[:4] = ((np.nan, -0.0), (0.25, np.nan), (0.5, np.nan), (0.75, np.nan))
    along = np.tile(np.asarray((1, 0), dtype=np.int64), (12, 1))
    along[:5] = ((1, 0), (0, 2), (1, 0), (3, 1), (2, 3))
    deviations = np.full((12,), 0.125, dtype=np.float64)
    deviations[:5] = (0.125, -0.0, 0.125, 0.375, 0.5)
    basis = np.zeros((4, 32), dtype=np.uint8)
    basis[:2] = np.arange(32, dtype=np.uint8)
    vertex_owners = np.asarray((0, 0, 0, 1, -1, -1, -1, -1), dtype=np.int32)
    edge_owners = np.asarray((0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=np.int32)
    cell_owners = np.asarray((0, 1, 1, 1), dtype=np.int32)
    arrays = {
        "vertex_global_ids": vertex_ids,
        "cell_global_ids": cell_ids,
        "cell_vertices": corners,
        "coordinates": coordinates,
        "cell_owners": cell_owners,
        "entity_global_ids_1": edge_ids,
        "entity_vertices_1": edges,
        "entity_active_0": np.arange(8) < 4,
        "entity_active_1": np.arange(12) < 5,
        "entity_active_2": np.arange(4) < 2,
        "initial/source_dimensions": np.asarray(
            (0, 1, 1, 2, -1, -1, -1, -1), dtype=np.int64
        ),
        "initial/source_parameters": parameters,
        "initial/cell_patches": np.asarray((4, 5, -1, -1), dtype=np.int64),
        "initial/edge_curves": np.asarray(
            (0, 1, -1, 2, 3, -1, -1, -1, -1, -1, -1, -1), dtype=np.int64
        ),
        "initial/edge_physical": np.asarray(
            (True, True, False, True, True, False, False, False, False, False, False, False)
        ),
        "initial/edge_along": along,
        "initial/edge_deviations": deviations,
        "source_basis": basis,
    }
    return (
        arrays,
        (vertex_ids[:, None], edges, cell_ids[:, None]),
        (vertex_ids, edge_ids, cell_ids),
        (vertex_owners, edge_owners, cell_owners),
        np.asarray((4, 5, 2), dtype=np.int64),
        np.ones((2, len(_SURFACE_CHECKS)), dtype=np.bool_),
    )


def _assert_bytes(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        observed, reference = np.asarray(left), np.asarray(right)
        assert observed.dtype == reference.dtype
        assert observed.shape == reference.shape
        assert observed.tobytes(order="C") == reference.tobytes(order="C")


def test_initial_join_matches_all_retained_bytes_and_digest() -> None:
    actual = jax.block_until_ready(_initial_content_values(*jax.device_put(_inputs())))
    expected = _expected()
    _assert_bytes(actual, expected)
    assert logical_array_value_collection_digest(
        actual[0]
    ) == logical_array_value_collection_digest(expected[0])


def test_initial_host_staging_preserves_topology_and_geometry_identity_recipes() -> None:
    raw, vertices, cells, curves = _inputs()
    vertex_rows = np.asarray((0, 1, 2, 6), dtype=np.int64)
    cell_rows = np.asarray((0, 2), dtype=np.int64)
    curve_rows = np.asarray((0, 1, 6, 7), dtype=np.int64)
    rows = _InitialOwnerRows(
        coordinates=raw["coordinates"][vertex_rows],
        vertex_ids=raw["vertex_ids"][vertex_rows],
        vertex_owners=np.zeros((4,), dtype=np.int32),
        corners=raw["corners"][cell_rows],
        cell_ids=raw["cell_ids"][cell_rows],
        vertex_banks=tuple(
            (name, bank[vertex_rows], -1 if bank.dtype.kind == "i" else 0)
            for name, bank in vertices.items()
        ),
        cell_banks=tuple((name, bank[cell_rows], -1) for name, bank in cells.items()),
        curve_edges=raw["curve_edges"][curve_rows],
        curve_along=raw["curve_along"][curve_rows],
        curve_values=raw["curve_values"][curve_rows],
        curve_banks=tuple((name, bank[curve_rows]) for name, bank in curves.items()),
        curve_uses=raw["curve_uses"],
        work_checks=np.ones((2,), dtype=np.bool_),
        global_vertex_count=4,
        global_cell_count=2,
    )
    sharding = _placement()

    def stage() -> tuple:
        return _collective_initial_content(
            rows,
            sharding,
            vertex_capacity=4,
            cell_capacity=2,
            block="initial-byte-contract",
            width=2,
        )

    actual = stage()
    with CompileCounter(capture_stack=True) as warm:
        repeated = stage()
        jax.block_until_ready(repeated)
    assert warm.count == 0, [event.to_record() for event in warm.events]
    _assert_bytes(actual[:4], repeated[:4])
    assert actual[4:7] == repeated[4:7]
    arrays = _expected()[0]
    shapes = {
        "cell_global_ids": (2,),
        "cell_vertices": (2, 3),
        "vertex_global_ids": (4,),
        "entity_global_ids_1": (5,),
        "entity_vertices_1": (5, 2),
    }
    topology = canonical_fingerprint(
        {
            "kind": "cell-mesh-topology",
            "dimension": 2,
            "blocks": [("initial-byte-contract", "triangle")],
            "arrays": logical_array_value_collection_digest(
                {name: arrays[name] for name in shapes}, logical_shapes=shapes
            ),
        }
    )
    geometry = canonical_fingerprint(
        {
            "kind": "cell-mesh-geometry",
            "topology": topology,
            "ambient_dimension": 2,
            "coordinates": logical_array_value_collection_digest(
                {"coordinates": arrays["coordinates"]},
                logical_shapes={"coordinates": (4, 2)},
            ),
        }
    )
    assert actual[4] == (4, 5, 2)
    assert actual[5:7] == (topology, geometry)


def test_initial_join_has_one_cold_compile_and_reuses_changed_active_counts() -> None:
    first = jax.device_put(_inputs())
    changed = _inputs()
    raw = changed[0]
    raw["vertex_ids"][6] = -1
    raw["cell_ids"][2] = -1
    raw["global_counts"][:] = (3, 1)
    raw["curve_edges"][6:8] = -1
    raw["curve_values"][6:8] = -1
    raw["curve_edges"][2] = (1, 2)
    raw["curve_values"][2] = 2
    raw["curve_along"][2] = (1, 2)
    second = jax.device_put(changed)
    jax.clear_caches()
    with CompileCounter() as cold:
        initial = jax.block_until_ready(_initial_content_values(*first))
    with CompileCounter(capture_stack=True) as warm:
        repeated = jax.block_until_ready(_initial_content_values(*first))
        reduced = jax.block_until_ready(_initial_content_values(*second))
    assert cold.count == 1
    assert warm.count == 0, [event.to_record() for event in warm.events]
    _assert_bytes(initial, repeated)
    np.testing.assert_array_equal(reduced[4], (3, 3, 1))
    assert np.all(np.asarray(reduced[5]))


@pytest.mark.parametrize(
    ("defect", "check"),
    (
        ("coordinates", 6),
        ("vertex_bank", 6),
        ("ownership", 5),
        ("global_counts", 5),
        ("coverage", 1),
        ("incidence", 7),
        ("orientation", 7),
    ),
)
def test_initial_join_retains_scientific_failure_admission(defect: str, check: int) -> None:
    values = _inputs()
    raw, vertices, _, _ = values
    if defect == "coordinates":
        raw["coordinates"][4, 0] += 0.125
    elif defect == "vertex_bank":
        vertices["source_parameters"][4, 0] += 0.125
    elif defect == "ownership":
        raw["vertex_owners"][0] = 1
    elif defect == "global_counts":
        raw["global_counts"][0] += 1
    elif defect == "coverage":
        raw["work_checks"][1, 0] = True
    elif defect == "incidence":
        raw["curve_uses"][0] = 2
    elif defect == "orientation":
        raw["corners"][2] = (1, 2, 3)
    result = jax.block_until_ready(_initial_content_values(*jax.device_put(values)))
    observed = np.asarray(result[5])
    assert not np.any(observed[:, check])
    assert not np.all(observed)
