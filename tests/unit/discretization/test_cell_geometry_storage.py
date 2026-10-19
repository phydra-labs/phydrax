#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import socket
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import (
    _cell_geometry as geometry_owner,
    CellBlock,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    CellGeometryStorageProjection,
    CellMesh,
    CellMeshStorage,
    CellValidityStatus,
    certify_cell_geometry_validity,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_enclosure import (
    coordinate_polynomials,
    evaluate,
)
from phydrax.discretization._mapped_geometry_measure import certified_cell_measures
from tests._support import cell_meshes
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.forced_devices import (
    gloo_rank_environment,
    run_on_forced_devices,
)
from tools.compile_counter import CompileCounter


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))


def test_nonrepresentable_midpoint_retains_positive_exact_map() -> None:
    source = np.asarray([[1.0], [np.nextafter(1.0, 2.0)]], dtype=np.float64)
    root = coordinate_lagrange_element("interval", 1)
    element = RestrictedCellGeometryElement(
        root,
        "interval",
        np.asarray([[0.5]], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
    )
    origin = CellGeometryRestrictionSource(
        canonical_fingerprint("source"),
        canonical_fingerprint("topology"),
        {"half": np.asarray([17], dtype=np.int64)},
        {"half": np.asarray([[23, 29]], dtype=np.int64)},
    )
    geometry = CellGeometrySpec(
        {"half": element},
        {"half": np.asarray([[0, 1]], dtype=np.int32)},
        source,
        restriction_source=origin,
    )
    # The sampled midpoint is exactly the left endpoint in FP64, unlike the map.
    mesh = CellMesh(
        np.asarray([[1.0], [1.0]], dtype=np.float64),
        (
            CellBlock(
                "half",
                "interval",
                np.asarray([[0, 1]], dtype=np.int32),
                global_ids=np.asarray([31], dtype=np.int64),
            ),
        ),
    )
    storage = cell_meshes.storage(mesh, geometry)
    local = CellMesh(mesh.coordinates, mesh.blocks, storage=storage)
    restored = storage.restore_geometry()
    certificate = certify_cell_geometry_validity(restored, mesh=local)
    assert int(certificate.status[0]) == CellValidityStatus.CERTIFIED_VALID
    assert float(certificate.determinant_lower[0]) > 0.0
    with coordinate_reference_budget() as budget:
        polynomials = coordinate_polynomials(
            restored.elements[0],
            np.asarray(restored.coordinates),
            budget=budget,
        )
        assert polynomials is not None
        exact_midpoint = (Fraction(source[0, 0]) + Fraction(source[1, 0])) / 2
        assert evaluate(polynomials[0], (Fraction(1),), budget=budget) == exact_midpoint
        assert Fraction(float(exact_midpoint)) != exact_midpoint
    assert cell_geometry_id(restored) == cell_geometry_id(geometry)
    with pytest.raises(ValueError, match="CellGeometrySpec|require_storage_geometry"):
        CellGeometrySpec.affine(local)


def test_copied_source_coefficient_mutation_is_rejected() -> None:
    root = coordinate_lagrange_element("triangle", 2)
    coefficients = np.asarray(root.reference_nodes).copy()
    coefficients[:, 1] += 0.125 * coefficients[:, 0] * coefficients[:, 1]
    mesh = CellMesh(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        (
            CellBlock(
                "curved",
                "triangle",
                np.asarray([[0, 1, 2]], dtype=np.int32),
                global_ids=np.asarray([37], dtype=np.int64),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"curved": root},
        {"curved": np.arange(root.local_dof_count, dtype=np.int32)[None]},
        coefficients,
    )
    storage = cell_meshes.storage(mesh, geometry)
    restored = storage.restore_geometry()
    copied = eqx.tree_at(
        lambda value: value.coordinates, restored, restored.coordinates.at[3, 1].add(0.01)
    )
    with pytest.raises(
        ValueError, match=r"CellGeometrySpec\.coordinates.*source coefficient binding"
    ):
        cell_geometry_id(copied)
    altered = eqx.tree_at(
        lambda value: value.coordinates, geometry, geometry.coordinates.at[3, 1].add(0.01)
    )
    with pytest.raises(
        ValueError,
        match=r"CellGeometrySpec\.logical_values.*original source coefficients",
    ):
        CellMeshStorage(
            storage.global_entity_counts,
            storage.entity_global_ids,
            storage.entity_owner,
            partition_index=0,
            partition_count=2,
            logical_topology_id=storage.logical_topology_id,
            logical_geometry_id=storage.logical_geometry_id,
            evidence_id=storage.evidence_id,
            logical_coordinate_geometry_id=storage.logical_coordinate_geometry_id,
            logical_arrays=storage.logical_arrays,
            local_coordinates=mesh.coordinates,
            local_blocks=mesh.blocks,
            global_coordinate_count=storage.global_coordinate_count,
            coordinate_global_ids=storage.coordinate_global_ids,
            coordinate_owner=storage.coordinate_owner,
            local_geometry=altered,
        )


@pytest.mark.parametrize(
    "degree,scale,area",
    [
        (1, 0.5, Fraction(1, 32)),
        (2, 0.5, Fraction(97, 3072)),
        (1, 0.1, Fraction(0.1) ** 4 / 2),
    ],
)
def test_nested_reference_map_preserves_original_basis(
    degree: int, scale: float, area: Fraction
) -> None:
    root = coordinate_lagrange_element("triangle", degree)
    coefficients = np.asarray(root.reference_nodes).copy()
    coefficients[:, 1] += 0.125 * coefficients[:, 0] * coefficients[:, 1]
    first = RestrictedCellGeometryElement(
        root,
        "triangle",
        np.eye(2, dtype=np.float64) * scale,
        np.zeros(2, dtype=np.float64),
    )
    element = RestrictedCellGeometryElement(
        first,
        "triangle",
        np.eye(2, dtype=np.float64) * scale,
        np.zeros(2, dtype=np.float64),
    )
    origin = CellGeometryRestrictionSource(
        canonical_fingerprint("original-source"),
        canonical_fingerprint("original-topology"),
        {"nested": np.asarray([17], dtype=np.int64)},
        {"nested": np.asarray([[23, 29, 41]], dtype=np.int64)},
    )
    geometry = CellGeometrySpec(
        {"nested": element},
        {"nested": np.arange(root.local_dof_count, dtype=np.int32)[None]},
        coefficients,
        restriction_source=origin,
    )
    values, _ = element.tabulate(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    )
    mesh = CellMesh(
        np.asarray(values) @ coefficients,
        (
            CellBlock(
                "nested",
                "triangle",
                np.asarray([[0, 1, 2]], dtype=np.int32),
                global_ids=np.asarray([47], dtype=np.int64),
            ),
        ),
    )
    storage = cell_meshes.storage(mesh, geometry)
    restored = storage.restore_geometry()
    local = CellMesh(mesh.coordinates, mesh.blocks, storage=storage)
    with coordinate_reference_budget() as budget:
        measure, error, exact = certified_cell_measures(local, restored, budget=budget)
    assert exact
    assert abs(Fraction(float(measure[0])) - area) <= Fraction(float(error[0]))
    np.testing.assert_array_equal(restored.coordinates, coefficients)
    corrupted = dict(storage.logical_arrays)
    corrupted["geometry/matrix/nested"] = (
        corrupted["geometry/matrix/nested"].at[0, 0, 0].set(0.5)
    )
    with pytest.raises(
        ValueError, match=r"CellGeometrySpec\.logical_values.*exact reference matrix"
    ):
        CellMeshStorage(
            storage.global_entity_counts,
            storage.entity_global_ids,
            storage.entity_owner,
            partition_index=0,
            partition_count=2,
            logical_topology_id=storage.logical_topology_id,
            logical_geometry_id=storage.logical_geometry_id,
            evidence_id=storage.evidence_id,
            logical_coordinate_geometry_id=storage.logical_coordinate_geometry_id,
            logical_arrays=tuple(sorted(corrupted.items())),
            local_coordinates=mesh.coordinates,
            local_blocks=mesh.blocks,
            global_coordinate_count=storage.global_coordinate_count,
            coordinate_global_ids=storage.coordinate_global_ids,
            coordinate_owner=storage.coordinate_owner,
            local_geometry=geometry,
        )


@pytest.mark.parametrize(
    "degree", [1, 2], ids=["odd-affine-control-count", "curved-control-map"]
)
def test_two_process_projection_preserves_unequal_owner_closure_maps(degree: int) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"

    def run_rank(rank: int) -> dict[str, object]:
        return run_on_forced_devices(
            "tests._support.discretization_process_scenarios",
            "projection_process",
            (str(degree),),
            devices=1,
            env=gloo_rank_environment(address, rank),
            timeout_seconds=120,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = tuple(executor.map(run_rank, range(2)))
    assert all(record["projection_valid"] for record in records)


def test_cold_projection_binds_original_coefficient_content(tmp_path: Path) -> None:
    element = coordinate_lagrange_element("triangle", 2)
    coefficients = np.asarray(element.reference_nodes).copy()
    coefficients[:, 1] += 0.125 * coefficients[:, 0] * coefficients[:, 1]
    mesh = CellMesh(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        (
            CellBlock(
                "curved",
                "triangle",
                np.asarray([[0, 1, 2]], dtype=np.int32),
                global_ids=np.asarray([37], dtype=np.int64),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"curved": element},
        {"curved": np.arange(element.local_dof_count, dtype=np.int32)[None]},
        coefficients,
    )
    seed = cell_meshes.storage(mesh, geometry)
    banks = dict(seed.logical_arrays)
    banks["closure/cell_ids"] = jnp.asarray([[37], [37]], dtype=jnp.int64)
    banks["closure/cell_valid"] = jnp.asarray([[True], [True]], dtype=jnp.bool_)
    arrays = tuple(sorted(banks.items()))
    projection = CellGeometryStorageProjection(
        arrays,
        cell_geometry_id(geometry),
        element.local_dof_count,
    )
    template = (projection, arrays, geometry)
    archive = tmp_path / "original-coordinate-map.eqx"
    eqx.tree_serialise_leaves(archive, template)
    projection, arrays, geometry = eqx.tree_deserialise_leaves(archive, template)
    entity_ids = tuple(entity.entity_ids for entity in mesh.topology.entity_sets)
    storage = CellMeshStorage(
        tuple(entity.count for entity in mesh.topology.entity_sets),
        entity_ids,
        tuple(np.zeros(value.shape, dtype=np.int32) for value in entity_ids),
        partition_index=0,
        partition_count=2,
        logical_topology_id=mesh.topology_id,
        logical_geometry_id=mesh.geometry_id,
        logical_coordinate_geometry_id=cell_geometry_id(geometry),
        evidence_id=seed.evidence_id,
        logical_arrays=arrays,
        local_coordinates=mesh.coordinates,
        local_blocks=mesh.blocks,
        global_coordinate_count=element.local_dof_count,
        coordinate_global_ids=seed.coordinate_global_ids,
        coordinate_owner=seed.coordinate_owner,
        local_geometry=geometry,
        geometry_projection=projection,
    )
    restored = storage.restore_geometry()
    local = CellMesh(mesh.coordinates, mesh.blocks, storage=storage)
    with coordinate_reference_budget() as budget:
        measure, error, _ = certified_cell_measures(local, restored, budget=budget)
    assert abs(Fraction(float(measure[0])) - Fraction(25, 48)) <= Fraction(
        float(error[0])
    )
    changed = dict(arrays)
    changed["geometry/coordinates"] = changed["geometry/coordinates"].at[3, 1].add(0.01)
    with pytest.raises(
        ValueError, match=r"CellGeometryStorageProjection\.source_content_id"
    ):
        projection.require_source(
            tuple(sorted(changed.items())), cell_geometry_id(geometry)
        )


@pytest.fixture(scope="module")
def logical_predicate_case() -> tuple[CellMesh, CellGeometrySpec, CellMeshStorage]:
    root = coordinate_lagrange_element("triangle", 1)
    first = RestrictedCellGeometryElement(root, "triangle", np.eye(2) * 0.1, np.zeros(2))
    element = RestrictedCellGeometryElement(
        first, "triangle", np.eye(2) * 0.1, np.zeros(2)
    )
    origin = CellGeometryRestrictionSource(
        canonical_fingerprint("predicate-source"),
        canonical_fingerprint("predicate-topology"),
        {"nested": np.asarray((17,), dtype=np.int64)},
        {"nested": np.asarray(((23, 29, 41),), dtype=np.int64)},
    )
    geometry = CellGeometrySpec(
        {"nested": element},
        {"nested": np.arange(root.local_dof_count, dtype=np.int32)[None]},
        np.asarray(root.reference_nodes),
        restriction_source=origin,
    )
    values, _ = element.tabulate(np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))))
    mesh = CellMesh(
        np.asarray(values) @ np.asarray(geometry.coordinates),
        (
            CellBlock(
                "nested",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((47,), dtype=np.int64),
            ),
        ),
    )
    return mesh, geometry, cell_meshes.storage(mesh, geometry)


def _require_predicate_case(
    case: tuple[CellMesh, CellGeometrySpec, CellMeshStorage],
    arrays: dict[str, jax.Array],
) -> None:
    mesh, geometry, storage = case
    geometry_owner.require_logical_geometry_lowering(
        geometry,
        mesh.blocks,
        tuple(arrays.items()),
        storage.global_coordinate_count,
        storage.coordinate_global_ids,
        storage.coordinate_owner,
        dict(storage.geometry_source_blocks),
    )


@pytest.mark.parametrize(
    "path,message",
    (
        ("geometry/coordinates", "original source coefficients"),
        ("geometry/coordinate_owners", "coordinate ownership"),
        ("geometry/routes/{bank}", "source coefficient routes"),
        ("geometry/matrix/{bank}", "exact reference matrix"),
        ("geometry/offset/{bank}", "exact reference offset"),
        ("geometry/reference_ancestry/{bank}", "exact composed ancestry"),
        ("geometry/parent_cell_ids/{bank}", "scientific root cell identities"),
        ("geometry/parent_vertex_ids/{bank}", "ordered scientific root corners"),
    ),
)
def test_lowering_fused_predicates_preserve_each_refusal(
    logical_predicate_case: tuple[CellMesh, CellGeometrySpec, CellMeshStorage],
    path: str,
    message: str,
) -> None:
    storage = logical_predicate_case[2]
    arrays = dict(storage.logical_arrays)
    _require_predicate_case(logical_predicate_case, arrays)
    bank = dict(storage.geometry_source_blocks)["nested"]
    name = path.format(bank=bank)
    value = np.asarray(arrays[name]).copy()
    value.flat[0] += 1
    arrays[name] = jax.device_put(value)
    with pytest.raises(ValueError) as caught:
        _require_predicate_case(logical_predicate_case, arrays)
    assert str(caught.value) == (
        f"CellGeometrySpec.logical_values must match logical {message}; got different local values."
    )


@pytest.mark.parametrize(
    "paths,message",
    (
        (
            ("geometry/coordinates", "geometry/coordinate_owners"),
            "original source coefficients",
        ),
        (("geometry/matrix/{bank}", "geometry/offset/{bank}"), "exact reference matrix"),
        (
            ("geometry/parent_cell_ids/{bank}", "geometry/parent_vertex_ids/{bank}"),
            "scientific root cell identities",
        ),
    ),
)
def test_lowering_fused_predicates_keep_refusal_order(
    logical_predicate_case: tuple[CellMesh, CellGeometrySpec, CellMeshStorage],
    paths: tuple[str, str],
    message: str,
) -> None:
    storage = logical_predicate_case[2]
    arrays = dict(storage.logical_arrays)
    bank = dict(storage.geometry_source_blocks)["nested"]
    for path in paths:
        name = path.format(bank=bank)
        value = np.asarray(arrays[name]).copy()
        value.flat[0] += 1
        arrays[name] = jax.device_put(value)
    with pytest.raises(ValueError) as caught:
        _require_predicate_case(logical_predicate_case, arrays)
    assert str(caught.value) == (
        f"CellGeometrySpec.logical_values must match logical {message}; got different local values."
    )


def test_fused_logical_equality_keeps_numpy_float_semantics_and_warm_identity() -> None:
    subnormal = np.nextafter(np.float64(0), np.float64(1))
    table = jax.device_put(np.asarray(((0.0, -0.0, subnormal, np.inf),)))
    expected = jax.device_put(np.asarray(((-0.0, 0.0, subnormal, np.inf),)))
    changed = jax.device_put(np.asarray(((-0.0, 0.0, 0.0, np.inf),)))
    nan = jax.device_put(np.asarray(((-0.0, 0.0, subnormal, np.nan),)))
    rows = jax.device_put(np.asarray((0,), dtype=np.int64))
    first = geometry_owner._logical_selected_values_equal(table, rows, expected)
    assert bool(first)
    with CompileCounter() as compiler:
        assert bool(geometry_owner._logical_selected_values_equal(table, rows, expected))
        assert not bool(
            geometry_owner._logical_selected_values_equal(table, rows, changed)
        )
        assert not bool(geometry_owner._logical_selected_values_equal(nan, rows, nan))
    assert compiler.count == 0


def test_fused_logical_row_lookup_preserves_absent_identity_refusal() -> None:
    ids = jax.device_put(np.asarray((1, 4, -1, -1), dtype=np.int64))
    missing = np.asarray((7,), dtype=np.int64)
    with pytest.raises(ValueError) as caught:
        geometry_owner._logical_geometry_rows(ids, missing)
    assert str(caught.value) == (
        "logical_geometry_rows.inputs must satisfy the condition: Local coordinate or cell identity is absent from logical geometry; got incompatible inputs."
    )


def test_lowering_reuses_fused_programs_and_rechecks_changed_values(
    logical_predicate_case: tuple[CellMesh, CellGeometrySpec, CellMeshStorage],
) -> None:
    arrays = dict(logical_predicate_case[2].logical_arrays)
    _require_predicate_case(logical_predicate_case, arrays)
    changed = dict(arrays)
    values = np.asarray(changed["geometry/coordinates"]).copy()
    values.flat[0] += 1.0
    changed["geometry/coordinates"] = jax.device_put(values)
    with CompileCounter() as compiler:
        _require_predicate_case(logical_predicate_case, arrays)
        with pytest.raises(ValueError, match="original source coefficients"):
            _require_predicate_case(logical_predicate_case, changed)
    assert compiler.count == 0
