# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Isolated discretization storage and integer-width scenarios."""

from fractions import Fraction

import jax
import numpy as np

from phydrax import config as pxconfig
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.discretization import (
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
from phydrax.discretization._coordinate_enclosure import coordinate_source_signature
from phydrax.discretization._mapped_geometry_measure import certified_cell_measures
from tests._support.coordinate_references import coordinate_reference_budget


def _digest(value: object) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(canonical_fingerprint(value)), dtype=np.uint8)


def projection_process(degree_text: str) -> dict[str, bool]:
    degree = int(degree_text)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    rank = jax.process_index()
    placement = Mesh(np.asarray(jax.devices()), ("part",))
    sharding = NamedSharding(placement, PartitionSpec("part"))
    replicated = NamedSharding(placement, PartitionSpec())
    element = coordinate_lagrange_element("triangle", degree)
    count = element.local_dof_count
    padding = ((count + 1) // 2) * 2 - count
    coordinate_ids = np.arange(100, 100 + count, dtype=np.int64)
    coefficients = np.asarray(element.reference_nodes).copy()
    coefficients[:, 1] += 0.125 * coefficients[:, 0] * coefficients[:, 1]
    original = CellGeometrySpec(
        {"root": element}, {"root": np.arange(count, dtype=np.int32)[None]}, coefficients
    )
    original_mesh = CellMesh(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        (
            CellBlock(
                "root",
                "triangle",
                np.asarray([[0, 1, 2]], dtype=np.int32),
                global_ids=np.asarray([7], dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray([23, 29, 41], dtype=np.int64),
    )
    basis = _digest(
        {
            "signature": coordinate_source_signature(element),
            "arrays": array_tree_fingerprint(element),
        }
    )
    host = {
        "geometry/coordinate_ids": np.concatenate(
            (coordinate_ids, np.full((padding,), -1, dtype=np.int64))
        ),
        "geometry/coordinate_owners": np.concatenate(
            (np.zeros((count,), dtype=np.int32), np.full((padding,), -1, dtype=np.int32))
        ),
        "geometry/coordinates": np.concatenate(
            (coefficients, np.zeros((padding, 2), dtype=np.float64))
        ),
        "geometry/cell_ids/root": np.asarray([17, 19], dtype=np.int64),
        "geometry/routes/root": np.tile(coordinate_ids, (2, 1)),
        "geometry/matrix/root": np.asarray(
            [np.eye(2) * 0.5, np.eye(2) * 0.25], dtype=np.float64
        ),
        "geometry/offset/root": np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float64),
        "geometry/parent_cell_ids/root": np.asarray([7, 7], dtype=np.int64),
        "geometry/parent_vertex_ids/root": np.asarray(
            [[23, 29, 41], [23, 29, 41]], dtype=np.int64
        ),
        "geometry/source_basis/root": np.tile(basis, (2, 1)),
        "geometry/source_geometry_id": _digest(cell_geometry_id(original)),
        "geometry/source_topology_id": _digest(original_mesh.topology_id),
        "closure/cell_ids": np.asarray([[17, 19], [19, -1]], dtype=np.int64),
        "closure/cell_valid": np.asarray([[True, True], [True, False]], dtype=np.bool_),
    }
    identities = ("geometry/source_geometry_id", "geometry/source_topology_id")
    arrays = tuple(
        (
            name,
            jax.make_array_from_callback(
                values.shape,
                replicated if name in identities else sharding,
                lambda selection, values=values: values[selection],
            ),
        )
        for name, values in sorted(host.items())
    )
    map_id = canonical_fingerprint(
        array_tree_fingerprint(
            {
                name: values
                for name, values in host.items()
                if name.startswith("geometry/") and name != "geometry/coordinate_owners"
            }
        )
    )
    projection = CellGeometryStorageProjection(arrays, map_id, count)
    packets = dict(projection.addressable_arrays(rank))
    cell_ids = np.asarray(packets["geometry/cell_ids/root"])
    np.testing.assert_array_equal(cell_ids, [17, 19] if rank == 0 else [19])
    np.testing.assert_array_equal(packets["geometry/coordinate_ids"], coordinate_ids)
    points, blocks, elements, routes, parents, corners, banks = [], [], {}, {}, {}, {}, {}
    for index, identifier in enumerate(cell_ids):
        name = f"local-{identifier}"
        restriction = RestrictedCellGeometryElement(
            element,
            "triangle",
            np.asarray(packets["geometry/matrix/root"])[index],
            np.asarray(packets["geometry/offset/root"])[index],
        )
        values, _ = restriction.tabulate(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        )
        points.append(np.asarray(values) @ coefficients)
        blocks.append(
            CellBlock(
                name,
                "triangle",
                np.asarray([[3 * index, 3 * index + 1, 3 * index + 2]], dtype=np.int32),
                global_ids=np.asarray([identifier], dtype=np.int64),
            )
        )
        elements[name] = restriction
        routes[name] = np.arange(count, dtype=np.int32)[None]
        parents[name] = np.asarray([7], dtype=np.int64)
        corners[name] = np.asarray([[23, 29, 41]], dtype=np.int64)
        banks[name] = "root"
    origin = CellGeometryRestrictionSource(
        cell_geometry_id(original),
        original_mesh.topology_id,
        parents,
        corners,
    )
    geometry = CellGeometrySpec(
        elements,
        routes,
        np.asarray(packets["geometry/coordinates"]),
        restriction_source=origin,
    )
    vertices = (
        np.arange(6, dtype=np.int64) if rank == 0 else np.arange(3, 6, dtype=np.int64)
    )
    probe = CellMesh(np.concatenate(points), blocks, vertex_global_ids=vertices)
    entity_ids = (vertices, vertices, cell_ids)
    entity_owners = (
        np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
        if rank == 0
        else np.ones(3, dtype=np.int32),
        np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
        if rank == 0
        else np.ones(3, dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32) if rank == 0 else np.ones(1, dtype=np.int32),
    )
    storage = CellMeshStorage(
        (6, 6, 2),
        entity_ids,
        entity_owners,
        partition_index=rank,
        partition_count=2,
        logical_topology_id=canonical_fingerprint("accepted-topology"),
        logical_geometry_id=canonical_fingerprint("accepted-corner-carrier"),
        logical_coordinate_geometry_id=map_id,
        evidence_id=canonical_fingerprint("accepted-coverage"),
        logical_arrays=arrays,
        local_coordinates=probe.coordinates,
        local_blocks=blocks,
        global_coordinate_count=count,
        coordinate_global_ids=packets["geometry/coordinate_ids"],
        coordinate_owner=packets["geometry/coordinate_owners"],
        local_geometry=geometry,
        geometry_source_blocks=banks,
        geometry_projection=projection,
    )
    mesh = CellMesh(probe.coordinates, blocks, storage=storage)
    restored = storage.restore_geometry()
    certificate = certify_cell_geometry_validity(restored, mesh=mesh)
    with coordinate_reference_budget() as budget:
        measure, error, _ = certified_cell_measures(mesh, restored, budget=budget)
    original_areas = (
        [Fraction(1, 8), Fraction(1, 32)]
        if degree == 1
        else [Fraction(49, 384), Fraction(103, 3072)]
    )
    expected = original_areas if rank == 0 else original_areas[1:]
    for actual, bound, exact in zip(measure, error, expected, strict=True):
        assert abs(Fraction(float(actual)) - exact) <= Fraction(float(bound))
    assert np.all(np.asarray(certificate.status) == CellValidityStatus.CERTIFIED_VALID)
    return {"projection_valid": True}


def integer_width_archive(
    boundary_archive: str, overflow_archive: str
) -> dict[str, bool]:
    import jax.numpy as jnp

    import phydrax as phx
    from phydrax._array_archive import read_array_archive

    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=False))
    assert not bool(pxconfig.x64_enabled())
    int32_max = np.iinfo(np.int32).max
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    quadrilaterals = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    boundary = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=quadrilaterals,
        vertex_global_ids=np.arange(int32_max - 3, int32_max + 1, dtype=np.int64),
        cell_global_ids=np.asarray((int32_max,), dtype=np.int64),
    )
    assert boundary.vertex_global_ids.dtype == jnp.int32
    assert boundary.cell_global_ids.dtype == jnp.int32
    assert int(boundary.vertex_global_ids[-1]) == int32_max
    try:
        phx.discretization.UnstructuredFiniteVolumePlan(
            vertices,
            quadrilaterals=quadrilaterals,
            vertex_global_ids=np.arange(int32_max + 1, int32_max + 5, dtype=np.int64),
        )
    except ValueError as error:
        assert "signed int32" in str(error)
    else:
        raise AssertionError(
            "integer_width_fixture.global_id must fit signed int32; got accepted overflow"
        )

    _, canonical_arrays = read_array_archive(boundary_archive)
    assert canonical_arrays["vertex_global_ids"].dtype == np.dtype(np.int64)
    assert canonical_arrays["cell_global_ids"].dtype == np.dtype(np.int64)
    restored = phx.discretization.read_unstructured_fv_archive(boundary_archive)
    assert restored.vertex_global_ids.dtype == jnp.int32
    assert restored.cell_global_ids.dtype == jnp.int32
    assert int(restored.vertex_global_ids[-1]) == int32_max
    try:
        phx.discretization.read_unstructured_fv_archive(overflow_archive)
    except ValueError as error:
        assert "signed int32" in str(error)
    else:
        raise AssertionError(
            "integer_width_fixture.archive must refuse signed int32 overflow; got accepted"
        )
    return {"integer_width_valid": True}


def cold_periodic_owners(path: str, content_id: str) -> dict[str, bool]:
    from phydrax.lifecycle._meshing_sources import read_meshing_source_closure

    roots = read_meshing_source_closure(path, expected_content_id=content_id)
    for root in roots:
        root.validate_restored()
    assert roots[0].scientific_keys == roots[1].scientific_keys
    return {"owner_keys_equal": True}


def cold_exact_rational_geometry(
    path: str,
    content_id: str,
    source_id: str,
    layout_id: str,
    coefficients: str,
    representatives: str,
) -> dict[str, bool | int]:
    from phydrax.discretization import ExactRationalCellGeometrySource
    from phydrax.lifecycle import read_meshing_source_closure

    mesh, geometry, parent = read_meshing_source_closure(
        path, expected_content_id=content_id
    )
    source = geometry.exact_source
    if not isinstance(source, ExactRationalCellGeometrySource):
        raise TypeError(
            "cold_source.owner must restore its rational action owner; got another type"
        )
    with coordinate_reference_budget() as budget:
        geometry.resolve(mesh, budget=budget)
        bank = geometry.source_coordinates(budget=budget)
        digest = canonical_fingerprint(
            tuple(
                tuple((value.numerator, value.denominator) for value in row)
                for row in bank
            )
        )
        retained = budget.retained_basis_bytes
    return {
        "source_identity": source.source_id == source_id,
        "layout_identity": geometry.geometry_layout_id == layout_id,
        "parent_sharing": source.parent is parent,
        "exact_coefficients": digest == coefficients,
        "representative_bytes": canonical_fingerprint(
            array_tree_fingerprint(geometry.coordinates)
        )
        == representatives,
        "dynamic_arrays": all(
            isinstance(value, jax.Array) for value in jax.tree_util.tree_leaves(source)
        ),
        "retained_bytes": retained,
    }


def cold_exact_rational_publication(
    path: str,
    content_id: str,
    result_id: str,
    source_id: str,
    coefficients: str,
    representatives: str,
) -> dict[str, bool | int | str]:
    from phydrax._fingerprint import array_tree_signature
    from phydrax._host_storage import retained_host_storage_profile
    from phydrax._meshcore import is_native_host_buffer
    from phydrax._model._structure import model_structure_recipe
    from phydrax.discretization import ExactRationalCellGeometrySource
    from phydrax.lifecycle import read_meshing_source_closure

    result, parent = read_meshing_source_closure(path, expected_content_id=content_id)
    source = result.geometry.exact_source
    if not isinstance(source, ExactRationalCellGeometrySource):
        raise TypeError(
            "cold_publication.source must retain its rational source owner; got another type"
        )
    with coordinate_reference_budget() as budget:
        bank = result.geometry.source_coordinates(budget=budget)
        digest = canonical_fingerprint(
            tuple(
                tuple((value.numerator, value.denominator) for value in row)
                for row in bank
            )
        )
        retained = budget.retained_basis_bytes
        peak = budget.peak_bytes_upper
        work = budget.work_units
    _, host_bytes, logical_bytes, _ = retained_host_storage_profile(
        source, {}, is_native_host_buffer
    )
    return {
        "result_identity": result.result_id == result_id,
        "source_identity": source.source_id == source_id,
        "parent_sharing": source.parent is parent,
        "exact_coefficients": digest == coefficients,
        "representative_bytes": canonical_fingerprint(
            array_tree_fingerprint(result.geometry.coordinates)
        )
        == representatives,
        "certificate_retained": result.certification is not None
        and result.certification.passed,
        "layout_id": result.geometry.geometry_layout_id,
        "source_id": source.source_id,
        "result_id": result.result_id,
        "source_recipe_id": canonical_fingerprint(model_structure_recipe(source)),
        "source_pytree_signature_id": canonical_fingerprint(array_tree_signature(source)),
        "source_dynamic_array_bytes": sum(
            value.nbytes for value in jax.tree_util.tree_leaves(source)
        ),
        "source_dynamic_array_leaves": len(jax.tree_util.tree_leaves(source)),
        "source_host_storage_upper": host_bytes,
        "source_logical_array_bytes": logical_bytes,
        "csr_array_bytes": sum(
            value.nbytes
            for value in (
                source.action_offsets,
                source.action_sources,
                source.action_weights,
            )
        ),
        "coefficient_bank_count": len(source.coefficients),
        "source_action_rows": len(source.action_offsets) - 1,
        "source_action_entries": len(source.action_sources),
        "retained_bytes": retained,
        "peak_bytes_upper": peak,
        "work_units": work,
    }


def cold_native_implicit_generation(
    path: str,
    content_id: str,
    source_id: str,
    result_id: str,
    grid_id: str,
    binding_id: str,
) -> dict[str, bool]:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.geometry._source_fidelity import ImplicitProjectionBoundarySource
    from phydrax.lifecycle import (
        read_meshing_source_closure,
        read_meshing_source_execution_controls,
    )
    from phydrax.lifecycle._meshing_sources import source_content_equal
    from phydrax.meshing import NativeImplicitSource

    controls = read_meshing_source_execution_controls(
        path, expected_content_id=content_id
    )
    records = read_meshing_source_closure(path, expected_content_id=content_id)
    source = records["generation_source"]
    result = records["generation_part"].carrier
    if not isinstance(source, NativeImplicitSource):
        raise TypeError(
            "cold_implicit.source must retain NativeImplicitSource; got another source owner"
        )
    report = result.certification
    if report is None or not isinstance(
        report.request.source, ImplicitProjectionBoundarySource
    ):
        raise ValueError(
            "cold_implicit.certificate must retain its analytic query; got no analytic source"
        )
    query = report.request.source
    query.profile.require_bound(source.geometry, result.coordinate_contract)
    return {
        "source_identity": source.source_id == source_id,
        "source_revision": controls.source_revision == source.source_revision,
        "grid_identity": source.grid.prepared_id == grid_id,
        "result_identity": result.result_id == result_id,
        "field_content": source_content_equal(
            query.profile.geometry, source.geometry, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
        ),
        "source_binding": controls.source_binding_id == binding_id,
        "certified": report.passed,
    }
