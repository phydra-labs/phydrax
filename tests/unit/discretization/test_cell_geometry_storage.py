#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import os
import socket
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.discretization._cell_geometry import (
    _nonrepresentable_reference_ancestry,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    CellGeometryStorageProjection,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import _certified_cell_measures
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh, CellMeshStorage
from phydrax.discretization._coordinate_enclosure import (
    coordinate_polynomials,
    coordinate_reference_chain,
    coordinate_source_signature,
    evaluate,
    RationalPolynomial,
)


def _digest(value: object) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(canonical_fingerprint(value)), dtype=np.uint8)


def _storage(mesh: CellMesh, geometry: CellGeometrySpec) -> CellMeshStorage:
    coordinate_ids = np.arange(geometry.coordinates.shape[0], dtype=np.int64) + 100
    owners = np.zeros(coordinate_ids.shape, dtype=np.int32)
    arrays = {
        "geometry/coordinate_ids": jnp.asarray(coordinate_ids),
        "geometry/coordinate_owners": jnp.asarray(owners),
        "geometry/coordinates": geometry.coordinates,
    }
    origin = geometry.restriction_source
    if origin is not None:
        arrays["geometry/source_geometry_id"] = jnp.asarray(
            _digest(origin.source_geometry_id)
        )
        arrays["geometry/source_topology_id"] = jnp.asarray(
            _digest(origin.source_topology_id)
        )
    for block, element, route in zip(
        mesh.blocks, geometry.elements, geometry.geometry_dofs, strict=True
    ):
        name = block.name
        root, expressions = coordinate_reference_chain(element)
        dimension = block.topological_dimension
        if any(isinstance(value, RationalPolynomial) for value in expressions):
            raise ValueError(
                "This storage fixture requires an actual polynomial reference action."
            )
        arguments = tuple(
            value for value in expressions if not isinstance(value, RationalPolynomial)
        )
        if any(sum(index) > 1 for value in arguments for index in value):
            raise ValueError(
                "This storage fixture requires an actual affine reference action."
            )
        matrix = np.asarray(
            [
                [
                    float(
                        value.get(
                            tuple(int(axis == column) for axis in range(dimension)),
                            Fraction(),
                        )
                    )
                    for column in range(dimension)
                ]
                for value in arguments
            ],
            dtype=np.float64,
        )
        offset = np.asarray(
            [float(value.get((0,) * dimension, Fraction())) for value in arguments],
            dtype=np.float64,
        )
        arrays[f"geometry/cell_ids/{name}"] = block.global_ids
        arrays[f"geometry/routes/{name}"] = jnp.asarray(coordinate_ids[np.asarray(route)])
        arrays[f"geometry/matrix/{name}"] = jnp.asarray(
            np.broadcast_to(matrix, (block.cell_count,) + matrix.shape)
        )
        arrays[f"geometry/offset/{name}"] = jnp.asarray(
            np.broadcast_to(offset, (block.cell_count,) + offset.shape)
        )
        ancestry = _nonrepresentable_reference_ancestry(element, matrix, offset)
        if ancestry is not None:
            digest = np.frombuffer(bytes.fromhex(ancestry), dtype=np.uint8)
            arrays[f"geometry/reference_ancestry/{name}"] = jnp.asarray(
                np.broadcast_to(digest, (block.cell_count, 32))
            )
        signature = {
            "signature": coordinate_source_signature(root),
            "arrays": array_tree_fingerprint(root),
        }
        arrays[f"geometry/source_basis/{name}"] = jnp.asarray(
            np.broadcast_to(_digest(signature), (block.cell_count, 32))
        )
        if origin is not None:
            arrays[f"geometry/parent_cell_ids/{name}"] = origin.block_parent_cell_ids[
                name
            ]
            arrays[f"geometry/parent_vertex_ids/{name}"] = origin.block_parent_vertex_ids[
                name
            ]
    entity_ids = tuple(entity.entity_ids for entity in mesh.topology.entity_sets)
    return CellMeshStorage(
        tuple(entity.count for entity in mesh.topology.entity_sets),
        entity_ids,
        tuple(np.zeros(value.shape, dtype=np.int32) for value in entity_ids),
        partition_index=0,
        partition_count=2,
        logical_topology_id=mesh.topology_id,
        logical_geometry_id=mesh.geometry_id,
        logical_coordinate_geometry_id=cell_geometry_id(geometry),
        evidence_id=canonical_fingerprint("collective"),
        logical_arrays=tuple(sorted(arrays.items())),
        local_coordinates=mesh.coordinates,
        local_blocks=mesh.blocks,
        global_coordinate_count=len(coordinate_ids),
        coordinate_global_ids=coordinate_ids,
        coordinate_owner=owners,
        local_geometry=geometry,
    )


def test_nonrepresentable_midpoint_retains_positive_exact_map() -> None:
    jax.config.update("jax_enable_x64", True)
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
    # The sampled midpoint is exactly the left endpoint in FP64, unlike the authoritative map.
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
    storage = _storage(mesh, geometry)
    local = CellMesh(mesh.coordinates, mesh.blocks, storage=storage)
    restored = storage.restore_geometry()
    certificate = certify_cell_geometry_validity(restored, mesh=local)
    assert int(certificate.status[0]) == CellValidityStatus.CERTIFIED_VALID
    assert float(certificate.determinant_lower[0]) > 0.0
    polynomials = coordinate_polynomials(
        restored.elements[0], np.asarray(restored.coordinates)
    )
    assert polynomials is not None
    exact_midpoint = (Fraction(source[0, 0]) + Fraction(source[1, 0])) / 2
    assert evaluate(polynomials[0], (Fraction(1),)) == exact_midpoint
    assert Fraction(float(exact_midpoint)) != exact_midpoint
    assert cell_geometry_id(restored) == cell_geometry_id(geometry)
    with pytest.raises(ValueError):
        CellGeometrySpec.affine(local)


def test_copied_source_coefficient_mutation_is_rejected() -> None:
    jax.config.update("jax_enable_x64", True)
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
    storage = _storage(mesh, geometry)
    restored = storage.restore_geometry()
    copied = eqx.tree_at(
        lambda value: value.coordinates, restored, restored.coordinates.at[3, 1].add(0.01)
    )
    with pytest.raises(ValueError, match="original source coefficient binding"):
        cell_geometry_id(copied)
    altered = eqx.tree_at(
        lambda value: value.coordinates, geometry, geometry.coordinates.at[3, 1].add(0.01)
    )
    with pytest.raises(ValueError, match="original source coefficients"):
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
    jax.config.update("jax_enable_x64", True)
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
    storage = _storage(mesh, geometry)
    restored = storage.restore_geometry()
    local = CellMesh(mesh.coordinates, mesh.blocks, storage=storage)
    measure, error, exact = _certified_cell_measures(local, restored)
    assert exact
    assert abs(Fraction(float(measure[0])) - area) <= Fraction(float(error[0]))
    np.testing.assert_array_equal(restored.coordinates, coefficients)
    corrupted = dict(storage.logical_arrays)
    corrupted["geometry/matrix/nested"] = (
        corrupted["geometry/matrix/nested"].at[0, 0, 0].set(0.5)
    )
    with pytest.raises(ValueError, match="exact reference matrix"):
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
    environment = dict(os.environ)
    root = Path(__file__).resolve().parents[3]
    environment.update(
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        PYTHONPATH=str(root),
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    scenario = (
        "import sys,jax,runpy; "
        "jax.distributed.initialize(coordinator_address=sys.argv[1],num_processes=2,"
        "process_id=int(sys.argv[2]),initialization_timeout=60); "
        "runpy.run_path(sys.argv[3])['_projection_process'](int(sys.argv[4])); "
        "jax.distributed.shutdown()"
    )
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                scenario,
                address,
                str(rank),
                str(Path(__file__).resolve()),
                str(degree),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for rank in range(2)
    ]
    try:
        results = [process.communicate(timeout=120) for process in processes]
    finally:
        for process in processes:
            process.kill()
            process.wait(timeout=10)
    for process, (stdout, stderr) in zip(processes, results, strict=True):
        if process.returncode:
            pytest.fail(f"{stdout}\n{stderr}", pytrace=False)


def _projection_process(degree: int) -> None:
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
    measure, error, _ = _certified_cell_measures(mesh, restored)
    original_areas = (
        [Fraction(1, 8), Fraction(1, 32)]
        if degree == 1
        else [Fraction(49, 384), Fraction(103, 3072)]
    )
    expected = original_areas if rank == 0 else original_areas[1:]
    for actual, bound, exact in zip(measure, error, expected, strict=True):
        assert abs(Fraction(float(actual)) - exact) <= Fraction(float(bound))
    assert np.all(np.asarray(certificate.status) == CellValidityStatus.CERTIFIED_VALID)


def test_cold_projection_binds_original_coefficient_content(tmp_path: Path) -> None:
    jax.config.update("jax_enable_x64", True)
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
    seed = _storage(mesh, geometry)
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
    measure, error, _ = _certified_cell_measures(local, restored)
    assert abs(Fraction(float(measure[0])) - Fraction(25, 48)) <= Fraction(
        float(error[0])
    )
    changed = dict(arrays)
    changed["geometry/coordinates"] = changed["geometry/coordinates"].at[3, 1].add(0.01)
    with pytest.raises(ValueError, match="scientific source banks"):
        projection.require_source(
            tuple(sorted(changed.items())), cell_geometry_id(geometry)
        )
