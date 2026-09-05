import os
import shlex
import shutil

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory
from phydrax.meshing.providers._tioga import TiogaOptions, TiogaProvider


def _grid(name, lower, upper, count, *, cavity=False, center=(0, 0, 0)):
    axis = np.linspace(lower, upper, count)
    coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    origins = np.stack(
        np.meshgrid(*(np.arange(count - 1),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    corners = np.array(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        )
    )
    cells = np.ravel_multi_index(
        (origins[:, None, :] + corners).transpose(2, 0, 1), (count,) * 3
    )
    if cavity:
        cells = cells[np.max(np.abs(coordinates[cells].mean(axis=1)), axis=1) > 0.5]
    retained = np.unique(cells)
    cells, coordinates = np.searchsorted(retained, cells), coordinates[retained]
    coordinates = coordinates + np.asarray(center)
    # Non-row-order, 64-bit source IDs intentionally overlap across parts.
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "hexes",
                "hexahedron",
                cells,
                global_ids=2**34 + 13 * np.arange(len(cells))[::-1],
            ),
        ),
        vertex_global_ids=2**33 + 17 * np.arange(len(coordinates))[::-1],
    )
    return MeshPart(name, certify_cell_mesh(mesh, SpatialCoordinateContract.si()))


@pytest.fixture(scope="module")
def overset_case():
    executable = shutil.which(os.environ.get("PHYDRAX_TIOGA_EXECUTABLE", "phydrax-tioga"))
    if executable is None:
        pytest.skip("Real phydrax-tioga native bridge is not installed")
    background = _grid("background", -5, 5, 21)
    parts, walls, overset = [background], [], []
    # Two recipient grids reuse IDs at different coordinates. On two ranks,
    # rank zero owns two blocks: neither local block index is a mesh tag.
    for name, center in (("body-left", (-2, 0, 0)), ("body-right", (2, 0, 0))):
        body = _grid(name, -1.5, 1.5, 13, cavity=True, center=center)
        mesh = body.carrier.mesh
        radius = np.max(np.abs(np.asarray(mesh.coordinates) - center), axis=1)
        walls.append(
            body.scope(0, np.asarray(mesh.vertex_global_ids)[np.isclose(radius, 0.5)])
        )
        overset.append(
            body.scope(0, np.asarray(mesh.vertex_global_ids)[np.isclose(radius, 1.5)])
        )
        parts.append(body)
    return executable, MeshAssembly(tuple(parts)), tuple(walls), tuple(overset)


@pytest.mark.parametrize("ranks", (1, 2))
def test_tioga_real_hole_cut_and_affine_transfer(overset_case, ranks):
    executable, assembly, walls, overset = overset_case
    launcher = os.environ.get("PHYDRAX_TIOGA_MPI_LAUNCHER", "mpiexec")
    if ranks > 1 and shutil.which(launcher) is None:
        pytest.skip("MPI launcher is not installed")
    options = TiogaOptions(
        executable=executable,
        ranks=ranks,
        mpi_launcher=launcher,
        mpi_arguments=tuple(
            shlex.split(os.environ.get("PHYDRAX_TIOGA_MPI_ARGUMENTS", ""))
        ),
        exclusion_layers=1,
    )
    result = TiogaProvider(options).execute(
        assembly, wall_scopes=walls, overset_scopes=overset
    )
    by_name = {status.part_name: status for status in result.blanking}
    background = by_name["background"]
    hole_mask = np.asarray(background.node_iblank) == 0
    coordinates = np.asarray(assembly.part("background").carrier.mesh.coordinates)
    assert np.any(hole_mask)
    distance_to_solids = np.max(
        np.abs(
            coordinates[hole_mask, None, :]
            - np.array(((-2, 0, 0), (2, 0, 0)))[None, :, :]
        ),
        axis=2,
    )
    assert np.all(np.min(distance_to_solids, axis=1) < 0.5)
    assert np.any(np.asarray(background.cell_iblank) == 0)
    receptor_total = sum(
        np.count_nonzero(np.asarray(status.node_iblank) < 0) for status in result.blanking
    )
    assert receptor_total > 0
    assert sum(item.donor_cell_ids.size for item in result.donors) == receptor_total
    assert {part.part_id for part in result.assembly.parts} == {
        part.part_id for part in assembly.parts
    }
    for status in result.blanking:
        part = assembly.part(status.part_name)
        np.testing.assert_array_equal(
            status.node_ids, part.carrier.mesh.vertex_global_ids
        )
        np.testing.assert_array_equal(
            status.cell_ids,
            np.concatenate(
                [np.asarray(block.global_ids) for block in part.carrier.mesh.blocks]
            ),
        )
    for link in result.assembly.couplings:
        source = assembly.part(link.source_scope.source_id)
        target = assembly.part(link.target_scope.source_id)
        source_points = np.asarray(source.point_coordinates(link.source_scope))
        target_points = np.asarray(target.point_coordinates(link.target_scope))
        values = 2 + source_points @ np.array((1.5, -2.5, 3.5))
        expected = 2 + target_points @ np.array((1.5, -2.5, 3.5))
        np.testing.assert_allclose(link.transfer(values), expected, rtol=0, atol=1e-8)
        record = next(
            item for item in result.donors if item.coupling_id == link.coupling_id
        )
        cells = {
            int(identifier): np.asarray(source.carrier.mesh.vertex_global_ids)[vertices]
            for block in source.carrier.mesh.blocks
            for identifier, vertices in zip(
                np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
            )
        }
        for cell, nodes in zip(
            np.asarray(record.donor_cell_ids), np.asarray(link.donor_ids), strict=True
        ):
            np.testing.assert_array_equal(
                np.sort(nodes[nodes >= 0]), np.sort(cells[int(cell)])
            )


def test_tioga_rejects_surface_cells_before_loading_native_dependency():
    mesh = CellMesh(
        np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        (CellBlock("surface", "triangle", np.array(((0, 1, 2),))),),
    )
    certified = certify_cell_mesh(mesh, SpatialCoordinateContract.si())
    assembly = MeshAssembly((MeshPart("a", certified), MeshPart("b", certified)))
    with pytest.raises(MeshingFailure) as failure:
        TiogaProvider(
            TiogaOptions(executable="nonexistent-tioga-for-unsupported-surface")
        ).execute(assembly)
    assert failure.value.category is MeshingFailureCategory.UNSUPPORTED_CAPABILITY
