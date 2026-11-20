#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""The native overset assembly against independent references.

Holes are checked by a NumPy solid-angle winding number away from the wall,
fringes by the cell rule, and every donor by brute-force location over all
cells of the other part (barycentric for tetrahedra, Newton for trilinear
hexahedra), as the F13 ledger checker does.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import _meshcore
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_complex import TetrahedralConnectivity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.discretization.fem import prepare_finite_element_field_reconstruction
from phydrax.meshing import _overset
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._coupling import CouplingSearchStatus
from phydrax.meshing._overset import (
    OversetConnectivityError,
    OversetPartSpec,
    OversetPolicy,
    OversetVertexStatus,
    prepare_overset_connectivity,
    prepare_overset_field_transfer,
)
from phydrax.meshing._scope import MeshingScope


pytestmark = pytest.mark.skipif(
    not _meshcore.meshcore_available(),
    reason="the native overset assembly needs meshcore",
)

_KUHN = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)
_HEX_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)
_TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def _grid(count: int, half: float, tets: bool) -> tuple[np.ndarray, np.ndarray]:
    side = count + 1
    axis = -half + 2.0 * half * np.arange(side, dtype=np.float64) / count
    k, j, i = np.meshgrid(*(np.arange(side),) * 3, indexing="ij")
    points = np.stack((axis[i.ravel()], axis[j.ravel()], axis[k.ravel()]), axis=1)
    k, j, i = (v.ravel() for v in np.meshgrid(*(np.arange(count),) * 3, indexing="ij"))
    corners = np.stack(
        [
            ((k + dk) * side + (j + dj)) * side + (i + di)
            for di, dj, dk in (
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            )
        ],
        axis=1,
    )
    if tets:
        corners = np.concatenate([corners[:, list(t)] for t in _KUHN])
    return points, corners


def _shell(
    count: int, layers: int, inner: float, outer: float
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.tan(np.pi / 4 * (-1 + 2.0 * np.arange(count + 1) / count))
    face, layer, j, i = np.meshgrid(
        np.arange(6),
        np.arange(layers + 1),
        np.arange(count + 1),
        np.arange(count + 1),
        indexing="ij",
    )
    a, b, one = angles[i], angles[j], np.ones(i.shape)
    rows = (
        (one, a, b),
        (-one, -a, b),
        (-a, one, b),
        (a, -one, b),
        (-b, a, one),
        (b, a, -one),
    )
    directions = np.select(
        [face[..., None] == side for side in range(6)],
        [np.stack(r, axis=-1) for r in rows],
    )
    radius = inner + (outer - inner) * layer / layers
    points = (
        directions / np.linalg.norm(directions, axis=-1)[..., None] * radius[..., None]
    )
    points = points.reshape((-1, 3))
    _, first, inverse = np.unique(
        np.rint(points * 1e9).astype(np.int64),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    ids = rank[inverse.reshape((-1,))].reshape(face.shape)
    f, l, jj, ii = (
        v.ravel()
        for v in np.meshgrid(
            np.arange(6),
            np.arange(layers),
            np.arange(count),
            np.arange(count),
            indexing="ij",
        )
    )
    cells = np.stack(
        [
            ids[f, l + dl, jj + dj, ii + di]
            for di, dj, dl in (
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            )
        ],
        axis=1,
    )
    vertices = points[first[order]]
    corners = vertices[cells]
    turn = (
        np.einsum(
            "ij,ij->i",
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0]),
            corners[:, 4] - corners[:, 0],
        )
        < 0
    )
    cells[turn] = cells[turn][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    return vertices, cells


def _part(name: str, points: np.ndarray, cells: np.ndarray) -> MeshPart:
    kind = "hexahedron" if cells.shape[1] == 8 else "tetrahedron"
    mesh = CellMesh(
        jnp.asarray(points, dtype=jnp.float64),
        (CellBlock(name, kind, jnp.asarray(cells, dtype=jnp.int32)),),
    )
    return MeshPart(
        name, phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    )


def _mesh(part: MeshPart) -> CellMesh:
    return _overset._cell_result(part).mesh


def _shell_scopes(part: MeshPart) -> tuple[MeshingScope, MeshingScope]:
    mesh = _mesh(part)
    entity_set = mesh.entity_set(2)
    identifiers = np.asarray(entity_set.entity_ids, dtype=np.int64)
    boundary = np.asarray(entity_set.subset("boundary").mask, dtype=np.bool_)
    connectivity = mesh.connectivity
    if isinstance(connectivity, (HexahedralConnectivity, TetrahedralConnectivity)):
        faces = np.asarray(connectivity.faces)
    elif isinstance(connectivity, SimplicialConnectivity):
        faces = np.asarray(connectivity.entities[2])
    else:
        raise TypeError("The shells are tetrahedral or hexahedral.")
    radius = np.linalg.norm(np.asarray(mesh.coordinates)[faces], axis=2).mean(axis=1)
    return (
        part.scope(2, identifiers[boundary & (radius < 0.9)]),
        part.scope(2, identifiers[boundary & (radius > 0.9)]),
    )


def _independent_location(corners: np.ndarray, point: np.ndarray) -> tuple[float, float]:
    """Distance outside the reference cell and the mapping residual."""
    if corners.shape[0] == 4:
        reference = np.linalg.solve((corners[1:] - corners[0]).T, point - corners[0])
        weights = np.concatenate(([1.0 - reference.sum()], reference))
        outside = float(max(-weights.min(), 0.0))
        return outside, float(np.linalg.norm(weights @ corners - point))
    bits = np.asarray(
        ((0, 1, 1, 0, 0, 1, 1, 0), (0, 0, 1, 1, 0, 0, 1, 1), (0, 0, 0, 0, 1, 1, 1, 1)),
        dtype=np.float64,
    )
    xi = np.full(3, 0.5)
    for _ in range(40):
        factors = np.where(bits == 1, xi[:, None], 1 - xi[:, None])
        shapes = np.prod(factors, axis=0)
        derivative = np.stack(
            [
                (2 * bits[a] - 1) * np.prod(np.delete(factors, a, axis=0), axis=0)
                for a in range(3)
            ],
            axis=1,
        )
        xi = xi - np.linalg.solve(corners.T @ derivative, shapes @ corners - point)
    factors = np.where(bits == 1, xi[:, None], 1 - xi[:, None])
    residual = float(np.linalg.norm(np.prod(factors, axis=0) @ corners - point))
    return float(max(-xi.min(), xi.max() - 1.0, 0.0)), residual


def _winding(triangles: np.ndarray, point: np.ndarray) -> float:
    v = triangles - point
    lengths = np.linalg.norm(v, axis=2)
    det = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
    den = (
        lengths[:, 0] * lengths[:, 1] * lengths[:, 2]
        + np.einsum("ij,ij->i", v[:, 0], v[:, 1]) * lengths[:, 2]
        + np.einsum("ij,ij->i", v[:, 1], v[:, 2]) * lengths[:, 0]
        + np.einsum("ij,ij->i", v[:, 2], v[:, 0]) * lengths[:, 1]
    )
    return float(np.sum(2.0 * np.arctan2(det, den)) / (4.0 * np.pi))


def _box_shell(tets: bool) -> tuple[np.ndarray, np.ndarray]:
    """A conforming grid shell between the cubes of half-widths 0.4 and 1.2."""
    points, cells = _grid(6, 1.2, tets)
    inner = np.all(np.max(np.abs(points[cells]), axis=2) <= 0.4 + 1e-12, axis=1)
    cells = cells[~inner]
    used, compact = np.unique(cells, return_inverse=True)
    return points[used], compact.reshape(cells.shape)


def _assembly(
    background_tets: bool, shell: str
) -> tuple[MeshPart, MeshPart, tuple[OversetPartSpec, OversetPartSpec]]:
    background = _part("background", *_grid(9, 1.8, background_tets))
    match shell:
        case "sphere":
            body = _part("shell", *_shell(4, 3, 0.5, 1.4))
        case "box-tet":
            body = _part("shell", *_box_shell(True))
        case "box-hex":
            body = _part("shell", *_box_shell(False))
        case other:
            raise ValueError(other)
    wall, boundary = _shell_scopes(body)
    specs = (
        OversetPartSpec("background"),
        OversetPartSpec("shell", wall=wall, boundary=boundary),
    )
    return background, body, specs


@pytest.mark.parametrize(
    ("background_tets", "shell"),
    [(True, "sphere"), (False, "sphere"), (True, "box-tet"), (False, "box-hex")],
)
def test_native_assembly_matches_independent_references(
    background_tets: bool, shell: str
) -> None:
    background, body, specs = _assembly(background_tets, shell)
    connectivity = prepare_overset_connectivity(
        MeshAssembly((background, body)),
        specs,
        policy=OversetPolicy(fringe_layers=1, fringe_donors=True),
    )
    connectivity.require_complete()
    parts = {"background": background, "shell": body}
    shell_points = np.asarray(_mesh(body).coordinates, dtype=np.float64)
    wall_rows = _overset._facet_rows(body, _shell_scopes(body)[0])
    wall = np.concatenate(
        [
            shell_points[wall_rows[:, [0, k, k + 1]]]
            for k in range(1, wall_rows.shape[1] - 1)
        ]
    )
    # Both walls are star-shaped about the origin: orient every triangle outward.
    normals = np.cross(wall[:, 1] - wall[:, 0], wall[:, 2] - wall[:, 0])
    inward = np.einsum("ij,ij->i", normals, wall.mean(axis=1)) < 0
    wall[inward] = wall[inward][:, [0, 2, 1]]
    blanking = connectivity.blanking_of("background")
    status = np.asarray(blanking.vertex_status)
    points = np.asarray(_mesh(background).coordinates, dtype=np.float64)
    contact = np.asarray(blanking.wall_contact, dtype=np.bool_)
    for row in np.flatnonzero(~contact):
        winding = abs(_winding(wall, points[row]))
        if 0.25 < winding < 0.75:
            continue  # on the wall
        assert (status[row] == OversetVertexStatus.HOLE) == (winding > 0.5)
    # Fringe rule: a cell with a hole vertex keeps no active vertex.
    cells = np.asarray(_mesh(background).blocks[0].vertices)
    touching = np.any(status[cells] == OversetVertexStatus.HOLE, axis=1)
    assert not np.any(status[cells[touching]] == OversetVertexStatus.ACTIVE)
    for receptor_name, donor_name in (("background", "shell"), ("shell", "background")):
        evidence = connectivity.receptors_of(receptor_name)
        assert np.all(np.asarray(evidence.found))
        receptor_points = np.asarray(_mesh(parts[receptor_name]).coordinates)
        donor_mesh = _mesh(parts[donor_name])
        donor_points = np.asarray(donor_mesh.coordinates, dtype=np.float64)
        donor_cells = np.asarray(donor_mesh.blocks[0].vertices)
        for row, cell in zip(
            np.asarray(evidence.receptor_rows),
            np.asarray(evidence.donor_cells),
            strict=True,
        ):
            outside, residual = _independent_location(
                donor_points[donor_cells[cell]], receptor_points[row]
            )
            assert outside <= 1e-12 and residual <= 1e-12


def test_native_hexahedral_overset_transfers_linear_fields_exactly() -> None:
    # Planar-faced hexahedra: the FE field query refuses warped tensor faces
    # without a curved support geometry.
    background, shell, specs = _assembly(False, "box-hex")
    connectivity = prepare_overset_connectivity(
        MeshAssembly((background, shell)),
        specs,
        policy=OversetPolicy(fringe_layers=1, fringe_donors=True),
    )
    connectivity.require_complete()
    reconstructions = {}
    fields = {}
    for name, part in (("background", background), ("shell", shell)):
        discretization = phx.discretization.FiniteElementPlan(
            _mesh(part),
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element("hexahedron", 1)
            ),
        ).prepare()
        reconstructions[name] = prepare_finite_element_field_reconstruction(
            discretization, "u"
        )
        x = np.asarray(discretization.dof_maps[0].dof_coordinates)
        fields[name] = jnp.asarray(1.0 + x[:, 0] + 2.0 * x[:, 1] + 3.0 * x[:, 2])
    transfer = prepare_overset_field_transfer(connectivity, reconstructions, "u")
    output = transfer.apply(fields)
    for name, part in (("background", background), ("shell", shell)):
        rows = np.asarray(connectivity.receptors_of(name).receptor_rows)
        x = np.asarray(_mesh(part).coordinates)[rows]
        np.testing.assert_allclose(
            output[name], 1.0 + x[:, 0] + 2.0 * x[:, 1] + 3.0 * x[:, 2], atol=1e-12
        )


def test_active_donor_policy_reports_orphans_as_evidence() -> None:
    background, shell, specs = _assembly(False, "sphere")
    status = np.asarray(
        prepare_overset_connectivity(
            MeshAssembly((background, shell)),
            specs,
            policy=OversetPolicy(fringe_layers=1, fringe_donors=True),
        )
        .receptors_of("shell")
        .status
    )
    assert np.all(status == int(CouplingSearchStatus.FOUND))
    connectivity = prepare_overset_connectivity(
        MeshAssembly((background, shell)), specs, policy=OversetPolicy(fringe_layers=1)
    )
    evidence = connectivity.receptors_of("shell")
    # Active donors only: the shell's boundary receptors inside fringe cells of
    # the background are reported orphans with their exclusion, never silent.
    assert connectivity.orphan_count > 0
    assert np.all(
        np.isin(
            np.asarray(evidence.status)[~np.asarray(evidence.found)],
            (int(CouplingSearchStatus.EXCLUDED_DONOR),),
        )
    )
    with pytest.raises(OversetConnectivityError, match="orphan receptors"):
        connectivity.require_complete()


def test_unproved_construction_reads_donor_coverage_unproved() -> None:
    background, shell, specs = _assembly(True, "sphere")
    policy = OversetPolicy(fringe_layers=1, fringe_donors=True)
    connectivity = prepare_overset_connectivity(
        MeshAssembly((background, shell)), specs, policy=policy, prove=False
    )
    assert not connectivity.donor_search_complete and not connectivity.complete
    with pytest.raises(
        OversetConnectivityError, match="donor coverage has not been proved"
    ):
        connectivity.require_complete()
    with pytest.raises(ValueError, match="prove=True"):
        connectivity.registration()
    proved = prepare_overset_connectivity(
        MeshAssembly((background, shell)), specs, policy=policy
    )
    for name in ("background", "shell"):
        np.testing.assert_array_equal(
            connectivity.blanking_of(name).vertex_status,
            proved.blanking_of(name).vertex_status,
        )
        np.testing.assert_array_equal(
            connectivity.receptors_of(name).status, proved.receptors_of(name).status
        )
    restored = proved.registration().prepare()
    assert restored.connectivity_id == proved.connectivity_id


@pytest.mark.parametrize(
    "disposition",
    [
        phx.meshing.CellMeshAuditDisposition.RECORD,
        phx.meshing.CellMeshAuditDisposition.SKIP,
    ],
)
@pytest.mark.parametrize("native", [True, False])
def test_overset_refuses_carriers_without_certified_validity(
    disposition: phx.meshing.CellMeshAuditDisposition,
    native: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Trilinear background hexahedra with positive volume but negative corner
    # Jacobians (one interior vertex pushed just past its neighbor), admitted
    # by an audit that records or skips invalid geometry.
    points, cells = _grid(9, 1.8, False)
    points[111, 0] += 0.45
    mesh = CellMesh(
        jnp.asarray(points, dtype=jnp.float64),
        (CellBlock("background", "hexahedron", jnp.asarray(cells, dtype=jnp.int32)),),
    )
    lenient = phx.meshing.CellMeshAuditPolicy(
        inconsistent_orientation=disposition,
        self_intersection=phx.meshing.CellMeshAuditDisposition.SKIP,
        invalid_geometry=disposition,
        unresolved=phx.meshing.CellMeshAuditDisposition.RECORD,
    )
    background = MeshPart(
        "background",
        phx.meshing.certify_cell_mesh(
            mesh, phx.SpatialCoordinateContract.si(), audit_policy=lenient
        ),
    )
    assert not _overset._cell_result(background).audit.validity.all_certified
    _, shell, specs = _assembly(False, "sphere")
    if not native:
        # The Python owner refuses the same carriers.
        monkeypatch.setattr(_meshcore, "meshcore_available", lambda: False)
    with pytest.raises(ValueError, match="require certified cell validity"):
        prepare_overset_connectivity(MeshAssembly((background, shell)), specs)


def test_python_owner_prepares_tetrahedra_without_meshcore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    background, shell, specs = _assembly(True, "box-tet")
    monkeypatch.setattr(_meshcore, "meshcore_available", lambda: False)
    connectivity = prepare_overset_connectivity(
        MeshAssembly((background, shell)), specs, policy=OversetPolicy(fringe_layers=1)
    )
    assert connectivity.donor_search_complete
    status = np.asarray(connectivity.blanking_of("background").vertex_status)
    assert np.any(status == OversetVertexStatus.HOLE)
