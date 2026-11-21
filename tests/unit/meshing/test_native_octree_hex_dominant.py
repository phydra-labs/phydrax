"""Hex-dominant meshing: the native kernel and the plc_octree_hex_dominant route."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import phydrax._meshcore as meshcore
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import PolyhedralConnectivity
from phydrax.discretization._cell_mesh import CellBlock
from phydrax.geometry._exact_polyhedral_geometry import determinant3
from phydrax.geometry._planar_chain_embedding import polytope_face_triangles
from phydrax.meshing import (
    BoundaryLayerControl,
    BoundaryLayerRoute,
    CellFamilyPolicy,
    CellMeshingTarget,
    LayerSchedule,
    MeshingFailure,
    MeshingScope,
    SizeControlStrength,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing.providers import (
    NativeMeshingOptions,
    NativeMeshingPlan,
    NativeMeshingProvider,
)
from phydrax.meshing.providers._native_sources import NativePlcSource


_QUADS = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)
_LAYERS = (0.02, 0.024)


def _cube_vertices() -> np.ndarray:
    return np.asarray(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
        dtype=np.float64,
    )


def _kernel_cube(layers: tuple[float, ...]) -> meshcore.HexDominant:
    triangles = np.asarray(
        [(a, b, c) for a, b, c, d in _QUADS] + [(a, c, d) for a, b, c, d in _QUADS],
        dtype=np.int32,
    )
    return meshcore.hex_dominant(
        _cube_vertices(),
        triangles,
        np.asarray(list(range(6)) * 2, dtype=np.int32),
        core_size=0.25,
        boundary_size=0.125,
        feature_angle_degrees=45.0,
        layer_thicknesses=np.asarray(layers, dtype=np.float64),
        prove=True,
    )


def _plc_cube() -> NativePlcSource:
    complex_ = PiecewiseLinearComplex(
        _cube_vertices(),
        tuple(np.asarray(quad, dtype=np.int64) for quad in _QUADS),
        np.arange(6, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1)),
        ("solid",),
        vertex_ids=tuple(f"v{row}" for row in range(8)),
        vertex_occurrence_paths=((),) * 8,
        facet_ids=tuple(f"f{row}" for row in range(6)),
        facet_occurrence_paths=((),) * 6,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    return NativePlcSource(complex_, "hex-dominant-cube", "r1")


def _specification(
    source: NativePlcSource,
    layers: tuple[float, ...] = (),
    *,
    families: CellFamilyPolicy = CellFamilyPolicy(required=("polyhedron",)),
) -> VolumeMeshingSpec:
    wall = MeshingScope.whole(source, 2)
    solid = MeshingScope.whole(source, 3)
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, families),
        wall,
        VolumeFillStrategy.HEX_DOMINANT_POLYHEDRAL_TRANSITION,
        size_controls=(
            UniformSizeControl(solid, 0.25, strength=SizeControlStrength.SOFT),
            UniformSizeControl(wall, 0.125, strength=SizeControlStrength.SOFT),
        ),
        layer_controls=(
            (
                BoundaryLayerControl(
                    wall,
                    LayerSchedule.geometric(
                        len(layers), layers[0], growth_rate=layers[1] / layers[0]
                    ),
                    route=BoundaryLayerRoute.PROVIDER,
                    volume_scope=solid,
                ),
            )
            if layers
            else ()
        ),
    )


def _plan(layers: tuple[float, ...] = ()) -> NativeMeshingPlan:
    source = _plc_cube()
    return NativeMeshingProvider(NativeMeshingOptions()).plan(
        source,
        _specification(source, layers),
        coordinate_contract=SpatialCoordinateContract.si(),
    )


def _fan_orientations(
    connectivity: PolyhedralConnectivity, points: np.ndarray
) -> set[int]:
    """Exact signs of the star tetrahedra (cell vertex average, fan triangle).

    Every face is the exact triangulation of its loop (the fan from its vertex
    average when non-planar), signed by its orientation in the cell.
    """
    exact = [tuple(Fraction(value) for value in row) for row in points.tolist()]
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64).tolist()
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int64).tolist()
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64).tolist()
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64).tolist()
    face_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int64).tolist()
    vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64).tolist()
    vertex_values = np.asarray(connectivity.cell_vertex_values, dtype=np.int64).tolist()
    signs: set[int] = set()
    for cell in range(connectivity.cell_count):
        corners = vertex_values[vertex_offsets[cell] : vertex_offsets[cell + 1]]
        centre = tuple(
            sum(exact[v][axis] for v in corners) / len(corners) for axis in range(3)
        )
        for slot in range(cell_offsets[cell], cell_offsets[cell + 1]):
            face = cell_faces[slot]
            loop = tuple(exact[v] for v in values[offsets[face] : offsets[face + 1]])
            for triangle in polytope_face_triangles(loop):
                volume = face_signs[slot] * determinant3(*triangle, centre)
                signs.add((volume > 0) - (volume < 0))
    return signs


def test_route_proves_the_cube_with_an_exact_wall() -> None:
    result = _plan().execute()

    certification = result.certification
    assert certification is not None and certification.passed
    assert certification.inputs.schedule.route == "volume_plc_fidelity"
    fidelity = certification.fidelity
    assert fidelity is not None and fidelity.status == "certified"
    mesh = result.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    connectivity = mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    boundary = np.flatnonzero(np.asarray(mesh.boundary_masks[2], dtype=np.bool_))
    # Every wall vertex lies on the cube, and the wall encloses the unit
    # volume (divergence theorem over the outward fans of the wall faces;
    # the cube is convex, so outward faces point away from its center).
    wall = np.unique(
        np.concatenate([values[offsets[f] : offsets[f + 1]] for f in boundary])
    )
    assert np.max(np.min(np.minimum(points[wall], 1.0 - points[wall]), axis=1)) == 0.0
    volume = 0.0
    for face in boundary:
        loop = points[values[offsets[face] : offsets[face + 1]]]
        normal = np.cross(loop[1] - loop[0], loop[2] - loop[0])
        sign = np.sign(np.dot(normal, loop.mean(axis=0) - 0.5))
        for k in range(1, loop.shape[0] - 1):
            volume += sign * float(np.dot(loop[0], np.cross(loop[k], loop[k + 1]))) / 6.0
    assert volume == pytest.approx(1.0, abs=1e-13)
    assert not any(isinstance(block, CellBlock) for block in mesh.blocks)
    # Every cell is a valid star of its exact fan faces about its vertex
    # average: one strict orientation over the whole mesh.
    assert _fan_orientations(connectivity, points) in ({1}, {-1})


def test_kernel_layers_lie_at_their_thicknesses_from_every_adjacent_wall() -> None:
    mesh = _kernel_cube(_LAYERS)

    assert mesh.status is meshcore.MeshcoreStatus.OK
    kinds = np.asarray(mesh.vertex_kinds, dtype=np.int64)
    roles = np.asarray(mesh.cell_roles, dtype=np.int64)
    assert np.count_nonzero(roles == 2) > 0
    points = np.asarray(mesh.points, dtype=np.float64)
    columns = np.count_nonzero((kinds >= 1) & (kinds <= 3))
    layer = points[kinds == 4]
    assert layer.shape[0] == len(_LAYERS) * columns
    # Layers follow the wall normals: every patch and corner column puts one
    # point at each cumulative thickness from the cube (on a corner column,
    # from each of its faces). A feature-edge column's points move deeper
    # along their offset direction where that repairs the cells across the
    # edge, never thinner than requested.
    curve = np.count_nonzero(kinds == 2)
    gaps = np.minimum(layer, 1.0 - layer)
    nearest = np.min(gaps, axis=1)
    assert np.min(nearest) >= _LAYERS[0] - 1e-14
    for height in np.cumsum(_LAYERS):
        level = np.abs(nearest - height) <= 1e-14
        assert columns - curve <= np.count_nonzero(level) <= columns


def test_route_publishes_layers_and_proves_them() -> None:
    result = _plan(_LAYERS).execute()

    certification = result.certification
    assert certification is not None and certification.passed
    roles = next(
        attribute
        for attribute in result.attributes
        if attribute.name == "octree_cell_role"
    )
    assert np.count_nonzero(np.asarray(roles.values) == 2) > 0


def test_unproved_candidate_refuses_proof() -> None:
    candidate = _plan().execute(prove=False)

    assert candidate.mesh.connectivity.cell_count == 432
    with pytest.raises(MeshingFailure, match="prove=False"):
        candidate.prove()


def test_route_requires_a_polyhedral_target() -> None:
    source = _plc_cube()
    hexahedral = CellFamilyPolicy(
        required=("hexahedron",),
        allowed_transitions=("prism", "polyhedron"),
        allow_mixed=True,
    )
    with pytest.raises(MeshingFailure, match="a polyhedral target"):
        NativeMeshingProvider(NativeMeshingOptions("plc_octree_hex_dominant")).plan(
            source,
            _specification(source, families=hexahedral),
            coordinate_contract=SpatialCoordinateContract.si(),
        )
