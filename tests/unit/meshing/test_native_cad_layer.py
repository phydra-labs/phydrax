"""Source-face boundary layers grown on native BRep volumes around a fixed core."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import point_triangle_distance
from phydrax.discretization.finite_volume import (
    prepare_polyhedral_finite_volume_geometry,
)
from phydrax.geometry._domain_boundary import _patch_chart_polygon
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep._constructors import brep_box, brep_cylinder, brep_sphere
from phydrax.meshing import NativeMeshingOptions
from phydrax.meshing._canonical import _entity_vertex_keys
from phydrax.meshing._contracts import MeshingFailure
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._trace import MeshingStageKind
from phydrax.meshing._volume_generation import NativeVolumeSchedule


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native layer/core meshing requires meshcore"
)

SCHEDULE = (0.01, 0.015)
LEVELS = np.concatenate(((0.0,), np.cumsum(SCHEDULE)))
CONTRACT = SpatialCoordinateContract.si()


def _box() -> Any:
    return brep_box(
        np.zeros(3), np.ones(3), coordinate_contract=CONTRACT, source_id="layer-box"
    )


def _sphere() -> Any:
    return brep_sphere(0.5, coordinate_contract=CONTRACT, source_id="layer-sphere")


def _cylinder() -> Any:
    return brep_cylinder(
        0.5, 1.0, coordinate_contract=CONTRACT, source_id="layer-cylinder"
    )


def _faces(model: Any, keep: Any) -> MeshingScope:
    """Source faces whose trim-polygon centroid satisfies ``keep``."""
    boundary = MeshingScope.whole(model, 2)
    domain = MeshingDomain.from_brep(model)
    selected = []
    for identifier in np.asarray(boundary.entity_ids, dtype=np.int64).tolist():
        entity_id, occurrence = domain.source_entity_key_for_scope(2, identifier)
        patch = domain.resolve_entity_id(2, entity_id, occurrence_path=occurrence)
        polygon = _patch_chart_polygon(domain, patch)
        assert polygon is not None
        center = np.mean(np.asarray(polygon[0], dtype=np.float64), axis=0)[None, :]
        if keep(domain.evaluate(np.asarray([patch]), center)[0]):
            selected.append(identifier)
    return MeshingScope(
        boundary.source_id,
        boundary.source_revision,
        boundary.entity_kind,
        2,
        boundary.entity_set_id,
        np.sort(np.asarray(selected, dtype=np.int64)),
    )


def _specification(
    model: Any, wall: MeshingScope, *, size: float, deviation: float, schedule: Any
) -> Any:
    boundary = MeshingScope.whole(model, 2)
    control = phx.meshing.BoundaryLayerControl(
        wall,
        phx.meshing.LayerSchedule(schedule),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        volume_scope=MeshingScope.whole(model, 3),
    )
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allowed_transitions=("pyramid",),
                allow_mixed=True,
            ),
        ),
        boundary,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                boundary, size, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        layer_controls=(control,),
        maximum_deviation=deviation,
    )


def _execute(model: Any, specification: Any) -> Any:
    options = NativeMeshingOptions(
        "cad_tetrahedral",
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
        ),
    )
    return phx.meshing.NativeMeshingProvider(options).plan(model, specification).execute()


def _blocks(mesh: Any, kind: str) -> np.ndarray:
    rows = [
        np.asarray(block.vertices, dtype=np.int64)
        for block in mesh.blocks
        if block.cell_kind == kind
    ]
    return np.concatenate(rows) if rows else np.empty((0, 0), dtype=np.int64)


def _exterior_triangles(mesh: Any) -> np.ndarray:
    """Exterior triangles of the result, oriented away from their owner cell."""
    connectivity = mesh.connectivity
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    owner = np.asarray(connectivity.face_owner, dtype=np.int64)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    centroids = {
        int(identifier): np.mean(points[row], axis=0)
        for block in mesh.blocks
        for identifier, row in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    triangles = []
    for face in np.flatnonzero(neighbor < 0).tolist():
        loop = values[offsets[face] : offsets[face + 1]]
        corners = points[loop]
        normal = np.sum(np.cross(corners, np.roll(corners, -1, axis=0)), axis=0)
        inward = centroids[int(cell_ids[owner[face]])] - np.mean(corners, axis=0)
        if np.dot(normal, inward) > 0.0:
            loop = loop[::-1]
        triangles.extend((loop[0], loop[k], loop[k + 1]) for k in range(1, loop.size - 1))
    return np.asarray(triangles, dtype=np.int64)


@jax.jit
def _nearest_squared(points: jax.Array, corners: jax.Array) -> jax.Array:
    shape = (points.shape[0], corners.shape[0], 3)
    squared = point_triangle_distance(
        jnp.broadcast_to(points[:, None, :], shape),
        *(jnp.broadcast_to(corners[None, :, corner], shape) for corner in range(3)),
    ).squared_distance
    return jnp.min(squared, axis=1)


def _distances(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the nearest of the given triangles."""
    # Equal point blocks (the last one padded) reuse one compiled kernel and keep
    # the pairwise table near 2**16 entries.
    block = max(1, 2**16 // corners.shape[0])
    count = points.shape[0]
    padded = np.concatenate(
        (points, np.repeat(points[:1], -count % block, axis=0)), axis=0
    )
    device_corners = jnp.asarray(corners)
    squared = np.concatenate(
        [
            np.asarray(
                _nearest_squared(
                    jnp.asarray(padded[start : start + block]), device_corners
                )
            )
            for start in range(0, padded.shape[0], block)
        ]
    )
    return np.sqrt(squared[:count])


def _assert_layered(result: Any, on_wall: Any, *, tolerance: float) -> int:
    """Check physical layer heights, layer count, and cap conformity; return wall size."""
    mesh = result.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    exterior = _exterior_triangles(mesh)
    wall = exterior[np.all(on_wall(points[exterior]), axis=1)]
    assert wall.shape[0] > 0
    prisms = _blocks(mesh, "prism")
    tetrahedra = _blocks(mesh, "simplex:3")
    assert prisms.shape[0] > 0 and tetrahedra.shape[0] > 0
    assert {block.cell_kind for block in mesh.blocks} <= {"prism", "simplex:3", "pyramid"}
    # Prism vertices 0-2 are the lower and 3-5 the upper triangle of one layer.
    lower = _distances(points[prisms[:, :3].reshape(-1)], points[wall]).reshape(-1, 3)
    upper = _distances(points[prisms[:, 3:].reshape(-1)], points[wall]).reshape(-1, 3)
    level = np.argmin(np.abs(lower[:, :1] - LEVELS[None, :-1]), axis=1)
    np.testing.assert_allclose(
        lower, np.repeat(LEVELS[level][:, None], 3, axis=1), rtol=0.0, atol=tolerance
    )
    np.testing.assert_allclose(
        upper,
        np.repeat(LEVELS[level + 1][:, None], 3, axis=1),
        rtol=0.0,
        atol=tolerance,
    )
    # Every wall triangle carries one prism per scheduled layer.
    np.testing.assert_array_equal(
        np.bincount(level, minlength=len(SCHEDULE)), wall.shape[0]
    )
    wall_keys = {tuple(sorted(row)) for row in wall.tolist()}
    assert {tuple(sorted(row)) for row in prisms[level == 0, :3].tolist()} == wall_keys

    # The outermost prism caps are exactly the layer/core interface: each is a
    # face of one core tetrahedron and of no other cell.
    cap_keys = [tuple(sorted(row)) for row in prisms[level == len(SCHEDULE) - 1, 3:]]
    tetrahedron_faces: dict[tuple[int, ...], int] = {}
    for cell in tetrahedra.tolist():
        for omit in range(4):
            key = tuple(sorted(cell[:omit] + cell[omit + 1 :]))
            tetrahedron_faces[key] = tetrahedron_faces.get(key, 0) + 1
    assert all(tetrahedron_faces.get(key) == 1 for key in cap_keys)
    patches = {patch.name: patch for patch in result.patches}
    assert patches["layer-core-interface"].scope.entity_ids.size == len(cap_keys)
    assert patches["wall"].scope.entity_ids.size == wall.shape[0]
    return wall.shape[0]


def _assert_certified_core(result: Any) -> None:
    assert result.certification is not None and result.certification.passed
    assert result.compliance.passed, result.compliance.issues
    validity = phx.discretization.certify_cell_geometry_validity(result.mesh)
    assert validity.invalid_count == 0 and validity.unresolved_count == 0
    assert validity.certified_valid_count == result.mesh.entity_set(3).count


def _finite_element_volume(mesh: Any) -> float:
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(block.cell_kind, 1)
            for block in mesh.blocks
        },
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    cells = discretization.mesh.entity_set(3)
    scope = phx.meshing.MeshingScope(
        discretization.mesh.mesh_id,
        discretization.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        3,
        cells.entity_set_id,
        np.asarray(cells.entity_ids, dtype=np.int64),
    )
    functional = phx.equations.FiniteElementFunctional(
        "volume",
        "u",
        lambda values, gradients, points, context: jnp.ones(values.shape[:2]),
        domain=discretization.integration_domain(
            "cell", phx.meshing.resolve_mesh_scope(discretization.mesh, scope)
        ),
    )
    return float(
        functional.evaluate(
            discretization, jnp.zeros((discretization.dof_maps[0].global_dof_count,))
        )
    )


def _enclosed_volume(mesh: Any) -> float:
    """Divergence-theorem volume of the outward exterior surface alone."""
    corners = np.asarray(mesh.coordinates, dtype=np.float64)[_exterior_triangles(mesh)]
    return float(np.sum(np.cross(corners[:, 0], corners[:, 1]) * corners[:, 2]) / 6.0)


def _box_wall_distance(points: np.ndarray) -> np.ndarray:
    return np.min(np.minimum(np.abs(points), np.abs(1.0 - points)), axis=-1)


def test_box_whole_boundary_wall_grows_certified_layers_around_a_tetrahedral_core() -> (
    None
):
    model = _box()
    result = _execute(
        model,
        _specification(
            model,
            MeshingScope.whole(model, 2),
            size=0.34,
            deviation=0.05,
            schedule=SCHEDULE,
        ),
    )

    _assert_certified_core(result)
    # The wall triangles tile the exact box faces, so distance to them is the
    # true wall distance of every column vertex.
    _assert_layered(
        result, lambda corners: _box_wall_distance(corners) == 0.0, tolerance=1.0e-9
    )
    assert _finite_element_volume(result.mesh) == pytest.approx(1.0, abs=1.0e-12)
    assert _enclosed_volume(result.mesh) == pytest.approx(1.0, abs=1.0e-12)
    # Columns diverge at the sharp edges, so their side faces are bilinear;
    # polyhedral FV decomposes those warped faces about their vertex centroids
    # and both neighbors integrate over the same triangles.
    finite_volume = prepare_polyhedral_finite_volume_geometry(result.mesh)
    warped = np.count_nonzero(finite_volume.face_quadrature_valid, axis=1) == 4
    assert np.any(warped)
    cell_volumes = np.asarray(finite_volume.cell_volumes, dtype=np.float64)
    assert np.all(cell_volumes > 0.0)
    assert float(np.sum(cell_volumes)) == pytest.approx(1.0, abs=1.0e-12)
    assert float(np.max(finite_volume.closure_residual)) <= 1.0e-14


def test_box_bottom_wall_layers_slide_along_fixed_sides_with_exact_charts() -> None:
    model = _box()
    wall = _faces(model, lambda point: point[2] == 0.0)
    assert wall.entity_ids.size == 1
    result = _execute(
        model,
        _specification(model, wall, size=0.34, deviation=0.05, schedule=SCHEDULE),
    )

    _assert_certified_core(result)
    _assert_layered(result, lambda corners: corners[..., 2] == 0.0, tolerance=1.0e-12)
    prisms = _blocks(result.mesh, "prism")
    heights = np.asarray(result.mesh.coordinates)[prisms.reshape(-1), 2]
    np.testing.assert_allclose(
        np.unique(np.round(heights, 12)), LEVELS, rtol=0.0, atol=1.0e-12
    )
    # Layer side strips belong to the patch of the side face they lie on.
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    row_of_vertex = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(result.mesh.vertex_global_ids))
    }
    face_vertices = dict(
        zip(
            np.asarray(result.mesh.entity_set(2).entity_ids).tolist(),
            _entity_vertex_keys(result.mesh, 2),
            strict=True,
        )
    )
    patches = {patch.name: patch for patch in result.patches}
    assert "layer-side" not in patches
    strips = 0
    for name, patch in patches.items():
        if not name.startswith("facet:"):
            continue
        corners = [
            points[[row_of_vertex[vertex] for vertex in face_vertices[face]]]
            for face in np.asarray(patch.scope.entity_ids).tolist()
        ]
        stacked = np.concatenate(corners)
        # Source faces are exact planes; the cap sits at the summed schedule.
        planes = [
            axis
            for axis in range(3)
            if (
                np.all(stacked[:, axis] == stacked[0, axis])
                and stacked[0, axis] in (0.0, 1.0)
            )
            or np.all(np.abs(stacked[:, axis] - LEVELS[-1]) <= 1.0e-12)
        ]
        assert len(planes) == 1, name
        strips += sum(
            bool(np.any((face[:, 2] > 0.0) & (face[:, 2] < LEVELS[-1])))
            for face in corners
        )
    assert strips > 0
    finite_volume = prepare_polyhedral_finite_volume_geometry(result.mesh)
    cell_volumes = np.asarray(finite_volume.cell_volumes, dtype=np.float64)
    assert np.all(cell_volumes > 0.0)
    assert float(np.sum(cell_volumes)) == pytest.approx(1.0, abs=1.0e-12)
    assert _finite_element_volume(result.mesh) == pytest.approx(1.0, abs=1.0e-12)
    assert _enclosed_volume(result.mesh) == pytest.approx(1.0, abs=1.0e-12)


def test_box_bottom_wall_refuses_a_schedule_that_meets_the_opposite_wall() -> None:
    model = _box()
    wall = _faces(model, lambda point: point[2] == 0.0)
    # Every bottom-wall column translates along +z, so the adjacent side planes
    # are parallel to the front and provably never crossed; they no longer limit
    # its height (396c7e272). The opposite top face does: fronts are probed up to
    # the total over the default proximity fraction (0.8), which here exceeds the
    # medial plane z = 0.5 between the bottom and top faces.
    schedule = (0.25, 0.35)
    assert sum(schedule) / 0.8 > 0.5

    with pytest.raises(MeshingFailure, match="collide") as failure:
        _execute(
            model,
            _specification(model, wall, size=0.34, deviation=0.05, schedule=schedule),
        )

    assert failure.value.stage == MeshingStageKind.LAYER_GENERATION.value
    locations = np.asarray(failure.value.locations, dtype=np.float64)
    assert locations.shape[0] > 0
    # The colliding columns stand on the wall.
    np.testing.assert_array_equal(locations[:, 2], 0.0)


def test_sphere_whole_wall_grows_smooth_layers_without_ridges() -> None:
    model = _sphere()
    result = _execute(
        model,
        _specification(
            model,
            MeshingScope.whole(model, 2),
            size=0.3,
            deviation=0.05,
            schedule=SCHEDULE,
        ),
    )

    _assert_certified_core(result)
    on_sphere = lambda corners: np.abs(np.linalg.norm(corners, axis=-1) - 0.5) <= 1e-9
    _assert_layered(result, on_sphere, tolerance=1.0e-9)
    enclosed = _enclosed_volume(result.mesh)
    assert _finite_element_volume(result.mesh) == pytest.approx(enclosed, abs=1.0e-12)
    # The faceted surface lies within the requested deviation of the sphere.
    exact = 4.0 / 3.0 * np.pi * 0.5**3
    assert abs(enclosed - exact) <= 4.0 * np.pi * 0.5**2 * 0.05


def test_cylinder_whole_wall_layers_close_at_its_circular_edges() -> None:
    model = _cylinder()
    result = _execute(
        model,
        _specification(
            model,
            MeshingScope.whole(model, 2),
            size=0.3,
            deviation=0.05,
            schedule=SCHEDULE,
        ),
    )

    _assert_certified_core(result)

    def on_cylinder(corners: np.ndarray) -> np.ndarray:
        radius = np.linalg.norm(corners[..., :2], axis=-1)
        height = corners[..., 2]
        return (np.abs(radius - 0.5) <= 1e-9) | (height == 0.0) | (height == 1.0)

    _assert_layered(result, on_cylinder, tolerance=1.0e-9)
    enclosed = _enclosed_volume(result.mesh)
    assert _finite_element_volume(result.mesh) == pytest.approx(enclosed, abs=1.0e-12)
    exact = np.pi * 0.5**2
    area = 2.0 * np.pi * 0.5 * (0.5 + 1.0)
    assert abs(enclosed - exact) <= area * 0.05
