"""Integer-cube PLC fixtures with explicit material and periodic facet identity."""

from collections.abc import Sequence

import numpy as np

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization import CellGeometrySpec, CellMesh, PolyhedralConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
)
from phydrax.meshing import MeshingScope, PeriodicConstraint, PiecewiseLinearComplex
from phydrax.meshing._polyhedral_adaptation import PolyhedralAdaptationOutcome
from phydrax.meshing._polyhedral_generation import (
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
    PolyhedralConstruction,
)


M = phx.meshing
POLYHEDRAL_CUBE_LOOPS = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)


def voxel_plc(
    cubes: Sequence[tuple[int, int, int]],
    regions: Sequence[int] | None = None,
) -> PiecewiseLinearComplex:
    """Build the oriented union of unit cubes with explicit material interfaces."""
    regions = [0] * len(cubes) if regions is None else regions
    vertices: list[tuple[int, int, int]] = []
    ids: dict[tuple[int, int, int], int] = {}
    faces: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]] = {}
    for cube, region in zip(cubes, regions, strict=True):
        corners = []
        for i in range(8):
            point = (
                cube[0] + (i & 1),
                cube[1] + ((i >> 1) & 1),
                cube[2] + ((i >> 2) & 1),
            )
            if point not in ids:
                ids[point] = len(vertices)
                vertices.append(point)
            corners.append(ids[point])
        for local in POLYHEDRAL_CUBE_LOOPS:
            face = tuple(corners[i] for i in local)
            faces.setdefault(tuple(sorted(face)), []).append((face, region))
    polygons, incidence = [], []
    for entries in faces.values():
        loop, region = entries[0]
        other = entries[1][1] if len(entries) == 2 else -1
        if other == region:
            continue
        polygons.append(loop)
        incidence.append((other, region))
    # Semantic groups deliberately span noncoplanar faces: they must not be
    # merged into a single invalid polygon by restricted assembly.
    pairs, facet_ids = np.unique(incidence, axis=0, return_inverse=True)
    return M.PiecewiseLinearComplex(
        vertices,
        polygons,
        facet_ids,
        pairs,
        tuple(f"material-{r}" for r in sorted(set(regions))),
    )


def periodic_cube_source() -> tuple[
    PiecewiseLinearComplex, tuple[PeriodicConstraint, ...]
]:
    """Return a unit-cube PLC with translations pairing all three facet axes."""
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float64
    )
    domain = PiecewiseLinearComplex(
        points,
        POLYHEDRAL_CUBE_LOOPS,
        np.arange(6, dtype=np.int64),
        [(-1, 0)] * 6,
        ("material-0",),
    )
    constraints = []
    for axis, source_facet, target_facet in ((0, 4, 5), (1, 2, 3), (2, 0, 1)):
        source = M.MeshingScope(
            "polyhedral-domain",
            "r1",
            M.MeshingEntityKind.GEOMETRY,
            2,
            f"source-{axis}",
            [source_facet],
        )
        target = M.MeshingScope(
            "polyhedral-domain",
            "r1",
            M.MeshingEntityKind.GEOMETRY,
            2,
            f"target-{axis}",
            [target_facet],
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[axis, 3] = 1
        constraints.append(
            M.PeriodicConstraint(
                source,
                target,
                matrix,
                tolerance=0,
                source_entity_ids=np.asarray([source_facet], dtype=np.int64),
                orientations=np.asarray([-1], dtype=np.int64),
            )
        )
    return domain, tuple(constraints)


def unequal_cube_seam_fixture() -> tuple[
    CellMesh, CellGeometrySpec, np.ndarray, MeshingScope, MeshingScope, np.ndarray
]:
    """Return cube geometry with unequal source/target seam subdivisions."""
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float64
    )
    faces = [*POLYHEDRAL_CUBE_LOOPS[:4], (0, 4, 6), (0, 6, 2), POLYHEDRAL_CUBE_LOOPS[5]]
    mesh = phx.discretization.CellMesh.from_polyhedra(
        points, [[np.asarray(face, dtype=np.int64) for face in faces]]
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    c = mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    offsets, values = np.asarray(c.face_vertex_offsets), np.asarray(c.face_vertex_values)
    facets = np.full(c.face_count, -1, dtype=np.int64)
    for face, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        corners = set(map(int, values[a:b]))
        if corners.issubset(POLYHEDRAL_CUBE_LOOPS[4]):
            facets[face] = 4
        if corners.issubset(POLYHEDRAL_CUBE_LOOPS[5]):
            facets[face] = 5
    source = M.MeshingScope(
        "cube-seam", "r1", M.MeshingEntityKind.GEOMETRY, 2, "left", [4]
    )
    target = M.MeshingScope(
        "cube-seam", "r1", M.MeshingEntityKind.GEOMETRY, 2, "right", [5]
    )
    matrix = np.eye(4)
    matrix[0, 3] = 1
    return mesh, geometry, facets, source, target, matrix


def periodic_fv_public_inputs(
    action_kind: str,
) -> tuple[
    M.NativePolyhedralSource,
    M.VolumeMeshingSpec,
    M.NativeMeshingOptions,
    np.ndarray,
]:
    """Return provider inputs shared by FV and contact diagnostics."""
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float64
    )
    domain = PiecewiseLinearComplex(
        points,
        POLYHEDRAL_CUBE_LOOPS,
        np.arange(6, dtype=np.int64),
        [(-1, 0)] * 6,
        ("material",),
    )
    source_scope = M.MeshingScope(
        "polyhedral-domain", "r1", M.MeshingEntityKind.GEOMETRY, 2, "left", [4]
    )
    target_scope = M.MeshingScope(
        "polyhedral-domain",
        "r1",
        M.MeshingEntityKind.GEOMETRY,
        2,
        "target",
        [2 if action_kind == "rotation" else 5],
    )
    action = np.eye(4, dtype=np.float64)
    action[0, 3] = 1
    if action_kind == "rotation":
        action = np.asarray(
            (
                (0.0, 1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
    elif action_kind == "screw":
        action[1, 1] = action[2, 2] = -1
        action[1:3, 3] = 1
    constraints = (
        M.PeriodicConstraint(
            source_scope,
            target_scope,
            action,
            tolerance=0,
            source_entity_ids=np.asarray([4], dtype=np.int64),
            orientations=np.asarray([-1], dtype=np.int64),
        ),
    )
    if action_kind == "translation":
        domain, constraints = periodic_cube_source()
    source = M.NativePolyhedralSource(
        domain,
        "polyhedral-domain",
        "r1",
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.zeros(2, dtype=np.float64),
    )
    root_scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "polyhedral-facets",
        np.arange(domain.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("polyhedron",))),
        root_scope,
        M.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            M.UniformSizeControl(root_scope, 10.0, strength=M.SizeControlStrength.SOFT),
        ),
        periodic_constraints=constraints,
    )
    return source, specification, M.NativeMeshingOptions("plc_restricted_power"), action


def boxes(xs: tuple[float, ...]) -> CellMesh:
    topology = reference_cell_topology("hexahedron")
    coordinates = np.asarray(
        [[x, y, z] for x in xs for z in (0.0, 1.0) for y in (0.0, 1.0)], dtype=np.float64
    )
    cells = []
    for interval in range(len(xs) - 1):
        vertices = [
            4 * (interval + int(x)) + 2 * int(z) + int(y) for x, y, z in topology.vertices
        ]
        cells.append(
            tuple(
                np.asarray([vertices[index] for index in face], dtype=np.int32)
                for face in topology.entities[2]
            )
        )
    return CellMesh.from_polyhedra(
        coordinates,
        cells,
        vertex_global_ids=np.arange(40, 40 + coordinates.shape[0], dtype=np.int64),
        cell_global_ids=np.arange(20, 20 + len(cells), dtype=np.int64),
    )


def periodic_adaptation_source() -> tuple[
    PiecewiseLinearComplex, tuple[PeriodicConstraint, ...], PolyhedralConstruction
]:
    """Prepare the two-site periodic cube used by polyhedral edit contracts."""
    domain, constraints = periodic_cube_source()
    with NativeExecutionScope(NativePolyhedralSchedule().native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain,
            sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
            periodic_constraints=constraints,
            scope=scope,
        )
    return domain, constraints, construction


def common_refinement_remap(
    source: CellMesh, target: CellMesh, outcome: PolyhedralAdaptationOutcome
) -> UnstructuredConservativeRemapPlan:
    common = outcome.common_refinement
    return UnstructuredConservativeRemapPlan(
        UnstructuredFiniteVolumePlan.from_cell_mesh(source).prepare(),
        UnstructuredFiniteVolumePlan.from_cell_mesh(target).prepare(),
        common.target_offsets,
        common.source_cells,
        common.volumes,
        method="common-refinement",
        provenance=common.refinement_id,
    )
