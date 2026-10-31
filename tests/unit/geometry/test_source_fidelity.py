from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._exact_algebraic_geometry import (
    coefficient_endpoint,
    RadicalElement,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._surface_chart_records import SurfaceChartWitness
from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._domain_boundary import (
    _point_triangle_squared,
    MeshingDomainBoundarySource,
)
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    PatchPoleUse,
)
from phydrax.geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from phydrax.geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryDistanceWitness,
    SourceBoundaryPointWitness,
)
from phydrax.geometry._source_fidelity import (
    _box_gap_upper,
    _centroid_candidate_rows,
    _DerivativeCache,
    _exact_triangle2,
    _nearest_triangle_distance_upper,
    _normal_deviation_for_cell,
    _partition_findings,
    _require_reference_chart_partition,
    _surface_target_chart_rows,
    _triangle_centroid_balls,
    _volume_boundary_facet_lookup,
    certify_source_fidelity,
    polyhedral_face_triangle_vertex_ids,
)
from phydrax.geometry._source_identity import source_entity_id as _source_entity_id
from phydrax.geometry.brep import brep_cylinder, brep_sphere
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    CircleCurve,
    LineCurve,
    PlanePatch,
    SpherePatch,
)
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.linalg._hermitian_spectral import fraction_sqrt_interval
from phydrax.units import METER, MILLIMETER


class _PwlBoundaryOwner:
    domain: PiecewiseLinearDomain
    edges: np.ndarray
    envelope_id: str
    scope_entity_ids_by_mesh_row: tuple[np.ndarray, ...]
    source_id: str
    source_revision: str

    def __init__(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        facets = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64)
        self.domain = PiecewiseLinearDomain(
            vertices,
            facets,
            np.tile(np.asarray(((0, -1),), dtype=np.int64), (facets.shape[0], 1)),
            ("solid",),
            source_id="pwl-curve-source",
        )
        self.source_id = self.domain.source_id
        self.source_revision = self.domain.source_revision
        self.envelope_id = "pwl-curve-owner"
        edge_rows = sorted(
            {
                tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
                for face in facets
                for index in range(3)
            }
        )
        self.edges = np.asarray(edge_rows, dtype=np.int64)
        self.scope_entity_ids_by_mesh_row = (
            np.asarray((101, 102, 103, 104), dtype=np.int64),
            np.asarray(tuple(201 + row for row in range(len(edge_rows))), dtype=np.int64),
            np.asarray(
                tuple(301 + row for row in range(facets.shape[0])), dtype=np.int64
            ),
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        return self.scope_entity_ids_by_mesh_row[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        return canonical_fingerprint(
            {
                "source": (self.source_id, self.source_revision),
                "dimension": dimension,
                "entities": tuple(
                    int(self.scope_entity_ids(dimension)[row])
                    for row in range(self.scope_entity_ids(dimension).size)
                ),
            }
        )

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        kinds = ("pwl-vertex", "pwl-edge", "pwl-facet")
        identifiers = self.scope_entity_ids(dimension)
        if dimension not in range(len(kinds)) or not np.any(identifiers == entity_id):
            raise ValueError("PWL test source entity is outside its retained scope.")
        return (
            _source_entity_id(
                self.source_id,
                self.source_revision,
                kinds[dimension],
                int(entity_id),
            ),
            (),
        )

    def require_current(self, /) -> None:
        if (self.domain.source_id, self.domain.source_revision) != (
            self.source_id,
            self.source_revision,
        ):
            raise ValueError(
                "PWL test owner no longer matches its retained source domain."
            )


def _domain() -> MeshingDomain:
    uses = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(3, LineCurve((0.0, 0.0), (0.0, 1.0)), 0.0, 1.0, sense=-1),
    )
    return MeshingDomain(
        (
            MeshingSurfacePatch(
                PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (uses,)
            ),
        ),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(2, 3),
            MeshingDomainCurve(0, 3),
        ),
        4,
        source_id="source-square",
        source_revision="source-revision-1",
        corner_points=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float64,
        ),
    )


def _polyhedral_query_domain() -> MeshingDomain:
    patches = []
    curves = []
    corners = []
    for patch_index, z in enumerate((0.0, 2.0)):
        corner = 4 * patch_index
        curve = 4 * patch_index
        uses = (
            PatchCurveUse(
                curve,
                LineCurve((0.0, 0.0), (1.0, 0.0)),
                0.0,
                1.0,
            ),
            PatchCurveUse(
                curve + 1,
                LineCurve((1.0, 0.0), (0.0, 1.0)),
                0.0,
                1.0,
            ),
            PatchCurveUse(
                curve + 2,
                LineCurve((1.0, 1.0), (-1.0, 0.0)),
                0.0,
                1.0,
            ),
            PatchCurveUse(
                curve + 3,
                LineCurve((0.0, 0.0), (0.0, 1.0)),
                0.0,
                1.0,
                sense=-1,
            ),
        )
        patches.append(
            MeshingSurfacePatch(
                PlanePatch((0.0, 0.0, z), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                (uses,),
            )
        )
        curves.extend(
            (
                MeshingDomainCurve(corner, corner + 1),
                MeshingDomainCurve(corner + 1, corner + 2),
                MeshingDomainCurve(corner + 2, corner + 3),
                MeshingDomainCurve(corner, corner + 3),
            )
        )
        corners.extend(
            (
                (0.0, 0.0, z),
                (1.0, 0.0, z),
                (1.0, 1.0, z),
                (0.0, 1.0, z),
            )
        )
    return MeshingDomain(
        tuple(patches),
        tuple(curves),
        8,
        source_id="polyhedral-fidelity-source",
        source_revision="polyhedral-fidelity-revision",
        source_indices=(
            (11, 17, 23, 41, 51, 59, 67, 73),
            (101, 202, 303, 404, 501, 502, 503, 504),
            (901, 900),
        ),
        source_occurrences=(
            ((),) * 8,
            ((),) * 8,
            ((), ()),
        ),
        corner_points=np.asarray(corners, dtype=np.float64),
    )


def _witness(
    mesh: CellMesh, geometry: CellGeometrySpec, domain: MeshingDomain, charts: np.ndarray
) -> SurfaceChartWitness:
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    scope_by_key = {
        domain.source_entity_key_for_scope(2, int(entity_id)): int(entity_id)
        for entity_id in domain.scope_entity_ids(2)
    }
    source_entity = domain.entity_id(2, 0)
    occurrence = domain.source_occurrences[2][0]
    root_id = scope_by_key[(source_entity, occurrence)]
    return SurfaceChartWitness(
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=ids,
        patches=np.zeros(ids.shape, dtype=np.int64),
        charts=charts,
        source_root_cell_ids=np.full(ids.shape, root_id, dtype=np.int64),
        source_piece_ids=tuple((0, int(domain.source_indices[2][0])) for _ in ids),
        geometry_entity_ids=tuple(source_entity for _ in ids),
        occurrence_paths=tuple(occurrence for _ in ids),
        root_mesh_cell_bindings=(None,) * ids.size,
    )


def _mesh(points: np.ndarray, triangles: np.ndarray) -> CellMesh:
    return CellMesh(
        points,
        (CellBlock("triangles", "triangle", triangles),),
        vertex_global_ids=np.arange(points.shape[0], dtype=np.int64),
    )


def test_polyhedral_surface_chart_uses_actual_triangulated_face_vertices() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    )
    mesh = CellMesh.from_polyhedra(
        points,
        (faces,),
        vertex_global_ids=np.arange(100, 108, dtype=np.int64),
        cell_global_ids=np.asarray((700,), dtype=np.int64),
    )
    domain = _polyhedral_query_domain()
    source = MeshingDomainBoundarySource(domain, (0,))
    geometry = CellGeometrySpec.affine(mesh)
    scope_by_key = {
        domain.source_entity_key_for_scope(2, int(scope_id)): int(scope_id)
        for scope_id in domain.scope_entity_ids(2)
    }
    patch_keys = tuple(
        (domain.entity_id(2, patch), domain.source_occurrences[2][patch])
        for patch in range(2)
    )
    scope_key_order = tuple(
        domain.source_entity_key_for_scope(2, int(scope_id))
        for scope_id in domain.scope_entity_ids(2)
    )
    assert scope_key_order == tuple(reversed(patch_keys))
    assert tuple(domain.source_indices[2]) == (901, 900)
    assert scope_by_key[patch_keys[0]] != scope_by_key[patch_keys[1]]
    scope_witness = _witness(
        mesh,
        geometry,
        domain,
        np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),)),
    )
    assert tuple(int(value) for value in scope_witness.source_root_cell_ids) == (
        scope_by_key[patch_keys[0]],
    )
    assert scope_witness.source_piece_ids == ((0, 901),)
    from phydrax.discretization._cell_complex import PolyhedralConnectivity

    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise AssertionError("Polyhedral fixture must retain PolyhedralConnectivity.")
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    face_row = next(
        row
        for row in range(face_ids.size)
        if set(
            vertex_ids[face_values[face_offsets[row] : face_offsets[row + 1]]].tolist()
        )
        == {100, 101, 102, 103}
    )
    target_triangles = np.asarray(
        [
            triangle
            for target_face, triangle in polyhedral_face_triangle_vertex_ids(
                mesh, geometry
            )
            if target_face == int(face_ids[face_row])
        ],
        dtype=np.int64,
    )
    coordinates_by_id = {
        int(identifier): points[row] for row, identifier in enumerate(vertex_ids)
    }
    source_parameters = np.asarray(
        [
            [coordinates_by_id[int(identifier)][:2] for identifier in triangle]
            for triangle in target_triangles
        ],
        dtype=np.float64,
    )
    parent_triangles = target_triangles
    parent_parameters = source_parameters
    target_reference_subtriangles = np.asarray(
        (
            ((0.0, 0.0), (0.5, 0.0), (0.0, 1.0)),
            ((0.5, 0.0), (1.0, 0.0), (0.0, 1.0)),
        ),
        dtype=np.float64,
    )
    target_triangles = np.repeat(parent_triangles, 2, axis=0)
    source_parameter_rows: list[np.ndarray] = []
    for chart in parent_parameters:
        midpoint = (chart[0] + chart[1]) / 2.0
        source_parameter_rows.extend(
            (
                np.stack((chart[0], midpoint, chart[2])),
                np.stack((midpoint, chart[1], chart[2])),
            )
        )
    source_parameters = np.asarray(source_parameter_rows, dtype=np.float64)
    target_reference_vertices = np.tile(
        target_reference_subtriangles, (parent_triangles.shape[0], 1, 1)
    )
    orientations = np.asarray(
        [
            1
            if (
                (source_parameters[row, 1, 0] - source_parameters[row, 0, 0])
                * (source_parameters[row, 2, 1] - source_parameters[row, 0, 1])
                - (source_parameters[row, 1, 1] - source_parameters[row, 0, 1])
                * (source_parameters[row, 2, 0] - source_parameters[row, 0, 0])
            )
            > 0.0
            else -1
            for row in range(target_triangles.shape[0])
        ],
        dtype=np.int8,
    )
    owner = int(np.asarray(connectivity.face_owner)[face_row])
    cell_id = int(np.asarray(mesh.entity_set(3).entity_ids)[owner])
    local_face = int(np.asarray(connectivity.face_owner_local)[face_row])
    source_entity = domain.entity_id(2, 0)
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_scope_id,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=np.full(target_triangles.shape[0], cell_id, dtype=np.int64),
        entity_global_ids=np.full(
            target_triangles.shape[0], int(face_ids[face_row]), dtype=np.int64
        ),
        local_entity_indices=np.full(
            target_triangles.shape[0], local_face, dtype=np.int32
        ),
        target_reference_vertices=target_reference_vertices,
        target_face_vertex_global_ids=target_triangles,
        source_chart_ids=tuple(source_entity for _ in target_triangles),
        source_entity_ids=tuple(source_entity for _ in target_triangles),
        occurrence_paths=tuple(() for _ in target_triangles),
        source_parameters=source_parameters,
        orientations=orientations,
    )

    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=witness
    )

    assert certificate.status == "certified"
    coverage = certificate.coverage
    assert coverage is not None
    assert coverage.semantics == "certified"
    assert coverage.simplices.shape == (2, 3, 3)
    assert certificate.mesh_to_source_upper == 0.0
    assert certificate.source_to_mesh_upper == 0.0
    wrong_triangles = target_triangles.copy()
    wrong_triangles[0, 0] = 104
    wrong_face = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_scope_id,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=np.full(target_triangles.shape[0], cell_id, dtype=np.int64),
        entity_global_ids=np.full(
            target_triangles.shape[0], int(face_ids[face_row]), dtype=np.int64
        ),
        local_entity_indices=np.full(
            target_triangles.shape[0], local_face, dtype=np.int32
        ),
        target_reference_vertices=np.broadcast_to(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            (target_triangles.shape[0], 3, 2),
        ),
        target_face_vertex_global_ids=wrong_triangles,
        source_chart_ids=tuple(source_entity for _ in wrong_triangles),
        source_entity_ids=tuple(source_entity for _ in wrong_triangles),
        occurrence_paths=tuple(() for _ in wrong_triangles),
        source_parameters=source_parameters,
        orientations=orientations,
    )
    with pytest.raises(
        ValueError, match="not in the canonical actual face triangulation"
    ):
        wrong_face.require_current(mesh, geometry, source)


def test_box_gap_upper_uses_cross_endpoints_for_independent_nonzero_width_intervals() -> (
    None
):
    target = RadicalElement.sqrt(2) * (1 << 52)
    target_lower = Fraction.from_float(coefficient_endpoint(target, upper=False))
    target_upper = Fraction.from_float(coefficient_endpoint(target, upper=True))
    assert target_lower < target_upper

    source_lower = float(target_lower - 2)
    source_upper = float(target_lower)
    assert source_lower < source_upper
    old_same_side_bound = max(
        abs(target_upper - Fraction.from_float(source_upper)),
        abs(target_lower - Fraction.from_float(source_lower)),
    )

    assert _box_gap_upper(source_lower, source_upper, target) > old_same_side_bound


def _curve_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: ParametricCurveBoundarySource,
    /,
    *,
    parameters: np.ndarray,
    occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
) -> SourceBoundaryChartWitness:
    cell_id = int(np.asarray(mesh.blocks[0].global_ids)[0])
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.query_id,
        chart_dimension=1,
        reference_dimension=1,
        cell_global_ids=np.asarray((cell_id,), dtype=np.int64),
        entity_global_ids=np.asarray((cell_id,), dtype=np.int64),
        local_entity_indices=np.asarray((-1,), dtype=np.int32),
        target_reference_vertices=np.asarray((((0.0,), (1.0,)),), dtype=np.float64),
        source_chart_ids=source.chart_ids,
        source_entity_ids=source.curve_entity_ids,
        occurrence_paths=source.occurrence_paths
        if occurrence_paths is None
        else occurrence_paths,
        source_parameters=parameters,
        orientations=np.asarray((1,), dtype=np.int8),
    )


def test_parametric_curve_fidelity_certifies_exact_map_and_refuses_omitted_source_interval() -> (
    None
):
    source = ParametricCurveBoundarySource(
        (LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        ((0.0, 1.0),),
        source_id="curve-source",
        source_revision="curve-revision-1",
        curve_entity_ids=("authored-edge",),
        occurrence_paths=((),),
        chart_ids=("authored-edge-chart",),
    )
    mesh = CellMesh(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64),
        (CellBlock("curve", "interval", np.asarray(((0, 1),)), global_ids=(17,)),),
        vertex_global_ids=np.asarray((29, 31), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    complete = _curve_witness(
        mesh,
        geometry,
        source,
        parameters=np.asarray((((0.0,), (1.0,)),), dtype=np.float64),
    )

    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=complete
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0
    assert certificate.source_to_mesh_upper == 0.0
    omitted = _curve_witness(
        mesh,
        geometry,
        source,
        parameters=np.asarray((((0.0,), (0.5,)),), dtype=np.float64),
    )
    with pytest.raises(ValueError):
        certify_source_fidelity(mesh, geometry, source, tolerance=1.0, witness=omitted)

    wrong_occurrence = _curve_witness(
        mesh,
        geometry,
        source,
        parameters=np.asarray((((0.0,), (1.0,)),), dtype=np.float64),
        occurrence_paths=(("other-occurrence",),),
    )
    with pytest.raises(ValueError):
        certify_source_fidelity(
            mesh, geometry, source, tolerance=0.0, witness=wrong_occurrence
        )


def test_parametric_curve_fidelity_accepts_explicit_curved_approximation_tolerance() -> (
    None
):
    source = ParametricCurveBoundarySource(
        (CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0),),
        ((0.0, float(np.pi / 2.0)),),
        source_id="circle-source",
        source_revision="circle-revision-1",
        curve_entity_ids=("authored-circle-arc",),
        occurrence_paths=((),),
        chart_ids=("authored-circle-chart",),
    )
    mesh = CellMesh(
        np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64),
        (CellBlock("chord", "interval", np.asarray(((0, 1),)), global_ids=(43,)),),
        vertex_global_ids=np.asarray((47, 53), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    witness = _curve_witness(
        mesh,
        geometry,
        source,
        parameters=np.asarray((((0.0,), (float(np.pi / 2.0),)),), dtype=np.float64),
    )

    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=1.5, witness=witness
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper <= 1.5
    assert certificate.source_to_mesh_upper <= 1.5


def test_curve_fidelity_binds_a_real_edge_of_a_surface_mesh() -> None:
    source = ParametricCurveBoundarySource(
        (LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        ((0.0, 1.0),),
        source_id="curve-feature-source",
        source_revision="curve-feature-revision",
        curve_entity_ids=("authored-feature-edge",),
        occurrence_paths=((),),
        chart_ids=("authored-feature-chart",),
    )
    mesh = CellMesh(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64),
        (CellBlock("surface", "triangle", np.asarray(((0, 1, 2),)), global_ids=(23,)),),
        vertex_global_ids=np.asarray((31, 37, 41), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    reference = reference_cell_topology("triangle")
    from phydrax.discretization._cell_complex import PolygonalConnectivity

    local_edge_index = next(
        index
        for index, edge in enumerate(reference.entities[1])
        if tuple(sorted(int(value) for value in edge)) == (0, 1)
    )
    local_edge = reference.entities[1][local_edge_index]
    target_vertices = np.asarray(reference.vertices, dtype=np.float64)[
        np.asarray(local_edge)
    ]
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise AssertionError("Triangle fixture must retain PolygonalConnectivity.")
    edge_values = np.asarray(connectivity.edges, dtype=np.int64)
    local_edge_values = np.sort(np.asarray(local_edge, dtype=np.int64))
    edge_rows = np.flatnonzero(
        np.all(np.sort(edge_values, axis=1) == local_edge_values[None, :], axis=1)
    )
    if edge_rows.shape != (1,):
        raise AssertionError(
            "Triangle fixture must have one canonical local edge entity."
        )
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_id = int(edge_ids[int(edge_rows[0])])
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.query_id,
        chart_dimension=1,
        reference_dimension=2,
        cell_global_ids=np.asarray((23,), dtype=np.int64),
        entity_global_ids=np.asarray((edge_id,), dtype=np.int64),
        local_entity_indices=np.asarray((local_edge_index,), dtype=np.int32),
        target_reference_vertices=target_vertices[None],
        source_chart_ids=source.chart_ids,
        source_entity_ids=source.curve_entity_ids,
        occurrence_paths=source.occurrence_paths,
        source_parameters=np.asarray((((0.0,), (1.0,)),), dtype=np.float64),
        orientations=np.asarray((1,), dtype=np.int8),
    )

    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=witness
    )

    assert certificate.status == "certified"
    assert certificate.source_cell_global_ids == (edge_id,)


def test_piecewise_linear_curve_fidelity_uses_complete_source_and_target_edges() -> None:
    from phydrax.discretization._cell_complex import TetrahedralConnectivity

    owner = _PwlBoundaryOwner()
    scope_id = int(owner.scope_entity_ids(1)[0])
    source = PiecewiseLinearBoundarySource(
        owner, 1, entity_ids=np.asarray((scope_id,), dtype=np.int64)
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)

    def make_witness(
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        *,
        source_parameters: np.ndarray | None = None,
        source_entity_ids: tuple[str, ...] | None = None,
    ) -> SourceBoundaryChartWitness:
        source_row = int(source.source_rows[0])
        source_edge = tuple(int(value) for value in owner.edges[source_row])
        topology = reference_cell_topology("tetrahedron")
        local_edge_rows = tuple(
            row
            for row in range(len(topology.entities[1]))
            if set(topology.entities[1][row]) == set(source_edge)
        )
        if len(local_edge_rows) != 1:
            raise AssertionError(
                "The selected source edge must be one target tetrahedron edge."
            )
        local_edge_index = local_edge_rows[0]
        local_edge = tuple(int(value) for value in topology.entities[1][local_edge_index])
        if local_edge == source_edge:
            exact_parameters = ((0.0,), (1.0,))
        elif local_edge == source_edge[::-1]:
            exact_parameters = ((1.0,), (0.0,))
        else:
            raise AssertionError(
                "Source and target edge endpoints must retain one vertex ancestry."
            )
        parameters = (
            np.asarray((exact_parameters,), dtype=np.float64)
            if source_parameters is None
            else source_parameters
        )
        reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
        target_reference = reference_vertices[np.asarray(local_edge, dtype=np.int64)]
        connectivity = mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise AssertionError(
                "The target tetrahedron must retain canonical edge connectivity."
            )
        edge_vertices = np.asarray(connectivity.edges, dtype=np.int64)
        edge_rows = tuple(
            row
            for row in range(edge_vertices.shape[0])
            if set(int(value) for value in edge_vertices[row]) == set(source_edge)
        )
        if len(edge_rows) != 1:
            raise AssertionError("The target edge must have one global entity identity.")
        edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
        block = mesh.blocks[0]
        if not isinstance(block, CellBlock):
            raise AssertionError("The target tetrahedron must use a standard CellBlock.")
        orientation = 1 if parameters[0, 1, 0] > parameters[0, 0, 0] else -1
        return SourceBoundaryChartWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id=source.query_id,
            chart_dimension=1,
            reference_dimension=3,
            cell_global_ids=np.asarray((int(block.global_ids[0]),), dtype=np.int64),
            entity_global_ids=np.asarray((int(edge_ids[edge_rows[0]]),), dtype=np.int64),
            local_entity_indices=np.asarray((local_edge_index,), dtype=np.int32),
            target_reference_vertices=target_reference[None],
            source_chart_ids=source.chart_ids,
            source_entity_ids=(
                source.source_entity_keys
                if source_entity_ids is None
                else source_entity_ids
            ),
            occurrence_paths=source.source_occurrence_paths,
            source_parameters=parameters,
            orientations=np.asarray((orientation,), dtype=np.int8),
        )

    mesh = CellMesh.from_tetrahedra(owner.domain.vertices, cells)
    geometry = CellGeometrySpec.affine(mesh)
    witness = make_witness(mesh, geometry)
    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=witness
    )
    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0
    assert certificate.source_to_mesh_upper == 0.0

    partial_parameters = np.array(witness.source_parameters, copy=True)
    partial_parameters[0, 1, 0] = 0.5
    partial = make_witness(mesh, geometry, source_parameters=partial_parameters)
    with pytest.raises(ValueError):
        certify_source_fidelity(mesh, geometry, source, tolerance=0.0, witness=partial)

    wrong_entity = make_witness(mesh, geometry, source_entity_ids=(source.source_id,))
    with pytest.raises(ValueError):
        certify_source_fidelity(
            mesh, geometry, source, tolerance=0.0, witness=wrong_entity
        )

    perturbed_vertices = np.array(owner.domain.vertices, copy=True)
    perturbed_vertices[int(owner.edges[int(source.source_rows[0]), 1]), 2] += 0.1
    perturbed_mesh = CellMesh.from_tetrahedra(perturbed_vertices, cells)
    perturbed_geometry = CellGeometrySpec.affine(perturbed_mesh)
    perturbed_witness = make_witness(perturbed_mesh, perturbed_geometry)
    perturbed = certify_source_fidelity(
        perturbed_mesh,
        perturbed_geometry,
        source,
        tolerance=0.0,
        witness=perturbed_witness,
    )
    assert perturbed.status == "unresolved"


def test_volume_boundary_chart_keeps_parent_reference_coordinates_and_local_face() -> (
    None
):
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh.from_tetrahedra(points, np.asarray(((0, 1, 2, 3),), dtype=np.int32))
    geometry = CellGeometrySpec.affine(mesh)
    cell_id = int(mesh.entity_set(3).entity_ids[0])
    local_face = 0
    face_id = _volume_boundary_facet_lookup(mesh)[(cell_id, local_face)][0]
    topology = reference_cell_topology("tetrahedron")
    reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
    face_indices = np.asarray(topology.entities[2][local_face], dtype=np.int64)
    target_reference = reference_vertices[face_indices]
    parameters = np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),), dtype=np.float64)

    def make_witness(
        target_local_face: int,
        target_reference_vertices: np.ndarray,
    ) -> SourceBoundaryChartWitness:
        return SourceBoundaryChartWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id="volume-chart-source",
            source_revision="volume-chart-revision",
            source_query_id="volume-chart-query",
            chart_dimension=2,
            reference_dimension=3,
            cell_global_ids=np.asarray((cell_id,), dtype=np.int64),
            entity_global_ids=np.asarray((face_id,), dtype=np.int64),
            local_entity_indices=np.asarray((target_local_face,), dtype=np.int32),
            target_reference_vertices=target_reference_vertices[None],
            source_chart_ids=("source-face-chart",),
            source_entity_ids=("source-face",),
            occurrence_paths=((),),
            source_parameters=parameters,
            orientations=np.asarray((1,), dtype=np.int8),
        )

    witness = make_witness(local_face, target_reference)
    assert all(len(point) == 3 for point in witness.target_reference_chart(0))
    cell_rows, face_charts, orientations = _surface_target_chart_rows(
        mesh, geometry, None, witness, full_source=False
    )
    assert cell_rows == ((0, 0),)
    assert face_charts == (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
    )
    assert orientations == (1,)

    wrong_local_face = make_witness(1, target_reference)
    with pytest.raises(ValueError, match="facet ancestry"):
        _surface_target_chart_rows(
            mesh, geometry, None, wrong_local_face, full_source=False
        )
    wrong_parent_reference = make_witness(
        local_face,
        reference_vertices[np.asarray(topology.entities[2][1], dtype=np.int64)],
    )
    with pytest.raises(ValueError, match="local reference facet"):
        _surface_target_chart_rows(
            mesh, geometry, None, wrong_parent_reference, full_source=False
        )


def test_parametric_curve_point_witness_uses_authored_endpoint_query() -> None:
    source = ParametricCurveBoundarySource(
        (LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        ((0.0, 1.0),),
        source_id="point-curve-source",
        source_revision="point-curve-revision-1",
        curve_entity_ids=("point-curve-edge",),
        occurrence_paths=((),),
        chart_ids=("point-curve-chart",),
        endpoint_source_keys=((("point-curve-start", ()), ("point-curve-end", ())),),
    )
    mesh = CellMesh(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64),
        (CellBlock("curve", "interval", np.asarray(((0, 1),)), global_ids=(59,)),),
        vertex_global_ids=np.asarray((61, 67), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    witness = SourceBoundaryPointWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.query_id,
        vertex_global_ids=np.asarray((61,), dtype=np.int64),
        source_entity_ids=("point-curve-start",),
        occurrence_paths=((),),
    )

    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=witness
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0


def test_source_point_witness_evaluates_target_cell_maps_and_refuses_perturbation() -> (
    None
):
    domain = _domain()
    source = MeshingDomainBoundarySource(domain, (0,))
    source_entity, occurrence = (
        domain.entity_id(0, 0),
        domain.source_occurrences[0][0],
    )
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )

    def witness(mesh: CellMesh, geometry: CellGeometrySpec) -> SourceBoundaryPointWitness:
        return SourceBoundaryPointWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id=source.source_scope_id,
            vertex_global_ids=np.asarray((17,), dtype=np.int64),
            source_entity_ids=(source_entity,),
            occurrence_paths=(occurrence,),
        )

    mesh = CellMesh(
        points,
        (CellBlock("surface", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)))),),
        vertex_global_ids=np.asarray((17, 19, 23, 29), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=witness(mesh, geometry)
    )
    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0
    certificate.require_current(mesh, geometry, source)

    wrong_occurrence = SourceBoundaryPointWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_scope_id,
        vertex_global_ids=np.asarray((17,), dtype=np.int64),
        source_entity_ids=(source_entity,),
        occurrence_paths=(("other-occurrence",),),
    )
    with pytest.raises(ValueError):
        certify_source_fidelity(
            mesh, geometry, source, tolerance=0.0, witness=wrong_occurrence
        )

    perturbed_points = points.copy()
    perturbed_points[0, 0] = 0.25
    perturbed_mesh = CellMesh(
        perturbed_points,
        (CellBlock("surface", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)))),),
        vertex_global_ids=np.asarray((17, 19, 23, 29), dtype=np.int64),
    )
    perturbed_geometry = CellGeometrySpec.affine(perturbed_mesh)
    perturbed = certify_source_fidelity(
        perturbed_mesh,
        perturbed_geometry,
        source,
        tolerance=0.0,
        witness=witness(perturbed_mesh, perturbed_geometry),
    )
    assert perturbed.status == "violated"
    assert perturbed.mesh_to_source_lower > 0.0


def test_root_bounded_cad_point_requires_explicit_approximation_tolerance() -> None:
    from phydrax.geometry.brep._boolean import boolean_brep
    from phydrax.geometry.brep._root_bindings import BRepVertexRoot

    coordinates = SpatialCoordinateContract(MILLIMETER)
    first = brep_cylinder(
        1.0, 4.0, base_center=(0.0, 0.0, -2.0), coordinate_contract=coordinates
    )
    second = brep_cylinder(
        0.8,
        4.0,
        base_center=(-2.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        coordinate_contract=coordinates,
    )
    model = boolean_brep(first, second, "intersection").model
    brep = model.geometry
    root_rows = tuple(
        row
        for row, root in enumerate(brep.vertex_roots)
        if isinstance(root, BRepVertexRoot) and brep.vertex_exact_points[row] is None
    )
    assert root_rows
    root_row = root_rows[0]
    root = brep.vertex_roots[root_row]
    if not isinstance(root, BRepVertexRoot):
        raise AssertionError(
            "The selected nonbinary source vertex must retain BRep root authority."
        )
    enclosure = np.asarray(root.point_enclosure(), dtype=np.float64)
    assert enclosure.shape == (2, 3)
    assert np.any(enclosure[0] < enclosure[1])

    domain = MeshingDomain.from_brep(model)
    domain_row = next(
        row
        for row, source_row in enumerate(domain.source_indices[0])
        if source_row == root_row
    )
    patch_row = None
    for candidate, patch in enumerate(domain.patches):
        corners: set[int] = set()
        for loop in patch.loops:
            for use in loop:
                if isinstance(use, PatchPoleUse):
                    corners.add(use.corner)
                else:
                    curve = domain.curves[use.curve]
                    corners.update((curve.start, curve.end))
        if domain_row in corners:
            patch_row = candidate
            break
    assert patch_row is not None
    source = MeshingDomainBoundarySource(domain, (patch_row,))
    expected_key = (
        domain.entity_id(0, domain_row),
        domain.source_occurrences[0][domain_row],
    )
    scope_id = next(
        int(identifier)
        for identifier in domain.scope_entity_ids(0)
        if domain.source_entity_key_for_scope(0, int(identifier)) == expected_key
    )
    source_entity, occurrence = domain.source_entity_key_for_scope(0, scope_id)
    target_point = np.mean(enclosure, axis=0)
    points = np.asarray(
        (target_point, target_point + (0.2, 0.0, 0.0), target_point + (0.0, 0.2, 0.1)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (CellBlock("root-point", "triangle", np.asarray(((0, 1, 2),))),),
        vertex_global_ids=np.asarray((101, 103, 107), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    witness = SourceBoundaryPointWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_scope_id,
        vertex_global_ids=np.asarray((101,), dtype=np.int64),
        source_entity_ids=(source_entity,),
        occurrence_paths=(occurrence,),
    )

    exact = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=witness
    )
    assert exact.status == "unresolved"
    bounded = certify_source_fidelity(
        mesh, geometry, source, tolerance=1.0, witness=witness
    )
    assert bounded.status == "certified"


def test_analytic_profile_distance_witness_is_continuous_without_claiming_coverage() -> (
    None
):
    center = (0.013, -0.021, 0.007)
    contract = SpatialCoordinateContract(METER)
    compiled = phx.geometry.Sphere(center, 0.4).compile()
    source = AnalyticImplicitProfile(
        compiled, contract, source_id="distance-only-profile"
    )
    origin = np.asarray(center, dtype=np.float64)
    points = np.asarray(
        (
            origin + (0.4, 0.0, 0.0),
            origin + (0.0, 0.4, 0.0),
            origin + (0.0, 0.0, 0.4),
            origin + (-0.4, -0.4, -0.4),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (CellBlock("restricted", "tetrahedron", np.asarray(((0, 1, 2, 3),))),),
        vertex_global_ids=np.asarray((211, 223, 227, 229), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    source_cover = source.boundary_cover(0.02)
    witness = SourceBoundaryDistanceWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.profile_id,
        sample_count=int(source_cover.points.shape[0]),
        cover_radius=0.02,
        target_facet_global_ids=np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64),
    )
    assert witness.cover_radius < source.cover_radius
    wrong_count = SourceBoundaryDistanceWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.profile_id,
        sample_count=witness.sample_count + 1,
        cover_radius=witness.cover_radius,
        target_facet_global_ids=witness.target_facet_global_ids,
    )
    with pytest.raises(ValueError, match="complete certified boundary cover"):
        wrong_count.require_current(mesh, geometry, source)

    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=2.0, witness=witness
    )

    assert certificate.status == "certified"
    assert certificate.coverage is None
    assert certificate.mesh_to_source_upper <= 2.0
    assert certificate.source_to_mesh_upper <= 2.0


@pytest.mark.parametrize("origin", (0.0, 1.0e6))
def test_pruned_reverse_distance_equals_the_exact_minimum_over_every_triangle(
    origin: float,
) -> None:
    generator = np.random.default_rng(7)
    corners = origin + generator.uniform(-1.0, 1.0, size=(60, 3, 3))
    # Near ties: translated copies sit a sub-ulp-scale step farther away.
    corners[30:40] = corners[:10] + 1.0e-12

    def exact(value: np.ndarray) -> tuple[Fraction, Fraction, Fraction]:
        return (
            Fraction.from_float(float(value[0])),
            Fraction.from_float(float(value[1])),
            Fraction.from_float(float(value[2])),
        )

    triangles = tuple(
        (exact(triangle[0]), exact(triangle[1]), exact(triangle[2]))
        for triangle in corners
    )
    centers, radii = _triangle_centroid_balls(triangles)
    center_values = np.asarray(
        [[float(value) for value in center] for center in centers], dtype=np.float64
    )
    radius_values = np.asarray([float(radius) for radius in radii], dtype=np.float64)
    center_scale = float(np.max(np.abs(center_values)))
    radius_bound = float(np.max(radius_values))
    vertex_values = np.asarray(
        [
            [[float(value) for value in vertex] for vertex in triangle]
            for triangle in triangles
        ],
        dtype=np.float64,
    )
    box_lower = np.nextafter(np.min(vertex_values, axis=1), -np.inf)
    box_upper = np.nextafter(np.max(vertex_values, axis=1), np.inf)
    samples = origin + generator.uniform(-1.5, 1.5, size=(25, 3))
    for row, rows in _centroid_candidate_rows(
        samples, center_values, center_scale, radius_bound
    ):
        sample = samples[row]
        point = exact(sample)
        unpruned = min(
            fraction_sqrt_interval(_point_triangle_squared(point, triangle))[1]
            for triangle in triangles
        )
        pruned = _nearest_triangle_distance_upper(
            point,
            sample,
            triangles,
            center_values,
            radius_values,
            rows,
            center_scale,
            radius_bound,
            box_lower,
            box_upper,
            vertex_values,
        )
        assert pruned == unpruned


def test_missing_fidelity_witness_retains_unbounded_failure_evidence() -> None:
    domain = _domain()
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    mesh = _mesh(points, np.asarray(((0, 1, 2),), dtype=np.int64))
    geometry = CellGeometrySpec.affine(mesh)
    source = MeshingDomainBoundarySource(domain, (0,))

    certificate = certify_source_fidelity(mesh, geometry, source, tolerance=0.0)

    assert certificate.status == "unresolved"
    assert certificate.mesh_to_source_lower == certificate.source_to_mesh_lower == 0.0
    assert np.isposinf(certificate.mesh_to_source_upper)
    assert np.isposinf(certificate.source_to_mesh_upper)
    assert any(
        finding.check == "source_fidelity_witness" for finding in certificate.findings
    )
    certificate.require_current(mesh, geometry, source)


def test_source_fidelity_certifies_complete_curved_chart_and_zero_normal_deviation() -> (
    None
):
    domain = _domain()
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    mesh = _mesh(points, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64))
    geometry = CellGeometrySpec.affine(mesh)
    chart = np.asarray(
        (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        dtype=np.float64,
    )
    source = MeshingDomainBoundarySource(domain, (0,))

    certificate = certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.0,
        maximum_normal_deviation=0.0,
        witness=_witness(mesh, geometry, domain, chart),
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0
    assert certificate.source_to_mesh_upper == 0.0
    assert certificate.maximum_normal_deviation_bound == 0.0
    certificate.require_current(mesh, geometry, source)


def test_source_fidelity_detects_omitted_source_chart_region() -> None:
    domain = _domain()
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)), dtype=np.float64
    )
    mesh = _mesh(points, np.asarray(((0, 1, 2),), dtype=np.int64))
    geometry = CellGeometrySpec.affine(mesh)
    chart = np.asarray((((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),), dtype=np.float64)
    source = MeshingDomainBoundarySource(domain, (0,))

    certificate = certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=1.0,
        witness=_witness(mesh, geometry, domain, chart),
    )

    assert certificate.status == "violated"
    assert any(
        finding.check == "source_chart_coverage" for finding in certificate.findings
    )


def test_source_normal_deviation_uses_actual_target_map_orientation() -> None:
    domain = _domain()
    angle = np.pi / 6.0
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, np.cos(angle), np.sin(angle)),
            (0.0, np.cos(angle), np.sin(angle)),
        ),
        dtype=np.float64,
    )
    mesh = _mesh(points, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64))
    geometry = CellGeometrySpec.affine(mesh)
    chart = np.asarray(
        (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        dtype=np.float64,
    )
    source = MeshingDomainBoundarySource(domain, (0,))

    certificate = certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=1.0,
        maximum_normal_deviation=0.1,
        witness=_witness(mesh, geometry, domain, chart),
    )

    assert certificate.status == "violated"
    assert certificate.maximum_normal_deviation_lower is not None
    assert certificate.maximum_normal_deviation_lower > 0.1


def test_curved_cell_normal_deviation_bound_is_first_order_in_cell_width() -> None:
    domain = MeshingDomain.from_brep(
        brep_cylinder(1.0, 2.0, coordinate_contract=SpatialCoordinateContract.si())
    )
    for width in (Fraction(1, 10), Fraction(1, 40), Fraction(1, 160)):
        triangle = (
            (Fraction(1), Fraction(1, 4)),
            (Fraction(1) + width, Fraction(1, 4)),
            (Fraction(1), Fraction(1, 4) + width),
        )
        points = np.asarray(
            domain.evaluate(
                np.full((3,), 1, dtype=np.int64),
                np.asarray([[float(u), float(v)] for u, v in triangle], dtype=np.float64),
            ),
            dtype=np.float64,
        )
        target = np.cross(points[1] - points[0], points[2] - points[0])
        bound = _normal_deviation_for_cell(
            domain,
            1,
            triangle,
            (
                Fraction(float(target[0])),
                Fraction(float(target[1])),
                Fraction(float(target[2])),
            ),
            _DerivativeCache(),
        )
        # The true deviation is below the angular span of the corners about the
        # axis. A first-order bound stays within a fixed multiple of that span,
        # while an O(sqrt(width)) bound exceeds it on the finer cells.
        span = float(np.ptp(np.unwrap(np.arctan2(points[:, 1], points[:, 0]))))
        assert bound is not None
        assert 0.0 <= float(bound[0]) <= float(bound[1]) <= 2.0 * span


def test_sphere_cell_normal_bound_splits_until_it_decides_the_limit() -> None:
    domain = MeshingDomain.from_brep(
        brep_sphere(1.0, coordinate_contract=SpatialCoordinateContract.si())
    )
    # A chart cell elongated in latitude: its box enclosure of the closed-form
    # normal exceeds the limit although the cell meets it.
    triangle = (
        (Fraction(3.8288160465625602), Fraction(0.9817477042468101)),
        (Fraction(3.877903431774901), Fraction(1.079922474671491)),
        (Fraction(3.730641276137879), Fraction(1.178097245096172)),
    )
    points = np.asarray(
        domain.evaluate(
            np.zeros((3,), dtype=np.int64),
            np.asarray([[float(u), float(v)] for u, v in triangle], dtype=np.float64),
        ),
        dtype=np.float64,
    )
    cross = np.cross(points[1] - points[0], points[2] - points[0])
    target = (
        Fraction(float(cross[0])),
        Fraction(float(cross[1])),
        Fraction(float(cross[2])),
    )
    limit = Fraction(1, 5)
    whole = _normal_deviation_for_cell(domain, 0, triangle, target, _DerivativeCache())
    split = _normal_deviation_for_cell(
        domain, 0, triangle, target, _DerivativeCache(), limit
    )
    assert whole is not None and split is not None
    assert whole[1] > limit
    assert whole[0] <= split[0] <= split[1] <= limit


def test_source_boundary_chart_witness_retains_non_dyadic_partition_exactly() -> None:
    a = (Fraction(0), Fraction(0))
    p = (Fraction(1, 3), Fraction(0))
    q = (Fraction(1, 3), Fraction(2, 3))
    b = (Fraction(1), Fraction(0))
    c = (Fraction(0), Fraction(1))
    triangles = ((a, p, c), (p, b, q), (p, q, c))
    exact_bank = tuple(
        tuple(tuple(value for value in point) for point in triangle)
        for triangle in triangles
    )
    float_bank = np.asarray(
        [
            [[float(value) for value in point] for point in triangle]
            for triangle in triangles
        ],
        dtype=np.float64,
    )

    def witness(*, exact: bool) -> SourceBoundaryChartWitness:
        exact_kwargs = (
            {
                "target_reference_rational_tokens": rational_chart_tokens(exact_bank),
                "source_parameter_rational_tokens": rational_chart_tokens(exact_bank),
            }
            if exact
            else {}
        )
        return SourceBoundaryChartWitness(
            mesh_id="non-dyadic-mesh",
            geometry_id="non-dyadic-geometry",
            topology_id="non-dyadic-topology",
            source_id="non-dyadic-source",
            source_revision="r1",
            source_query_id="query-1",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=np.zeros((3,), dtype=np.int64),
            entity_global_ids=np.zeros((3,), dtype=np.int64),
            local_entity_indices=np.full((3,), -1, dtype=np.int64),
            target_reference_vertices=float_bank,
            source_chart_ids=("source-chart",) * 3,
            source_entity_ids=("source-face",) * 3,
            occurrence_paths=((),) * 3,
            source_parameters=float_bank,
            orientations=np.ones((3,), dtype=np.int8),
            **exact_kwargs,
        )

    exact_witness = witness(exact=True)
    exact_rows = tuple(
        _exact_triangle2(exact_witness.source_parameter_chart(row))
        for row in range(len(exact_witness.source_chart_ids))
    )
    _require_reference_chart_partition(
        exact_rows,
        (
            (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            ),
        ),
    )
    assert any(
        Fraction(1, 3) == coordinate
        for row in range(len(exact_witness.source_chart_ids))
        for point in exact_witness.source_parameter_chart(row)
        for coordinate in point
    )

    rounded_witness = witness(exact=False)
    assert exact_witness.witness_id != rounded_witness.witness_id
    rounded_rows = tuple(
        _exact_triangle2(rounded_witness.source_parameter_chart(row))
        for row in range(len(rounded_witness.source_chart_ids))
    )
    with pytest.raises(ValueError):
        _require_reference_chart_partition(
            rounded_rows,
            (
                (
                    (Fraction(0), Fraction(0)),
                    (Fraction(1), Fraction(0)),
                    (Fraction(0), Fraction(1)),
                ),
            ),
        )


def test_source_partition_respects_authored_patch_orientation() -> None:
    base = _domain()
    reversed_domain = MeshingDomain(
        tuple(
            MeshingSurfacePatch(
                patch.surface, patch.loops, orientation=-patch.orientation
            )
            for patch in base.patches
        ),
        base.curves,
        base.corner_count,
        source_id=base.source_id,
        source_revision=base.source_revision,
        corner_points=np.asarray(base.corner_points),
    )
    source = MeshingDomainBoundarySource(reversed_domain, (0,))
    triangles = tuple(
        triangle
        for patch, values in source.boundary_parameter_triangles()
        if patch == 0
        for triangle in values
    )
    target_triangles = tuple(tuple(reversed(triangle)) for triangle in triangles)
    mesh_vertex_by_uv = {
        tuple(Fraction.from_float(float(value)) for value in point[:2]): row
        for row, point in enumerate(np.asarray(base.corner_points))
    }
    mesh_triangles = np.asarray(
        [
            tuple(mesh_vertex_by_uv[point] for point in triangle)
            for triangle in target_triangles
        ],
        dtype=np.int64,
    )
    mesh = _mesh(np.asarray(base.corner_points, dtype=np.float64), mesh_triangles)
    geometry = CellGeometrySpec.affine(mesh)
    witness = _witness(
        mesh,
        geometry,
        reversed_domain,
        np.asarray(target_triangles, dtype=np.float64),
    )

    assert _partition_findings(source, witness) == ()

    positive_source = MeshingDomainBoundarySource(base, (0,))
    positive_witness = _witness(
        mesh,
        geometry,
        base,
        np.asarray(target_triangles, dtype=np.float64),
    )
    positive_findings = _partition_findings(positive_source, positive_witness)
    assert any(
        finding.check == "source_chart_orientation" and finding.status == "violated"
        for finding in positive_findings
    )


def _sphere_lune_domain(width: float, /) -> MeshingDomain:
    """One sphere lune ``[0, width] x [-pi/2, pi/2]`` with both poles collapsed."""
    half_pi = 0.5 * np.pi
    uses = (
        PatchPoleUse(0, LineCurve((0.0, -half_pi), (width, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, LineCurve((width, -half_pi), (0.0, np.pi)), 0.0, 1.0),
        PatchPoleUse(1, LineCurve((width, half_pi), (-width, 0.0)), 0.0, 1.0),
        PatchCurveUse(0, LineCurve((0.0, -half_pi), (0.0, np.pi)), 0.0, 1.0, sense=-1),
    )
    return MeshingDomain(
        (
            MeshingSurfacePatch(
                SpherePatch(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    1.0,
                ),
                (uses,),
            ),
        ),
        (MeshingDomainCurve(0, 1), MeshingDomainCurve(0, 1)),
        2,
        source_id="source-sphere-lune",
        source_revision="source-revision-1",
        tolerance=1.0e-12,
        corner_points=np.asarray(((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), dtype=np.float64),
    )


def test_collapsed_pole_slivers_complete_the_partition_within_their_cap_bound() -> None:
    width = 2.0 * np.pi / 3.0
    half_pi = 0.5 * np.pi
    band = 1.0
    domain = _sphere_lune_domain(width)
    source = MeshingDomainBoundarySource(domain, (0,))
    # Chart rows: a middle band and two fan cells per pole. Each fan cell keeps
    # its own chart point on the collapsed pole side, leaving three slivers per
    # pole between consecutive pole-side points.
    b0, bm, b1 = (0.0, -band), (0.5 * width, -band), (width, -band)
    t0, tm, t1 = (0.0, band), (0.5 * width, band), (width, band)
    south = ((0.25 * width, -half_pi), (0.75 * width, -half_pi))
    north = ((0.25 * width, half_pi), (0.75 * width, half_pi))
    charts = (
        (south[0], bm, b0),
        (south[1], b1, bm),
        (b0, bm, tm),
        (b0, tm, t0),
        (bm, b1, t1),
        (bm, t1, tm),
        (north[0], t0, tm),
        (north[1], tm, t1),
    )
    vertex_rows = {b0: 0, bm: 1, b1: 2, t0: 3, tm: 4, t1: 5}
    surface = domain.patches[0].surface

    def physical(points: np.ndarray, /) -> np.ndarray:
        return np.asarray(surface.evaluate(jnp.asarray(points)), dtype=np.float64)

    points = np.concatenate(
        (
            physical(np.asarray(tuple(vertex_rows), dtype=np.float64)),
            np.asarray(domain.corner_points, dtype=np.float64),
        )
    )

    def witness_for(rows: tuple[int, ...], /) -> SurfaceChartWitness:
        triangles = np.asarray(
            [
                tuple(
                    6 if point in south else 7 if point in north else vertex_rows[point]
                    for point in charts[row]
                )
                for row in rows
            ],
            dtype=np.int64,
        )
        mesh = _mesh(points, triangles)
        return _witness(
            mesh,
            CellGeometrySpec.affine(mesh),
            domain,
            np.asarray([charts[row] for row in rows], dtype=np.float64),
        )

    complete = witness_for(tuple(range(len(charts))))
    exact_charts = tuple(
        _exact_triangle2(tuple((Fraction(u), Fraction(v)) for u, v in chart))
        for chart in charts
    )
    slivers = source.collapsed_pole_slivers(0, exact_charts)

    assert len(slivers) == 6
    # Owners are the fan cells holding a pole-side edge to the sliver apex.
    assert {owner for owner, _ in slivers} == {0, 1, 6, 7}
    assert _partition_findings(source, complete) == ()

    rng = np.random.default_rng(7)
    for _owner, sliver in slivers:
        corners = np.asarray(
            [[float(value) for value in point] for point in sliver], dtype=np.float64
        )
        weights = rng.dirichlet(np.ones(3), size=256)
        samples = physical(np.concatenate((corners, weights @ corners)))
        sampled_diameter = float(
            np.max(np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1))
        )
        assert source.pole_sliver_cap_bound(0, sliver) >= Fraction(sampled_diameter)

    missing_fan = witness_for(tuple(range(1, len(charts))))
    assert any(
        finding.check == "source_chart_coverage" and finding.status == "violated"
        for finding in _partition_findings(source, missing_fan)
    )


def test_curved_trim_reference_follows_target_chords_within_certified_lenses() -> None:
    arc = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    angle = 1.0
    end = np.asarray(arc.evaluate(jnp.asarray((0.0, angle))), dtype=np.float64)[1]
    uses = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, arc, 0.0, angle),
        PatchCurveUse(2, LineCurve(end, -end), 0.0, 1.0),
    )
    domain = MeshingDomain(
        (
            MeshingSurfacePatch(
                PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                (uses,),
            ),
        ),
        (MeshingDomainCurve(0, 1), MeshingDomainCurve(1, 2), MeshingDomainCurve(2, 0)),
        3,
        source_id="source-circular-sector",
        source_revision="source-revision-1",
        tolerance=1.0e-12,
    )
    source = MeshingDomainBoundarySource(domain, (0,))
    parameters = np.linspace(0.0, angle, 5, dtype=np.float64)
    chain_points = np.asarray(arc.evaluate(jnp.asarray(parameters)), dtype=np.float64)
    chain = tuple((Fraction(float(u)), Fraction(float(v))) for u, v in chain_points)
    origin = (Fraction(0), Fraction(0))
    fan = tuple((origin, chain[row], chain[row + 1]) for row in range(4))

    declared, lenses = source.trim_partition_reference(0, fan)

    def area(triangle: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
        (ax, ay), (bx, by), (cx, cy) = triangle
        return abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2

    # The reference keeps the target chords, so the exact measures agree; the
    # curved trim enters only through the certified lenses.
    assert sum(map(area, declared), Fraction(0)) == sum(map(area, fan), Fraction(0))
    assert [lens.chord for lens in lenses] == [
        (chain[row], chain[row + 1]) for row in range(4)
    ]
    for row, lens in enumerate(lenses):
        samples = np.asarray(
            arc.evaluate(jnp.linspace(parameters[row], parameters[row + 1], 65)),
            dtype=np.float64,
        )
        first, second = chain_points[row], chain_points[row + 1]
        chord = second - first
        along = np.clip((samples - first) @ chord / (chord @ chord), 0.0, 1.0)
        sagitta = float(
            np.max(np.linalg.norm(samples - (first + along[:, None] * chord), axis=1))
        )
        lower = np.asarray([float(value) for value in lens.lower], dtype=np.float64)
        upper = np.asarray([float(value) for value in lens.upper], dtype=np.float64)
        assert sagitta <= float(lens.parameter_bound) < 8.0 * sagitta
        assert np.all(samples >= lower) and np.all(samples <= upper)

    with pytest.raises(ValueError, match="boundary chain"):
        source.trim_partition_reference(0, fan[:1] + fan[2:])


def test_exact_ruled_surface_map_keeps_offset_and_bilinear_coefficients() -> None:
    from phydrax.geometry._source_interpolation import (
        _exact_surface_interpolation_bound,
        exact_surface_coordinate_expressions,
    )
    from phydrax.geometry.brep._patches import OffsetCurve, RuledSurface

    surface = RuledSurface(
        OffsetCurve(
            LineCurve((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            Fraction(1, 3),
            (0.0, 0.0, 1.0),
        ),
        LineCurve((0.0, 1.0, 0.0), (1.25, 0.0, 0.0)),
    )
    expressions = exact_surface_coordinate_expressions(surface)

    assert expressions is not None
    assert expressions[0][(1, 1)] == Fraction(1, 4)
    lower = coefficient_endpoint(expressions[0][(0, 0)], upper=False)
    upper = coefficient_endpoint(expressions[0][(0, 0)], upper=True)
    assert lower < upper

    triangle = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    deviation = _exact_surface_interpolation_bound(expressions, triangle)
    assert 0.0 < deviation < 1.0


def _triangular_surface_fidelity_inputs(
    surface: AbstractSurfacePatch,
    source_id: str,
) -> tuple[CellMesh, CellGeometrySpec, MeshingDomainBoundarySource, SurfaceChartWitness]:
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    corners = np.asarray(surface.evaluate(jnp.asarray(uv)), dtype=np.float64)
    uses = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, LineCurve((1.0, 0.0), (-1.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((0.0, 0.0), (0.0, 1.0)), 0.0, 1.0, sense=-1),
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(surface, (uses,)),),
        (MeshingDomainCurve(0, 1), MeshingDomainCurve(1, 2), MeshingDomainCurve(0, 2)),
        3,
        source_id=source_id,
        source_revision=f"{source_id}:revision",
        corner_points=corners,
    )
    mesh = _mesh(corners, np.asarray(((0, 1, 2),), dtype=np.int64))
    geometry = CellGeometrySpec.affine(mesh)
    witness = _witness(mesh, geometry, domain, uv[None, ...])
    return mesh, geometry, MeshingDomainBoundarySource(domain, (0,)), witness


def test_source_fidelity_uses_exact_affine_offset_rail_coefficients() -> None:
    from phydrax.geometry.brep._patches import OffsetCurve, RuledSurface

    surface = RuledSurface(
        OffsetCurve(
            LineCurve((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            Fraction(1, 3),
            (0.0, 0.0, 1.0),
        ),
        LineCurve((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
    )
    mesh, geometry, source, witness = _triangular_surface_fidelity_inputs(
        surface, "offset-ruled-affine-source"
    )

    exact = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=witness
    )
    assert exact.status != "certified"
    bounded = certify_source_fidelity(
        mesh, geometry, source, tolerance=1.0e-14, witness=witness
    )
    assert bounded.status == "certified"
    assert 0.0 < bounded.source_to_mesh_upper <= 1.0e-14


def test_source_fidelity_bounds_exact_bilinear_ruled_surface_interpolation() -> None:
    from phydrax.geometry._source_interpolation import (
        exact_surface_coordinate_expressions,
    )
    from phydrax.geometry.brep._patches import OffsetCurve, RuledSurface

    surface = RuledSurface(
        OffsetCurve(
            LineCurve((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            Fraction(1, 3),
            (0.0, 0.0, 1.0),
        ),
        LineCurve((0.0, 1.0, 0.0), (1.25, 0.0, 0.0)),
    )
    expressions = exact_surface_coordinate_expressions(surface)
    assert expressions is not None
    assert expressions[0][(1, 1)] == Fraction(1, 4)
    mesh, geometry, source, witness = _triangular_surface_fidelity_inputs(
        surface, "offset-ruled-bilinear-source"
    )

    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.5, witness=witness
    )
    assert certificate.status == "certified", (
        certificate.findings,
        certificate.source_to_mesh_upper,
    )
    assert 0.0 < certificate.source_to_mesh_upper < 0.5


def test_source_fidelity_uses_exact_nonbinary_placed_curve_coefficients() -> None:
    from phydrax.geometry._source_interpolation import (
        exact_surface_coordinate_expressions,
    )
    from phydrax.geometry.brep._assembly import BRepAffinePlacement
    from phydrax.geometry.brep._patches import RuledSurface
    from phydrax.geometry.brep._placed import PlacedCurve

    translation = Fraction(1, 3)
    placement = BRepAffinePlacement(
        (
            (Fraction(1), Fraction(0), Fraction(0), translation),
            (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        )
    )
    first = PlacedCurve(
        LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        np.eye(3, dtype=np.float64),
        np.asarray((float(translation), 0.0, 0.0), dtype=np.float64),
        affine=placement,
    )
    surface = RuledSurface(first, LineCurve((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)))
    expressions = exact_surface_coordinate_expressions(surface)
    assert expressions is not None
    assert expressions[0][(0, 0)] == translation
    mesh, geometry, source, witness = _triangular_surface_fidelity_inputs(
        surface, "placed-exact-ruled-source"
    )

    exact = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=witness
    )
    assert exact.status != "certified"
    bounded = certify_source_fidelity(
        mesh, geometry, source, tolerance=1.0e-14, witness=witness
    )
    assert bounded.status == "certified"


def test_source_fidelity_uses_exact_normal_offset_surface_coefficients() -> None:
    from phydrax.geometry._source_interpolation import (
        exact_surface_coordinate_expressions,
    )
    from phydrax.geometry.brep._patches import OffsetSurface

    surface = OffsetSurface(
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        Fraction(1, 3),
    )
    expressions = exact_surface_coordinate_expressions(surface)
    assert expressions is not None
    assert expressions[2][(0, 0)] == Fraction(1, 3)
    mesh, geometry, source, witness = _triangular_surface_fidelity_inputs(
        surface, "exact-offset-surface-source"
    )

    exact = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.0, witness=witness
    )
    assert exact.status != "certified"
    bounded = certify_source_fidelity(
        mesh, geometry, source, tolerance=1.0e-14, witness=witness
    )
    assert bounded.status == "certified", bounded.findings
