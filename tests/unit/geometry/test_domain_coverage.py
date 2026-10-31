from fractions import Fraction
from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import ExpressionInput
from phydrax.discretization._mapped_locator import PreparedMappedCellLocator
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._view_support import (
    mapped_mesh_support_query,
    MappedFieldSupport,
    prepare_mapped_field_support,
)
from phydrax.discretization._views import FieldQueryStatus
from phydrax.discretization.fem import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    prepare_finite_element_field_reconstruction,
)
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.geometry._domain_coverage import (
    certify_domain_coverage,
    DomainCoverageCertificate,
    PiecewiseLinearDomain,
)
from phydrax.geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._meshing_domain import MeshingDomain, MeshingDomainRegion
from phydrax.geometry.brep import brep_box
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._overset import (
    OversetCellStatus,
    OversetPartSpec,
    OversetPolicy,
    OversetVertexStatus,
    prepare_overset_connectivity,
)
from phydrax.units import MILLIMETER


def _split_square() -> tuple[
    PiecewiseLinearDomain, CellMesh, CellGeometrySpec, np.ndarray
]:
    vertices = np.asarray(
        ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    facets = np.asarray(
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)),
        dtype=np.int64,
    )
    facet_regions = np.asarray(
        ((0, -1), (1, -1), (1, -1), (1, -1), (0, -1), (0, -1), (0, 1)),
        dtype=np.int64,
    )
    domain = PiecewiseLinearDomain(
        vertices,
        facets,
        facet_regions,
        ("left", "right"),
        source_id="split-square",
    )
    cells = np.asarray(((0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)), dtype=np.int32)
    mesh = CellMesh.from_triangles(
        vertices,
        cells,
        vertex_global_ids=np.asarray((11, 13, 17, 19, 23, 29), dtype=np.int64),
        cell_global_ids=np.asarray((101, 202, 303, 404), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    return domain, mesh, geometry, np.asarray((0, 0, 1, 1), dtype=np.int64)


def _embedding_report(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> GlobalEmbeddingCertificate:
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    return certify_global_embedding(mesh, geometry, validity)


def _embedding(mesh: CellMesh, geometry: CellGeometrySpec) -> GlobalEmbeddingCertificate:
    certificate = _embedding_report(mesh, geometry)
    assert certificate.status == "certified"
    return certificate


def _cube_boundary_triangles(
    vertices: np.ndarray, /, *, inward: bool = False
) -> list[tuple[int, int, int]]:
    topology = reference_cell_topology("hexahedron")
    center = np.mean(vertices, axis=0)
    result: list[tuple[int, int, int]] = []
    for face in topology.entities[2]:
        face_ids = tuple(int(value) for value in face)
        face_center = np.mean(vertices[np.asarray(face_ids, dtype=np.int64)], axis=0)
        for local in ((0, 1, 2), (0, 2, 3)):
            triangle = [face_ids[index] for index in local]
            points = vertices[np.asarray(triangle, dtype=np.int64)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            if float(np.dot(normal, face_center - center)) < 0.0:
                triangle[1], triangle[2] = triangle[2], triangle[1]
            if inward:
                triangle[1], triangle[2] = triangle[2], triangle[1]
            result.append((triangle[0], triangle[1], triangle[2]))
    return result


def _cad_box_domain() -> tuple[MeshingDomain, CellMesh, CellGeometrySpec]:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract(MILLIMETER),
    )
    domain = MeshingDomain.from_brep(model)
    vertices = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "cube",
                "hexahedron",
                np.arange(8, dtype=np.int32)[None, :],
                global_ids=(811,),
            ),
        ),
    )
    return domain, mesh, CellGeometrySpec.affine(mesh)


def _wrongly_wound_cad_domain(domain: MeshingDomain, /) -> MeshingDomain:
    region = domain.regions[0]
    flipped_patch = region.boundary[0][0]
    flipped = MeshingDomainRegion(
        region.name,
        tuple(
            (patch, -orientation if patch == flipped_patch else orientation)
            for patch, orientation in region.boundary
        ),
    )
    return MeshingDomain(
        domain.patches,
        domain.curves,
        domain.corner_points.shape[0],
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        regions=(flipped,),
        tolerance=domain.tolerance,
        accuracy=domain.accuracy,
        source_indices=domain.source_indices,
        source_occurrences=domain.source_occurrences,
        region_source_indices=domain.region_source_indices,
        region_source_occurrences=domain.region_source_occurrences,
        source_kinds=domain.source_kinds,
        authority_id=domain.authority_id,
        brep_authority=domain.brep_authority,
        corner_points=domain.corner_points,
    )


def _mapped_cube(
    *,
    child_count: int = 2,
    chart_mode: str = "split",
) -> tuple[MappedReferenceDomain, CellMesh, CellGeometrySpec, np.ndarray]:
    topology = reference_cell_topology("hexahedron")
    root_points = np.asarray(topology.vertices, dtype=np.float64)
    root_mesh = CellMesh(
        root_points,
        (
            CellBlock(
                "root",
                "hexahedron",
                np.arange(8, dtype=np.int32)[None, :],
                global_ids=np.asarray((71,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(501, 509, dtype=np.int64),
    )
    source_geometry = CellGeometrySpec.affine(root_mesh)
    facets = []
    for face in topology.entities[2]:
        if len(face) == 3:
            facets.append(face)
        else:
            facets.extend(((face[0], face[1], face[2]), (face[0], face[2], face[3])))
    reference_domain = PiecewiseLinearDomain(
        root_points,
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (len(facets), 1)),
        ("solid",),
        source_id="mapped-cube-chart",
    )
    domain = MappedReferenceDomain(
        reference_domain,
        root_mesh,
        source_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="mapped-cube",
        source_revision="r1",
        coordinate_contract=SpatialCoordinateContract.si(),
    )

    reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
    scale = np.asarray((0.5, 1.0, 1.0), dtype=np.float64)
    if chart_mode == "split":
        origins = (0.0, 0.5)[:child_count]
    elif chart_mode == "overlap":
        origins = (0.0, 0.25)
    elif chart_mode == "outside":
        origins = (0.75,)
    else:
        raise ValueError(f"Unsupported test chart mode {chart_mode!r}.")
    child_count = len(origins)
    x_coordinates = sorted(
        {
            float(reference_vertex[0] * scale[0] + origin)
            for origin in origins
            for reference_vertex in reference_vertices
        }
    )
    node_positions = tuple(
        (x, y, z) for x, y, z in product(x_coordinates, (0.0, 1.0), (0.0, 1.0))
    )
    node_index = {point: index for index, point in enumerate(node_positions)}

    def target_node(reference_vertex: np.ndarray, origin: float, /) -> int:
        point = reference_vertex * scale + (origin, 0.0, 0.0)
        key = (float(point[0]), float(point[1]), float(point[2]))
        return node_index[key]

    cell_vertices = np.asarray(
        tuple(
            tuple(
                target_node(reference_vertex, origin)
                for reference_vertex in reference_vertices
            )
            for origin in origins
        ),
        dtype=np.int32,
    )
    target_points = np.asarray(node_positions, dtype=np.float64)
    target_ids = np.arange(901, 901 + child_count, dtype=np.int64)
    target_mesh = CellMesh(
        target_points,
        (CellBlock("children", "hexahedron", cell_vertices, global_ids=target_ids),),
        vertex_global_ids=np.arange(1001, 1001 + len(node_positions), dtype=np.int64),
    )
    root_element = source_geometry.elements[0]
    assert isinstance(root_element, FiniteElementSpec)
    zero_power: tuple[int, ...] = (0, 0, 0)
    first_power: tuple[int, ...] = (1, 0, 0)
    second_power: tuple[int, ...] = (0, 1, 0)
    third_power: tuple[int, ...] = (0, 0, 1)
    chart_arguments: tuple[tuple[ExpressionInput, ...], ...] = tuple(
        (
            {zero_power: Fraction(str(origin)), first_power: Fraction(1, 2)},
            {zero_power: Fraction(0), second_power: Fraction(1)},
            {zero_power: Fraction(0), third_power: Fraction(1)},
        )
        for origin in origins
    )
    target_element = PerCellRestrictedCellGeometryElement(
        root_element,
        "hexahedron",
        target_ids,
        np.full((child_count,), 71, dtype=np.int64),
        chart_arguments,
    )
    source_routes = np.asarray(source_geometry.geometry_dofs[0], dtype=np.int64)[0]
    target_routes = np.tile(source_routes, (child_count, 1))
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        root_mesh.topology_id,
        {"children": np.full((child_count,), 71, dtype=np.int64)},
        {
            "children": np.tile(
                np.asarray(root_mesh.vertex_global_ids, dtype=np.int64),
                (child_count, 1),
            )
        },
    )
    target_geometry = CellGeometrySpec(
        {"children": target_element},
        {"children": target_routes},
        np.asarray(source_geometry.coordinates),
        restriction_source=restriction,
    )
    return domain, target_mesh, target_geometry, np.zeros((child_count,), dtype=np.int64)


def test_source_partition_coverage_uses_oriented_interfaces_and_dense_regions() -> None:
    domain, mesh, geometry, regions = _split_square()
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        regions,
        embedding=_embedding(mesh, geometry),
    )

    assert certificate.status == "certified"
    assert certificate.cell_regions.tolist() == [0, 0, 1, 1]
    assert certificate.region_ids == ("left", "right")
    assert certificate.requested_region_measures == (
        Fraction(1, 2),
        Fraction(1, 2),
    )
    assert certificate.achieved_region_measures == certificate.requested_region_measures
    assert certificate.covered_source_facet_count == certificate.source_facet_count == 7
    assert certificate.binding.mesh_id == mesh.mesh_id


def test_coverage_identity_preserves_duplicate_region_names_by_occurrence() -> None:
    source, mesh, geometry, regions = _split_square()
    domain = PiecewiseLinearDomain(
        source.vertices,
        source.facets,
        source.facet_regions,
        ("material", "material"),
        source_id="split-square-instances",
        region_occurrence_paths=(("occurrence-a",), ("occurrence-b",)),
    )
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        regions,
        embedding=_embedding(mesh, geometry),
    )

    assert certificate.status == "certified", certificate.findings
    assert certificate.region_ids == ("material", "material")
    assert certificate.region_occurrence_paths == (("occurrence-a",), ("occurrence-b",))
    assert len(set(certificate.source_region_degree_ids)) == 2
    restored = DomainCoverageCertificate.from_stored_fields(certificate, domain)
    assert restored.certificate_id == certificate.certificate_id


def test_exact_polyhedral_source_coverage_uses_oriented_face_loops() -> None:
    vertices = np.asarray(
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
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    source_facets = tuple(
        triangle
        for face in faces
        for triangle in (
            (face[0], face[1], face[2]),
            (face[0], face[2], face[3]),
        )
    )
    domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(source_facets, dtype=np.int64),
        np.asarray(((0, -1),) * len(source_facets), dtype=np.int64),
        ("solid",),
        source_id="polyhedral-cube-source",
    )
    mesh = CellMesh.from_polyhedra(
        vertices,
        [[np.asarray(face, dtype=np.int32) for face in faces]],
    )
    geometry = CellGeometrySpec.affine(mesh)
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.asarray((0,), dtype=np.int64),
        embedding=_embedding(mesh, geometry),
    )

    assert certificate.status == "certified"
    assert certificate.requested_region_measures == (Fraction(1),)
    assert certificate.achieved_region_measures == (Fraction(1),)
    assert certificate.covered_source_facet_count == certificate.source_facet_count == 12
    assert len(certificate.source_region_degree_ids) == 1


def test_three_dimensional_source_boundary_has_exact_facet_coverage() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64),
        np.asarray(((0, -1),) * 4, dtype=np.int64),
        ("solid",),
        source_id="tetrahedral-domain",
    )
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "tetra", "tetrahedron", np.asarray(((0, 1, 2, 3),)), global_ids=(907,)
            ),
        ),
        vertex_global_ids=np.asarray((31, 37, 41, 43), dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.asarray((0,), dtype=np.int64),
        embedding=_embedding(mesh, geometry),
    )

    assert certificate.status == "certified"
    assert certificate.requested_region_measures == (Fraction(1, 6),)
    assert certificate.achieved_region_measures == (Fraction(1, 6),)
    assert certificate.covered_source_facet_count == 4


def test_exact_planar_cad_domain_uses_authored_source_faces() -> None:
    domain, mesh, geometry = _cad_box_domain()
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.asarray((0,), dtype=np.int64),
        embedding=_embedding(mesh, geometry),
    )

    assert certificate.status == "certified", certificate.findings
    assert certificate.domain_id == domain.domain_id
    assert certificate.region_occurrence_paths == domain.region_source_occurrences
    assert certificate.source_id == domain.source_id
    assert certificate.requested_region_measures == (Fraction(1),)
    assert certificate.achieved_region_measures == certificate.requested_region_measures
    assert certificate.coverage_method == "exact_oriented_facets"
    assert certificate.source_coverage_id is not None
    assert certificate.covered_source_facet_count == certificate.source_facet_count
    restored = DomainCoverageCertificate.from_stored_fields(certificate, domain)
    assert restored.certificate_id == certificate.certificate_id


def test_cad_domain_coverage_revalidates_retained_brep_authority() -> None:
    domain, mesh, geometry = _cad_box_domain()
    authority = domain.brep_authority
    assert authority is not None
    vertex_points = jnp.asarray(authority.geometry.vertex_points)
    stale_geometry = eqx.tree_at(
        lambda model: model.geometry.vertex_points,
        authority,
        vertex_points.at[0, 0].add(0.125),
    )
    stale_domain = eqx.tree_at(lambda value: value.brep_authority, domain, stale_geometry)

    with pytest.raises(ValueError, match="content has changed"):
        certify_domain_coverage(
            mesh,
            geometry,
            stale_domain,
            np.asarray((0,), dtype=np.int64),
            embedding=_embedding(mesh, geometry),
        )


def test_brep_domain_refuses_reoriented_source_authority_at_construction() -> None:
    domain, _, _ = _cad_box_domain()

    with pytest.raises(ValueError, match="complete oriented BRep solid boundary"):
        _wrongly_wound_cad_domain(domain)


def test_exact_source_coverage_rejects_an_omitted_cavity() -> None:
    topology = reference_cell_topology("hexahedron")
    reference = np.asarray(topology.vertices, dtype=np.float64)
    inner = 0.25 + 0.5 * reference
    vertices = np.concatenate((reference, inner), axis=0)
    outer_facets = _cube_boundary_triangles(reference)
    cavity_facets = [
        tuple(vertex + 8 for vertex in triangle)
        for triangle in _cube_boundary_triangles(inner, inward=True)
    ]
    source = PiecewiseLinearDomain(
        vertices,
        np.asarray((*outer_facets, *cavity_facets), dtype=np.int64),
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (24, 1)),
        ("material-with-cavity",),
        source_id="nested-cube-source",
    )
    mesh = CellMesh(
        reference,
        (
            CellBlock(
                "outer-carrier", "hexahedron", np.arange(8, dtype=np.int32)[None, :]
            ),
        ),
    )
    geometry = CellGeometrySpec.affine(mesh)
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        source,
        np.asarray((0,), dtype=np.int64),
        embedding=_embedding(mesh, geometry),
    )

    assert certificate.status == "violated"
    assert certificate.requested_region_measures == (Fraction(7, 8),)
    assert certificate.achieved_region_measures == (Fraction(1),)
    assert any(finding.check == "uncovered_boundary" for finding in certificate.findings)


def test_adaptive_implicit_volume_coverage_revalidates_stratified_roots() -> None:
    from phydrax.discretization import TensorGridPlan, UniformAxisSpec
    from phydrax.geometry import Sphere
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
    from phydrax.geometry._source_fidelity import certify_source_fidelity
    from phydrax.geometry.implicit import (
        AdaptiveImplicitSurfacePolicy,
        AnalyticImplicitProfile,
        discover_adaptive_implicit_surface,
    )
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
    from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
    from phydrax.meshing._volume_generation import NativeVolumeSchedule
    from phydrax.meshing.providers._native_implicit import prepare_native_implicit
    from phydrax.meshing.providers._native_options import NativeMeshingOptions
    from phydrax.meshing.providers._native_sources import NativeImplicitSource

    geometry_source = Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="coverage-implicit-sphere"
    ).compile()
    grid = TensorGridPlan(
        tuple(UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))
    source_id, source_revision = "coverage-implicit", "coverage-implicit-r1"
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08, root_tolerance=1.0e-8
    )
    lower_bounds = []
    upper_bounds = []
    for axis in grid.axes:
        bounds = axis.bounds
        if bounds is None:
            raise AssertionError(
                "Prepared implicit discovery axes must retain physical bounds."
            )
        lower_bounds.append(bounds[0])
        upper_bounds.append(bounds[1])
    discovery_domain = np.asarray([lower_bounds, upper_bounds], dtype=np.float64)
    discovery = discover_adaptive_implicit_surface(
        geometry_source,
        domain=discovery_domain,
        policy=policy,
        source_id=source_id,
    )
    # The sphere distance field is singular at its centre; the exact profile
    # proves that singular set lies away from the zero-set cover.
    profile = AnalyticImplicitProfile(
        geometry_source,
        SpatialCoordinateContract.si(),
        source_id=source_id,
        source_revision=source_revision,
    )
    source = NativeImplicitSource(
        geometry_source,
        source_id,
        source_revision,
        discovery=discovery,
        authoring_profile=profile,
    )
    boundary_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        # A size hint: the adaptive tetrahedral route reports, rather than
        # matches exactly, its edge-length statistics.
        size_controls=(
            UniformSizeControl(boundary_scope, 0.7, strength=SizeControlStrength.SOFT),
        ),
        maximum_deviation=0.8,
    )
    coordinate_contract = SpatialCoordinateContract.si()
    options = NativeMeshingOptions(
        "implicit_adaptive_tetrahedral",
        implicit_policy=policy,
        volume_schedule=NativeVolumeSchedule(),
    )
    construction = prepare_native_implicit(
        source, specification, options, coordinate_contract
    ).execute()
    request = construction.certification
    assert request.cell_regions is not None
    assert request.fidelity_source is not None
    assert request.fidelity_witness is not None
    assert request.fidelity_tolerance is not None
    embedding = _embedding(construction.mesh, construction.geometry)
    fidelity = certify_source_fidelity(
        construction.mesh,
        construction.geometry,
        request.fidelity_source,
        tolerance=request.fidelity_tolerance,
        witness=request.fidelity_witness,
    )
    certificate = certify_domain_coverage(
        construction.mesh,
        construction.geometry,
        discovery,
        request.cell_regions,
        embedding=embedding,
        source=request.fidelity_source,
        fidelity=fidelity,
    )

    assert certificate.status == "certified"
    assert certificate.coverage_method == "source_boundary_homotopy"
    assert certificate.region_ids == ("interior",)
    assert certificate.covered_source_facet_count == certificate.source_facet_count
    assert len(certificate.source_region_degree_ids) == 1
    assert certificate.source_region_degree_ids[0]
    assert certificate.source_distance_upper is not None
    assert certificate.source_distance_upper <= request.fidelity_tolerance

    witness = request.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    implicit_root_edge_keys = witness.implicit_root_edge_keys
    if implicit_root_edge_keys is None:
        raise AssertionError(
            "Adaptive implicit source witness must retain root-edge ancestry."
        )
    changed_roots = np.array(implicit_root_edge_keys, copy=True)
    changed_roots[0, 0, 0, 0] += 1
    invalid_witness = SourceBoundaryChartWitness(
        mesh_id=witness.mesh_id,
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        chart_dimension=witness.chart_dimension,
        reference_dimension=witness.reference_dimension,
        cell_global_ids=witness.cell_global_ids,
        entity_global_ids=witness.entity_global_ids,
        local_entity_indices=witness.local_entity_indices,
        target_reference_vertices=witness.target_reference_vertices,
        source_chart_ids=witness.source_chart_ids,
        source_entity_ids=witness.source_entity_ids,
        occurrence_paths=witness.occurrence_paths,
        source_parameters=witness.source_parameters,
        orientations=witness.orientations,
        implicit_cell_keys=witness.implicit_cell_keys,
        implicit_root_edge_keys=changed_roots,
    )
    with pytest.raises(ValueError):
        certify_source_fidelity(
            construction.mesh,
            construction.geometry,
            request.fidelity_source,
            tolerance=request.fidelity_tolerance,
            witness=invalid_witness,
        )
    original_witness = fidelity.witness
    object.__setattr__(fidelity, "witness", invalid_witness)
    try:
        invalid_coverage = certify_domain_coverage(
            construction.mesh,
            construction.geometry,
            discovery,
            request.cell_regions,
            embedding=embedding,
            source=request.fidelity_source,
            fidelity=fidelity,
        )
        assert invalid_coverage.status == "violated"
        assert any(
            finding.check == "source_fidelity_premise"
            for finding in invalid_coverage.findings
        )
    finally:
        object.__setattr__(fidelity, "witness", original_witness)


def test_source_partition_coverage_rejects_a_missing_material_interface() -> None:
    domain, mesh, geometry, regions = _split_square()
    embedding = _embedding(mesh, geometry)
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.zeros_like(regions),
        embedding=embedding,
    )

    assert certificate.status == "violated"
    assert any(
        finding.check in ("uncovered_boundary", "unmatched_boundary_facet")
        for finding in certificate.findings
    )
    assert any(finding.check == "region_measure" for finding in certificate.findings)
    reversed_certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        np.asarray((1, 1, 0, 0), dtype=np.int64),
        embedding=embedding,
    )
    assert reversed_certificate.status == "violated"
    assert any(
        finding.check == "interface_orientation"
        for finding in reversed_certificate.findings
    )


def test_coverage_restoration_recomputes_the_bound_witness() -> None:
    domain, mesh, geometry, regions = _split_square()
    embedding = _embedding(mesh, geometry)
    certificate = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding
    )

    restored = DomainCoverageCertificate.from_stored_fields(certificate, domain)
    assert restored.certificate_id == certificate.certificate_id
    with pytest.raises(ValueError, match="cell_regions"):
        certificate.require_bound(
            mesh,
            geometry,
            domain,
            np.zeros_like(regions),
            embedding=embedding,
        )


def test_stored_status_cannot_override_recomputed_coverage() -> None:
    domain, mesh, geometry, regions = _split_square()
    embedding = _embedding(mesh, geometry)
    certificate = certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding
    )
    original_status = certificate.status
    object.__setattr__(certificate, "status", "violated")
    try:
        with pytest.raises(ValueError, match="mathematical witness"):
            DomainCoverageCertificate.from_stored_fields(certificate, domain)
    finally:
        object.__setattr__(certificate, "status", original_status)


def test_mapped_child_charts_certify_containment_disjointness_and_completeness() -> None:
    domain, mesh, geometry, regions = _mapped_cube()
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        regions,
        embedding=_embedding(mesh, geometry),
    )

    assert certificate.status == "certified"
    assert certificate.requested_region_measures == (Fraction(1),)
    assert certificate.achieved_region_measures == (Fraction(1),)
    assert certificate.chart_certificate_id is not None


def test_mapped_child_chart_with_missing_half_is_not_complete() -> None:
    domain, mesh, geometry, regions = _mapped_cube(child_count=1)
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        regions,
        embedding=_embedding(mesh, geometry),
    )

    assert certificate.status != "certified"
    assert any(
        finding.check in ("mapped_chart_completeness", "region_measure")
        for finding in certificate.findings
    )


def test_mapped_child_chart_with_positive_volume_overlap_is_violated() -> None:
    domain, mesh, geometry, regions = _mapped_cube(chart_mode="overlap")
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        regions,
        embedding=_embedding_report(mesh, geometry),
    )

    assert certificate.status == "violated"
    assert any(
        finding.check == "mapped_chart_overlap" for finding in certificate.findings
    )


def test_mapped_child_chart_outside_root_is_violated() -> None:
    domain, mesh, geometry, regions = _mapped_cube(chart_mode="outside")
    certificate = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        regions,
        embedding=_embedding_report(mesh, geometry),
    )

    assert certificate.status == "violated"
    assert any(
        finding.check == "mapped_chart_containment" for finding in certificate.findings
    )


def test_mapped_child_tabulation_preserves_compiled_values_and_derivatives() -> None:
    _, _, geometry, _ = _mapped_cube()
    element = geometry.elements[0]
    assert isinstance(element, PerCellRestrictedCellGeometryElement)
    coefficients = geometry.coordinates[geometry.geometry_dofs[0][0]]
    point = jnp.asarray((0.2, 0.3, 0.4), dtype=jnp.float64)

    def evaluate(
        element: PerCellRestrictedCellGeometryElement,
        coefficients: Array,
        point: Array,
    ) -> tuple[Array, Array]:
        values, gradients = element.tabulate_cell(0, point[None, :])
        return values[0] @ coefficients, coefficients.T @ gradients[0]

    position, derivative = eqx.filter_jit(evaluate)(element, coefficients, point)
    automatic, _ = eqx.filter_jit(jax.jacfwd(evaluate, argnums=2))(
        element, coefficients, point
    )
    np.testing.assert_allclose(position, (0.1, 0.3, 0.4), atol=1.0e-14)
    expected = np.diag(np.asarray((0.5, 1.0, 1.0), dtype=np.float64))
    np.testing.assert_allclose(derivative, expected, atol=1.0e-14)
    np.testing.assert_allclose(automatic, expected, atol=1.0e-14)


def test_mapped_coverage_binds_prepared_field_queries_to_exact_map() -> None:
    domain, mesh, geometry, regions = _mapped_cube()
    embedding = _embedding(mesh, geometry)
    coverage = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        regions,
        embedding=embedding,
    )
    assert coverage.status == "certified"
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("u", lagrange_element("hexahedron", 1)),
        coordinate_spec=geometry,
    ).prepare()
    reconstruction = prepare_finite_element_field_reconstruction(
        discretization, "u", support_geometry=coverage
    )
    coefficients = jnp.asarray(discretization.dof_maps[0].dof_coordinates)[:, 0]
    query = reconstruction.prepare_query(
        np.asarray(((0.25, 0.5, 0.5),)), derivative=(1, 0, 0)
    )

    support = query.reconstruction.support_geometry
    assert isinstance(support, MappedFieldSupport)
    assert support.coverage_id == coverage.certificate_id
    assert query.evidence.status.tolist() == [int(FieldQueryStatus.VALID)]
    np.testing.assert_allclose(query.apply(coefficients), (1.0,), atol=1e-10)

    membership = mapped_mesh_support_query(
        support,
        np.asarray(((0.25, 0.5, 0.5), (0.0, 0.5, 0.5), (1.5, 0.5, 0.5))),
    )
    assert membership.successful.tolist() == [True, True, True]
    assert membership.inside.tolist() == [True, True, False]
    assert membership.boundary.tolist() == [False, True, False]
    assert membership.outside.tolist() == [False, False, True]


def test_mapped_support_authority_drives_native_overset_wall_membership() -> None:
    domain, mesh, geometry, regions = _mapped_cube()
    source_result = certify_cell_mesh(
        mesh,
        SpatialCoordinateContract.si(),
        geometry=geometry,
    )
    mesh, geometry = source_result.mesh, source_result.geometry
    coverage = certify_domain_coverage(
        mesh,
        geometry,
        domain,
        regions,
        embedding=_embedding(mesh, geometry),
    )
    locator = PreparedMappedCellLocator.from_cell_geometry(mesh, geometry, 0)
    support = prepare_mapped_field_support(mesh, geometry, locator, coverage)
    source_part = MeshPart("solid", source_result)
    source_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    source_boundary = np.asarray(
        mesh.boundary_masks[mesh.topological_dimension - 1], dtype=np.bool_
    )
    wall = source_part.scope(2, source_ids[source_boundary])
    target_points = np.asarray(
        ((0.2, 0.4, 0.4), (0.3, 0.4, 0.4), (0.2, 0.5, 0.4), (0.2, 0.4, 0.5)),
        dtype=np.float64,
    )
    target_mesh = CellMesh(
        target_points,
        (CellBlock("probe", "tetrahedron", np.asarray(((0, 1, 2, 3),), dtype=np.int32)),),
    )
    target_result = certify_cell_mesh(
        target_mesh,
        SpatialCoordinateContract.si(),
        geometry=CellGeometrySpec.affine(target_mesh),
    )
    target_part = MeshPart("probe", target_result)
    target_ids = np.asarray(target_mesh.entity_set(2).entity_ids, dtype=np.int64)
    target_boundary = np.asarray(
        target_mesh.boundary_masks[target_mesh.topological_dimension - 1],
        dtype=np.bool_,
    )
    boundary = target_part.scope(2, target_ids[target_boundary])
    connectivity = prepare_overset_connectivity(
        MeshAssembly((source_part, target_part)),
        (
            OversetPartSpec("solid", wall=wall, solid_query=support),
            OversetPartSpec("probe", boundary=boundary),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )

    assert connectivity.complete
    source_mask = connectivity.blanking_of("probe")
    assert np.all(np.asarray(source_mask.vertex_status) == int(OversetVertexStatus.HOLE))
    assert np.all(np.asarray(source_mask.cell_status) == int(OversetCellStatus.HOLE))
