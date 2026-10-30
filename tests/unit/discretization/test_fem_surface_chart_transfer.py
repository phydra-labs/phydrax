from __future__ import annotations

from typing import Literal

from phydrax import config as pxconfig


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    prepare_surface_chart_deformation,
)
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._surface_chart_records import (
    PreparedSurfaceChartDeformation,
    SurfaceChartWitness,
)
from phydrax.discretization.fem import (
    FiniteElementDeRhamComplex,
    prepare_sphere_chart_compatible_transfer,
    prepare_surface_chart_compatible_transfer,
    prepare_surface_chart_field_transfer,
    prepare_surface_chart_finite_volume_contents,
)
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from phydrax.geometry import Sphere
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._sphere_material_atlas import (
    prepare_sphere_chart_deformation,
    SphereMaterialCellAtlas,
)
from phydrax.geometry.brep._patches import LineCurve, TorusPatch
from phydrax.geometry.implicit import AnalyticImplicitProfile


_TORUS_UV = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64)
_SOURCE_TRIANGLES = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
_TARGET_TRIANGLES = np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32)

_SPHERE_CHART_COORDINATES = np.asarray(
    ((0.2, 0.3), (1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.5, 2.0), (-0.3, -0.4)),
    dtype=np.float64,
)


def _torus_points(surface: TorusPatch, /) -> np.ndarray:
    return np.stack(
        tuple(
            np.asarray(surface.evaluate(jnp.asarray(point)), dtype=np.float64)
            for point in _TORUS_UV
        )
    )


def _torus_domain() -> tuple[MeshingDomain, TorusPatch]:
    surface = TorusPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        2.0,
        0.5,
    )
    pcurves = (
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((1.0, 1.0), (-1.0, 0.0)),
        LineCurve((0.0, 1.0), (0.0, -1.0)),
    )
    uses = tuple(
        PatchCurveUse(index, curve, 0.0, 1.0) for index, curve in enumerate(pcurves)
    )
    corners = _torus_points(surface)
    domain = MeshingDomain(
        (MeshingSurfacePatch(surface, (uses,)),),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(2, 3),
            MeshingDomainCurve(3, 0),
        ),
        4,
        source_id="surface-chart-transfer-torus",
        source_revision="torus-chart-r1",
        corner_points=corners,
    )

    return domain, surface


_TORUS_DOMAIN, _TORUS_SURFACE = _torus_domain()


def _torus_mesh(
    surface: TorusPatch,
    triangles: np.ndarray,
    cell_ids: tuple[int, int],
) -> CellMesh:
    coordinates = _torus_points(surface)
    return CellMesh.from_triangles(
        coordinates,
        triangles,
        vertex_global_ids=np.asarray((501, 503, 509, 521), dtype=np.int64),
        cell_global_ids=np.asarray(cell_ids, dtype=np.int64),
    )


def _chart_complex_mesh(
    physical_mesh: CellMesh,
    chart_coordinates: np.ndarray,
    /,
    *,
    orient_positive: bool = False,
) -> CellMesh:
    triangles = np.asarray(physical_mesh.blocks[0].vertices, dtype=np.int32).copy()
    if orient_positive:
        first = chart_coordinates[triangles[:, 1]] - chart_coordinates[triangles[:, 0]]
        second = chart_coordinates[triangles[:, 2]] - chart_coordinates[triangles[:, 0]]
        reverse = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0] < 0.0
        previous = triangles[reverse, 1].copy()
        triangles[reverse, 1] = triangles[reverse, 2]
        triangles[reverse, 2] = previous
    return CellMesh.from_triangles(
        chart_coordinates,
        triangles,
        vertex_global_ids=np.asarray(physical_mesh.vertex_global_ids, dtype=np.int64),
        cell_global_ids=np.asarray(physical_mesh.blocks[0].global_ids, dtype=np.int64),
    )


def _surface_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MeshingDomain,
    triangles: np.ndarray,
    root_ids: tuple[int, int],
) -> SurfaceChartWitness:
    block = mesh.blocks[0]
    identifiers = np.asarray(block.global_ids, dtype=np.int64)
    charts = _TORUS_UV[triangles]
    return SurfaceChartWitness(
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=identifiers,
        patches=np.zeros((2,), dtype=np.int32),
        charts=charts,
        source_root_cell_ids=np.asarray(root_ids, dtype=np.int64),
        source_piece_ids=((0, int(root_ids[0])), (0, int(root_ids[1]))),
        geometry_entity_ids=(domain.entity_id(2, 0),) * 2,
        occurrence_paths=(domain.source_occurrences[2][0],) * 2,
        root_mesh_cell_bindings=(None, None),
    )


def _torus_transfer_case() -> tuple[
    MeshingDomain,
    CellGeometrySpec,
    CellGeometrySpec,
    CellMesh,
    CellMesh,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    PreparedSurfaceChartDeformation,
]:
    domain, surface = _TORUS_DOMAIN, _TORUS_SURFACE
    source_mesh = _torus_mesh(surface, _SOURCE_TRIANGLES, (101, 103))
    target_mesh = _torus_mesh(surface, _TARGET_TRIANGLES, (107, 109))
    source_geometry = CellGeometrySpec.affine(source_mesh)
    target_geometry = CellGeometrySpec.affine(target_mesh)
    source_witness = _surface_witness(
        source_mesh, source_geometry, domain, _SOURCE_TRIANGLES, (101, 103)
    )
    target_witness = _surface_witness(
        target_mesh, target_geometry, domain, _TARGET_TRIANGLES, (101, 103)
    )
    cover = MeshingDomainBoundarySource(domain, (0,))
    deformation = prepare_surface_chart_deformation(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        domain,
        source_witness=source_witness,
        target_witness=target_witness,
        source_chart_cover=cover,
        target_chart_cover=cover,
        maximum_fidelity=3.0,
        maximum_displacement=3.0,
    )
    return (
        domain,
        source_geometry,
        target_geometry,
        source_mesh,
        target_mesh,
        tuple(_TORUS_UV[_SOURCE_TRIANGLES]),
        tuple(_TORUS_UV[_TARGET_TRIANGLES]),
        deformation,
    )


def _material_values(degree: int, points: np.ndarray, /) -> np.ndarray:
    u, v = points[:, 0], points[:, 1]
    if degree == 0:
        return (1.0 + 0.2 * u - 0.3 * v + 0.15 * u**2 + 0.07 * u * v)[:, None]
    if degree == 1:
        return np.column_stack((0.4 + u + 0.2 * v + 0.1 * u * v, -0.3 + v + 0.15 * u**2))
    return (0.7 + 0.4 * u - 0.2 * v)[:, None]


def _form_coefficients(
    complex_: FiniteElementDeRhamComplex,
    charts: tuple[np.ndarray, ...],
    degree: int,
) -> np.ndarray:
    element = complex_.discretization.elements[degree][0]
    basis = element.form_basis
    if basis is None:
        raise TypeError("Chart reproduction oracle requires a canonical form basis.")
    dofs = complex_.discretization.dof_maps[degree]
    routes = np.asarray(dofs.cell_dofs[0], dtype=np.int64)
    transforms = np.asarray(dofs.cell_transforms[0], dtype=np.float64)
    result = np.full((dofs.global_dof_count,), np.nan, dtype=np.float64)
    reference = np.asarray(basis.functional_points, dtype=np.float64)
    for cell, chart in enumerate(charts):
        frame = (chart[1:] - chart[:1]).T
        material_points = chart[0] + reference @ frame.T
        values = _material_values(degree, material_points)
        if degree == 1:
            values = values @ frame
        elif degree == 2:
            values = values * np.linalg.det(frame)
        local = np.asarray(
            basis.interpolate(jnp.asarray(values, dtype=jnp.float64)),
            dtype=np.float64,
        )
        canonical = np.linalg.solve(transforms[cell], local)
        cell_routes = routes[cell]
        shared = np.isfinite(result[cell_routes])
        np.testing.assert_allclose(
            result[cell_routes][shared], canonical[shared], atol=5e-11, rtol=5e-11
        )
        result[cell_routes] = canonical
    if not np.all(np.isfinite(result)):
        raise ValueError("Chart reproduction oracle did not assign every global DOF.")
    return result


def _physical_top_form_coefficients(
    complex_: FiniteElementDeRhamComplex,
    charts: tuple[np.ndarray, ...],
    cell_measures: np.ndarray,
) -> np.ndarray:
    element = complex_.discretization.elements[2][0]
    basis = element.form_basis
    if basis is None:
        raise TypeError("Surface DG oracle requires a canonical top-form basis.")
    dofs = complex_.discretization.dof_maps[2]
    routes = np.asarray(dofs.cell_dofs[0], dtype=np.int64)
    transforms = np.asarray(dofs.cell_transforms[0], dtype=np.float64)
    reference = np.asarray(basis.functional_points, dtype=np.float64)
    result = np.full((dofs.global_dof_count,), np.nan, dtype=np.float64)
    for cell, triangle in enumerate(charts):
        frame = (triangle[1:] - triangle[:1]).T
        material_points = triangle[0] + reference @ frame.T
        area_density = 2.0 * float(cell_measures[cell])
        values = _material_values(2, material_points) * area_density
        local = np.asarray(
            basis.interpolate(jnp.asarray(values, dtype=jnp.float64)),
            dtype=np.float64,
        )
        canonical = np.linalg.solve(transforms[cell], local)
        cell_routes = routes[cell]
        shared = np.isfinite(result[cell_routes])
        np.testing.assert_allclose(
            result[cell_routes][shared], canonical[shared], atol=5e-11, rtol=5e-11
        )
        result[cell_routes] = canonical
    if not np.all(np.isfinite(result)):
        raise ValueError("Surface DG oracle did not assign every global DOF.")
    return result


def _reference_top_form_integral(
    complex_: FiniteElementDeRhamComplex, values: np.ndarray, /
) -> float:
    element = complex_.discretization.elements[2][0]
    basis = element.form_basis
    if basis is None:
        raise TypeError("Surface integral oracle requires a canonical top-form basis.")
    dofs = complex_.discretization.dof_maps[2]
    routes = np.asarray(dofs.cell_dofs[0], dtype=np.int64)
    transforms = np.asarray(dofs.cell_transforms[0], dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    first, second = np.meshgrid(nodes, nodes, indexing="ij")
    points = np.column_stack((first.reshape(-1), ((1.0 - first) * second).reshape(-1)))
    quadrature = (weights[:, None] * weights[None, :] * (1.0 - first)).reshape(-1)
    basis_values = np.asarray(basis.tabulate_components(points)[0], dtype=np.float64)
    total = 0.0
    for cell, cell_routes in enumerate(routes):
        local = transforms[cell] @ values[cell_routes]
        field = np.einsum("qd,d->q", basis_values[:, :, 0], local)
        total += float(np.dot(quadrature, field))
    return total


def _fv(mesh: CellMesh) -> UnstructuredFiniteVolumeDiscretization:
    block = mesh.blocks[0]
    return UnstructuredFiniteVolumeDiscretization(
        UnstructuredFiniteVolumePlan(
            mesh.coordinates,
            triangles=block.vertices,
            vertex_global_ids=mesh.vertex_global_ids,
            cell_global_ids=block.global_ids,
        )
    )


@pytest.mark.parametrize("twist", ("untwisted", "twisted"), ids=("hcurl", "hdiv"))
def test_torus_chart_forms_reproduce_polynomials_and_commute(
    twist: Literal["untwisted", "twisted"],
) -> None:
    (
        _,
        _,
        _,
        source_mesh,
        target_mesh,
        source_charts,
        target_charts,
        deformation,
    ) = _torus_transfer_case()
    source_chart_mesh = _chart_complex_mesh(source_mesh, _TORUS_UV)
    target_chart_mesh = _chart_complex_mesh(target_mesh, _TORUS_UV)
    source = FiniteElementDeRhamComplex(source_chart_mesh, family="trimmed", order=2)
    target = FiniteElementDeRhamComplex(target_chart_mesh, family="trimmed", order=2)
    prepared = prepare_surface_chart_compatible_transfer(
        source, target, deformation, form_twist=twist
    )
    assert prepared.deformation_id == deformation.deformation_id
    assert np.all(np.asarray(prepared.commuting_defects) < 2.0e-8)
    for degree in range(3):
        old = _form_coefficients(source, source_charts, degree)
        expected = _form_coefficients(target, target_charts, degree)
        actual = prepared.complex_map.map(degree).mv(jnp.asarray(old))
        np.testing.assert_allclose(actual, expected, atol=2.0e-8, rtol=2.0e-8)
    generator = np.random.default_rng(137)
    for degree in range(2):
        values = generator.normal(size=source._absolute.space(degree).size)
        left = target._absolute.differentials[degree].mv(
            prepared.complex_map.map(degree).mv(jnp.asarray(values))
        )
        right = prepared.complex_map.map(degree + 1).mv(
            source._absolute.differentials[degree].mv(jnp.asarray(values))
        )
        np.testing.assert_allclose(left, right, atol=2.0e-8, rtol=2.0e-8)
    load = jnp.linspace(-0.4, 1.1, prepared.complex_map.target.space(1).size)
    raw = prepared.complex_map.map(1).transpose_mv(load)
    hilbert = prepared.complex_map.adjoint().map(1).mv(load)
    assert np.linalg.norm(np.asarray(raw - hilbert)) > 1.0e-7


def test_surface_dg_and_fv_chart_transfers_conserve_physical_content() -> None:
    (
        _,
        _,
        _,
        source_mesh,
        target_mesh,
        _,
        _,
        deformation,
    ) = _torus_transfer_case()
    source_chart_mesh = _chart_complex_mesh(source_mesh, _TORUS_UV)
    target_chart_mesh = _chart_complex_mesh(target_mesh, _TORUS_UV)
    source = FiniteElementDeRhamComplex(source_chart_mesh, family="trimmed", order=2)
    target = FiniteElementDeRhamComplex(target_chart_mesh, family="trimmed", order=2)
    dg = prepare_surface_chart_field_transfer(source, target, deformation, semantics="dg")
    assert dg.degree == 2
    old = _physical_top_form_coefficients(
        source,
        tuple(_TORUS_UV[_SOURCE_TRIANGLES]),
        np.asarray(deformation.source_cell_measures, dtype=np.float64),
    )
    transferred = dg.complex_map.map(2).mv(jnp.asarray(old))
    assert np.all(np.isfinite(np.asarray(transferred)))
    np.testing.assert_allclose(
        _reference_top_form_integral(source, old),
        _reference_top_form_integral(target, np.asarray(transferred)),
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    assert np.all(np.asarray(dg.commuting_defects) < 2.0e-8)

    source_fv, target_fv = _fv(source_mesh), _fv(target_mesh)
    for finite_volume in (source_fv, target_fv):
        vertices = np.asarray(finite_volume.vertices, dtype=np.float64)
        triangles = np.asarray(finite_volume.triangles, dtype=np.int64)
        normals = np.cross(
            vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
            vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
        )
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        connectivity = finite_volume.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError("Surface FV tests require polygonal edge topology.")
        edges = np.asarray(connectivity.edges, dtype=np.int64)
        edge_tangents = vertices[edges[:, 1]] - vertices[edges[:, 0]]
        owners = np.asarray(finite_volume.owner_cells, dtype=np.int64)
        conormals = np.asarray(finite_volume.area_vectors, dtype=np.float64)
        np.testing.assert_allclose(
            np.einsum("fd,fd->f", conormals, normals[owners]),
            0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            np.einsum("fd,fd->f", conormals, edge_tangents),
            0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            np.linalg.norm(conormals, axis=1),
            np.linalg.norm(edge_tangents, axis=1),
            atol=2.0e-12,
        )
        widths = np.asarray(finite_volume.directional_control_volume_widths())
        assert widths.shape == (finite_volume.cell_count, 2)
        assert np.all(np.isfinite(widths) & (widths > 0.0))
        assert float(finite_volume.quality.maximum_closure_residual) < 2.0e-12
    transfer = prepare_surface_chart_finite_volume_contents(
        source_fv, target_fv, deformation
    )
    old_density = jnp.asarray((1.25, -0.35), dtype=jnp.float64)
    target_density = transfer.primal.mv(old_density)
    source_content = jnp.sum(old_density * transfer.source_measures)
    target_content = jnp.sum(target_density * transfer.target_measures)
    np.testing.assert_allclose(target_content, source_content, atol=2.0e-11, rtol=2.0e-11)
    assert float(transfer.conservation_defect) < 2.0e-11
    load = jnp.asarray((0.7, -1.1), dtype=jnp.float64)
    raw = transfer.primal.transpose_mv(load)
    hilbert = transfer.hilbert_adjoint.mv(load)
    assert np.linalg.norm(np.asarray(raw - hilbert)) > 1.0e-5


def _octahedron(rotation: np.ndarray, /) -> CellMesh:
    vertices = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    triangles = np.asarray(
        (
            (4, 0, 2),
            (4, 2, 1),
            (4, 1, 3),
            (4, 3, 0),
            (5, 2, 0),
            (5, 1, 2),
            (5, 3, 1),
            (5, 0, 3),
        ),
        dtype=np.int32,
    )
    return CellMesh.from_triangles(
        vertices @ rotation.T,
        triangles,
        vertex_global_ids=np.asarray((601, 607, 613, 617, 619, 631), dtype=np.int64),
        cell_global_ids=np.asarray(tuple(range(701, 709)), dtype=np.int64),
    )


def _sphere_chart_mesh(mesh: CellMesh, /) -> CellMesh:
    return _chart_complex_mesh(mesh, _SPHERE_CHART_COORDINATES)


def _sphere_atlas(
    profile: AnalyticImplicitProfile, mesh: CellMesh, /
) -> tuple[SphereMaterialCellAtlas, CellGeometrySpec]:
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity)
    return SphereMaterialCellAtlas(profile, mesh, geometry, validity, embedding), geometry


def _sphere_quadratic_h1(complex_: FiniteElementDeRhamComplex, /) -> np.ndarray:
    element = complex_.discretization.elements[0][0]
    basis = element.form_basis
    if basis is None:
        raise TypeError("Sphere reproduction requires a canonical scalar basis.")
    dofs = complex_.discretization.dof_maps[0]
    routes = np.asarray(dofs.cell_dofs[0], dtype=np.int64)
    transforms = np.asarray(dofs.cell_transforms[0], dtype=np.float64)
    points = np.asarray(basis.functional_points, dtype=np.float64)
    result = np.full((dofs.global_dof_count,), np.nan, dtype=np.float64)
    triangles = np.asarray(complex_.mesh.blocks[0].vertices, dtype=np.int64)
    coordinates = np.asarray(complex_.mesh.coordinates, dtype=np.float64)
    for cell, vertices in enumerate(triangles):
        barycentric = np.column_stack((1.0 - points.sum(axis=1), points))
        physical_x = barycentric @ coordinates[vertices, 0]
        local = np.asarray(
            basis.interpolate(jnp.asarray((physical_x**2)[:, None], dtype=jnp.float64)),
            dtype=np.float64,
        )
        canonical = np.linalg.solve(transforms[cell], local)
        cell_routes = routes[cell]
        shared = np.isfinite(result[cell_routes])
        np.testing.assert_allclose(
            result[cell_routes][shared], canonical[shared], atol=2.0e-11, rtol=2.0e-11
        )
        result[cell_routes] = canonical
    if not np.all(np.isfinite(result)):
        raise ValueError("Sphere polynomial oracle did not assign every global DOF.")
    return result


def test_sphere_projective_chart_reproduces_polynomials_and_commutes() -> None:
    profile = AnalyticImplicitProfile(
        Sphere((0.0, 0.0, 0.0), 1.0).compile(),
        SpatialCoordinateContract.si(),
        source_id="surface-chart-unit-sphere",
        source_revision="sphere-r1",
    )
    identity = np.eye(3, dtype=np.float64)
    source_atlas, _ = _sphere_atlas(profile, _octahedron(identity))
    source_chart_mesh = _sphere_chart_mesh(source_atlas.source_mesh)
    source = FiniteElementDeRhamComplex(source_chart_mesh, family="trimmed", order=2)
    same_deformation = prepare_sphere_chart_deformation(source_atlas, source_atlas)
    same_transfer = prepare_sphere_chart_compatible_transfer(
        source, source, same_deformation
    )
    source_values = _sphere_quadratic_h1(source)
    reproduced = same_transfer.complex_map.map(0).mv(jnp.asarray(source_values))
    np.testing.assert_allclose(reproduced, source_values, atol=2.0e-10, rtol=2.0e-10)

    angle = 0.31
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    target_atlas, _ = _sphere_atlas(profile, _octahedron(rotation))
    target_chart_mesh = _sphere_chart_mesh(target_atlas.source_mesh)
    target = FiniteElementDeRhamComplex(target_chart_mesh, family="trimmed", order=2)
    deformation = prepare_sphere_chart_deformation(source_atlas, target_atlas)
    projective = prepare_sphere_chart_compatible_transfer(source, target, deformation)
    constant = jnp.ones((source._absolute.space(0).size,), dtype=jnp.float64)
    np.testing.assert_allclose(
        projective.complex_map.map(0).mv(constant),
        jnp.ones((target._absolute.space(0).size,), dtype=jnp.float64),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    assert np.all(np.asarray(projective.commuting_defects) < 2.0e-8)
    generator = np.random.default_rng(91)
    for degree in range(2):
        values = generator.normal(size=source._absolute.space(degree).size)
        left = target._absolute.differentials[degree].mv(
            projective.complex_map.map(degree).mv(jnp.asarray(values))
        )
        right = projective.complex_map.map(degree + 1).mv(
            source._absolute.differentials[degree].mv(jnp.asarray(values))
        )
        np.testing.assert_allclose(left, right, atol=2.0e-8, rtol=2.0e-8)
