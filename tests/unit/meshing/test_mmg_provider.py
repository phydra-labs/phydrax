import itertools
import os
import shutil
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing import (
    MmgLagrangianMode,
    MmgLagrangianMotion,
    MmgLevelSet,
    MmgOptions,
    MmgProvider,
)


pytestmark = pytest.mark.meshing_mmg

CONTRACT = phx.SpatialCoordinateContract(phx.units.METER)
Failure = phx.meshing.MeshingFailure
Category = phx.meshing.MeshingFailureCategory

requires_worker = pytest.mark.skipif(
    shutil.which(os.environ.get("PHYDRAX_MMG_WORKER", "phydrax-mmg-worker")) is None,
    reason="phydrax-mmg-worker is not built (set PHYDRAX_MMG_WORKER)",
)


@pytest.fixture(scope="module")
def provider() -> Any:
    with MmgProvider() as value:
        yield value


def _grid(dimension: Any, count: Any, *, drop: Any = None) -> Any:
    """Structured unit square (two triangles per square) or cube (Kuhn tetrahedra)."""
    axes = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(*([axes] * dimension), indexing="ij"), -1).reshape(
        -1, dimension
    )
    strides = (count + 1) ** np.arange(dimension - 1, -1, -1)
    origins = np.stack(
        np.meshgrid(*([np.arange(count)] * dimension), indexing="ij"), -1
    ).reshape(-1, dimension)
    if drop is not None:
        origins = origins[~drop(origins / count)]
    cells = []
    for order in itertools.permutations(range(dimension)):
        path = np.zeros((dimension + 1, dimension), dtype=np.int64)
        for step, axis in enumerate(order):
            path[step + 1 :, axis] += 1
        cells.append((origins[:, None, :] + path[None]) @ strides)
    cells = np.concatenate(cells)
    used, cells = np.unique(cells, return_inverse=True)
    points, cells = points[used], cells.reshape(-1, dimension + 1)
    corners = points[cells]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0
    cells[negative, :2] = cells[negative, 1::-1]
    return points, cells


def _mesh(points: Any, cells: Any, **identities: Any) -> Any:
    constructor = (
        phx.discretization.CellMesh.from_tetrahedra
        if cells.shape[1] == 4
        else phx.discretization.CellMesh.from_triangles
    )
    return constructor(points, cells, **identities)


def _scope(mesh: Any, dimension: Any, mask: Any) -> Any:
    entities = mesh.entity_set(dimension)
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        np.asarray(entities.entity_ids)[mask],
    )


def _facets(mesh: Any) -> Any:
    connectivity = mesh.connectivity
    return np.asarray(
        connectivity.faces if mesh.topological_dimension == 3 else connectivity.edges
    )


def _cell_corners(mesh: Any, cell_ids: Any = None) -> Any:
    points = np.asarray(mesh.coordinates)
    return np.concatenate(
        [
            points[
                np.asarray(block.vertices)[
                    slice(None)
                    if cell_ids is None
                    else np.isin(np.asarray(block.global_ids), cell_ids)
                ]
            ]
            for block in mesh.blocks
        ]
    )


def _cell_measures(mesh: Any, cell_ids: Any = None) -> Any:
    corners = _cell_corners(mesh, cell_ids)
    edges = corners[:, 1:] - corners[:, :1]
    if corners.shape[1] == 4:
        return np.linalg.det(edges) / 6
    if corners.shape[2] == 2:
        return np.linalg.det(edges) / 2
    return np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1) / 2


def _vertex_attribute(provider: Any, mesh: Any, name: Any, values: Any) -> Any:
    return phx.meshing.MeshAttribute(
        name, phx.meshing.MeshAttributeRole.USER, provider.vertex_scope(mesh), values
    )


def _uniform_metric(provider: Any, mesh: Any, size: Any) -> Any:
    dimension = mesh.ambient_dimension
    return phx.meshing.MeshMetricField(
        provider.vertex_scope(mesh),
        np.tile(np.eye(dimension) / size**2, (len(mesh.coordinates), 1, 1)),
        minimum_size=size,
        maximum_size=size,
    )


def _two_regions(dimension: Any, count: Any) -> Any:
    """Unit square/cube split at x = 0.5 with inlet (x = 0) and outlet (x = 1) patches."""
    points, cells = _grid(dimension, count)
    mesh = _mesh(points, cells)
    left = points[cells].mean(axis=1)[:, 0] < 0.5
    facet_x = points[_facets(mesh)][:, :, 0]
    zones = tuple(
        phx.meshing.MeshZone(
            name, phx.meshing.MeshZoneRole.REGION, _scope(mesh, dimension, mask)
        )
        for name, mask in (("left", left), ("right", ~left))
    )
    patches = tuple(
        phx.meshing.MeshPatch(
            name, _scope(mesh, dimension - 1, np.all(facet_x == x, axis=1))
        )
        for name, x in (("inlet", 0.0), ("outlet", 1.0))
    )
    return phx.meshing.certify_cell_mesh(mesh, CONTRACT, zones=zones, patches=patches)


def _by_name(items: Any, name: Any) -> Any:
    return next(item for item in items if item.name == name)


def _members(mesh: Any, dimension: Any, scope: Any) -> Any:
    return np.isin(np.asarray(mesh.entity_set(dimension).entity_ids), scope.entity_ids)


@requires_worker
@pytest.mark.parametrize("kind", ("planar", "surface", "volume"))
def test_mmg_routes_preserve_domain_measure_without_inventing_ids(
    provider: Any, kind: Any
) -> None:
    if kind == "planar":
        mesh = phx.discretization.CellMesh.from_triangles(
            np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
            np.array(((0, 1, 2), (0, 2, 3))),
            vertex_global_ids=np.array((90, 7, 52, 11)),
            cell_global_ids=np.array((102, 55)),
        )
        expected = 1.0
    else:
        points = np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        if kind == "surface":
            mesh = phx.discretization.CellMesh.from_triangles(
                points, np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
            )
            expected = 1.5 + np.sqrt(3.0) / 2
        else:
            mesh = phx.discretization.CellMesh.from_tetrahedra(
                points, np.array(((0, 1, 2, 3),))
            )
            expected = 1 / 6
    source = phx.meshing.certify_cell_mesh(mesh, CONTRACT)
    result = provider.adapt(
        source,
        metric=_uniform_metric(provider, source.mesh, 0.25),
        options=MmgOptions(hausdorff_distance=0.005),
    )
    measures = _cell_measures(result.mesh.mesh)
    if kind != "surface":
        assert np.all(measures > 0)
    assert measures.sum() == pytest.approx(expected, rel=0.02)
    assert result.mesh.audit.passed and result.mesh.compliance.passed
    assert result.metric_representation == "scalar"
    assert result.metric.scope.source_id == result.mesh.mesh.mesh_id
    assert not np.intersect1d(
        result.mesh.mesh.vertex_global_ids, source.mesh.vertex_global_ids
    ).size
    assert not np.intersect1d(
        result.mesh.mesh.blocks[0].global_ids, source.mesh.blocks[0].global_ids
    ).size
    assert (
        result.mesh.derivative_mode is phx.meshing.MeshingDerivativeMode.NONDIFFERENTIABLE
    )
    assert phx.meshing.MeshingCapability.LINEAGE not in result.mesh.provider.capabilities
    assert result.mesh.adapter_reports[0].losses


@requires_worker
def test_mmg_honors_a_rotated_anisotropic_tensor_metric(provider: Any) -> None:
    source = phx.meshing.certify_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
            np.array(((0, 1, 2), (0, 2, 3))),
        ),
        CONTRACT,
    )
    rotation = np.array(((0.8, -0.6), (0.6, 0.8)))
    tensor = rotation @ np.diag((9.0, 64.0)) @ rotation.T
    metric = phx.meshing.MeshMetricField(
        provider.vertex_scope(source.mesh),
        np.tile(tensor, (4, 1, 1)),
        minimum_size=0.1,
        maximum_size=1.0,
        maximum_anisotropy=10.0,
    )
    result = provider.adapt(source, metric=metric)
    assert result.metric_representation == "tensor"
    points = np.asarray(result.mesh.mesh.coordinates)
    edges = np.asarray(result.mesh.mesh.connectivity.edges)
    displacement = points[edges[:, 1]] - points[edges[:, 0]]
    normalized = np.sqrt(np.sum((displacement @ tensor) * displacement, axis=1))
    # Dropping the off-diagonal entry or swapping tensor coefficients changes
    # the physical edge lengths in this rotated metric.
    assert np.quantile(normalized, 0.95) < 2.0
    assert np.median(normalized) > 0.5
    np.testing.assert_allclose(np.asarray(result.metric.values)[0], tensor, rtol=1e-9)


@requires_worker
def test_mmg_surface_tensor_adaptation_returns_an_ambient_spd_metric(
    provider: Any,
) -> None:
    # Tetrahedron edges are ridges, where Mmg stores metrics in a tangent frame.
    source = phx.meshing.certify_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            np.array(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            ),
            np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))),
        ),
        CONTRACT,
    )
    rotation = np.array(((0.8, -0.6, 0.0), (0.6, 0.8, 0.0), (0.0, 0.0, 1.0)))
    tensor = rotation @ np.diag((4.0, 36.0, 16.0)) @ rotation.T
    metric = phx.meshing.MeshMetricField(
        provider.vertex_scope(source.mesh),
        np.tile(tensor, (4, 1, 1)),
        minimum_size=1 / 6,
        maximum_size=0.5,
    )
    result = provider.adapt(source, metric=metric)
    assert result.metric_representation == "tensor"
    assert _cell_measures(result.mesh.mesh).sum() == pytest.approx(
        1.5 + np.sqrt(3.0) / 2, rel=1e-12
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(result.metric.values))
    assert np.all(eigenvalues > 0.0)
    assert np.all(1 / np.sqrt(eigenvalues) <= 0.5 * (1 + 1e-9))


@requires_worker
def test_mmg_binds_metric_rows_by_sorted_vertex_ids(provider: Any) -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.array(((0, 1, 2), (0, 2, 3))),
        vertex_global_ids=np.array((90, 7, 52, 11)),
    )
    source = phx.meshing.certify_cell_mesh(mesh, CONTRACT)
    # Scope rows follow sorted IDs (7, 11, 52, 90): only ID 90 at (0, 0) is fine.
    sizes = np.array((0.4, 0.4, 0.4, 0.05))
    metric = phx.meshing.MeshMetricField(
        provider.vertex_scope(source.mesh),
        np.eye(2)[None] / sizes[:, None, None] ** 2,
        minimum_size=0.05,
        maximum_size=0.4,
    )
    points = np.asarray(provider.adapt(source, metric=metric).mesh.mesh.coordinates)
    near_origin = np.count_nonzero(np.all(points < 0.25, axis=1))
    near_opposite = np.count_nonzero(np.all(points > 0.75, axis=1))
    assert near_origin > 4 * max(near_opposite, 1)


def test_mmg_refuses_a_metric_from_another_numeric_mesh_revision() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.array(((0, 1, 2),)),
        numeric_version="source",
    )
    other = phx.meshing.certify_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            np.asarray(mesh.coordinates),
            np.asarray(mesh.blocks[0].vertices),
            numeric_version="other",
        ),
        CONTRACT,
    )
    provider = MmgProvider(executable="/nonexistent/phydrax-mmg-worker")
    with pytest.raises(Failure) as caught:
        provider.plan(other, metric=_uniform_metric(provider, mesh, 0.25))
    assert caught.value.category is Category.INVALID_SOURCE
    assert provider.worker.launches == 0


@requires_worker
def test_mmg_reuses_one_worker_session_across_adaptations() -> None:
    source = _two_regions(2, 2)
    with MmgProvider() as provider:
        metric = _uniform_metric(provider, source.mesh, 0.2)
        first = provider.adapt(source, metric=metric)
        second = provider.adapt(source, metric=metric)
        assert provider.worker.launches == 1
    assert first.session.session_id == second.session.session_id
    assert first.session.identity_id == second.session.identity_id
    assert (first.session.sequence, second.session.sequence) == (1, 2)
    assert first.session.peak_rss_bytes > 0
    assert first.mesh.runtime.actual_version == second.mesh.runtime.actual_version


@requires_worker
def test_mmg_timeout_refuses_and_relaunches_a_fresh_session() -> None:
    source = phx.meshing.certify_cell_mesh(_mesh(*_grid(3, 2)), CONTRACT)
    with MmgProvider() as provider:
        with pytest.raises(Failure) as caught:
            provider.adapt(
                source,
                metric=_uniform_metric(provider, source.mesh, 0.01),
                limits=phx.meshing.MeshingLimits(maximum_wall_seconds=0.2),
            )
        assert caught.value.category is Category.TIMED_OUT
        result = provider.adapt(
            source, metric=_uniform_metric(provider, source.mesh, 0.4)
        )
        assert provider.worker.launches == 2
    assert result.session.sequence == 1


@requires_worker
@pytest.mark.parametrize("dimension", (2, 3))
def test_mmg_retains_region_zones_and_boundary_patches(
    provider: Any, dimension: Any
) -> None:
    source = _two_regions(dimension, 2)
    size = 0.3 if dimension == 3 else 0.1
    result = provider.adapt(source, metric=_uniform_metric(provider, source.mesh, size))
    mesh = result.mesh.mesh
    assert {zone.name for zone in result.mesh.zones} == {"left", "right"}
    assert {patch.name for patch in result.mesh.patches} == {"inlet", "outlet"}
    for name in ("left", "right"):
        cells = np.asarray(_by_name(result.mesh.zones, name).scope.entity_ids)
        assert _cell_measures(mesh, cells).sum() == pytest.approx(0.5, abs=1e-12)
        centroids = _cell_corners(mesh, cells).mean(axis=1)
        assert np.all((centroids[:, 0] < 0.5) == (name == "left"))
    facet_x = np.asarray(mesh.coordinates)[_facets(mesh)][:, :, 0]
    for name, x in (("inlet", 0.0), ("outlet", 1.0)):
        selected = _members(
            mesh, dimension - 1, _by_name(result.mesh.patches, name).scope
        )
        assert np.array_equal(selected, np.all(facet_x == x, axis=1))
    kinds = {reference.kind for reference in result.references.references}
    assert kinds == {"region", "boundary"}
    assert all(item.target_count > 0 for item in result.references.references)


@requires_worker
@pytest.mark.parametrize("dimension", (2, 3))
def test_mmg_interpolates_linear_fields_with_transfer_evidence(
    provider: Any, dimension: Any
) -> None:
    source = phx.meshing.certify_cell_mesh(_mesh(*_grid(dimension, 2)), CONTRACT)
    gradient = np.arange(1.0, dimension + 1.0)
    points = np.asarray(source.mesh.coordinates)
    fields = (
        _vertex_attribute(provider, source.mesh, "pressure", 0.5 + points @ gradient),
        _vertex_attribute(provider, source.mesh, "velocity", points[:, ::-1] * 2.0),
    )
    size = 0.25 if dimension == 3 else 0.1
    result = provider.adapt(
        source, metric=_uniform_metric(provider, source.mesh, size), fields=fields
    )
    target = np.asarray(result.mesh.mesh.coordinates)
    pressure = _by_name(result.mesh.attributes, "pressure")
    velocity = _by_name(result.mesh.attributes, "velocity")
    np.testing.assert_allclose(
        np.asarray(pressure.values), 0.5 + target @ gradient, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(velocity.values), target[:, ::-1] * 2.0, rtol=0, atol=1e-12
    )
    evidence = result.fields
    assert evidence.method == "p1-barycentric-closest-simplex"
    assert evidence.source_configuration == "source"
    assert evidence.field_names == ("pressure", "velocity")
    assert evidence.located_count == len(target)
    assert evidence.projected_count == 0


@requires_worker
def test_mmg_level_set_discretizes_the_contour_and_keeps_regions(provider: Any) -> None:
    source = _two_regions(2, 4)
    points = np.asarray(source.mesh.coordinates)
    # A P1-exact contour from (0, 0.3) to (1, 0.1) crossing both regions.
    level_set = MmgLevelSet(
        _vertex_attribute(
            provider, source.mesh, "phi", points[:, 1] + 0.2 * points[:, 0] - 0.3
        ),
        interface="contour",
        interior="below",
        exterior="above",
    )
    result = provider.adapt(
        source,
        level_set=level_set,
        options=MmgOptions(hausdorff_distance=1e-3, minimum_size=0.02, maximum_size=0.1),
    )
    mesh = result.mesh.mesh
    below = np.asarray(_by_name(result.mesh.labels, "below").scope.entity_ids)
    above = np.asarray(_by_name(result.mesh.labels, "above").scope.entity_ids)
    for name, expected_below in (("left", 0.125), ("right", 0.075)):
        cells = np.asarray(_by_name(result.mesh.zones, name).scope.entity_ids)
        assert _cell_measures(mesh, cells).sum() == pytest.approx(0.5, abs=1e-12)
        assert _cell_measures(mesh, np.intersect1d(cells, below)).sum() == (
            pytest.approx(expected_below, abs=1e-12)
        )
    assert _cell_measures(mesh, above).sum() == pytest.approx(0.8, abs=1e-12)
    contour = _members(mesh, 1, _by_name(result.mesh.patches, "contour").scope)
    on_contour = np.asarray(mesh.coordinates)[np.unique(_facets(mesh)[contour])]
    np.testing.assert_allclose(
        on_contour[:, 1] + 0.2 * on_contour[:, 0], 0.3, rtol=0, atol=1e-12
    )
    assert {patch.name for patch in result.mesh.patches} == {"inlet", "outlet", "contour"}
    sides = [
        item
        for item in result.references.references
        if item.kind in ("interior", "exterior")
    ]
    assert len(sides) == 4 and all(item.target_count > 0 for item in sides)


@requires_worker
def test_mmg_keeps_required_vertices(provider: Any) -> None:
    source = phx.meshing.certify_cell_mesh(_mesh(*_grid(2, 4)), CONTRACT)
    points = np.asarray(source.mesh.coordinates)
    required = np.all(points == (0.25, 0.75), axis=1) | np.all(
        points == (0.5, 0.5), axis=1
    )
    result = provider.adapt(
        source,
        metric=_uniform_metric(provider, source.mesh, 0.7),
        required=(_scope(source.mesh, 0, required),),
    )
    target = np.asarray(result.mesh.mesh.coordinates)
    for point in points[required]:
        assert np.any(np.all(target == point, axis=1))
    assert len(target) < len(points)
    assert result.references.required_vertices == 2
    assert result.references.retained_required_vertices == 2


def _holed_square(provider: Any) -> Any:
    """Unit square with a centered square hole [0.4, 0.6]^2 as the moving boundary."""
    points, cells = _grid(
        2, 10, drop=lambda origin: np.all((origin >= 0.35) & (origin < 0.55), axis=1)
    )
    mesh = _mesh(points, cells)
    hole = np.all(np.abs(points[_facets(mesh)] - 0.5) <= 0.1 + 1e-12, axis=(1, 2))
    patch = phx.meshing.MeshPatch("hole", _scope(mesh, 1, hole))
    source = phx.meshing.certify_cell_mesh(mesh, CONTRACT, patches=(patch,))
    on_hole = np.all(np.abs(points - 0.5) <= 0.1 + 1e-12, axis=1)
    displacement = np.where(on_hole[:, None], (0.05, 0.0), 0.0)
    motion = MmgLagrangianMotion(
        _vertex_attribute(provider, source.mesh, "displacement", displacement),
        patch.scope,
        MmgLagrangianMode.DISPLACE,
    )
    return source, motion


def _lagrangian_available(provider: Any) -> Any:
    return provider.worker.identity.reported["lagrangian"]["mmg2d"]


@requires_worker
def test_mmg_lagrangian_motion_moves_the_boundary_and_carries_fields(
    provider: Any,
) -> None:
    if not _lagrangian_available(provider):
        pytest.skip("Mmg was built without USE_ELAS")
    source, motion = _holed_square(provider)
    x = np.asarray(source.mesh.coordinates)[:, 0].copy()
    result = provider.adapt(
        source, motion=motion, fields=(_vertex_attribute(provider, source.mesh, "x", x),)
    )
    mesh = result.mesh.mesh
    assert _cell_measures(mesh).sum() == pytest.approx(0.96, abs=1e-12)
    hole = _members(mesh, 1, _by_name(result.mesh.patches, "hole").scope)
    moved = np.unique(_facets(mesh)[hole])
    target = np.asarray(mesh.coordinates)
    assert target[moved, 0].min() == pytest.approx(0.45, abs=1e-12)
    assert target[moved, 0].max() == pytest.approx(0.65, abs=1e-12)
    carried = np.asarray(_by_name(result.mesh.attributes, "x").values)
    np.testing.assert_allclose(carried[moved], target[moved, 0] - 0.05, atol=1e-12)
    outer = np.any((target == 0.0) | (target == 1.0), axis=1)
    np.testing.assert_allclose(carried[outer], target[outer, 0], atol=1e-12)
    assert result.fields.source_configuration == "lagrangian-displaced"


@requires_worker
def test_mmg_lagrangian_motion_is_refused_without_elas(provider: Any) -> None:
    if _lagrangian_available(provider):
        pytest.skip("Mmg was built with USE_ELAS")
    source, motion = _holed_square(provider)
    with pytest.raises(Failure) as caught:
        provider.adapt(source, motion=motion)
    assert caught.value.category is Category.UNSUPPORTED_CAPABILITY
    assert "USE_ELAS" in str(caught.value)
