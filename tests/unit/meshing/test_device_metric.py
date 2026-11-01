#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax.discretization import AdaptiveSimplexPolicy, AdaptiveSimplexStatus
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.meshing import (
    _device_metric as device_metric,
    _local_metric as local_metric,
    adapt_device_metric,
    commit_device_metric_adaptation,
    execute_mesh_adaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingScope,
    MeshMetricField,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    MetricMeshAdaptation,
    prepare_device_metric_adaptation,
    prepare_mesh_adaptation,
    RelocationMeshAdaptation,
)
from phydrax.meshing._device_metric import (
    _compiled_adapt,
    _work,
    DeviceMetricEvidence,
)
from phydrax.meshing._local_metric import LocalMetricEvidence
from phydrax.meshing._metric import _constant_preserving_metric_mean


def test_internal_metric_mean_preserves_identical_tensor_exactly() -> None:
    tensor = jnp.asarray(((3.0, 0.4), (0.4, 2.0)), dtype=jnp.float64)
    samples = jnp.broadcast_to(tensor, (3, 3, 2, 2))
    weights = jnp.asarray(((0.1, 0.2, 0.7), (1.0, 0.0, 0.0), (0.3, 0.3, 0.4)))
    actual = jax.jit(_constant_preserving_metric_mean)(samples, weights)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(samples[:, 0]))


_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)
# One capacity bucket for every test: one compiled executable per worker.
_DEVICE = AdaptiveSimplexPolicy(vertex_capacity=256, cell_capacity=512)


def _grid(count: Any, /, *, perturbation: Any = 0.0, shear: Any = 0.0) -> Any:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, count + 1), np.linspace(0.0, 1.0, count + 1))
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    offsets = np.where(interior[:, None], perturbation * np.asarray((1.0, -0.6)), 0.0)
    shift = np.sin(7.0 * points[:, :1] + 3.0 * points[:, 1:])
    points = points + offsets * shift / count
    points[:, 0] += shear * points[:, 1]
    index = np.arange((count + 1) ** 2).reshape((count + 1, count + 1))
    a = index[:-1, :-1].ravel()
    b = index[:-1, 1:].ravel()
    c = index[1:, 1:].ravel()
    d = index[1:, :-1].ravel()
    triangles = np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))
    return points, triangles


def _scope(mesh: Any, dimension: Any, entity_ids: Any, /) -> Any:
    entities = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        np.asarray(entity_ids, dtype=np.int64),
    )


def _source(
    count: Any = 4,
    /,
    *,
    perturbation: Any = 0.0,
    shear: Any = 0.0,
    organized: Any = False,
) -> Any:
    points, triangles = _grid(count, perturbation=perturbation, shear=shear)
    mesh = phx.discretization.CellMesh.from_triangles(points, triangles)
    contract = phx.SpatialCoordinateContract.si()
    mesh = phx.meshing.certify_cell_mesh(mesh, contract).mesh
    if not organized:
        return phx.meshing.certify_cell_mesh(mesh, contract)
    coordinates = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.entity_set(2).entity_ids)
    centroids = np.concatenate(
        [coordinates[np.asarray(block.vertices)].mean(axis=1) for block in mesh.blocks]
    )
    # ty: ignore[unresolved-attribute]
    edges = np.asarray(mesh.connectivity.edges)
    bottom = np.all(coordinates[edges][:, :, 1] == 0.0, axis=1)
    zones = (
        MeshZone(
            "left", MeshZoneRole.REGION, _scope(mesh, 2, cells[centroids[:, 0] < 0.5])
        ),
        MeshZone(
            "right", MeshZoneRole.REGION, _scope(mesh, 2, cells[centroids[:, 0] > 0.5])
        ),
    )
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids)
    patches = (MeshPatch("bottom", _scope(mesh, 1, edge_ids[bottom])),)
    return phx.meshing.certify_cell_mesh(mesh, contract, patches=patches, zones=zones)


def _periodic_source() -> Any:
    shape = 3
    vertex_rows = np.arange(shape**2, dtype=np.int64).reshape(shape, shape)
    lattice = np.stack(
        np.meshgrid(np.arange(shape), np.arange(shape), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    coordinates = lattice.astype(np.float64) / 2.0
    indices = np.arange(shape, dtype=np.int64)
    representatives = vertex_rows[indices[:, None] % 2, indices[None, :] % 2].reshape(-1)
    shifts = np.stack(
        np.meshgrid(indices // 2, indices // 2, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    triangles = []
    for i in range(shape - 1):
        for j in range(shape - 1):
            a = int(vertex_rows[i, j])
            b = int(vertex_rows[i + 1, j])
            c = int(vertex_rows[i + 1, j + 1])
            d = int(vertex_rows[i, j + 1])
            triangles.extend(((a, b, c), (a, c, d)))
    block = CellBlock(
        "triangles",
        "triangle",
        np.asarray(triangles, dtype=np.int32),
        global_ids=np.arange(100, 100 + len(triangles), dtype=np.int64),
    )
    vertex_ids = np.arange(shape**2, dtype=np.int64)
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)
    periodic = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.eye(2, dtype=np.float64)),
        representatives,
        shifts,
    )
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=vertex_ids,
        periodic_topology=periodic,
    )
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _metric(source: Any, tensor: Any, /) -> Any:
    mesh = source.mesh
    vertices = mesh.entity_set(0)
    scope = _scope(mesh, 0, vertices.entity_ids)
    values = np.broadcast_to(tensor, (vertices.entity_ids.shape[0], 2, 2))
    return MeshMetricField(scope, values, minimum_size=1.0e-3, maximum_size=10.0)


def _policy(device: Any = _DEVICE, /, **options: Any) -> Any:
    return MeshAdaptationPolicy(
        MeshAdaptationRoute.DEVICE_METRIC_2D, device_policy=device, **options
    )


def _adapt(source: Any, request: Any, /, **options: Any) -> Any:
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=_policy(**options))
    )


def _signed_areas(mesh: Any, /) -> Any:
    points = np.asarray(mesh.coordinates)
    areas = []
    for block in mesh.blocks:
        a, b, c = np.moveaxis(points[np.asarray(block.vertices)], 1, 0)
        areas.append(
            0.5
            * (
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            )
        )
    return np.concatenate(areas)


def _edge_metric_lengths(mesh: Any, tensor: Any, /) -> Any:
    points = np.asarray(mesh.coordinates)
    edges = np.asarray(mesh.connectivity.edges)
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    return np.sqrt(np.sum((delta @ tensor) * delta, axis=1))


def test_native_metric_route_escapes_the_split_collapse_hysteresis() -> None:
    # Regular passes stall on one short birth edge whose collapses all create an
    # edge above sqrt(2); the stall escape collapses it (the next split repairs
    # the long edge) and lowers the stall potential, so the host route completes
    # on the same input as the device route.
    tensor = np.diag((1.0 / 0.5**2, 1.0 / 0.1**2))
    source = _source(perturbation=0.3)
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MetricMeshAdaptation(_metric(source, tensor)),
            policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_METRIC_2D),
        )
    )

    assert result.status is MeshAdaptationStatus.COMPLETE
    evidence = result.evidence
    assert isinstance(evidence, LocalMetricEvidence)
    assert evidence.converged and not evidence.stalled and not evidence.stalled_edges
    result.target.audit.require_passed()
    lengths = _edge_metric_lengths(result.target.mesh, tensor)
    assert np.all((lengths >= _LOWER - 1.0e-12) & (lengths <= _UPPER + 1.0e-12))
    assert np.all(_signed_areas(result.target.mesh) > 0.0)


def test_device_metric_adaptation_reaches_a_certified_unit_mesh() -> None:
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    rotated = rotation @ np.diag((1.0 / 0.45**2, 1.0 / 0.22**2)) @ rotation.T
    for tensor, perturbation in [
        (np.eye(2) / 0.2**2, 0.0),
        (rotated, 0.2),
        (np.diag((1.0 / 0.5**2, 1.0 / 0.1**2)), 0.3),
    ]:
        source = _source(perturbation=perturbation)
        result = _adapt(source, MetricMeshAdaptation(_metric(source, tensor)))

        assert result.route is MeshAdaptationRoute.DEVICE_METRIC_2D
        assert result.status is MeshAdaptationStatus.COMPLETE
        assert result.evidence.converged and result.evidence.unit_fraction == 1.0
        assert result.evidence.splits > 0
        result.target.audit.require_passed()
        lengths = _edge_metric_lengths(result.target.mesh, tensor)
        assert np.all((lengths >= _LOWER - 1.0e-12) & (lengths <= _UPPER + 1.0e-12))
        areas = _signed_areas(result.target.mesh)
        assert np.all(areas > 0.0)
        assert np.isclose(np.sum(areas), 1.0, rtol=0.0, atol=1.0e-14)

        def field(points: Any) -> Any:
            return 2.0 + 3.0 * points[:, 0] - 1.5 * points[:, 1]

        target = result.target.mesh
        if result.transfer is None:
            assert not np.all(
                np.any(np.asarray(result.stencil.valid, dtype=np.bool_), axis=1)
            )
            continue
        values = result.stencil.apply(
            np.asarray(source.mesh.vertex_global_ids),
            field(np.asarray(source.mesh.coordinates)),
        )
        order = np.argsort(np.asarray(target.vertex_global_ids))
        rows = order[
            np.searchsorted(
                np.asarray(target.vertex_global_ids)[order],
                np.asarray(result.stencil.target_global_ids),
            )
        ]
        expected = field(np.asarray(target.coordinates)[rows])
        np.testing.assert_allclose(np.asarray(values), expected, rtol=0.0, atol=1.0e-13)


def test_device_metric_scenario_1() -> None:
    source = _source(organized=True)
    result = _adapt(source, MetricMeshAdaptation(_metric(source, np.eye(2) / 0.12**2)))
    mesh = result.target.mesh

    assert result.status is MeshAdaptationStatus.COMPLETE
    areas = _signed_areas(mesh)
    cell_ids = np.asarray(mesh.entity_set(2).entity_ids)
    zone_areas = {
        zone.name: np.sum(areas[np.isin(cell_ids, np.asarray(zone.scope.entity_ids))])
        for zone in result.target.zones
    }
    assert zone_areas == pytest.approx({"left": 0.5, "right": 0.5}, abs=1.0e-14)
    points = np.asarray(mesh.coordinates)
    edges = np.asarray(mesh.connectivity.edges)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids)
    (patch,) = result.target.patches
    patch_edges = edges[np.isin(edge_ids, np.asarray(patch.scope.entity_ids))]
    assert np.all(points[patch_edges][:, :, 1] == 0.0)
    patch_length = np.sum(np.abs(np.diff(points[patch_edges][:, :, 0], axis=1)))
    assert patch_length == pytest.approx(1.0, abs=1.0e-14)
    source = _source(perturbation=0.3)
    request = MetricMeshAdaptation(
        _metric(source, np.diag((1.0 / 0.3**2, 1.0 / 0.12**2)))
    )

    first = _adapt(source, request)
    second = _adapt(source, request)
    prepared = prepare_device_metric_adaptation(source, request, policy=_policy())
    update = adapt_device_metric(prepared.layout, prepared.state)
    epoch = commit_device_metric_adaptation(prepared, update.state)

    assert first.result_id == second.result_id == epoch.result_id
    assert first.target.mesh.mesh_id == second.target.mesh.mesh_id
    source = _source(6, perturbation=0.35)
    result = _adapt(
        source, RelocationMeshAdaptation(_metric(source, np.eye(2) / 0.17**2))
    )
    evidence = result.evidence

    assert evidence.relocations > 0
    assert evidence.splits == evidence.collapses == evidence.flips == 0
    assert result.target.mesh.topology_id == source.mesh.topology_id
    assert not np.array_equal(
        np.asarray(result.target.mesh.coordinates), np.asarray(source.mesh.coordinates)
    )
    assert np.all(_signed_areas(result.target.mesh) > 0.0)
    source_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    source_coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
    target_coordinates = np.asarray(result.target.mesh.coordinates, dtype=np.float64)
    target_vertex_ids = np.asarray(result.target.mesh.vertex_global_ids, dtype=np.int64)
    source_rows = {
        int(identifier): row for row, identifier in enumerate(source_ids.tolist())
    }
    target_rows = {
        int(identifier): row for row, identifier in enumerate(target_vertex_ids.tolist())
    }
    stencil = result.stencil
    assert stencil is not None
    target_ids = np.asarray(stencil.target_global_ids, dtype=np.int64)
    sources = np.asarray(stencil.source_global_ids, dtype=np.int64)
    weights = np.asarray(stencil.weights, dtype=np.float64)
    valid = np.asarray(stencil.valid, dtype=np.bool_)
    for row, identifier in enumerate(target_ids.tolist()):
        if not valid[row].any():
            assert result.transfer is None
            continue
        exact_sum = sum(stencil.exact_weights[row], Fraction(0))
        assert exact_sum == 1
        source_row = source_rows.get(identifier)
        if source_row is not None and not np.array_equal(
            source_coordinates[source_row],
            target_coordinates[target_rows[identifier]],
        ):
            support = sources[row, valid[row]]
            assert not (
                support.size == 1
                and int(support[0]) == identifier
                and weights[row, valid[row]][0] == 1.0
            )
    source = _source()
    request = MetricMeshAdaptation(_metric(source, np.eye(2) / 0.2**2))
    tiny = AdaptiveSimplexPolicy(vertex_capacity=25, cell_capacity=32)
    prepared = prepare_device_metric_adaptation(source, request, policy=_policy(tiny))
    overflow = _compiled_adapt(prepared.layout, prepared.state)
    capacity_status = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
    assert int(overflow.report.status) & capacity_status
    assert not bool(overflow.report.failed)
    source_work = _work(prepared.state)
    rolled_back_work = _work(overflow.state)
    for name in source_work._fields:
        if name == "flags":
            continue
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(getattr(rolled_back_work, name))),
            np.asarray(jax.device_get(getattr(source_work, name))),
        )
    update = adapt_device_metric(prepared.layout, prepared.state)
    report = jax.device_get(update.report)

    assert not bool(report.failed)
    assert int(report.splits) > 0
    assert update.layout.vertex_capacity > prepared.layout.vertex_capacity
    assert update.layout.cell_capacity > prepared.layout.cell_capacity
    continued = adapt_device_metric(update.layout, update.state)
    assert bool(jax.device_get(continued.report.converged))
    assert int(jax.device_get(continued.report.passes)) == 0
    result = commit_device_metric_adaptation(prepared, continued.state)
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert isinstance(result.evidence, DeviceMetricEvidence)
    assert result.evidence.converged
    default = _adapt(source, request)
    assert isinstance(default.evidence, DeviceMetricEvidence)
    assert result.target.mesh.mesh_id == default.target.mesh.mesh_id
    assert result.result_id == default.result_id
    assert result.evidence.evidence_id == default.evidence.evidence_id
    result.target.audit.require_passed()
    source = _source()
    mesh = source.mesh
    edges = mesh.entity_set(1)
    protected = _scope(mesh, 1, edges.entity_ids)
    result = _adapt(
        source,
        MetricMeshAdaptation(_metric(source, np.eye(2) / 0.1**2)),
        relocation=False,
        protected_scopes=(protected,),
    )

    assert result.status is MeshAdaptationStatus.STALLED
    assert not result.status.converged
    assert result.target.result_id == source.result_id
    assert result.transition is result.lineage is result.transfer is None
    assert result.evidence.stalled and not result.evidence.converged
    assert result.evidence.out_of_range_edges > 0
    # The slanted sides x = y / 2 and x = 1 + y / 2 are exactly straight, but
    # FILTERED_DEVICE certifies zero orientations only structurally: collapsing
    # or sliding a vertex of a slanted side needs host resolution.
    source = _source(8, shear=0.5)
    request = MetricMeshAdaptation(_metric(source, np.eye(2) / 0.3**2))
    prepared = prepare_device_metric_adaptation(source, request, policy=_policy())
    update = adapt_device_metric(prepared.layout, prepared.state)
    report = jax.device_get(update.report)

    assert AdaptiveSimplexStatus(int(report.status)) & (
        AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION
    )
    assert not report.failed
    assert report.rejected_uncertain > 0
    result = commit_device_metric_adaptation(prepared, update.state)
    assert isinstance(result.evidence, DeviceMetricEvidence)
    status = result.evidence.status
    assert isinstance(status, AdaptiveSimplexStatus)
    assert status & AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION
    assert result.evidence.collapses > 0
    points = np.asarray(source.mesh.coordinates)
    identifiers = np.asarray(source.mesh.vertex_global_ids)
    slanted = np.isclose(points[:, 0], 0.5 * points[:, 1]) | np.isclose(
        points[:, 0], 1.0 + 0.5 * points[:, 1]
    )
    target = result.target.mesh
    target_ids = np.asarray(target.vertex_global_ids)
    order = np.argsort(target_ids)
    rows = order[np.searchsorted(target_ids[order], identifiers[slanted])]
    np.testing.assert_array_equal(target_ids[rows], identifiers[slanted])
    np.testing.assert_array_equal(np.asarray(target.coordinates)[rows], points[slanted])
    areas = _signed_areas(target)
    assert np.all(areas > 0.0)
    assert np.isclose(np.sum(areas), 1.0, rtol=0.0, atol=1.0e-14)


def _assert_same_arrays(first: Any, second: Any, /) -> None:
    for before, after in zip(
        jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))


def test_device_metric_periodic_operations_preserve_complete_orbits() -> None:
    source = _periodic_source()
    angle = np.deg2rad(23.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    tensor = rotation @ np.diag((1.0 / 0.25**2, 1.0 / 0.4**2)) @ rotation.T
    result = _adapt(source, MetricMeshAdaptation(_metric(source, tensor)))

    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.converged and result.evidence.splits > 0
    target = result.target.mesh
    assert target.periodic_topology is not None
    target.periodic_topology.require_lifted(
        target.topology, target.lifted_topology_id, target.blocks
    )
    target.periodic_topology.require_coordinates(target.coordinates)
    lengths = _edge_metric_lengths(target, tensor)
    assert np.all(lengths >= _LOWER) and np.all(lengths <= _UPPER)
    np.testing.assert_allclose(np.sum(_signed_areas(target)), 1.0, rtol=0.0, atol=1e-12)


def test_device_metric_rejects_non_equivariant_periodic_spd_data() -> None:
    source = _periodic_source()
    mesh = source.mesh
    scope = _scope(mesh, 0, mesh.entity_set(0).entity_ids)
    values = np.broadcast_to(
        np.eye(2, dtype=np.float64), (mesh.vertex_global_ids.size, 2, 2)
    ).copy()
    values[-1, 0, 0] = 2.0
    request = MetricMeshAdaptation(
        MeshMetricField(scope, values, minimum_size=1.0e-3, maximum_size=10.0)
    )

    with pytest.raises(ValueError, match="equivariantly"):
        prepare_device_metric_adaptation(source, request, policy=_policy())


def test_host_and_device_optimal_collapse_relocate_the_survivor() -> None:
    # The short birth edge is 0.6 long. Either endpoint collapse produces a
    # spoke of length 1.5, whereas merging at the origin leaves all edges in
    # the unit band. Hull vertices and the six source triangles are unchanged.
    points = np.asarray(
        ((-1.2, 0.0), (-0.3, 0.0), (0.3, 0.0), (1.2, 0.0), (0.0, 0.72), (0.0, -0.72))
    )
    triangles = np.asarray(
        ((0, 1, 4), (1, 2, 4), (2, 3, 4), (1, 0, 5), (2, 1, 5), (3, 2, 5)), dtype=np.int32
    )
    mesh = CellMesh.from_triangles(points, triangles)
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    tensors = np.broadcast_to(np.eye(2), (6, 2, 2))
    edge_count = mesh.entity_set(1).count
    host, host_source = local_metric._initial_state(
        mesh,
        tensors,
        np.zeros(6, dtype=np.int64),
        np.zeros(edge_count, dtype=np.int64),
        np.zeros(edge_count, dtype=np.bool_),
        np.zeros(6, dtype=np.bool_),
        PredicateMode.EXACT,
    )
    births = host.source_rows.copy()
    births[1:3] = -1
    host = host._replace(source_rows=births)
    topology = local_metric._topology(host, PredicateMode.EXACT)
    candidates = local_metric._collapse_candidates(
        host, topology, PredicateMode.EXACT, False
    )
    assert not np.any(candidates.status == local_metric._OK)
    host, _, count = local_metric._optimal_collapse_round(
        host, topology, local_metric._Counts(), PredicateMode.EXACT, None, host_source
    )
    assert count == 1
    assert local_metric._unit(local_metric._topology(host, PredicateMode.EXACT))
    np.testing.assert_allclose(host.points[1], (0.0, 0.0), rtol=0.0, atol=1e-15)

    prepared = prepare_device_metric_adaptation(
        source, MetricMeshAdaptation(_metric(source, np.eye(2))), policy=_policy()
    )
    work = _work(prepared.state)
    work = work._replace(source_rows=work.source_rows.at[1:3].set(-1))
    updated, _, overflow, count = eqx.filter_jit(device_metric._optimal_collapse_round)(
        work, jnp.asarray(False), prepared.layout
    )
    assert int(count) == 1 and not int(overflow)
    assert bool(device_metric._unit(updated))
    np.testing.assert_allclose(
        np.asarray(updated.coordinates[1]), (0.0, 0.0), rtol=0.0, atol=1e-15
    )
    np.testing.assert_array_equal(np.asarray(updated.vertex_active[:6]), host.alive)


def test_host_optimal_collapse_batches_preserve_candidate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diamond = np.asarray(
        ((-1.2, 0.0), (-0.3, 0.0), (0.3, 0.0), (1.2, 0.0), (0.0, 0.72), (0.0, -0.72))
    )
    cells = np.asarray(
        ((0, 1, 4), (1, 2, 4), (2, 3, 4), (1, 0, 5), (2, 1, 5), (3, 2, 5)),
        dtype=np.int32,
    )
    points = np.concatenate([diamond + (3.0 * index, 0.0) for index in range(5)])
    mesh = CellMesh.from_triangles(
        points, np.concatenate([cells + 6 * index for index in range(5)])
    )
    state, source = local_metric._initial_state(
        mesh,
        np.broadcast_to(np.eye(2), (30, 2, 2)),
        np.zeros(30, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.bool_),
        np.zeros(30, dtype=np.bool_),
        PredicateMode.EXACT,
    )
    topology = local_metric._topology(state, PredicateMode.EXACT)
    short = np.flatnonzero(topology.lengths < _LOWER)
    assert short.size == 5
    original = local_metric._optimal_collapse_batch
    reference = original(state, topology, PredicateMode.EXACT, None, source, short)
    sizes: list[int] = []

    def observed(
        state: local_metric._State,
        topology: local_metric._Topology,
        mode: PredicateMode,
        context: local_metric._PeriodicContext | None,
        source: local_metric._Source,
        edges: np.ndarray,
    ) -> local_metric._OptimalCollapses:
        sizes.append(edges.size)
        return original(state, topology, mode, context, source, edges)

    monkeypatch.setattr(local_metric, "_OPTIMAL_BLOCK", 2)
    monkeypatch.setattr(local_metric, "_optimal_collapse_batch", observed)
    batched = local_metric._optimal_collapse_search(
        state, topology, PredicateMode.EXACT, None, source, short
    )
    assert sizes == [2, 2, 1]
    for name in ("edge", "removed", "kept", "point", "metric", "longest"):
        np.testing.assert_array_equal(getattr(batched, name), getattr(reference, name))
    assert batched.reason == reference.reason
    for first, second in zip(batched.stencils, reference.stencils, strict=True):
        assert first is not None and second is not None
        for left, right in zip(first[:3], second[:3], strict=True):
            np.testing.assert_array_equal(left, right)
        assert first.exact == second.exact


@pytest.mark.parametrize("scale, feasible", [(0.999, True), (1.001, False)])
def test_host_optimal_collapse_searches_between_seed_parameters(
    scale: float,
    feasible: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    center = -0.2 + 0.4 * (17.0 / 64.0)
    points = np.asarray(
        (
            (center - 0.8 * scale, 0.6 * scale),
            (-0.2, 0.0),
            (0.2, 0.0),
            (center + 0.8 * scale, -0.6 * scale),
            (center, 0.65),
            (center, -0.65),
        )
    )
    cells = np.asarray(
        ((0, 1, 4), (1, 2, 4), (2, 3, 4), (1, 0, 5), (2, 1, 5), (3, 2, 5)),
        dtype=np.int32,
    )
    mesh = CellMesh.from_triangles(points, cells)
    state, source = local_metric._initial_state(
        mesh,
        np.broadcast_to(2.0 * np.eye(2), (6, 2, 2)),
        np.zeros(6, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.bool_),
        np.zeros(6, dtype=np.bool_),
        PredicateMode.EXACT,
    )
    exact_points = list(source.exact_points)
    exact_points[1] = (exact_points[1][0] + Fraction(1, 2**60), exact_points[1][1])
    assert float(exact_points[1][0]) == points[1, 0]
    source = source._replace(exact_points=tuple(exact_points))
    topology = local_metric._topology(state, PredicateMode.EXACT)
    short = np.flatnonzero(topology.lengths < _LOWER)
    assert short.size == 1
    seeds = local_metric._optimal_collapse_batch(
        state, topology, PredicateMode.EXACT, None, source, short
    )
    assert seeds.reason == ("unresolved",)
    result = local_metric._optimal_collapse_search(
        state, topology, PredicateMode.EXACT, None, source, short
    )
    if feasible:
        width = np.sqrt(1.0 - (0.6 * scale) ** 2) - 0.8 * scale
        assert 0.0 < width < 0.4 / 64.0
        assert result.reason == ("",)
        assert abs(result.point[0, 0] - center) < width
        stencil = result.stencils[0]
        assert stencil is not None
        by_id = dict(zip(source.vertex_ids.tolist(), source.exact_points, strict=True))
        exact_x = sum(
            (weight * by_id[identifier][0] for identifier, weight in stencil.exact),
            Fraction(0),
        )
        expected = exact_points[1][0] + Fraction(17, 64) * (
            exact_points[2][0] - exact_points[1][0]
        )
        assert exact_x == expected
        assert result.point[0, 0] == float(expected)
        stale_points = state.points.copy()
        stale_points[0, 0] += 0.001
        stale = state._replace(points=stale_points)
        stale_topology = local_metric._topology(stale, PredicateMode.EXACT)
        refused = local_metric._optimal_collapse_search(
            stale, stale_topology, PredicateMode.EXACT, None, source, short
        )
        assert refused.reason == ("unresolved",)

        def exhausted(*_args: object) -> None:
            raise MemoryError

        monkeypatch.setattr(local_metric, "_optimal_remainder", exhausted)
        resource = local_metric._optimal_collapse_search(
            state, topology, PredicateMode.EXACT, None, source, short
        )
        assert resource.reason == ("resource",)
    else:
        assert result.reason == ("long",)


def test_host_optimal_collapse_opaque_field_seed_miss_is_unresolved() -> None:
    points = np.asarray(
        ((-1.2, 0.0), (-0.3, 0.0), (0.3, 0.0), (1.2, 0.0), (0.0, 0.72), (0.0, -0.72))
    )
    cells = np.asarray(
        ((0, 1, 4), (1, 2, 4), (2, 3, 4), (1, 0, 5), (2, 1, 5), (3, 2, 5)),
        dtype=np.int32,
    )
    mesh = CellMesh.from_triangles(points, cells)

    def opaque(query: np.ndarray, rows: np.ndarray, weights: np.ndarray, /) -> np.ndarray:
        del rows, weights
        # A narrow dip halfway between adjacent lattice seeds.
        scale = np.where(np.abs(query[:, 0] - 0.009375) < 0.001, 0.01, 100.0)
        return scale[:, None, None] * np.eye(2)

    state, source = local_metric._initial_state(
        mesh,
        np.broadcast_to(np.eye(2), (6, 2, 2)),
        np.zeros(6, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.bool_),
        np.zeros(6, dtype=np.bool_),
        PredicateMode.EXACT,
        metric_field=opaque,
    )
    topology = local_metric._topology(state, PredicateMode.EXACT)
    short = np.flatnonzero(topology.lengths < _LOWER)
    assert short.size == 1
    result = local_metric._optimal_collapse_search(
        state, topology, PredicateMode.EXACT, None, source, short
    )
    assert result.reason == ("missing-certification",)

    class NoncontractingField:
        def __call__(
            self, query: np.ndarray, rows: np.ndarray, weights: np.ndarray, /
        ) -> np.ndarray:
            del rows, weights
            return np.broadcast_to(100.0 * np.eye(2), (query.shape[0], 2, 2))

        def segment_range(
            self,
            rows: tuple[int, ...],
            lower: tuple[Fraction, ...],
            upper: tuple[Fraction, ...],
            chart_lower: np.ndarray,
            chart_upper: np.ndarray,
            /,
        ) -> tuple[np.ndarray, np.ndarray]:
            del rows, lower, upper, chart_lower, chart_upper
            return np.zeros((2, 2)), 100.0 * np.eye(2)

    wide = source._replace(metric_field=NoncontractingField())
    result = local_metric._optimal_collapse_search(
        state, topology, PredicateMode.EXACT, None, wide, short
    )
    assert result.reason == ("missing-certification",)


def test_current_metric_segment_encloses_vertex_ball_interpolation() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    mesh = CellMesh.from_triangles(points, np.asarray(((0, 1, 2), (0, 2, 3))))
    tensors = np.asarray(
        (
            ((2.0, 0.0), (0.0, 1.0)),
            ((3.0, 0.0), (0.0, 1.5)),
            ((4.0, 0.0), (0.0, 2.0)),
            ((1.0, 0.0), (0.0, 3.0)),
        )
    )
    state, source = local_metric._initial_state(
        mesh,
        tensors,
        np.zeros(4, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.bool_),
        np.zeros(4, dtype=np.bool_),
        PredicateMode.EXACT,
    )
    topology = local_metric._topology(state, PredicateMode.EXACT)
    prepared = local_metric._PreparedMetricEnclosures(state.metric)
    bounds = local_metric._current_segment_metric(
        state, topology, source, 0, 2, Fraction(1, 4), Fraction(3, 4), prepared
    )
    assert bounds is not None
    queries = np.asarray(((0.25, 0.25), (0.5, 0.5), (0.75, 0.75)))
    actual = local_metric._proposal_metric(
        state, topology, np.zeros(3, dtype=np.int64), queries
    )
    assert np.all(actual >= bounds[0])
    assert np.all(actual <= bounds[1])


def test_source_box_detects_single_and_crossing_cells() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    mesh = CellMesh.from_triangles(points, np.asarray(((0, 1, 2), (0, 2, 3))))
    _, source = local_metric._initial_state(
        mesh,
        np.broadcast_to(np.eye(2), (4, 2, 2)),
        np.zeros(4, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.int64),
        np.zeros(mesh.entity_set(1).count, dtype=np.bool_),
        np.zeros(4, dtype=np.bool_),
        PredicateMode.EXACT,
    )
    covered, inside = local_metric._source_box_pieces(
        source, np.asarray((0.7, 0.2)), np.asarray((0.8, 0.3))
    )
    assert covered and len(inside) == 1
    covered, crossing = local_metric._source_box_pieces(
        source, np.asarray((0.2, 0.2)), np.asarray((0.3, 0.3))
    )
    assert not covered and len(crossing) == 2
    exact = list(source.exact_points)
    exact[1] = (Fraction(1) + Fraction(1, 2**54), Fraction(0))
    source = source._replace(exact_points=tuple(exact))
    covered, shifted = local_metric._source_box_pieces(
        source, np.asarray((1.0, 0.0)), np.asarray((1.0, 0.0))
    )
    assert covered
    piece = next(piece for piece in shifted if piece.rows == (0, 1, 2))
    assert piece.weight_upper[1] == 1 / (Fraction(1) + Fraction(1, 2**54))




def test_candidate_box_encloses_rounded_and_exact_affine_values() -> None:
    origin = np.asarray((-0.3, 0.7))
    target = np.asarray((0.8, -0.2))
    lower, upper = local_metric._candidate_box(origin, target, 0.1, 0.2)
    for t in (0.1, 0.125, 0.175, 0.2):
        rounded = origin + t * (target - origin)
        for axis in range(2):
            exact = Fraction(float(origin[axis])) + Fraction(t) * (
                Fraction(float(target[axis])) - Fraction(float(origin[axis]))
            )
            assert Fraction(float(lower[axis])) <= exact <= Fraction(float(upper[axis]))
            assert lower[axis] <= rounded[axis] <= upper[axis]
    exact_origin = (Fraction(1, 3), Fraction(-2, 7))
    exact_target = (Fraction(5, 11), Fraction(4, 9))
    lower, upper = local_metric._exact_candidate_box(
        exact_origin, exact_target, Fraction(1, 4), Fraction(3, 4)
    )
    for t in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
        for axis in range(2):
            exact = exact_origin[axis] + t * (exact_target[axis] - exact_origin[axis])
            assert Fraction(float(lower[axis])) <= exact <= Fraction(float(upper[axis]))
            assert lower[axis] <= float(exact) <= upper[axis]
