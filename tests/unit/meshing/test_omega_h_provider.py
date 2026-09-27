import functools
import os
import shlex
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

import phydrax as phx


M = phx.meshing


def _worker() -> str | None:
    return shutil.which(
        os.environ.get("PHYDRAX_OMEGA_H_WORKER", "phydrax-omega-h-worker")
    )


@functools.cache
def _mpi_launcher() -> tuple[str, ...] | None:
    """Deployment launcher from PHYDRAX_OMEGA_H_MPI_LAUNCHER, if it can start 2 ranks."""
    launcher = tuple(
        shlex.split(os.environ.get("PHYDRAX_OMEGA_H_MPI_LAUNCHER", "mpiexec"))
    )
    if not launcher or shutil.which(launcher[0]) is None:
        return None
    probe = subprocess.run(
        [*launcher, "-n", "2", sys.executable, "-c", "pass"],
        capture_output=True,
        timeout=120,
    )
    return launcher if probe.returncode == 0 else None


def requires_worker(test: Any) -> Any:
    """Mark a test that runs the real worker; skip when it is not built."""
    skip = pytest.mark.skipif(
        _worker() is None, reason="phydrax-omega-h-worker is not built"
    )
    return pytest.mark.meshing_omega_h(skip(test))


def _launcher_or_skip() -> tuple[str, ...]:
    launcher = _mpi_launcher()
    if launcher is None:
        pytest.skip("No MPI launcher can start two ranks on this host")
    return launcher


def _grid(n: int) -> phx.discretization.CellMesh:
    """Unit square as 2 n^2 positively oriented triangles with sparse global IDs."""
    xs = np.linspace(0.0, 1.0, n + 1)
    x, y = np.meshgrid(xs, xs, indexing="xy")
    points = np.column_stack((x.ravel(), y.ravel()))
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    a = (j * (n + 1) + i).ravel()
    b, c, d = a + 1, a + n + 2, a + n + 1
    triangles = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
    return phx.discretization.CellMesh.from_triangles(
        points,
        triangles,
        vertex_global_ids=11 + 7 * np.arange(len(points), dtype=np.int64),
        cell_global_ids=17 + 13 * np.arange(len(triangles), dtype=np.int64),
    )


def _scope(mesh: Any, dimension: int, ids: Any) -> M.MeshingScope:
    return M.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        np.asarray(ids, dtype=np.int64),
    )


def _cells(mesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(global IDs, areas, centroids) of every triangle."""
    coordinates = np.asarray(mesh.coordinates)
    vertices = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    corners = coordinates[vertices]
    areas = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 2.0
    return ids, areas, corners.mean(axis=1)


def _regions(mesh: Any) -> Any:
    """Certified source with left/right region zones and an inlet patch at x=0."""
    ids, _, centroids = _cells(mesh)
    left = M.MeshZone(
        "left", M.MeshZoneRole.REGION, _scope(mesh, 2, ids[centroids[:, 0] < 0.5])
    )
    right = M.MeshZone(
        "right", M.MeshZoneRole.REGION, _scope(mesh, 2, ids[centroids[:, 0] > 0.5])
    )
    edges = np.asarray(mesh.connectivity.edges)
    on_inlet = np.all(np.asarray(mesh.coordinates)[edges][:, :, 0] == 0.0, axis=1)
    inlet = M.MeshPatch(
        "inlet",
        _scope(mesh, 1, np.asarray(mesh.entity_set(1).entity_ids)[on_inlet]),
        adjacent_zone_ids=(left.zone_id,),
    )
    return M.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), zones=(left, right), patches=(inlet,)
    )


def _metric(mesh: Any, size: Any) -> M.MeshMetricField:
    """Isotropic metric of edge length ``size(x)`` at vertices in sorted-ID order."""
    order = np.argsort(np.asarray(mesh.vertex_global_ids))
    sizes = np.asarray(size(np.asarray(mesh.coordinates)[order]), dtype=np.float64)
    return M.MeshMetricField(
        _scope(mesh, 0, np.asarray(mesh.vertex_global_ids)[order]),
        np.eye(2) * sizes[:, None, None] ** -2,
        minimum_size=float(sizes.min()),
        maximum_size=float(sizes.max()),
        maximum_anisotropy=1.0,
    )


def _uniform(size: float) -> Any:
    return lambda points: np.full(len(points), size)


@requires_worker
def test_real_omega_h_refines_and_preserves_domain_measure() -> None:
    mesh = _grid(2)
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    options = M.OmegaHOptions(
        min_quality_allowed=0.25, min_quality_desired=0.35, should_coarsen=False
    )

    with M.OmegaHProvider(_worker()) as provider:
        result = provider.execute(source, _metric(mesh, _uniform(0.1)), options=options)

    # ty: ignore[unresolved-attribute]
    _, areas, _ = _cells(result.target.mesh)
    assert np.all(areas > 0.0)
    assert np.sum(areas) == pytest.approx(1.0, abs=1e-12)
    assert areas.size > 8
    # ty: ignore[unresolved-attribute]
    assert result.target.audit.passed
    assert result.lineage_status == "unknown"
    # ty: ignore[unresolved-attribute]
    runtime = result.target.runtime
    assert "aggregate_output_bytes" in runtime.enforced_limits
    assert {"worker_address_space", "worker_peak_resident_audit"} & set(
        runtime.enforced_limits
    )
    assert "native_adaptation_intermediate_entities" in runtime.unenforced_limits
    evidence = result.evidence
    assert evidence.options == options
    assert (evidence.min_quality_allowed, evidence.min_quality_desired) == (0.25, 0.35)
    assert evidence.cell_count == areas.size
    assert 0.0 < evidence.minimum_quality <= evidence.maximum_quality <= 1.0
    assert 0.0 < evidence.minimum_length <= evidence.maximum_length
    assert len(result.partitions) == 1
    assert bool(np.all(result.partitions[0].cell_owned))
    # ty: ignore[unresolved-attribute]
    assert np.all(np.linalg.eigvalsh(np.asarray(result.metric.values)) > 0.0)


def test_omega_h_contracts() -> None:
    with pytest.raises(ValueError):
        M.OmegaHOptions(min_length_desired=2.0, max_length_desired=1.0)
    with pytest.raises(ValueError):
        M.OmegaHOptions(min_quality_allowed=0.5, min_quality_desired=0.4)
    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        M.OmegaHOptions(should_swap=1)
    mesh = _grid(1)
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())

    with pytest.raises(M.MeshingFailure) as failure:
        M.OmegaHProvider("nonexistent-omega-h").execute(
            source,
            _metric(mesh, _uniform(1.0)),
            limits=M.MeshingLimits(maximum_vertices=1),
        )

    assert failure.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    mesh = _grid(2)
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    moved = mesh.with_coordinates(
        np.asarray(mesh.coordinates) * 2.0, numeric_version="moved"
    )

    with pytest.raises(ValueError, match="Metric scope"):
        M.OmegaHProvider("nonexistent-omega-h").execute(
            source, _metric(moved, _uniform(0.5))
        )
    mesh = _grid(1)
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())

    with pytest.raises(M.MeshingFailure) as failure:
        M.OmegaHProvider("nonexistent-omega-h", mpi_launcher=()).execute(
            source, _metric(mesh, _uniform(1.0)), ranks=2
        )

    assert failure.value.category is M.MeshingFailureCategory.UNSUPPORTED_CAPABILITY


@requires_worker
def test_omega_h_rank_outputs_share_one_aggregate_byte_budget() -> None:
    for ranks in [1, 2]:
        launcher = _launcher_or_skip() if ranks > 1 else ("mpiexec",)
        mesh = _grid(2)
        source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())

        with M.OmegaHProvider(_worker(), mpi_launcher=launcher) as provider:
            with pytest.raises(M.MeshingFailure) as failure:
                provider.execute(
                    source,
                    _metric(mesh, _uniform(0.04)),
                    ranks=ranks,
                    limits=M.MeshingLimits(maximum_data_bytes=20_000),
                )

        assert failure.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED


@requires_worker
def test_omega_h_reuses_one_worker_session() -> None:
    mesh = _grid(2)
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())

    with M.OmegaHProvider(_worker()) as provider:
        first = provider.execute(source, _metric(mesh, _uniform(0.25)))
        second = provider.execute(source, _metric(mesh, _uniform(0.2)))
        launches = provider.worker(1).launches

    assert launches == 1
    assert first.evidence.session_id == second.evidence.session_id
    assert first.evidence.identity_id == second.evidence.identity_id
    assert (first.evidence.sequence, second.evidence.sequence) == (1, 2)
    assert second.evidence.cell_count > first.evidence.cell_count


@requires_worker
def test_omega_h_timeout_fails_and_next_call_relaunches() -> None:
    mesh = _grid(2)
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())

    with M.OmegaHProvider(_worker()) as provider:
        with pytest.raises(M.MeshingFailure) as failure:
            provider.execute(
                source,
                _metric(mesh, _uniform(0.003)),
                limits=M.MeshingLimits(maximum_wall_seconds=0.05),
            )
        timed_out_launches = provider.worker(1).launches
        recovered = provider.execute(source, _metric(mesh, _uniform(0.25)))
        launches = provider.worker(1).launches

    assert failure.value.category is M.MeshingFailureCategory.TIMED_OUT
    assert (timed_out_launches, launches) == (1, 2)
    # ty: ignore[unresolved-attribute]
    assert recovered.target.audit.passed


@requires_worker
def test_omega_h_transfers_linear_and_conservative_fields() -> None:
    mesh = _grid(8)
    source = _regions(mesh)
    vertex_ids = np.sort(np.asarray(mesh.vertex_global_ids))
    points = np.asarray(mesh.coordinates)[np.argsort(np.asarray(mesh.vertex_global_ids))]
    cell_ids, areas, centroids = _cells(mesh)
    order = np.argsort(cell_ids)
    density = (np.where(centroids[:, 0] < 0.5, 1.0, 5.0) + centroids[:, 1] ** 2)[order]
    linear = M.OmegaHField(
        "temperature",
        M.OmegaHFieldTransfer.LINEAR,
        _scope(mesh, 0, vertex_ids),
        np.column_stack((2.0 * points[:, 0] - 3.0 * points[:, 1] + 1.0, points[:, 1])),
    )
    conserved = M.OmegaHField(
        "density",
        M.OmegaHFieldTransfer.CONSERVE,
        _scope(mesh, 2, cell_ids[order]),
        density,
    )
    # Fine near x = 0, coarse elsewhere: the run both refines and coarsens.
    metric = _metric(mesh, lambda x: 0.03 + 0.4 * x[:, 0])

    with M.OmegaHProvider(_worker()) as provider:
        result = provider.execute(source, metric, fields=(linear, conserved))

    # ty: ignore[unresolved-attribute]
    target = result.target.mesh
    temperature, transferred = result.fields
    target_points = np.asarray(target.coordinates)[
        np.searchsorted(
            np.asarray(target.vertex_global_ids),
            np.asarray(temperature.scope.entity_ids),
            sorter=np.argsort(np.asarray(target.vertex_global_ids)),
        )
    ]
    np.testing.assert_allclose(
        np.asarray(temperature.values)[:, 0],
        2.0 * target_points[:, 0] - 3.0 * target_points[:, 1] + 1.0,
        rtol=0.0,
        atol=1e-12,
    )
    target_ids, target_areas, target_centroids = _cells(target)
    target_density = np.asarray(transferred.values)[
        np.searchsorted(np.asarray(transferred.scope.entity_ids), target_ids)
    ]
    before = np.sum(density * areas[order])
    after = np.sum(target_density * target_areas)
    assert after == pytest.approx(before, rel=1e-12)
    source_left = centroids[order, 0] < 0.5
    target_left = target_centroids[:, 0] < 0.5
    for source_side, target_side in (
        (source_left, target_left),
        (~source_left, ~target_left),
    ):
        assert np.sum(
            target_density[target_side] * target_areas[target_side]
        ) == pytest.approx(np.sum((density * areas[order])[source_side]), rel=1e-12)
    methods = {value.name: value.method for value in result.evidence.fields}
    assert methods == {
        "temperature": "OMEGA_H_LINEAR_INTERP",
        "density": "OMEGA_H_CONSERVE",
    }
    density_evidence = result.evidence.fields[1]
    # ty: ignore[not-subscriptable]
    assert float(density_evidence.integral_after[0]) == pytest.approx(
        # ty: ignore[not-subscriptable]
        float(density_evidence.integral_before[0]),
        rel=1e-12,
    )
    # ty: ignore[not-subscriptable]
    assert float(density_evidence.integral_before[0]) == pytest.approx(before, rel=1e-12)


@requires_worker
def test_omega_h_retains_regions_and_patches_by_class() -> None:
    mesh = _grid(8)
    source = _regions(mesh)
    # A coarser metric forces collapses next to the region interface and inlet.
    metric = _metric(mesh, lambda x: 0.2 + 0.15 * x[:, 1])

    with M.OmegaHProvider(_worker()) as provider:
        result = provider.execute(source, metric)

    target = result.target
    # ty: ignore[unresolved-attribute]
    ids, areas, centroids = _cells(target.mesh)
    # ty: ignore[unresolved-attribute]
    zones = {zone.name: zone for zone in target.zones}
    assert set(zones) == {"left", "right"}
    for name, inside in (
        ("left", centroids[:, 0] < 0.5),
        ("right", centroids[:, 0] > 0.5),
    ):
        members = np.isin(ids, np.asarray(zones[name].scope.entity_ids))
        assert np.array_equal(members, inside)
        assert np.sum(areas[members]) == pytest.approx(0.5, abs=1e-12)
    # ty: ignore[unresolved-attribute]
    (inlet,) = target.patches
    # ty: ignore[unresolved-attribute]
    edges = np.asarray(target.mesh.connectivity.edges)
    patch_edges = edges[
        np.isin(
            # ty: ignore[unresolved-attribute]
            np.asarray(target.mesh.entity_set(1).entity_ids),
            np.asarray(inlet.scope.entity_ids),
        )
    ]
    # ty: ignore[unresolved-attribute]
    endpoints = np.asarray(target.mesh.coordinates)[patch_edges]
    assert np.all(endpoints[:, :, 0] == 0.0)
    assert np.sum(np.abs(endpoints[:, 1, 1] - endpoints[:, 0, 1])) == pytest.approx(1.0)
    assert inlet.adjacent_zone_ids == (zones["left"].zone_id,)
    # ty: ignore[unresolved-attribute]
    assert target.audit.passed
    assert areas.size < 128
    classification = result.classification
    assert set(classification.cell_zones) == {"left", "right"}
    assert classification.facet_patches == (("inlet",),)


@requires_worker
def test_omega_h_distributed_partitions_and_gathered_carrier() -> None:
    launcher = _launcher_or_skip()
    mesh = _grid(4)
    source = _regions(mesh)
    metric = _metric(mesh, _uniform(0.08))

    with M.OmegaHProvider(_worker(), mpi_launcher=launcher) as provider:
        local = provider.execute(source, metric, ranks=2)
        gathered = provider.execute(source, metric, ranks=2, gather=True)
        launches = provider.worker(2).launches

    assert launches == 1
    assert local.target is None and local.distribution is None
    assert len(local.partitions) == 2
    owned_vertices = np.concatenate(
        [
            np.asarray(part.vertex_global_ids)[np.asarray(part.vertex_owned)]
            for part in local.partitions
        ]
    )
    owned_cells = np.concatenate(
        [
            np.asarray(part.cell_global_ids)[np.asarray(part.cell_owned)]
            for part in local.partitions
        ]
    )
    assert (
        np.unique(owned_vertices).size
        == owned_vertices.size
        == local.evidence.vertex_count
    )
    assert np.unique(owned_cells).size == owned_cells.size == local.evidence.cell_count
    for part in local.partitions:
        ghosts = np.asarray(part.cell_ghosts)
        assert 0 < np.count_nonzero(ghosts) < ghosts.size
        assert np.all(np.asarray(part.cell_owner_ranks)[ghosts] != part.rank)
        assert np.all(
            np.asarray(part.vertex_owner_ranks)[np.asarray(part.vertex_ghosts)]
            != part.rank
        )

    target = gathered.target
    # ty: ignore[unresolved-attribute]
    assert target.audit.passed
    # ty: ignore[unresolved-attribute]
    _, areas, _ = _cells(target.mesh)
    assert np.sum(areas) == pytest.approx(1.0, abs=1e-12)
    # ty: ignore[unresolved-attribute]
    assert {zone.name for zone in target.zones} == {"left", "right"}
    distribution = gathered.distribution
    # ty: ignore[unresolved-attribute]
    assert distribution.partition.part_count == 2
    # ty: ignore[unresolved-attribute]
    for part, halo in zip(gathered.partitions, distribution.halo_global_ids, strict=True):
        ghost_ids = np.asarray(part.cell_global_ids)[np.asarray(part.cell_ghosts)]
        assert np.array_equal(np.sort(ghost_ids), np.asarray(halo))
