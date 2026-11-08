"""Cold archive round trips of uncertified results and route adaptation results."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.meshing import (
    CellMeshingResult,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingScope,
    MeshMetricField,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from phydrax.meshing._model_codec import (
    MODEL_ARCHIVE_LIMITS,
    read_model_archive,
    read_model_artifact,
    restore_model_archive_value,
    write_model_archive,
    write_model_artifact,
)


def _certified(mesh: CellMesh, /) -> CellMeshingResult:
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _triangle_grid(columns: int, rows: int, /) -> CellMesh:
    xs = np.linspace(0.0, 1.0, columns + 1)
    ys = np.linspace(0.0, 1.0, rows + 1)
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2))
    cells = []
    for j, i in itertools.product(range(rows), range(columns)):
        a = j * (columns + 1) + i
        b, c, d = a + 1, a + columns + 2, a + columns + 1
        cells.extend(((a, b, c), (a, c, d)))
    return CellMesh.from_triangles(points, np.asarray(cells, dtype=np.int32))


def _kuhn_cube() -> CellMesh:
    size = 2
    axis = np.linspace(0.0, 1.0, size + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = points.reshape((-1, 3))
    cells = []
    for origin in itertools.product(range(size), repeat=3):
        for order in itertools.permutations(range(3)):
            corner = list(origin)
            path = [(corner[0] * (size + 1) + corner[1]) * (size + 1) + corner[2]]
            for direction in order:
                corner[direction] += 1
                path.append((corner[0] * (size + 1) + corner[1]) * (size + 1) + corner[2])
            cells.append(path)
    rows = np.asarray(cells, dtype=np.int32)
    corners = points[rows]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    rows[negative] = rows[negative][:, [1, 0, 2, 3]]
    return CellMesh.from_tetrahedra(points, rows)


def _metric_request(source: CellMeshingResult, diagonal: tuple[float, ...], /) -> Any:
    mesh = source.mesh
    vertices = mesh.entity_set(0)
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        np.asarray(vertices.entity_ids, dtype=np.int64),
    )
    values = np.broadcast_to(
        np.diag(diagonal), (vertices.entity_ids.size, len(diagonal), len(diagonal))
    ).copy()
    return MetricMeshAdaptation(
        MeshMetricField(
            scope, values, minimum_size=1.0e-3, maximum_size=10.0, maximum_anisotropy=10.0
        )
    )


def _adapt(source: CellMeshingResult, request: Any, route: Any, /) -> Any:
    policy = MeshAdaptationPolicy(route, relocation=False)
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )


def _round_trip(path: Path, value: Any, expected_type: type, /) -> Any:
    write_model_artifact(path, value, limits=MODEL_ARCHIVE_LIMITS)
    return read_model_artifact(
        path,
        expected_type=expected_type,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )


def _refuses_tamper(
    path: Path,
    expected_type: type,
    field_path: tuple[str | int, ...],
    change: Callable[[Any], Any],
    /,
) -> None:
    manifest, arrays = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)
    node = manifest["model_recipe"]
    for name in field_path:
        node = node["items"][name] if isinstance(name, int) else node["fields"][name]
    node["value"] = change(node["value"])
    changed = write_model_archive(
        path.with_name(f"tampered-{'-'.join(map(str, field_path))}.phx"),
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            changed,
            expected_type=expected_type,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )


def _assert_same_adaptation(restored: Any, original: Any, /) -> None:
    assert type(restored) is MeshAdaptationResult
    assert restored.result_id == original.result_id
    assert restored.prepared_id == original.prepared_id
    assert restored.status is original.status
    assert restored.source.result_id == original.source.result_id
    assert restored.target.result_id == original.target.result_id
    identity = "result_id" if hasattr(original.evidence, "result_id") else "evidence_id"
    assert getattr(restored.evidence, identity) == getattr(original.evidence, identity)
    for name in ("transition", "metric", "transfer", "hierarchy"):
        stored, rebuilt = getattr(original, name), getattr(restored, name)
        assert (stored is None) == (rebuilt is None)
    if original.transition is not None:
        assert restored.transition.transition_id == original.transition.transition_id
    if original.metric is not None:
        assert restored.metric.metric_id == original.metric.metric_id
    if original.transfer is not None:
        assert restored.transfer.transfer_id == original.transfer.transfer_id
    if original.hierarchy is not None:
        assert restored.hierarchy.hierarchy_id == original.hierarchy.hierarchy_id


def _flip_hex(value: str, /) -> str:
    return ("1" if value[0] == "0" else "0") + value[1:]


def test_uncertified_result_round_trips_and_refuses_changed_audit(
    tmp_path: Path,
) -> None:
    original = _certified(_triangle_grid(2, 2))
    assert original.certification is None
    path = tmp_path / "uncertified.phx"
    restored = _round_trip(path, original, CellMeshingResult)
    assert restored.certification is None
    assert restored.result_id == original.result_id
    assert restored.audit.report_id == original.audit.report_id
    assert restored.quality.report_id == original.quality.report_id
    _refuses_tamper(path, CellMeshingResult, ("audit", "report_id"), _flip_hex)
    _refuses_tamper(
        path, CellMeshingResult, ("audit", "policy", "minimum_measure"), lambda _: 0.5
    )
    _refuses_tamper(path, CellMeshingResult, ("result_id",), _flip_hex)


def test_uncertified_result_restores_its_custom_audit_policy(tmp_path: Path) -> None:
    from phydrax.meshing import CellMeshAuditPolicy

    policy = CellMeshAuditPolicy(minimum_mean_ratio=0.01)
    original = phx.meshing.certify_cell_mesh(
        _triangle_grid(2, 2),
        phx.SpatialCoordinateContract.si(),
        audit_policy=policy,
    )
    assert original.audit.policy_id == policy.policy_id
    path = tmp_path / "custom.phx"
    restored = _round_trip(path, original, CellMeshingResult)
    assert restored.result_id == original.result_id
    assert restored.audit.report_id == original.audit.report_id
    assert restored.audit.policy.policy_id == policy.policy_id
    assert restored.audit.policy.minimum_mean_ratio == 0.01
    _refuses_tamper(
        path,
        CellMeshingResult,
        ("audit", "policy", "minimum_mean_ratio"),
        lambda value: 0.02,
    )
    _refuses_tamper(path, CellMeshingResult, ("audit", "policy", "policy_id"), _flip_hex)


def test_overset_registration_with_uncertified_parts_restores(tmp_path: Path) -> None:
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing import (
        MeshAssembly,
        MeshPart,
        OversetPartSpec,
        OversetPolicy,
        prepare_overset_connectivity,
    )
    from phydrax.meshing._model_codec import (
        overset_registration_recipe,
        restore_overset_registration,
    )

    def certified(points: Any, /) -> CellMeshingResult:
        mesh = CellMesh.from_triangles(
            np.asarray(points, dtype=np.float64), np.asarray(((0, 1, 2),), np.int32)
        )
        return _certified(mesh)

    receptor_result = certified(((0.2, 0.2), (0.4, 0.2), (0.2, 0.4)))
    donor = MeshPart("donor", certified(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))))
    receptor = MeshPart("receptor", receptor_result)
    facets = receptor_result.mesh.entity_set(1)
    boundary = np.asarray(facets.subset("boundary").mask, dtype=np.bool_)
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor, receptor)),
        (
            OversetPartSpec("donor"),
            OversetPartSpec(
                "receptor",
                boundary=receptor.scope(
                    1, np.asarray(facets.entity_ids, dtype=np.int64)[boundary]
                ),
            ),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    values: dict[str, Any] = {
        "overset:connectivity": overset_registration_recipe(connectivity.registration())
    }
    for item in connectivity.assembly.parts:
        values[f"overset:part:{item.name}"] = item.carrier
    composition = Composition(
        tuple(
            CompositionEntry(
                values.get(entry.entry_id, entry.value),
                entry_id=entry.entry_id,
                role=entry.role,
                owner_id=entry.owner_id,
                structure_id=entry.structure_id,
                revision_id=entry.revision_id,
                semantics_id=entry.semantics_id,
                dependencies=entry.dependencies,
            )
            for entry in connectivity.composition_entries()
        ),
        boundary_id="test:overset",
    )
    restored = _round_trip(tmp_path / "overset.phx", composition, Composition)
    recipe = restored.value("overset:connectivity")
    parts = tuple(
        MeshPart(name, restored.value(f"overset:part:{name}"))
        for name, _ in recipe.part_bindings
    )
    for item in parts:
        assert isinstance(item.carrier, CellMeshingResult)
        assert item.carrier.certification is None
    rebuilt = restore_overset_registration(recipe, parts).prepare()
    rebuilt.require_complete()
    assert rebuilt.connectivity_id == connectivity.connectivity_id
    assert rebuilt.assembly.assembly_id == connectivity.assembly.assembly_id


@pytest.mark.parametrize(
    "route",
    (MeshAdaptationRoute.NATIVE_METRIC_3D, MeshAdaptationRoute.DEVICE_METRIC_3D),
)
def test_tetra_metric_adaptation_round_trips_and_refuses_tamper(
    tmp_path: Path, route: MeshAdaptationRoute
) -> None:
    source = _certified(_kuhn_cube())
    original = _adapt(source, _metric_request(source, (4.0, 3.8, 3.6)), route)
    assert original.status is MeshAdaptationStatus.COMPLETE
    assert original.transition is not None
    path = tmp_path / f"{route.value}.phx"
    _assert_same_adaptation(_round_trip(path, original, MeshAdaptationResult), original)
    _refuses_tamper(path, MeshAdaptationResult, ("prepared_id",), _flip_hex)
    _refuses_tamper(path, MeshAdaptationResult, ("evidence", "evidence_id"), _flip_hex)
    _refuses_tamper(
        path, MeshAdaptationResult, ("transition", "transition_id"), _flip_hex
    )


def test_metric_cube_archives_are_byte_identical_without_wall_time(
    tmp_path: Path,
) -> None:
    source = _certified(_kuhn_cube())
    request = _metric_request(source, (4.0, 3.8, 3.6))
    route = MeshAdaptationRoute.NATIVE_METRIC_3D
    first, second = (_adapt(source, request, route) for _ in range(2))
    assert first.result_id == second.result_id
    assert first.elapsed_seconds is not None and second.elapsed_seconds is not None
    paths = [
        write_model_artifact(
            tmp_path / f"run-{index}.phx", value, limits=MODEL_ARCHIVE_LIMITS
        )
        for index, value in enumerate((first, second))
    ]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    restored = read_model_artifact(
        paths[0],
        expected_type=MeshAdaptationResult,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    _assert_same_adaptation(restored, first)
    assert restored.elapsed_seconds is None
    # An archive that carries wall time is not canonical.
    _refuses_tamper(paths[0], MeshAdaptationResult, ("elapsed_seconds",), lambda _: 0.25)


def test_native_metric_2d_adaptation_round_trips(tmp_path: Path) -> None:
    source = _certified(_triangle_grid(3, 3))
    original = _adapt(
        source,
        _metric_request(source, (36.0, 36.0)),
        MeshAdaptationRoute.NATIVE_METRIC_2D,
    )
    assert original.transition is not None
    path = tmp_path / "metric-2d.phx"
    _assert_same_adaptation(_round_trip(path, original, MeshAdaptationResult), original)
    _refuses_tamper(path, MeshAdaptationResult, ("metric", "metric_id"), _flip_hex)


def test_bisection_adaptation_round_trips_with_hierarchy(tmp_path: Path) -> None:
    source = _certified(_triangle_grid(2, 2))
    cell_ids = np.asarray(source.mesh.blocks[0].global_ids, dtype=np.int64)
    request = MarkedMeshAdaptation(cell_ids[:1], np.zeros((0,), dtype=np.int64))
    original = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            request,
            policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION),
        )
    )
    assert original.hierarchy is not None and original.transition is not None
    path = tmp_path / "bisection.phx"
    _assert_same_adaptation(_round_trip(path, original, MeshAdaptationResult), original)
    _refuses_tamper(path, MeshAdaptationResult, ("hierarchy", "hierarchy_id"), _flip_hex)


def test_distributed_bisection_restores_its_distribution_transition(
    tmp_path: Path,
) -> None:
    from phydrax.meshing import (
        MeshPart,
        MeshPartitionKind,
        MeshPartitionPolicy,
        prepare_mesh_distribution,
    )

    source = _certified(_triangle_grid(4, 4))
    partition = MeshPartitionPolicy(MeshPartitionKind.MORTON, 2)
    distribution = prepare_mesh_distribution(MeshPart("domain", source), policy=partition)
    cell_ids = np.asarray(source.mesh.blocks[0].global_ids, dtype=np.int64)
    original = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(cell_ids[:3], np.zeros((0,), dtype=np.int64)),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION,
                distribution=distribution,
                partition_policy=partition,
            ),
        )
    )
    assert original.distribution is not None
    path = tmp_path / "distributed.phx"
    restored = _round_trip(path, original, MeshAdaptationResult)
    _assert_same_adaptation(restored, original)
    assert restored.distribution.transition_id == original.distribution.transition_id
    _refuses_tamper(
        path, MeshAdaptationResult, ("distribution", "transition_id"), _flip_hex
    )
    _refuses_tamper(
        path,
        MeshAdaptationResult,
        ("distribution", "rebalanced"),
        lambda value: not value,
    )


def _l_polyhedra() -> CellMesh:
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    )
    corners = list(itertools.product((0, 1), repeat=3))
    cube = (
        corners[0],
        corners[4],
        corners[6],
        corners[2],
        corners[1],
        corners[5],
        corners[7],
        corners[3],
    )
    points: list[tuple[int, int, int]] = []
    rows: dict[tuple[int, int, int], int] = {}
    cells = []
    for dx, dy in ((0, 0), (1, 0), (0, 1)):
        local = []
        for x, y, z in cube:
            point = (x + dx, y + dy, z)
            if point not in rows:
                rows[point] = len(points)
                points.append(point)
            local.append(rows[point])
        cells.append(
            tuple(np.asarray([local[i] for i in face], dtype=np.int32) for face in faces)
        )
    return CellMesh.from_polyhedra(
        np.asarray(points, dtype=np.float64),
        cells,
        vertex_global_ids=np.arange(100, 100 + len(points), dtype=np.int64),
        cell_global_ids=np.asarray((10, 20, 30), dtype=np.int64),
    )


def test_polyhedral_plane_split_round_trips_with_conservative_remap(
    tmp_path: Path,
) -> None:
    from phydrax.meshing import (
        PolyhedralAdaptationOperation,
        PolyhedralAdaptationRequest,
    )

    source = _certified(_l_polyhedra())
    original = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            PolyhedralAdaptationRequest(
                PolyhedralAdaptationOperation.PLANE_SPLIT,
                planes=np.asarray(((0.0, 1.0, 0.0, 0.5),), dtype=np.float64),
            ),
            policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_POLYHEDRAL),
        )
    )
    assert original.conservative_remap is not None
    path = tmp_path / "polyhedral.phx"
    restored = _round_trip(path, original, MeshAdaptationResult)
    _assert_same_adaptation(restored, original)
    assert restored.conservative_remap.plan_id == original.conservative_remap.plan_id
    _refuses_tamper(
        path, MeshAdaptationResult, ("conservative_remap", "plan_id"), _flip_hex
    )
    _refuses_tamper(path, MeshAdaptationResult, ("hierarchy", 7), _flip_hex)

    manifest, arrays = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)

    def tamper_parent_map(node: Any, /) -> bool:
        if isinstance(node, dict):
            if str(node.get("type", "")).endswith(":_PolyhedralExactCoordinateSource"):
                node["fields"]["parent_map_id"]["value"] = _flip_hex(
                    node["fields"]["parent_map_id"]["value"]
                )
                return True
            return any(tamper_parent_map(child) for child in node.values())
        if isinstance(node, list):
            return any(tamper_parent_map(child) for child in node)
        return False

    assert tamper_parent_map(manifest["model_recipe"])
    forged = write_model_archive(
        tmp_path / "forged-parent-map.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            forged,
            expected_type=MeshAdaptationResult,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )


def test_polyhedral_exact_coordinate_source_refuses_changed_inputs() -> None:
    import copy

    from phydrax.meshing import (
        PolyhedralAdaptationOperation,
        PolyhedralAdaptationRequest,
    )
    from phydrax.meshing._polyhedral_adaptation import _PolyhedralExactCoordinateSource

    source = _certified(_l_polyhedra())
    target = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            PolyhedralAdaptationRequest(
                PolyhedralAdaptationOperation.PLANE_SPLIT,
                planes=np.asarray(((0.0, 1.0, 0.0, 0.5),), dtype=np.float64),
            ),
            policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_POLYHEDRAL),
        )
    ).target
    stored = target.geometry.source_map
    assert type(stored) is _PolyhedralExactCoordinateSource
    rebuilt = _PolyhedralExactCoordinateSource.from_stored_fields(
        stored, mesh=target.mesh
    )
    assert rebuilt.source_map_id == stored.source_map_id
    for name, value in (
        ("parent_map_id", _flip_hex(stored.parent_map_id)),
        ("vertex_ids", np.asarray(stored.vertex_ids)[::-1].copy()),
    ):
        forged = copy.copy(stored)
        object.__setattr__(forged, name, value)
        with pytest.raises(ValueError, match="exact coordinate source changed"):
            _PolyhedralExactCoordinateSource.from_stored_fields(forged, mesh=target.mesh)


def test_source_preserving_adaptation_round_trips(tmp_path: Path) -> None:
    source = _certified(_triangle_grid(2, 2))
    request = MarkedMeshAdaptation(
        np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    )
    original = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            request,
            policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION),
        )
    )
    assert original.status is MeshAdaptationStatus.UNCHANGED
    restored = _round_trip(tmp_path / "unchanged.phx", original, MeshAdaptationResult)
    _assert_same_adaptation(restored, original)
    assert restored.target is restored.source


def test_mmg_provider_adaptation_restores_without_rerunning_mmg(
    tmp_path: Path,
) -> None:
    """A provider result is plain retained data; restore never runs Mmg."""
    from phydrax.interchange import AdapterLoss, AdapterReport, AdapterStatus
    from phydrax.meshing._adaptation import (
        _compliance,
        _RouteOutcome,
        _StoredAdaptationBinding,
        _unknown_lineage,
    )
    from phydrax.meshing._lineage import CellMeshTransition, MeshTransitionKind
    from phydrax.meshing.providers._mmg import (
        MmgAdaptationResult,
        MmgProvider,
        MmgReference,
        MmgReferenceRetention,
        MmgSessionEvidence,
    )

    source = _certified(_triangle_grid(2, 2))
    accepted = _certified(_triangle_grid(3, 3))
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "mmg2d",
        "phydrax-cell-mesh",
        source_id=source.mesh.mesh_id,
        target_id=accepted.mesh.mesh_id,
        coordinate_mapping=("identity",),
        assumptions=("Entity correspondence is not supplied.",),
        losses=(
            AdapterLoss(
                "cells",
                "import",
                "transformed",
                "Mmg renumbers every cell.",
                changes_interpretation=False,
            ),
        ),
    )
    target = CellMeshingResult(
        accepted.mesh,
        accepted.geometry,
        accepted.coordinate_contract,
        accepted.audit,
        accepted.quality,
        accepted.compliance,
        accepted.trace,
        accepted.provider,
        accepted.runtime,
        accepted.derivative_mode,
        accepted.provenance,
        adapter_reports=(report,),
    )
    vertex_count = target.mesh.entity_set(0).entity_ids.size
    metric = MeshMetricField(
        MmgProvider.vertex_scope(target.mesh),
        np.broadcast_to(4.0 * np.eye(2), (vertex_count, 2, 2)).copy(),
        minimum_size=0.1,
        maximum_size=1.0,
    )
    evidence = MmgAdaptationResult(
        "fabricated-mmg-plan",
        target,
        metric,
        "tensor",
        None,
        MmgReferenceRetention(
            (MmgReference("region", 1, ("domain",), source_count=8, target_count=18),),
            required_vertices=0,
            retained_required_vertices=0,
        ),
        MmgSessionEvidence(
            "fabricated-session",
            "fabricated-worker",
            sequence=1,
            ranks=1,
            peak_rss_bytes=1,
            elapsed_seconds=0.25,
        ),
    )
    lineage = _unknown_lineage(source.mesh, target.mesh)
    binding = _StoredAdaptationBinding(
        MeshAdaptationRoute.MMG, source, "fabricated-prepared"
    )
    original = MeshAdaptationResult(
        binding,
        _RouteOutcome(
            MeshAdaptationStatus.COMPLETE,
            target,
            CellMeshTransition(
                source.mesh.mesh_id,
                source.mesh.topology_id,
                target,
                lineage,
                MeshTransitionKind.REMESH,
            ),
            lineage,
            None,
            None,
            metric,
            evidence,
            None,
        ),
        _compliance(binding, target),
        None,
        0.25,
    )
    path = tmp_path / "mmg.phx"
    restored = _round_trip(path, original, MeshAdaptationResult)
    _assert_same_adaptation(restored, original)
    assert restored.elapsed_seconds is None
    assert restored.evidence.session.elapsed_seconds is None
    assert restored.target.adapter_reports[0].report_id == report.report_id
    _refuses_tamper(
        path, MeshAdaptationResult, ("evidence", "session", "evidence_id"), _flip_hex
    )
    _refuses_tamper(
        path,
        MeshAdaptationResult,
        ("evidence", "session", "elapsed_seconds"),
        lambda _: 0.25,
    )
    _refuses_tamper(
        path,
        MeshAdaptationResult,
        ("target", "adapter_reports", 0, "report_id"),
        _flip_hex,
    )


def _omega_h_adaptation() -> tuple[MeshAdaptationResult, dict[str, Any]]:
    """A fabricated serial Omega_h result; the host has no Omega_h worker."""
    from phydrax._fingerprint import canonical_fingerprint
    from phydrax.meshing._adaptation import (
        _compliance,
        _RouteOutcome,
        _StoredAdaptationBinding,
        _unknown_lineage,
    )
    from phydrax.meshing._lineage import CellMeshTransition, MeshTransitionKind
    from phydrax.meshing.providers._omega_h import (
        _entity_scope,
        _GatheredTarget,
        _RankArrays,
        OmegaHAdaptationEvidence,
        OmegaHAdaptationResult,
        OmegaHClassification,
        OmegaHField,
        OmegaHFieldEvidence,
        OmegaHFieldTransfer,
        OmegaHOptions,
        OmegaHPartition,
        OmegaHTransferredField,
    )

    source = _certified(_triangle_grid(2, 2))
    target = _certified(_triangle_grid(3, 3))
    source_vertices = np.asarray(source.mesh.entity_set(0).entity_ids, dtype=np.int64)
    source_cells = np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
    vertices = np.asarray(target.mesh.entity_set(0).entity_ids, dtype=np.int64)
    cells = np.asarray(target.mesh.entity_set(2).entity_ids, dtype=np.int64)
    vertex_scope = _entity_scope(target.mesh, 0, vertices)
    cell_scope = _entity_scope(target.mesh, 2, cells)
    linear = OmegaHField(
        "temperature",
        OmegaHFieldTransfer.LINEAR,
        _entity_scope(source.mesh, 0, source_vertices),
        np.linspace(0.0, 1.0, source_vertices.size),
    )
    conserve = OmegaHField(
        "density",
        OmegaHFieldTransfer.CONSERVE,
        _entity_scope(source.mesh, 2, source_cells),
        np.full(source_cells.size, 2.0),
    )
    options = OmegaHOptions()
    record = {
        "cell_count": int(cells.size),
        "classification": {"generated_facet_classes": 1},
        "facet_count": 0,
        "fields": [
            {"name": "field-0", "method": "OMEGA_H_LINEAR_INTERP"},
            {
                "name": "field-1",
                "method": "OMEGA_H_CONSERVE",
                "integral_before": [2.0],
                "integral_after": [2.0],
            },
        ],
        "iterations": 3,
        "maximum_length": 1.25,
        "maximum_quality": 1.0,
        "minimum_length": 0.75,
        "minimum_quality": 0.5,
        "options": {
            **options.worker_record()["adapt"],
            "feature_angle": options.feature_angle,
            "gradation_rate": options.gradation_rate,
            "maximum_iterations": options.maximum_iterations,
            "min_quality_allowed": 0.3,
            "min_quality_desired": 0.4,
        },
        "ranks": 1,
        "vertex_count": int(vertices.size),
    }
    field_evidence = (
        OmegaHFieldEvidence(
            linear.name,
            linear.transfer,
            "OMEGA_H_LINEAR_INTERP",
            None,
            None,
            None,
            linear.field_id,
        ),
        OmegaHFieldEvidence(
            conserve.name,
            conserve.transfer,
            "OMEGA_H_CONSERVE",
            conserve.diffusion_tolerance,
            np.asarray([2.0]),
            np.asarray([2.0]),
            conserve.field_id,
        ),
    )
    evidence = OmegaHAdaptationEvidence(
        options,
        field_evidence,
        record,
        session_id="fabricated-session",
        identity_id="fabricated-worker",
        sequence=1,
        peak_rss_bytes=1,
    )
    # The canonical record hashes exactly as the worker's JSON record.
    assert evidence.evidence_id == canonical_fingerprint(
        {
            "kind": "omega-h-adaptation-evidence",
            "options": options.options_id,
            "record": record,
            "fields": [value.evidence_id for value in field_evidence],
            "identity": "fabricated-worker",
        }
    )
    metric = MeshMetricField(
        vertex_scope,
        np.broadcast_to(4.0 * np.eye(2), (vertices.size, 2, 2)).copy(),
        minimum_size=0.1,
        maximum_size=1.0,
    )
    transferred = (
        OmegaHTransferredField(
            "temperature",
            OmegaHFieldTransfer.LINEAR,
            vertex_scope,
            np.linspace(0.0, 1.0, vertices.size),
        ),
        OmegaHTransferredField(
            "density", OmegaHFieldTransfer.CONSERVE, cell_scope, np.full(cells.size, 2.0)
        ),
    )
    rows = target.mesh.blocks[0].vertices
    partition = OmegaHPartition(
        0,
        _RankArrays(
            vertices,
            np.asarray(target.mesh.coordinates, dtype=np.float64),
            np.asarray(metric.values),
            np.zeros(vertices.size, dtype=np.int32),
            np.arange(vertices.size, dtype=np.int32),
            cells,
            np.asarray(rows, dtype=np.int32),
            np.zeros(cells.size, dtype=np.int32),
            np.zeros(cells.size, dtype=np.int32),
            np.arange(cells.size, dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            tuple(np.asarray(field.values) for field in transferred),
        ),
    )
    classification = OmegaHClassification(
        cell_blocks=(target.mesh.blocks[0].name,),
        cell_zones=(None,),
        cell_labels=((),),
        facet_patches=(),
        facet_zones=(),
        facet_labels=(),
        feature_angle=options.feature_angle,
    )
    omega_h = OmegaHAdaptationResult(
        source,
        "fabricated-input-metric",
        (partition,),
        classification,
        evidence,
        _GatheredTarget(
            target,
            metric,
            transferred,
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
        ),
        None,
        target.provider,
        target.runtime,
        target.provenance,
    )
    lineage = _unknown_lineage(source.mesh, target.mesh)
    binding = _StoredAdaptationBinding(
        MeshAdaptationRoute.OMEGA_H, source, "fabricated-prepared"
    )
    result = MeshAdaptationResult(
        binding,
        _RouteOutcome(
            MeshAdaptationStatus.COMPLETE,
            target,
            CellMeshTransition(
                source.mesh.mesh_id,
                source.mesh.topology_id,
                target,
                lineage,
                MeshTransitionKind.REMESH,
            ),
            lineage,
            None,
            None,
            metric,
            omega_h,
            None,
        ),
        _compliance(binding, target),
        None,
        0.25,
    )
    return result, record


def _refuses_record_tamper(path: Path, /) -> None:
    manifest, arrays = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)
    record = manifest["model_recipe"]["fields"]["evidence"]["fields"]["evidence"][
        "fields"
    ]["record"]
    assert record["kind"] == "frozendict"
    (value,) = (item for key, item in record["items"] if key["value"] == "iterations")
    value["value"] += 1
    changed = write_model_archive(
        path.with_name("tampered-record.phx"),
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            changed,
            expected_type=MeshAdaptationResult,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )


def test_omega_h_adaptation_restores_without_rerunning_omega_h(tmp_path: Path) -> None:
    from phydrax.meshing.providers._omega_h import OmegaHAdaptationResult

    original, record = _omega_h_adaptation()
    path = tmp_path / "omega-h.phx"
    restored = _round_trip(path, original, MeshAdaptationResult)
    _assert_same_adaptation(restored, original)
    evidence, stored = restored.evidence, original.evidence
    assert isinstance(evidence, OmegaHAdaptationResult)
    assert isinstance(stored, OmegaHAdaptationResult)
    assert evidence.result_id == stored.result_id
    assert evidence.evidence.evidence_id == stored.evidence.evidence_id
    assert evidence.evidence.record == stored.evidence.record
    assert evidence.evidence.record["fields"][1]["integral_after"] == (2.0,)
    assert [value.field_id for value in evidence.evidence.fields] == [
        value.field_id for value in stored.evidence.fields
    ]
    assert [value.field_id for value in evidence.fields] == [
        value.field_id for value in stored.fields
    ]
    assert evidence.partitions[0].partition_id == stored.partitions[0].partition_id
    assert evidence.metric is not None and stored.metric is not None
    assert evidence.metric.metric_id == stored.metric.metric_id
    assert record["iterations"] == evidence.evidence.iterations

    _refuses_record_tamper(path)
    for field_path in (
        ("evidence", "evidence", "fields", 0, "field_id"),
        ("evidence", "partitions", 0, "partition_id"),
        ("evidence", "classification", "classification_id"),
    ):
        _refuses_tamper(path, MeshAdaptationResult, field_path, _flip_hex)


def test_two_epoch_material_chain_restores_every_epoch_map(tmp_path: Path) -> None:
    from tests.unit.meshing.test_transition_acceptance import (
        _bisection_hierarchy,
        _curved_material_refinement_case,
    )

    case = _curved_material_refinement_case()
    route, refined = case[3], case[5]
    refined_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64)
            for block in refined.target.mesh.blocks
        )
    )
    coarsened = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            refined.target,
            MarkedMeshAdaptation(
                (), refined_ids, hierarchy=_bisection_hierarchy(refined)
            ),
            policy=route,
            prior_transition=refined.transition,
        )
    )
    assert coarsened.transition is not None
    chain = coarsened.transition.volume_material_chain
    assert chain is not None and len(chain.maps) == 2
    for name, original in (("refined", refined), ("coarsened", coarsened)):
        path = tmp_path / f"{name}.phx"
        restored = _round_trip(path, original, MeshAdaptationResult)
        _assert_same_adaptation(restored, original)
        assert restored.transition is not None and original.transition is not None
        restored_chain = restored.transition.volume_material_chain
        original_chain = original.transition.volume_material_chain
        assert restored_chain is not None and original_chain is not None
        assert restored_chain.chain_id == original_chain.chain_id
    # The first epoch's source state exists only inside the chain; forge the
    # identity of its retained coverage certificate.
    intermediate = chain.maps[0].source_coverage.certificate_id
    manifest, arrays = read_model_archive(
        tmp_path / "coarsened.phx", limits=MODEL_ARCHIVE_LIMITS
    )
    node = manifest["model_recipe"]
    for name in ("transition", "volume_material_chain", "maps"):
        node = node["fields"][name]
    forged = 0

    def forge(item: Any, /) -> None:
        nonlocal forged
        if isinstance(item, dict):
            if item.get("value") == intermediate:
                item["value"] = _flip_hex(intermediate)
                forged += 1
            for child in item.values():
                forge(child)
        elif isinstance(item, list):
            for child in item:
                forge(child)

    forge(node["items"][0]["fields"]["source_coverage"])
    assert forged
    changed = write_model_archive(
        tmp_path / "forged-coverage.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            changed,
            expected_type=MeshAdaptationResult,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )


def test_surface_metric_adaptation_round_trips_with_its_chart_witness(
    tmp_path: Path,
) -> None:
    from phydrax.meshing._implicit_association_transfer import (
        ImplicitAssociationTransfer,
    )
    from phydrax.meshing._surface_metric import SurfaceMetricOutcome
    from tests.unit.meshing.test_surface_metric import (
        _coarsening_metric,
        _implicit_sphere_source,
    )

    profile, source, _ = _implicit_sphere_source(0.5)
    values = _coarsening_metric(source.mesh)
    vertex_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(vertex_ids, kind="stable")
    metric = MeshMetricField(
        MeshingScope(
            source.mesh.mesh_id,
            source.mesh.numeric_version,
            MeshingEntityKind.MESH,
            0,
            source.mesh.entity_set(0).entity_set_id,
            vertex_ids[order],
        ),
        values[order],
        minimum_size=0.1,
        maximum_size=0.6,
        maximum_anisotropy=2.0,
    )
    original = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MetricMeshAdaptation(metric),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_METRIC_SURFACE,
                association_transfer=ImplicitAssociationTransfer(profile),
                relocation=True,
            ),
        )
    )
    assert isinstance(original.evidence, SurfaceMetricOutcome)
    assert original.surface_chart_witness is not None
    path = tmp_path / "surface.phx"
    restored = _round_trip(path, original, MeshAdaptationResult)
    assert restored.result_id == original.result_id
    assert restored.target.result_id == original.target.result_id
    assert restored.transition is not None and original.transition is not None
    assert restored.transition.transition_id == original.transition.transition_id
    assert restored.metric is not None and original.metric is not None
    assert restored.metric.metric_id == original.metric.metric_id
    assert (
        restored.evidence.evidence.evidence_id == original.evidence.evidence.evidence_id
    )
    _refuses_tamper(
        path,
        MeshAdaptationResult,
        ("evidence", 3, "evidence_id"),
        _flip_hex,
    )
    _refuses_tamper(path, MeshAdaptationResult, ("evidence", 5), lambda value: value + 1)


def test_hp_adaptation_cold_archive_and_coarsening(tmp_path: Path) -> None:
    import gc
    import subprocess
    import sys
    import weakref

    import jax.numpy as jnp

    from phydrax.discretization import CellBlock
    from phydrax.discretization.fem import (
        finite_element_hp_transfer_plan,
        initial_finite_element_hp_topology,
        prepare_finite_element_hp_epoch,
    )
    from phydrax.lifecycle._composition_rebind import Composition, CompositionEntry

    mesh = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            CellBlock(
                "quad",
                "quadrilateral",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((10,), dtype=np.int64),
            ),
        ),
    )
    topology, geometry = initial_finite_element_hp_topology(mesh, 2, 16)
    epoch = prepare_finite_element_hp_epoch(topology, geometry, "u", conformity="L2")
    source = _certified(epoch.mesh)
    request = MarkedMeshAdaptation(np.asarray((10,), dtype=np.int64), hierarchy=epoch)
    epoch_id = epoch.epoch_id
    epoch_reference = weakref.ref(epoch)
    result = _adapt(source, request, MeshAdaptationRoute.HP)
    del request, epoch
    gc.collect()
    assert epoch_reference() is None
    path = tmp_path / "hp-refined.phx"
    restored = _round_trip(path, result, MeshAdaptationResult)
    assert restored.result_id == result.result_id
    source_epoch = restore_model_archive_value(restored.evidence.source_epoch)
    assert source_epoch.epoch_id == epoch_id
    refined = restored.evidence.refinement
    target_epoch = prepare_finite_element_hp_epoch(
        refined.topology, refined.geometry, "u", conformity="L2"
    )
    transfer = finite_element_hp_transfer_plan(
        source_epoch,
        target_epoch,
        refined.lineage,
        "u",
        "h-refinement",
    )
    values = jnp.zeros((topology.capacity, transfer.primal.shape[2])).at[0].set(1.0)
    transferred = np.asarray(transfer.apply_primal(values))
    for slot, count in zip(transfer.target_slots, transfer.target_dof_count, strict=True):
        np.testing.assert_allclose(transferred[int(slot), : int(count)], 1.0)
    _refuses_tamper(
        path,
        MeshAdaptationResult,
        ("evidence", "refinement", "lineage", "lineage_id"),
        _flip_hex,
    )
    _refuses_tamper(
        path,
        MeshAdaptationResult,
        ("evidence", "refinement", "geometry", "geometry_id"),
        _flip_hex,
    )
    state = Composition(
        (
            CompositionEntry(
                result,
                entry_id="adaptation",
                role="discretization",
                owner_id="hp",
                structure_id=topology.plan_id,
                revision_id=result.result_id,
                semantics_id="hp-adaptation",
            ),
            CompositionEntry(
                values,
                entry_id="field",
                role="physical-state",
                owner_id="hp-field",
                structure_id=topology.plan_id,
                revision_id="field-revision",
                semantics_id="primal",
            ),
        ),
        boundary_id="accepted",
    )
    state_path = tmp_path / "hp-state.phx"
    write_model_artifact(state_path, state, limits=MODEL_ARCHIVE_LIMITS)
    script = """
import sys
import numpy as np
from phydrax.discretization.fem import finite_element_hp_transfer_plan, prepare_finite_element_hp_epoch
from phydrax.lifecycle._composition_rebind import Composition
from phydrax.meshing import MarkedMeshAdaptation, MeshAdaptationPolicy, MeshAdaptationResult, MeshAdaptationRoute, execute_mesh_adaptation, prepare_mesh_adaptation
from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value, write_model_artifact
state = read_model_artifact(sys.argv[1], expected_type=Composition, restore=restore_model_archive_value, limits=MODEL_ARCHIVE_LIMITS)
result = state.value("adaptation")
values = state.value("field")
assert result.result_id == sys.argv[3]
assert np.all(np.asarray(values)[0] == 1.0) and np.all(np.asarray(values)[1:] == 0.0)
refined = result.evidence.refinement
source_epoch = restore_model_archive_value(result.evidence.source_epoch)
prepared_target = prepare_finite_element_hp_epoch(refined.topology, refined.geometry, "u", conformity="L2")
transfer = finite_element_hp_transfer_plan(source_epoch, prepared_target, refined.lineage, "u", "h-refinement")
mapped = np.asarray(transfer.apply_primal(values))
for slot, count in zip(transfer.target_slots, transfer.target_dof_count, strict=True):
    np.testing.assert_allclose(mapped[int(slot), :int(count)], 1.0)
epoch = result.hierarchy
children = np.asarray(epoch.topology.cell_global_ids)[1:5]
request = MarkedMeshAdaptation((), children, hierarchy=epoch)
prepared = prepare_mesh_adaptation(result.target, request, policy=MeshAdaptationPolicy(MeshAdaptationRoute.HP, relocation=False))
coarsened = execute_mesh_adaptation(prepared)
assert coarsened.hierarchy.topology.active_count == 1
write_model_artifact(sys.argv[2], coarsened, limits=MODEL_ARCHIVE_LIMITS)
"""
    coarsened_path = tmp_path / "hp-coarsened.phx"
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(state_path),
            str(coarsened_path),
            result.result_id,
        ],
        check=True,
    )
    coarsened = read_model_artifact(
        coarsened_path,
        expected_type=MeshAdaptationResult,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert coarsened.hierarchy.topology.active_count == 1
    coarsened_evidence = coarsened.evidence.refinement
    coarsened_source = restore_model_archive_value(coarsened.evidence.source_epoch)
    coarsened_source = prepare_finite_element_hp_epoch(
        coarsened_source.topology,
        coarsened_source.geometry,
        "u",
        conformity="L2",
    )
    coarsened_target = prepare_finite_element_hp_epoch(
        coarsened_evidence.topology, coarsened_evidence.geometry, "u", conformity="L2"
    )
    reverse = finite_element_hp_transfer_plan(
        coarsened_source,
        coarsened_target,
        coarsened_evidence.lineage,
        "u",
        "h-coarsening",
    )
    reverse_values = jnp.zeros(
        (coarsened_source.topology.capacity, reverse.primal.shape[2])
    )
    for slot, count in zip(reverse.source_slots, reverse.source_dof_count, strict=True):
        reverse_values = reverse_values.at[int(slot), : int(count)].set(1.0)
    mapped = np.asarray(reverse.apply_primal(reverse_values))
    for slot, count in zip(reverse.target_slots, reverse.target_dof_count, strict=True):
        np.testing.assert_allclose(mapped[int(slot), : int(count)], 1.0)


def test_hp_nonconforming_cell_mesh_is_refused_by_audit() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization.fem import (
        initial_finite_element_hp_topology,
        prepare_finite_element_hp_epoch,
    )
    from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory

    mesh = CellMesh(
        np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
        ),
        (
            CellBlock(
                "quads",
                "quadrilateral",
                np.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=np.int32),
                global_ids=np.asarray((10, 20), dtype=np.int64),
            ),
        ),
    )
    topology, geometry = initial_finite_element_hp_topology(mesh, 2, 16)
    epoch = prepare_finite_element_hp_epoch(topology, geometry, "u", conformity="L2")
    with pytest.raises(MeshingFailure) as failure:
        _adapt(
            _certified(epoch.mesh),
            MarkedMeshAdaptation(np.asarray((10,), dtype=np.int64), hierarchy=epoch),
            MeshAdaptationRoute.HP,
        )
    assert failure.value.category is MeshingFailureCategory.AUDIT_FAILED
