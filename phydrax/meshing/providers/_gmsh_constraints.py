#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh mesh-conformity constraints: periodic node maps and protected entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...discretization import CellMesh
from ...discretization._periodic_topology import PeriodicIsometryGroup
from ...geometry.brep import BRepModel
from .._contracts import MeshingFailure, MeshingFailureCategory
from .._controls import ProtectedFeature
from .._trace import MeshingStageKind
from ._gmsh_elements import _evaluate_element_maps, _local_connectivity
from ._gmsh_evidence import _connectivity_face_rows, _EvidenceSection
from ._gmsh_import import (
    _match_entities,
    _resolve_entities,
    _scope_samples,
    _source_scale,
)


def _set_periodic(gmsh: Any, plan: Any, shape: Any, /) -> Any:
    records = []
    slaves_used = set()
    for constraint in plan.specification.periodic_constraints:
        dimension = constraint.source_scope.entity_dimension
        masters = _resolve_entities(gmsh, plan.source, constraint.source_scope)
        candidates = _resolve_entities(gmsh, plan.source, constraint.target_scope)
        transform = np.asarray(constraint.transform)
        samples = _scope_samples(plan.source, constraint.source_scope)
        transformed = tuple(
            points @ transform[:3, :3].T + transform[:3, 3] for points in samples
        )
        slaves = _match_entities(
            gmsh, dimension, transformed, candidates, constraint.tolerance
        )
        if set(masters) & set(slaves) or any(
            (dimension, tag) in slaves_used for tag in slaves
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Periodic self-pairs and multiply constrained slave entities are unsupported.",
            )
        gmsh.model.mesh.setPeriodic(
            dimension, list(slaves), list(masters), transform.reshape(-1).tolist()
        )
        slaves_used.update((dimension, tag) for tag in slaves)
        records.append((constraint, masters, slaves))
    return tuple(records)


def _verify_bound_orbit_cycles(
    pairs: list[tuple[int, int, np.ndarray]],
    coordinates: dict[int, np.ndarray],
    tolerance: float,
    /,
) -> int:
    """Validate node-bound isometry cycles, including finite fixed-point stabilizers."""

    adjacency: dict[int, list[tuple[int, np.ndarray]]] = {}
    for master, slave, transform in pairs:
        adjacency.setdefault(master, []).append((slave, transform))
        adjacency.setdefault(slave, []).append((master, np.linalg.inv(transform)))
    frames: dict[int, np.ndarray] = {}
    stabilizers = 0
    for root in adjacency:
        if root in frames:
            continue
        frames[root] = np.eye(4)
        queue = [root]
        for current in queue:
            for target, transform in adjacency[current]:
                candidate = transform @ frames[current]
                if target not in frames:
                    frames[target] = candidate
                    queue.append(target)
                    continue
                cycle = np.linalg.inv(frames[target]) @ candidate
                if np.max(np.abs(cycle - np.eye(4))) <= tolerance:
                    continue
                # A rotation axis may end on the paired boundaries: its node
                # orbit has a finite stabilizer, not a contradictory translation.
                homogeneous = np.append(coordinates[root], 1.0)
                if np.linalg.norm((cycle @ homogeneous - homogeneous)[:3]) > tolerance:
                    raise ValueError(
                        "A bound periodic node orbit has a conflicting transformation cycle."
                    )
                group = PeriodicIsometryGroup(cycle[None], tolerance=tolerance)
                if group.orders[0] == 0:
                    raise ValueError(
                        "A periodic cycle has nonzero translational holonomy."
                    )
                stabilizers += 1
    return stabilizers


def _audit_periodic(
    gmsh: Any, records: Any, node_tags: Any, points: Any, /
) -> _EvidenceSection:
    requested = []
    achieved = []
    bound_pairs: list[tuple[int, int, np.ndarray]] = []
    threshold = min(
        (constraint.tolerance for constraint, _, _ in records), default=1.0e-10
    )
    transforms = [np.asarray(constraint.transform) for constraint, _, _ in records]
    if transforms:
        try:
            for first, transform in enumerate(transforms):
                if np.max(np.abs(transform - np.eye(4))) > threshold:
                    PeriodicIsometryGroup(transform[None], tolerance=threshold)
                for other in transforms[first + 1 :]:
                    if np.max(np.abs(transform @ other - other @ transform)) > threshold:
                        raise ValueError(
                            "Periodic boundary transformations have a conflicting composed cycle."
                        )
        except ValueError as error:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                str(error),
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            ) from error
    for constraint, masters, slaves in records:
        dimension = constraint.source_scope.entity_dimension
        transform = np.asarray(constraint.transform)
        residual = 0.0
        pair_count = 0
        for master, slave in zip(masters, slaves, strict=True):
            actual_master, slave_nodes, master_nodes, actual_transform = (
                gmsh.model.mesh.getPeriodicNodes(
                    dimension, slave, includeHighOrderNodes=True
                )
            )
            slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
            master_nodes = np.asarray(master_nodes, dtype=np.int64)
            expected_slave, _, _ = gmsh.model.mesh.getNodes(
                dimension, slave, includeBoundary=True
            )
            expected_master, _, _ = gmsh.model.mesh.getNodes(
                dimension, master, includeBoundary=True
            )
            if (
                actual_master != master
                or not np.array_equal(np.sort(slave_nodes), np.unique(expected_slave))
                or not np.array_equal(np.sort(master_nodes), np.unique(expected_master))
                or not np.allclose(
                    np.asarray(actual_transform).reshape((4, 4)),
                    transform,
                    rtol=0.0,
                    atol=constraint.tolerance,
                )
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "Gmsh periodic correspondence is not a complete high-order node bijection.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                )
            slave_points = points[_local_connectivity(node_tags, slave_nodes)]
            master_points = points[_local_connectivity(node_tags, master_nodes)]
            mapped = master_points @ transform[:3, :3].T + transform[:3, 3]
            residual = max(
                residual,
                float(np.max(np.linalg.norm(slave_points - mapped, axis=1), initial=0.0)),
            )
            pair_count += slave_nodes.size
            bound_pairs.extend(
                (int(first), int(second), transform)
                for first, second in zip(master_nodes, slave_nodes, strict=True)
            )
        if residual > constraint.tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Periodic node residual exceeds the exact requested tolerance.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        key = f"periodic:{constraint.constraint_id}"
        requested.append((f"{key}:tolerance", constraint.tolerance))
        achieved.extend(
            (
                (f"{key}:maximum_residual", residual),
                (f"{key}:node_pairs", float(pair_count)),
            )
        )
    try:
        stabilizers = _verify_bound_orbit_cycles(
            bound_pairs,
            {
                int(tag): np.asarray(point)
                for tag, point in zip(node_tags, points, strict=True)
            },
            threshold,
        )
    except ValueError as error:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            str(error),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        ) from error
    achieved.append(("periodic:fixed_point_cycle_count", float(stabilizers)))
    return _EvidenceSection(tuple(requested), tuple(achieved))


@dataclass(frozen=True, slots=True)
class _ProtectedEntities:
    feature: ProtectedFeature
    tags: tuple[int, ...]
    embedded_count: int


def _entity_samples(gmsh: Any, dimension: int, tag: int, /) -> np.ndarray:
    if dimension == 0:
        return np.asarray(gmsh.model.getValue(0, tag, []), dtype=np.float64).reshape(
            (1, 3)
        )
    lower, upper = gmsh.model.getParametrizationBounds(1, tag)
    parameters = np.linspace(float(lower[0]), float(upper[0]), 7)[1:-1]
    return np.asarray(
        gmsh.model.getValue(1, tag, parameters.tolist()), dtype=np.float64
    ).reshape((-1, 3))


def _contains(
    gmsh: Any, dimension: int, tag: int, points: np.ndarray, tolerance: float, /
) -> Any:
    if dimension == 3:
        return gmsh.model.isInside(3, tag, points.reshape(-1)) == points.shape[0]
    closest, _ = gmsh.model.getClosestPoint(dimension, tag, points.reshape(-1))
    closest = np.asarray(closest, dtype=np.float64).reshape((-1, 3))
    return bool(
        closest.shape == points.shape
        and np.max(np.linalg.norm(closest - points, axis=1)) <= tolerance
        and gmsh.model.isInside(dimension, tag, closest.reshape(-1)) == points.shape[0]
    )


def _embedding_host(
    gmsh: Any, dimension: int, tag: int, top_dimension: int, tolerance: float, /
) -> tuple[int, int]:
    """Embed a free entity into the lowest-dimensional meshed entity containing it."""
    points = _entity_samples(gmsh, dimension, tag)
    for host_dimension in range(dimension + 1, top_dimension + 1):
        hosts = tuple(
            host
            for _, host in gmsh.model.getEntities(host_dimension)
            if _contains(gmsh, host_dimension, host, points, tolerance)
        )
        if len(hosts) > 1:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "A free protected entity lies in more than one host entity.",
                stage=MeshingStageKind.CONTROL_RESOLUTION.value,
            )
        if hosts:
            return host_dimension, hosts[0]
    raise MeshingFailure(
        MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
        "A free protected entity lies in no meshed host entity.",
        stage=MeshingStageKind.CONTROL_RESOLUTION.value,
    )


def _free_entities(gmsh: Any, top_dimension: int, /) -> set[tuple[int, int]]:
    return {
        (dimension, tag)
        for dimension in range(top_dimension)
        for _, tag in gmsh.model.getEntities(dimension)
        if not len(gmsh.model.getAdjacencies(dimension, tag)[0])
    }


def _apply_protected_features(
    gmsh: Any,
    source: BRepModel,
    shape: Any,
    specification: Any,
    embedding_allowed: bool,
    /,
) -> tuple[_ProtectedEntities, ...]:
    """Embed free protected entities into their hosts; discard other free entities.

    Free CAD curves and points bound no meshed entity, so only an explicit
    protection request gives them meshing semantics.
    """
    top_dimension = specification.target.topological_dimension
    tolerance = 1.0e-7 * _source_scale(source)
    free = _free_entities(gmsh, top_dimension)
    records = []
    embeddings: dict[tuple[int, int, int], list[int]] = {}
    for feature in specification.protected_features:
        dimension = feature.scope.entity_dimension
        tags = _resolve_entities(gmsh, source, feature.scope)
        embedded = 0
        for tag in tags:
            if (dimension, tag) not in free:
                continue
            if not embedding_allowed:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Free protected entities cannot be embedded on strict semantic, swept, or periodic paths.",
                    stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                )
            host_dimension, host = _embedding_host(
                gmsh, dimension, tag, top_dimension, tolerance
            )
            embeddings.setdefault((host_dimension, host, dimension), []).append(tag)
            embedded += 1
        records.append(_ProtectedEntities(feature, tags, embedded))
    protected = {
        (dimension, tag) for (_, _, dimension), tags in embeddings.items() for tag in tags
    }
    discarded = sorted(free - protected, reverse=True)
    if discarded:
        gmsh.model.occ.remove(discarded, recursive=True)
        gmsh.model.occ.synchronize()
    for (host_dimension, host, dimension), tags in sorted(embeddings.items()):
        gmsh.model.mesh.embed(dimension, sorted(set(tags)), host_dimension, host)
    return tuple(records)


def _entity_elements(gmsh: Any, dimension: int, tag: int, /) -> Any:
    element_types, _, node_blocks = gmsh.model.mesh.getElements(dimension, tag)
    for element_type, node_values in zip(element_types, node_blocks, strict=True):
        _, _, _, count, _, corners = gmsh.model.mesh.getElementProperties(
            int(element_type)
        )
        yield (
            int(element_type),
            np.asarray(node_values, dtype=np.int64).reshape((-1, int(count))),
            int(corners),
        )


def _mesh_entity_rows(mesh: CellMesh, dimension: int, /) -> set[tuple[int, ...]]:
    if dimension == 1:
        # ty: ignore[unresolved-attribute]
        rows = np.asarray(mesh.connectivity.edges, dtype=np.int64)
    else:
        # ty: ignore[invalid-argument-type]
        rows = _connectivity_face_rows(mesh.connectivity)
    return {tuple(sorted(int(value) for value in row)) for row in rows}


def _protected_deviation(
    gmsh: Any,
    dimension: int,
    tag: int,
    node_tags: np.ndarray,
    points: np.ndarray,
    source_to_corner: np.ndarray,
    mesh_rows: set[tuple[int, ...]],
    /,
) -> tuple[float, int]:
    if dimension == 0:
        nodes, _, _ = gmsh.model.mesh.getNodes(0, tag)
        local = _local_connectivity(node_tags, np.asarray(nodes, dtype=np.int64))
        if local.size != 1 or source_to_corner[local[0]] < 0:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A protected point is not one canonical mesh vertex.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        target = np.asarray(gmsh.model.getValue(0, tag, []), dtype=np.float64)
        return float(np.linalg.norm(points[local[0]] - target)), 1
    deviation = 0.0
    count = 0
    for element_type, nodes, corners in _entity_elements(gmsh, dimension, tag):
        local = _local_connectivity(node_tags, nodes)
        canonical = source_to_corner[local[:, :corners]]
        if np.any(canonical < 0) or any(
            tuple(sorted(int(value) for value in row)) not in mesh_rows
            for row in canonical
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A protected CAD entity is not represented by canonical mesh entities.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        # Interior samples of Gmsh's own element maps, in Gmsh reference coordinates.
        reference = (
            np.asarray(((-0.5,), (0.0,), (0.5,)), dtype=np.float64)
            if dimension == 1
            else np.full((1, 2), 1.0 / 3.0, dtype=np.float64)
        )
        samples = _evaluate_element_maps(
            gmsh, element_type, points[local], reference
        ).reshape((-1, 3))
        closest, _ = gmsh.model.getClosestPoint(dimension, tag, samples.reshape(-1))
        distances = np.linalg.norm(
            np.asarray(closest, dtype=np.float64).reshape((-1, 3)) - samples, axis=1
        )
        deviation = max(deviation, float(np.max(distances, initial=0.0)))
        count += nodes.shape[0]
    if not count:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A protected CAD entity generated no mesh entities.",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    return deviation, count


def _audit_protected_features(
    gmsh: Any,
    records: tuple[_ProtectedEntities, ...],
    mesh: CellMesh,
    node_tags: np.ndarray,
    points: np.ndarray,
    source_to_corner: np.ndarray,
    tolerance: float,
    /,
) -> _EvidenceSection:
    requested = []
    achieved = []
    issues = []
    rows = {
        dimension: _mesh_entity_rows(mesh, dimension)
        for dimension in {record.feature.scope.entity_dimension for record in records}
        if dimension > 0
    }
    for record in records:
        feature = record.feature
        dimension = feature.scope.entity_dimension
        deviation = 0.0
        count = 0
        for tag in record.tags:
            value, entities = _protected_deviation(
                gmsh,
                dimension,
                tag,
                node_tags,
                points,
                source_to_corner,
                rows.get(dimension, set()),
            )
            deviation = max(deviation, value)
            count += entities
        key = f"protected:{feature.feature_id}"
        requested.append((f"{key}:maximum_deviation", feature.maximum_deviation))
        achieved.extend(
            (
                (f"{key}:maximum_deviation", deviation),
                (f"{key}:mesh_entity_count", float(count)),
                (f"{key}:embedded_entity_count", float(record.embedded_count)),
            )
        )
        if feature.hard and deviation > feature.maximum_deviation + tolerance:
            issues.append(f"protected_deviation:{feature.feature_id}")
    return _EvidenceSection(tuple(requested), tuple(achieved), tuple(issues))
