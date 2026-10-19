# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import model_array_bytes
from phydrax.meshing import metis_identity
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _runtime_identity,
    qualification_failure,
)


class _DistributionTransferRecord(TypedDict):
    maximum_parent_copy_error: float
    integral_before: float
    integral_after: float
    relative_integral_residual: float


class _DistributionTransitionRecord(TypedDict):
    status: str
    transition_id: str
    lineage_id: str
    source_distribution_id: str
    target_distribution_id: str
    target_provenance: str
    maximum_imbalance: float
    rebalanced: bool
    imbalance_after: float
    imbalance_bound: float
    edge_cut_after: int
    halo_replicas_after: int
    migration_counts: list[list[int]]
    migrated_cells: int
    migration_volume: float
    message_slots: int
    preserved_cells: int
    checks_passed: list[str]
    transfer: _DistributionTransferRecord
    resources: dict[str, float | int]


def _distribution_lattice(cells_per_axis: int, /) -> phx.discretization.CellMesh:
    """Counterclockwise lattice triangles of the unit square."""
    axis = np.linspace(0.0, 1.0, cells_per_axis + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    index = np.arange((cells_per_axis + 1) ** 2, dtype=np.int32).reshape(
        (cells_per_axis + 1, cells_per_axis + 1)
    )
    lower, right = index[:-1, :-1].reshape((-1,)), index[1:, :-1].reshape((-1,))
    upper, left = index[1:, 1:].reshape((-1,)), index[:-1, 1:].reshape((-1,))
    triangles = np.concatenate(
        (np.stack((lower, right, upper), axis=1), np.stack((lower, upper, left), axis=1))
    )
    return phx.discretization.CellMesh.from_triangles(points, triangles)


def _distribution_cells(
    mesh: phx.discretization.CellMesh, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native-order global IDs, signed areas, and centroids of triangle cells."""
    triangles = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = np.asarray(mesh.coordinates)[triangles]
    edges = corners[:, 1:] - corners[:, :1]
    areas = 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    return ids, areas, corners.mean(axis=1)


def _distribution_edge_cut(
    mesh: phx.discretization.CellMesh, owner: np.ndarray, /
) -> int:
    """Edge-sharing cell pairs with different owners, recounted from the triangles."""
    triangles = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int32)
    edges = np.sort(triangles[:, corners], axis=2).reshape((-1, 2)).astype(np.int64)
    keys = edges[:, 0] * mesh.coordinates.shape[0] + edges[:, 1]
    order = np.argsort(keys, kind="stable")
    cells = np.repeat(np.arange(triangles.shape[0]), 3)[order]
    shared = keys[order][1:] == keys[order][:-1]
    return int(np.count_nonzero(owner[cells[:-1][shared]] != owner[cells[1:][shared]]))


def _distribution_rows(native: np.ndarray, identifiers: np.ndarray, /) -> np.ndarray:
    """Rows of ``identifiers`` in ``native``; every identifier must be present."""

    order = np.argsort(native, kind="stable")
    located = np.searchsorted(native, identifiers, sorter=order)
    rows = order[np.minimum(located, native.size - 1)]
    if not np.array_equal(native[rows], identifiers):
        raise qualification_failure(
            "_distribution_rows",
            ("Lineage identifiers are missing from the native cell order."),
            {"identifiers": identifiers, "native": native, "rows": rows},
        )
    return rows


def _distribution_route(
    part: phx.meshing.MeshPart,
    policy: phx.meshing.MeshPartitionPolicy,
    weights: np.ndarray,
    bound: float,
    /,
    **ownership: np.ndarray,
) -> tuple[phx.meshing.MeshDistribution, dict[str, object]]:
    """Partition ``part`` and check its evidence against an independent recount."""

    distribution, stage = _measured(
        lambda: phx.meshing.prepare_mesh_distribution(
            part, policy=policy, cell_weights=weights, **ownership
        )
    )

    owner = np.asarray(distribution.partition.cell_owner)

    evidence = distribution.evidence
    part_weights = np.bincount(owner, weights=weights, minlength=policy.part_count)

    halos = tuple(np.asarray(rows) for rows in distribution.halo_rows)

    carrier = part.carrier
    if not isinstance(carrier, phx.meshing.CellMeshingResult):
        raise TypeError(
            f"_distribution_route.carrier must be a CellMeshingResult; got {type(carrier).__name__!r}."
        )
    edge_cut = _distribution_edge_cut(carrier.mesh, owner)
    ghost_owned = any(np.any(owner[rows] == rank) for rank, rows in enumerate(halos))
    if (
        np.any(part_weights <= 0.0)
        or not np.allclose(np.asarray(evidence.part_weights), part_weights, rtol=1e-14)
        or float(evidence.imbalance) > bound + 1e-12
        or int(evidence.edge_cut) != edge_cut
        or int(evidence.halo_replicas) != sum(rows.size for rows in halos)
        or ghost_owned
    ):
        raise qualification_failure(
            "_distribution_route",
            (f"{policy.kind.value} distribution evidence failed its recount."),
            {
                "bound": bound,
                "edge_cut": edge_cut,
                "evidence": evidence,
                "ghost_owned": ghost_owned,
                "halos": halos,
                "part_weights": part_weights,
            },
        )
    return distribution, {
        "status": "passed",
        "distribution_id": distribution.distribution_id,
        "evidence_id": evidence.evidence_id,
        "provenance": evidence.provenance,
        "part_weights": part_weights.tolist(),
        "imbalance": float(evidence.imbalance),
        "imbalance_bound": float(bound),
        "edge_cut": edge_cut,
        "halo_replicas": int(evidence.halo_replicas),
        "resources": {
            **stage,
            "retained_bytes": model_array_bytes(distribution),
        },
    }


def _distribution_transition(
    source: phx.meshing.MeshDistribution,
    target_part: phx.meshing.MeshPart,
    lineage: phx.meshing.MeshLineage,
    policy: phx.meshing.MeshPartitionPolicy,
    weights: np.ndarray,
    source_areas: np.ndarray,
    density: np.ndarray,
    /,
) -> _DistributionTransitionRecord:
    """Carry ``source`` through ``lineage``; check migration and cell-data transfer."""

    transition, stage = _measured(
        lambda: phx.meshing.prepare_distribution_transition(
            source, target_part, lineage, policy=policy, cell_weights=weights
        )
    )

    target = transition.target

    carrier = target_part.carrier
    if not isinstance(carrier, phx.meshing.CellMeshingResult):
        raise TypeError(
            f"_distribution_transition.carrier must be a CellMeshingResult; got {type(carrier).__name__!r}."
        )
    mesh = carrier.mesh
    target_ids, target_areas, _ = _distribution_cells(mesh)
    source_ids = np.asarray(source.cell_global_ids)
    cells = next(record for record in lineage.entities if record.dimension == 2)
    lineage_sources = np.asarray(cells.source_global_ids)
    lineage_targets = np.asarray(cells.target_global_ids)
    parents = np.full(target_ids.shape, -1, dtype=np.int64)
    parents[_distribution_rows(target_ids, lineage_targets)] = _distribution_rows(
        source_ids, lineage_sources
    )

    if np.any(parents < 0) or not np.all(np.asarray(transition.target_defined)):
        raise qualification_failure(
            "_distribution_transition",
            ("Bisection lineage left a target cell without a parent."),
            {"parents": parents, "transition": transition},
        )

    moved = np.asarray(transition.transfer(jnp.asarray(density)))
    copy_error = float(np.max(np.abs(moved - density[parents])))
    integral_before = float(np.sum(density * source_areas))
    integral_after = float(np.sum(moved * target_areas))
    residual = abs(integral_after - integral_before) / abs(integral_before)
    parts = policy.part_count
    sender = np.asarray(source.partition.cell_owner, dtype=np.int64)[parents]
    receiver = np.asarray(target.partition.cell_owner, dtype=np.int64)
    remote = sender != receiver
    counts = np.bincount(sender * parts + receiver, minlength=parts * parts)

    slots = np.arange(transition.send.capacity)

    sent = np.concatenate([np.asarray(transition.sent_by(rank)) for rank in range(parts)])
    received = np.concatenate(
        [np.asarray(transition.received_by(rank)) for rank in range(parts)]
    )
    preserved = np.asarray(cells.relation_kinds) == int(
        phx.meshing.EntityLineageKind.PRESERVED
    )
    bound = (
        1.0 + parts * np.max(weights) / np.sum(weights)
        if transition.rebalanced
        else policy.maximum_imbalance
    )
    checks = {
        "exact_parent_copy": copy_error == 0.0,
        "integral_preserved": residual <= 1e-12,
        "migration_counts": np.array_equal(
            np.asarray(transition.migration_counts),
            counts.reshape((parts, parts)),
        ),
        "migrated_cells": int(transition.migrated_cells) == np.count_nonzero(remote),
        "migration_volume": np.isclose(
            float(transition.migration_volume),
            np.sum(weights[remote]),
            rtol=1e-14,
            atol=0.0,
        ),
        "message_slots": np.array_equal(np.sort(sent), slots)
        and np.array_equal(np.sort(received), slots),
        "imbalance_bound": float(target.evidence.imbalance) <= bound + 1e-12,
        "edge_cut": int(target.evidence.edge_cut)
        == _distribution_edge_cut(mesh, receiver),
        "source_ids_unchanged": np.array_equal(
            np.asarray(transition.source.cell_global_ids),
            source_ids,
        ),
        "target_ids_native": np.array_equal(
            np.asarray(target.cell_global_ids), target_ids
        ),
        "preserved_ids_unchanged": np.array_equal(
            lineage_sources[preserved], lineage_targets[preserved]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise qualification_failure(
            "_distribution_transition",
            (f"Distribution transition failed checks: {failed}."),
            {"failed": failed},
        )
    return {
        "status": "passed",
        "transition_id": transition.transition_id,
        "lineage_id": transition.lineage_id,
        "source_distribution_id": source.distribution_id,
        "target_distribution_id": target.distribution_id,
        "target_provenance": target.evidence.provenance,
        "maximum_imbalance": policy.maximum_imbalance,
        "rebalanced": transition.rebalanced,
        "imbalance_after": float(target.evidence.imbalance),
        "imbalance_bound": float(bound),
        "edge_cut_after": int(target.evidence.edge_cut),
        "halo_replicas_after": int(target.evidence.halo_replicas),
        "migration_counts": counts.reshape((parts, parts)).tolist(),
        "migrated_cells": int(transition.migrated_cells),
        "migration_volume": float(transition.migration_volume),
        "message_slots": int(slots.size),
        "preserved_cells": int(np.count_nonzero(preserved)),
        "checks_passed": sorted(checks),
        "transfer": {
            "maximum_parent_copy_error": copy_error,
            "integral_before": integral_before,
            "integral_after": integral_after,
            "relative_integral_residual": residual,
        },
        "resources": {
            **stage,
            "retained_bytes": model_array_bytes(transition),
        },
    }


def qualify_distribution() -> dict[str, object]:

    kind = phx.meshing.MeshPartitionKind
    parts = 4
    source, certify_stage = _measured(
        lambda: phx.meshing.certify_cell_mesh(_distribution_lattice(8), _contract())
    )
    part = phx.meshing.MeshPart("distribution-square", source)

    cell_ids, areas, centroids = _distribution_cells(source.mesh)
    # The right half costs twice as much, so balanced parts differ in cell count.
    weights = np.where(centroids[:, 0] > 0.5, 2.0, 1.0)
    curve_bound = 1.0 + parts * np.max(weights) / np.sum(weights)
    distributions, routes = {}, {}
    for route in (kind.MORTON, kind.HILBERT):
        distributions[route], routes[route.value] = _distribution_route(
            part, phx.meshing.MeshPartitionPolicy(route, parts), weights, curve_bound
        )
    graph = phx.meshing.MeshPartitionPolicy(kind.GRAPH, parts, maximum_imbalance=1.1)
    _, routes[kind.GRAPH.value] = _distribution_route(
        part, graph, weights, graph.maximum_imbalance
    )
    # METIS is an explicit comparison route; loading its library is the only
    # way to learn whether it is usable.
    try:
        metis = metis_identity()
    except phx.meshing.MetisUnavailableError as error:
        routes[kind.METIS.value] = {
            "status": "missing-dependency",
            "dependency": "METIS",
            "detail": f"{error} Install METIS 5 and set PHYDRAX_METIS_LIBRARY.",
        }
    else:
        comparison = phx.meshing.MeshPartitionPolicy(
            kind.METIS, parts, maximum_imbalance=1.1
        )
        _, record = _distribution_route(
            part, comparison, weights, comparison.maximum_imbalance
        )
        routes[kind.METIS.value] = {**record, "metis": metis}
    # Horizontal strips stand in for ownership computed by an external partitioner.
    strips = np.minimum(np.floor(centroids[:, 1] * parts), parts - 1).astype(np.int32)
    strip_weights = np.bincount(strips, weights=weights, minlength=parts)
    provided, routes[kind.PROVIDER.value] = _distribution_route(
        part,
        phx.meshing.MeshPartitionPolicy(kind.PROVIDER, parts),
        weights,
        np.max(strip_weights) / np.mean(strip_weights),
        ownership=strips,
    )
    if not np.array_equal(np.asarray(provided.partition.cell_owner), strips):
        raise qualification_failure(
            "qualify_distribution",
            ("The PROVIDER route did not keep the supplied ownership."),
            {"provided": provided, "strips": strips},
        )

    marked = cell_ids[np.all(centroids < 0.3, axis=1)]
    adaptation, adaptation_stage = _measured(
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                source,
                phx.meshing.MarkedMeshAdaptation(marked),
                policy=phx.meshing.MeshAdaptationPolicy(
                    phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
                ),
            )
        )
    )

    target = adaptation.target

    if adaptation.lineage is None or not target.audit.passed:
        raise qualification_failure(
            "qualify_distribution",
            ("Native bisection did not refine the marked subregion."),
            {"adaptation": adaptation, "target": target},
        )
    target_part = phx.meshing.MeshPart(part.name, target)
    _, _, target_centroids = _distribution_cells(target.mesh)
    target_weights = np.where(target_centroids[:, 0] > 0.5, 2.0, 1.0)
    density = 1.0 + centroids[:, 0] + 2.0 * centroids[:, 1] ** 2
    hilbert = distributions[kind.HILBERT]
    transitions = {
        name: _distribution_transition(
            hilbert,
            target_part,
            adaptation.lineage,
            phx.meshing.MeshPartitionPolicy(
                kind.HILBERT, parts, maximum_imbalance=tolerance
            ),
            target_weights,
            areas,
            density,
        )
        for name, tolerance in (("inherited", 2.0), ("rebalanced", 1.05))
    }
    inherited, rebalanced = transitions["inherited"], transitions["rebalanced"]
    if (
        inherited["rebalanced"]
        or inherited["migrated_cells"] != 0
        or not rebalanced["rebalanced"]
        or rebalanced["migrated_cells"] == 0
    ):
        raise qualification_failure(
            "qualify_distribution",
            ("Transition ownership ignored the maximum-imbalance rule."),
            {"inherited": inherited, "rebalanced": rebalanced},
        )
    return {
        "status": "passed",
        "source": {
            **_result_identity(source),
            "part_id": part.part_id,
            "distribution_id": hilbert.distribution_id,
        },
        "target": {
            **_result_identity(target),
            "part_id": target_part.part_id,
            "adaptation_result_id": adaptation.result_id,
            "lineage_id": adaptation.lineage.lineage_id,
        },
        "runtime": _runtime_identity(),
        "quality": {
            "before": _quality_summary(source.quality),
            "after": _quality_summary(target.quality),
        },
        "routes": routes,
        "adaptation": {
            "route": adaptation.route.value,
            "status": adaptation.status.value,
            "marked_cells": int(marked.size),
        },
        "transitions": transitions,
        "transfer": {name: value["transfer"] for name, value in transitions.items()},
        "conservation": {
            name: value["transfer"]["relative_integral_residual"]
            for name, value in transitions.items()
        },
        "resources": {
            "certification": certify_stage,
            "adaptation": adaptation_stage,
            "routes": {
                name: value["resources"]
                for name, value in routes.items()
                if "resources" in value
            },
            "transitions": {
                name: value["resources"] for name, value in transitions.items()
            },
        },
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_distribution()
