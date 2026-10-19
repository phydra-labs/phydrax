# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Prepare split-lineage weights and compose surface root-reference ancestry."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

import numpy as np

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.geometry._surface_source_support as surface_support_owner
import phydrax.meshing._association_routes as association_routes_owner
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._lineage_records as lineage_owner
import phydrax.meshing._result as result_owner

from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry._rational import rational_dot
from ..typing import checked


type SurfaceVertexWeights = tuple[tuple[int, Fraction], ...]


def _support_weights(
    points: np.ndarray, point: np.ndarray, /
) -> tuple[Fraction, ...] | None:
    """Rational least-squares barycentric weights of ``point`` on a segment or triangle support."""
    # This one/two-dimensional rational kernel is immutable host preparation.
    # Retaining it preserves the split-functional arithmetic and admission
    # order; a budgeted native solve would add previously uncharged work.
    origin = tuple(Fraction(float(value)) for value in points[0])
    edges = tuple(
        tuple(
            Fraction(float(value)) - base for value, base in zip(row, origin, strict=True)
        )
        for row in points[1:]
    )
    offset = tuple(
        Fraction(float(value)) - base for value, base in zip(point, origin, strict=True)
    )
    gram = tuple(tuple(rational_dot(a, b) for b in edges) for a in edges)
    rhs = tuple(rational_dot(edge, offset) for edge in edges)
    if len(edges) == 1:
        if gram[0][0] == 0:
            return None
        local: tuple[Fraction, ...] = (rhs[0] / gram[0][0],)
    else:
        determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
        if determinant == 0:
            return None
        local = (
            (rhs[0] * gram[1][1] - gram[0][1] * rhs[1]) / determinant,
            (gram[0][0] * rhs[1] - gram[1][0] * rhs[0]) / determinant,
        )
    return (1 - sum(local, Fraction(0)), *local)


@checked
def created_weights(
    lineage: lineage_owner.MeshLineage,
    source_mesh: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    /,
) -> dict[int, SurfaceVertexWeights]:
    """Position of every created vertex in its split-lineage support simplex.

    Lineage alone names the support (a source edge or cell); the rational weights
    locate the target carrier point in it. Any off-support carrier
    deviation is measured afresh by the restriction proof.
    """
    record = lineage.entity_lineage(0)
    split = np.asarray(record.relation_kinds, dtype=np.int32) == int(
        lineage_owner.EntityLineageKind.SPLIT_FROM
    )
    supports: dict[int, set[int]] = {}
    for parent, child in zip(
        np.asarray(record.source_global_ids, dtype=np.int64)[split].tolist(),
        np.asarray(record.target_global_ids, dtype=np.int64)[split].tolist(),
        strict=True,
    ):
        supports.setdefault(child, set()).add(parent)
    children = np.asarray(sorted(supports), dtype=np.int64)
    child_rows = association_state_owner.entity_rows(target, 0, children)
    if np.any(child_rows < 0):
        raise association_routes_owner.AssociationPropagationError(
            "SurfaceSubdivision.lineage.vertices must name target vertices; got absent target vertex IDs",
            children[child_rows < 0],
        )
    source_coordinates = np.asarray(source_mesh.coordinates, dtype=np.float64)
    target_coordinates = np.asarray(target.coordinates, dtype=np.float64)
    weights: dict[int, SurfaceVertexWeights] = {}
    for child, child_row in zip(children.tolist(), child_rows.tolist(), strict=True):
        witness = np.asarray((child,), dtype=np.int64)
        ordered = np.asarray(sorted(supports[child]), dtype=np.int64)
        rows = association_state_owner.entity_rows(source_mesh, 0, ordered)
        if ordered.size not in (2, 3) or np.any(rows < 0):
            raise association_routes_owner.AssociationPropagationError(
                "SurfaceSubdivision.created_vertices must retain source segment or triangle support; got missing split support",
                witness,
            )
        local = _support_weights(source_coordinates[rows], target_coordinates[child_row])
        if local is None or any(value < 0 for value in local):
            raise association_routes_owner.AssociationPropagationError(
                "SurfaceSubdivision.created_vertices must lie in the split support simplex; got deficient or off-simplex support",
                witness,
            )
        weights[child] = tuple(zip(ordered.tolist(), local, strict=True))
    return weights


@checked
def lineage_ancestry(
    source_proof: surface_support_owner.SurfaceRestrictionRows,
    lineage: lineage_owner.MeshLineage,
    target: cell_mesh_owner.CellMesh,
    weights: Mapping[int, SurfaceVertexWeights],
    /,
) -> dict[int, surface_support_owner.SurfaceCellAncestry]:
    """Serial target roots and root-reference corners from topology lineage.

    Preserved vertices keep their proved source-cell reference corner; a created
    vertex is the weighted reference point of its split-lineage support in the
    same scientific root cell. Coordinates never select a root.
    """
    known = {
        cell: (root, dict(zip(vertices, corners, strict=True)))
        for cell, root, vertices, corners in zip(
            source_proof.cell_ids.tolist(),
            source_proof.root_cell_ids.tolist(),
            source_proof.vertex_ids,
            source_proof.reference_corners,
            strict=True,
        )
    }
    cell_record = lineage.entity_lineage(2)
    cell_parents: dict[int, set[int]] = {}
    for parent, child in zip(
        np.asarray(cell_record.source_global_ids, dtype=np.int64).tolist(),
        np.asarray(cell_record.target_global_ids, dtype=np.int64).tolist(),
        strict=True,
    ):
        cell_parents.setdefault(child, set()).add(parent)
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    ancestry: dict[int, surface_support_owner.SurfaceCellAncestry] = {}
    for block in target.blocks:
        for identifier, vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(),
            np.asarray(block.vertices),
            strict=True,
        ):
            witness = np.asarray((identifier,), dtype=np.int64)
            parents = sorted(cell_parents.get(identifier, (identifier,)))
            if any(parent not in known for parent in parents):
                raise association_routes_owner.AssociationPropagationError(
                    "SurfaceSubdivision.cells must retain proved source-cell lineage; got unknown parent cells",
                    witness,
                )
            roots = {known[parent][0] for parent in parents}
            if len(roots) != 1:
                raise association_routes_owner.AssociationPropagationError(
                    "SurfaceSubdivision.cells must retain one source root; got merged source roots",
                    witness,
                )
            positions: dict[int, tuple[Fraction, Fraction]] = {}
            for parent in parents:
                for vertex, corner in known[parent][1].items():
                    if positions.setdefault(vertex, corner) != corner:
                        raise association_routes_owner.AssociationPropagationError(
                            "SurfaceSubdivision.siblings must agree on root-reference vertex positions; got different sibling positions",
                            witness,
                        )
            corners: list[tuple[Fraction, Fraction]] = []
            for vertex in target_vertex_ids[vertices].tolist():
                if vertex in positions:
                    corners.append(positions[vertex])
                    continue
                support = weights.get(vertex, ())
                if not support or any(end not in positions for end, _ in support):
                    raise association_routes_owner.AssociationPropagationError(
                        "SurfaceSubdivision.created_vertices must retain split support in its root cell; got missing root-reference support",
                        witness,
                    )
                corners.append(
                    (
                        sum(
                            (weight * positions[end][0] for end, weight in support),
                            Fraction(0),
                        ),
                        sum(
                            (weight * positions[end][1] for end, weight in support),
                            Fraction(0),
                        ),
                    )
                )
            ancestry[identifier] = surface_support_owner.SurfaceCellAncestry(
                roots.pop(), tuple(corners)
            )
    return ancestry


@checked
def subdivision_proof(
    support: surface_support_owner.PreparedSurfaceSourceSupport,
    lineages: tuple[lineage_owner.MeshLineage, ...],
    targets: tuple[cell_mesh_owner.CellMesh, ...],
    geometry: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    surface_support_owner.SurfaceRestrictionRows,
    tuple[dict[int, SurfaceVertexWeights], ...],
]:
    """Compose serial lineage epochs from the accepted root to the final target.

    Each epoch's ancestry is taken from the previous epoch's proved root
    references, so every proof stays in rational root-reference coordinates.
    """

    root = support.original
    if not isinstance(root, result_owner.CellMeshingResult):
        raise ValueError(
            "SurfaceSubdivision.support.root must be an accepted serial surface result; got different root kind"
        )
    proof = support.prove_native_restrictions(root.mesh, root.geometry, budget=budget)
    previous = root.mesh
    epochs: list[dict[int, SurfaceVertexWeights]] = []
    for index, (lineage, target) in enumerate(zip(lineages, targets, strict=True)):
        weights = created_weights(lineage, previous, target)
        ancestry = lineage_ancestry(proof, lineage, target, weights)
        epoch_geometry = (
            geometry
            if index == len(targets) - 1
            else cell_geometry_owner.CellGeometrySpec.affine(target)
        )
        if index == len(targets) - 1 and epoch_geometry.restriction_source is not None:
            proof = support.prove_native_restrictions(
                target, epoch_geometry, budget=budget
            )
            restricted_roots = dict(
                zip(proof.cell_ids.tolist(), proof.root_cell_ids.tolist(), strict=True)
            )
            if set(restricted_roots) != set(ancestry) or any(
                restricted_roots[identifier] != witness.root_cell_id
                for identifier, witness in ancestry.items()
            ):
                raise ValueError(
                    "SurfaceSubdivision.restrictions must agree with serial root ancestry; got different restriction roots"
                )
        else:
            proof = support.prove_native_restrictions(
                target,
                epoch_geometry,
                ancestry=ancestry,
                budget=budget,
            )
        epochs.append(weights)
        previous = target
    return proof, tuple(epochs)
