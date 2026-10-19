#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host edge evidence and resource/compliance admission for native routes."""

from __future__ import annotations

from fractions import Fraction
from time import monotonic

import numpy as np

import phydrax.discretization as cell_mesh_owner
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from ..._geometry_predicates import exact_charge
from ...discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from ...discretization._cell_mesh import CellMesh
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._rational import fraction_endpoint
from ...linalg._hermitian_spectral import fraction_sqrt_interval
from ...typing import checked
from .. import _result as result_owner
from .._sizing import (
    size_statistic_quantile,
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from .._trace import MeshingStageKind


@checked
def check_deadline(
    started: float, limits: MeshingLimits, stage: MeshingStageKind, /
) -> None:
    """Refuse publication once the enforced wall-time budget has elapsed."""

    elapsed = monotonic() - started
    if elapsed > limits.maximum_wall_seconds:
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT,
            "execution.elapsed_seconds must respect limits.maximum_wall_seconds; got excess wall time",
            stage=stage.value,
            requested=(("maximum_wall_seconds", limits.maximum_wall_seconds),),
            achieved=(("elapsed_seconds", elapsed),),
        )


def unique_edges(cells: np.ndarray, cell_kind: str, /) -> np.ndarray:
    """Sorted edge pairs of one explicitly declared reference-cell family."""
    topology = reference_cell_topology(cell_kind)
    if cells.ndim != 2 or cells.shape[1] != len(topology.vertices):
        raise ValueError(
            "cells.shape must match the declared reference topology; got incompatible cell rows"
        )
    local = np.asarray(topology.entities[1], dtype=np.int32)
    pairs = cells[:, local].reshape((-1, 2))
    return np.unique(np.sort(pairs, axis=1), axis=0)


def _source_edge_bank(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, ...], ...]:
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    if geometry.coordinates.shape != mesh.coordinates.shape or not np.array_equal(
        np.asarray(geometry.coordinates), np.asarray(mesh.coordinates)
    ):
        raise ValueError(
            "source_edge_vectors.geometry must retain the mesh representative bank; got different coordinate rows"
        )
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if (
            block.cell_kind != "tetrahedron"
            or element.element_id
            != coordinate_lagrange_element("tetrahedron", 1).element_id
            or not np.array_equal(np.asarray(route), np.asarray(block.vertices))
        ):
            raise ValueError(
                "source_edge_vectors.geometry must have degree-one affine tetrahedral vertex routes; got another coordinate layout"
            )
    bank = geometry.rational_source_coordinates(budget=budget)
    if len(bank) != mesh.coordinates.shape[0]:
        raise ValueError(
            "source_edge_vectors.geometry must provide one source row per mesh vertex; got a different source bank"
        )
    return bank


@checked
def source_edge_vectors(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    edges: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Evaluate affine source endpoint differences before binary64 execution."""
    bank = _source_edge_bank(mesh, geometry, budget=budget)
    bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for row in bank
        for value in row
    )
    with exact_charge(
        12 * edges.shape[0],
        3 * edges.shape[0] * (128 + (4 * bits + 7) // 8),
        budget=budget,
    ):
        vectors = np.asarray(
            [
                [
                    float(bank[int(last)][axis] - bank[int(first)][axis])
                    for axis in range(mesh.ambient_dimension)
                ]
                for first, last in edges
            ],
            dtype=np.float64,
        )
        budget.retain_basis((vectors,))
    return vectors


@checked
def source_edge_arc_bounds(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    edges: np.ndarray,
    /,
    *,
    length_scale: float = 1.0,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Enclose straight source-edge lengths separately from numerical scores."""
    bank = _source_edge_bank(mesh, geometry, budget=budget)
    divisor = Fraction(length_scale)
    if divisor <= 0:
        raise ValueError(
            f"source_edge_arc_bounds.length_scale must be positive; got {length_scale!r}"
        )
    bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for row in bank
        for value in row
    )
    with exact_charge(
        16 * edges.shape[0],
        16 * edges.shape[0] + 1024 + 8 * bits,
        budget=budget,
    ):
        bounds = np.empty((edges.shape[0], 2), dtype=np.float64)
        for index, (first, last) in enumerate(edges):
            squared = sum(
                (
                    (bank[int(last)][axis] - bank[int(first)][axis]) ** 2
                    for axis in range(mesh.ambient_dimension)
                ),
                Fraction(0),
            )
            lower, upper = fraction_sqrt_interval(squared, budget=budget)
            bounds[index] = (
                fraction_endpoint(lower / divisor, upper=False),
                fraction_endpoint(upper / divisor, upper=True),
            )
    budget.retain_basis((bounds,))
    return bounds


@checked
def edge_size_evidence(
    vertices: np.ndarray,
    edges: np.ndarray,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    mesh: CellMesh | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, float]:
    """Edge lengths and incident growth from the declared coordinate geometry."""
    if geometry is None:
        vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    else:
        if mesh is None or budget is None:
            raise ValueError(
                "edge_size_evidence.geometry must have a bound mesh and coordinate budget; got missing source evaluation inputs"
            )
        if not np.array_equal(vertices, np.asarray(mesh.coordinates)):
            raise ValueError(
                "edge_size_evidence.vertices must retain the bound mesh representatives; got different coordinates"
            )
        vectors = source_edge_vectors(mesh, geometry, edges, budget=budget)
    lengths = np.linalg.norm(vectors, axis=1)
    return lengths, edge_growth_evidence(lengths, edges, vertices.shape[0])


def edge_growth_evidence(
    lengths: np.ndarray, edges: np.ndarray, vertex_count: int, /
) -> float:
    """Incident-edge growth of an finite positive physical length bank."""
    if not lengths.size or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "edge_evidence.lengths must be finite, positive, and nonempty; got invalid lengths",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    minimum = np.full((vertex_count,), np.inf, dtype=np.float64)
    maximum = np.zeros((vertex_count,), dtype=np.float64)
    np.minimum.at(minimum, edges[:, 0], lengths)
    np.minimum.at(minimum, edges[:, 1], lengths)
    np.maximum.at(maximum, edges[:, 0], lengths)
    np.maximum.at(maximum, edges[:, 1], lengths)
    active = np.isfinite(minimum) & (minimum > 0.0)
    return float(np.max(maximum[active] / minimum[active], initial=1.0))


@checked
def uniform_size_compliance(
    control: UniformSizeControl,
    policy: SizeCompliancePolicy,
    lengths: np.ndarray,
    growth: float,
    /,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[str]]:
    """Requested, achieved, and failed quantities of one uniform size control.

    ``lengths`` is a NumPy host array; source evaluation transfers its output
    before this boundary. Quantiles and reductions remain host operations.

    Soft controls are recorded; hard controls fail when a requested target
    statistic, bound, or growth rate is missed beyond the compliance tolerance.
    """

    key = f"size:{control.control_id}"
    requested = [(f"{key}:target_size", control.target_size)]
    optional = (
        ("minimum_size", control.minimum_size),
        ("maximum_size", control.maximum_size),
        ("maximum_growth_rate", control.maximum_growth_rate),
    )
    requested.extend(
        (f"{key}:{name}", value) for name, value in optional if value is not None
    )
    achieved = [
        (f"{key}:minimum_edge", float(np.min(lengths))),
        (f"{key}:maximum_edge", float(np.max(lengths))),
        (f"{key}:maximum_local_edge_ratio", growth),
    ]
    statistics: dict[str, float] = {}
    for statistic in policy.target_statistics:
        value = float(np.quantile(lengths, size_statistic_quantile(statistic)))
        statistics[statistic] = value
        achieved.append((f"{key}:{statistic}_edge", value))
    issues: list[str] = []
    match control.strength:
        case SizeControlStrength.HARD:
            tolerance = policy.tolerance

            for statistic, value in statistics.items():
                if abs(value - control.target_size) > tolerance(control.target_size):
                    issues.append(f"target_size_{statistic}:{control.control_id}")
            if control.minimum_size is not None and float(
                np.min(lengths)
            ) < control.minimum_size - tolerance(control.minimum_size):
                issues.append(f"minimum_size:{control.control_id}")
            if control.maximum_size is not None and float(
                np.max(lengths)
            ) > control.maximum_size + tolerance(control.maximum_size):
                issues.append(f"maximum_size:{control.control_id}")
            if (
                control.maximum_growth_rate is not None
                and growth
                > control.maximum_growth_rate + tolerance(control.maximum_growth_rate)
            ):
                issues.append(f"maximum_growth_rate:{control.control_id}")
        case SizeControlStrength.SOFT:
            pass
        case _:
            raise ValueError(
                f"control.strength must be hard or soft; got {control.strength!r}"
            )
    return requested, achieved, issues


@checked
def require_compliance(compliance: result_owner.MeshingComplianceReport, /) -> None:
    """Refuse a construction that misses a hard request, with its evidence."""

    if not compliance.passed:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "compliance.issues must be empty; got " + "; ".join(compliance.issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            requested=compliance.requested,
            achieved=compliance.achieved,
        )


@checked
def simplex_entity_limits(
    vertices: np.ndarray,
    cells: np.ndarray,
    limits: MeshingLimits,
    stage: MeshingStageKind,
    /,
    *,
    cell_kind: str,
) -> None:
    """Refuse constructed simplices beyond the entity and data budgets.

    Runs on the host construction arrays before any mesh carrier is built.
    """

    edges = unique_edges(cells, cell_kind).shape[0]
    topology = reference_cell_topology(cell_kind)
    faces = 0
    if topology.dimension >= 2:
        for width in sorted({len(face) for face in topology.entities[2]}):
            local = np.asarray(
                tuple(face for face in topology.entities[2] if len(face) == width),
                dtype=np.int32,
            )
            rows = cells[:, local].reshape((-1, width))
            faces += np.unique(np.sort(rows, axis=1), axis=0).shape[0]
    checks = (
        ("vertices", vertices.shape[0], limits.maximum_vertices),
        ("edges", edges, limits.maximum_edges),
        ("faces", faces, limits.maximum_faces),
        ("cells", cells.shape[0], limits.maximum_cells),
        ("connectivity_entries", cells.size, limits.maximum_connectivity_entries),
        ("data_bytes", vertices.nbytes + cells.nbytes, limits.maximum_data_bytes),
    )
    exceeded = tuple(name for name, count, limit in checks if count > limit)
    if exceeded:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "construction.resources must respect the declared limits; got excess "
            + ", ".join(exceeded),
            stage=stage.value,
            requested=tuple((f"maximum_{name}", limit) for name, _, limit in checks),
            achieved=tuple((name, count) for name, count, _ in checks),
        )


@checked
def published_entity_limits(
    mesh: cell_mesh_owner.CellMesh, limits: MeshingLimits, /
) -> None:
    """Admit every cell and oriented-incidence index buffer."""
    dimension = mesh.topological_dimension
    entries = sum(block.vertices.size for block in mesh.blocks) + sum(
        incidence.relation.source_indices.size + incidence.relation.target_indices.size
        for incidence in mesh.topology.incidences
    )
    counts = (
        ("vertices", mesh.coordinates.shape[0], limits.maximum_vertices),
        (
            "edges",
            mesh.entity_set(1).entity_ids.size if dimension >= 1 else 0,
            limits.maximum_edges,
        ),
        (
            "faces",
            mesh.entity_set(2).entity_ids.size if dimension >= 2 else 0,
            limits.maximum_faces,
        ),
        (
            "cells",
            sum(block.vertices.shape[0] for block in mesh.blocks),
            limits.maximum_cells,
        ),
        ("connectivity_entries", entries, limits.maximum_connectivity_entries),
    )
    if any(actual > maximum for _, actual, maximum in counts):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "publication.entities must respect limits entity and connectivity maxima; got excess entities",
            stage=MeshingStageKind.CANONICALIZATION.value,
            requested=tuple((f"maximum_{name}", maximum) for name, _, maximum in counts),
            achieved=tuple((name, actual) for name, actual, _ in counts),
        )
