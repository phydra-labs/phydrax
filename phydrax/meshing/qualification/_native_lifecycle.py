# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction
from time import monotonic
from typing import final

import numpy as np

from ..._meshcore import NativeExecutionMemoryCounters, NativeExecutionScope
from ..._strict import StrictModule
from ...discretization import CellGeometrySpec, CellMesh
from ...discretization._cell_geometry import (
    cell_geometry_preparation_bounds,
    cell_geometry_preparation_query_upper,
    coordinate_lagrange_element,
    owned_coordinate_preparation,
    require_published_affine_carrier,
)
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._coordinate_enclosure import (
    affine_simplex_corner_images,
    CoordinateSourceBank,
)
from ...discretization._exact_algebraic_geometry import rational_coefficient
from ...typing import checked
from .._association import PlcAssociationTransfer
from .._contracts import MeshingLimits, VolumeMeshingSpec
from .._measurements import NativeMeshingPhaseRecorder
from .._native_preparation import native_preparation_ledger
from .._result import CellMeshingResult
from .._trace import MeshingStageKind
from .._volume_generation import (
    native_volume_execution_budget,
    native_volume_source_execution_budget,
    prepare_plc_source,
    PreparedPlcSource,
)
from ..providers._native import NativeMeshingProvider
from ..providers._native_options import NativeMeshingOptions
from ..providers._native_publication import bind_native_execution_result
from ..providers._native_route_checks import simplex_entity_limits
from ..providers._native_sources import NativePlcSource, NativePolyhedralSource


type QualificationFailureFactory = Callable[[str, str, dict[str, object]], Exception]


@checked
def declared_geometry_edge_lengths(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    edges: np.ndarray,
    /,
    *,
    failure_factory: QualificationFailureFactory,
) -> np.ndarray:
    """Evaluate affine-source lengths with authenticated carrier publication.

    Source preparation verifies bounded coefficient representatives. Restricted
    affine roots separately verify retained bytes and once-rounded descendants.
    Neither publication law requires retained carriers to round the source map.
    """
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if geometry.exact_source is None and geometry.periodic_source is None:
        elements, routes, _ = geometry.resolve(mesh)
        if np.array_equal(np.asarray(geometry.coordinates), points) and all(
            element.element_id
            == coordinate_lagrange_element(block.cell_kind, 1).element_id
            and np.array_equal(np.asarray(route), np.asarray(block.vertices))
            for block, element, route in zip(mesh.blocks, elements, routes, strict=True)
        ):
            return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    work, memory = cell_geometry_preparation_bounds(geometry)
    cell_count = sum(block.cell_count for block in mesh.blocks)
    visits = cell_count + edges.shape[0] + 1
    with owned_coordinate_preparation(
        work + 512 * visits,
        memory + 4096 * visits,
        maximum_geometry_queries=cell_geometry_preparation_query_upper(geometry),
    ) as budget:
        budget.reserve(16 * edges.shape[0] + 8 * cell_count, 4096 * visits)
        vertices = _authenticated_affine_corners(
            mesh, geometry, points, budget, failure_factory
        )
        return _once_rounded_edge_lengths(mesh, edges, vertices, failure_factory)


def _admit_affine_corner(
    vertex: int,
    image: tuple[Fraction, ...],
    vertices: dict[int, tuple[Fraction, ...]],
    coefficients: CoordinateSourceBank,
    representatives: np.ndarray,
    points: np.ndarray,
    published_roots: bool,
    failure_factory: QualificationFailureFactory,
) -> None:
    """Admit one shared corner under retained or published-root carrier laws."""
    previous = vertices.get(vertex)
    retained_coefficient = (
        vertex < len(coefficients)
        and image == coefficients[vertex]
        and np.array_equal(
            representatives[vertex].view(np.uint64),
            points[vertex].view(np.uint64),
        )
    )
    if (
        previous is not None
        and previous != image
        or not published_roots
        and not retained_coefficient
        and not np.array_equal(
            np.asarray([float(value) for value in image], dtype=np.float64),
            points[vertex],
        )
    ):
        raise failure_factory(
            "native_lifecycle.source_edge_oracle",
            "native_lifecycle.source_edge_map must agree on shared vertices and their authenticated representatives; got a different geometry",
            {"vertex": vertex},
        )
    vertices[vertex] = image


def _authenticated_affine_corners(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    points: np.ndarray,
    budget: CoordinateEnclosureBudget,
    failure_factory: QualificationFailureFactory,
) -> dict[int, tuple[Fraction, ...]]:
    """Authenticate carriers and collect affine corners in block/cell order."""
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    coefficients = geometry.source_coordinates(budget=budget)
    representatives = np.asarray(geometry.coordinates, dtype=np.float64)
    restriction = geometry.restriction_source
    published_roots = (
        restriction is not None
        and restriction.carrier_publication == "published_affine_roots"
    )
    if published_roots:
        require_published_affine_carrier(geometry, mesh, budget=budget)
    vertices: dict[int, tuple[Fraction, ...]] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for cell, dofs in zip(
            np.asarray(block.vertices).tolist(),
            np.asarray(route).tolist(),
            strict=True,
        ):
            images = affine_simplex_corner_images(
                element, tuple(coefficients[index] for index in dofs), budget=budget
            )
            if images is None or len(images) != len(cell):
                raise failure_factory(
                    "native_lifecycle.source_edge_oracle",
                    "native_lifecycle.source_edge_map must declare affine simplex corner images; got an unsupported coordinate map",
                    {"geometry_layout_id": geometry.geometry_layout_id},
                )
            for vertex, image in zip(cell, images, strict=True):
                rational_image = tuple(
                    rational_coefficient(value, role="native_lifecycle.source_edge_map")
                    for value in image
                )
                _admit_affine_corner(
                    vertex,
                    rational_image,
                    vertices,
                    coefficients,
                    representatives,
                    points,
                    published_roots,
                    failure_factory,
                )
    return vertices


def _once_rounded_edge_lengths(
    mesh: CellMesh,
    edges: np.ndarray,
    vertices: dict[int, tuple[Fraction, ...]],
    failure_factory: QualificationFailureFactory,
) -> np.ndarray:
    """Publish exact rational norms with one final binary64 rounding."""
    if any(int(vertex) not in vertices for vertex in edges.ravel()):
        raise failure_factory(
            "native_lifecycle.source_edge_oracle",
            "native_lifecycle.source_edge_map must cover the declared edge endpoints; got an unbound vertex",
            {},
        )
    bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for image in vertices.values()
        for value in image
    )
    # A d<=3 squared norm has denominator height at most 12*bits.
    # This precision also separates binary64 midpoint squares at the
    # extreme IEEE exponents before the single binary64 publication.
    with localcontext() as context:
        context.prec = 8 * bits + 1024
        lengths = []
        for first, last in edges:
            squared = sum(
                (
                    (vertices[int(last)][axis] - vertices[int(first)][axis]) ** 2
                    for axis in range(mesh.ambient_dimension)
                ),
                Fraction(0),
            )
            lengths.append(
                float((Decimal(squared.numerator) / Decimal(squared.denominator)).sqrt())
            )
    return np.asarray(lengths, dtype=np.float64)


@final
class QualifiedSimplexArrays(StrictModule):
    points: np.ndarray
    cells: np.ndarray


@checked
def qualify_simplex_resources(
    result: CellMeshingResult, limits: MeshingLimits, /
) -> QualifiedSimplexArrays:
    """Prepare host simplex arrays and enforce their declared entity resources."""
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    cells = np.concatenate([np.asarray(block.vertices) for block in result.mesh.blocks])
    simplex_entity_limits(
        points,
        cells,
        limits,
        MeshingStageKind.GEOMETRY_AUDIT,
        cell_kind=result.mesh.blocks[0].cell_kind,
    )
    return QualifiedSimplexArrays(points=points, cells=cells)


@final
class NativeConsumerResult[T](StrictModule):
    result: CellMeshingResult
    value: T


@checked
def execute_native_consumer[T](
    result: CellMeshingResult,
    limits: MeshingLimits,
    call: Callable[[], T],
    /,
    *,
    failure_factory: QualificationFailureFactory,
    target: Callable[[T], CellMeshingResult] | None = None,
) -> NativeConsumerResult[T]:
    """Continue ended evidence without counting its imported work twice.

    The native scope admits the source chain before dispatch. Its published
    phase excludes that matched overlap because the retained preparation
    evidence already contributes the same work, queries, and elapsed time.
    """
    receipt = result.execution_evidence
    if receipt is None or receipt.owner_id is None:
        raise failure_factory(
            "_corpus_native_consumer",
            "Native corpus consumer requires owner-bound ended root execution evidence.",
            {"receipt": receipt},
        )
    receipt.require_valid()
    started = monotonic() - float(np.asarray(receipt.total_elapsed_seconds))
    with (
        NativeExecutionScope(limits.native_limits()) as budget,
        native_volume_source_execution_budget(
            limits, receipt, receipt.owner_id, scope=budget
        ),
    ):
        overlap = native_preparation_ledger(budget).overlap(budget, receipt)
        value = call()
    if budget.evidence is None:
        raise failure_factory(
            "_corpus_native_consumer",
            "Native corpus consumer did not retain ended scope evidence.",
            {"budget": budget},
        )
    bound = bind_native_execution_result(
        result if target is None else target(value),
        overlap.normalize(budget.evidence),
        limits,
        started,
        preparation_evidence=receipt,
    )
    return NativeConsumerResult(result=bound, value=value)


@final
class PreparedPlcAssociations(StrictModule):
    association_transfer: PlcAssociationTransfer
    prepared_id: str
    preparation_counters: tuple[tuple[str, int], ...]
    memory_counters: NativeExecutionMemoryCounters | None
    prepared_source: PreparedPlcSource


@checked
def prepare_plc_associations(
    source: NativePlcSource,
    result: CellMeshingResult,
    limits: MeshingLimits,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> PreparedPlcAssociations:
    """Prepare PLC strata within execution limits and validate association support."""
    with NativeExecutionScope(limits.native_limits()) as scope:
        prepared = prepare_plc_source(
            source.complex,
            source.source_id,
            source.source_revision,
            result.coordinate_contract,
            limits=limits,
            scope=scope,
            record_phase=record_phase,
        )
    prepared.association_transfer.source_associations(result)
    return PreparedPlcAssociations(
        association_transfer=prepared.association_transfer,
        prepared_id=prepared.prepared_id,
        preparation_counters=prepared.preparation_counters,
        memory_counters=prepared.memory_counters,
        prepared_source=prepared,
    )


@checked
def execute_native_capacity_trial(
    source: NativePolyhedralSource,
    request: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    result: CellMeshingResult,
    /,
    *,
    operation_started: float,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Execute one reduced-limit publication with the request's elapsed-time debit."""
    with native_volume_execution_budget(
        request.limits, operation_started=operation_started
    ):
        return (
            NativeMeshingProvider(options)
            .plan(
                source,
                request,
                coordinate_contract=result.coordinate_contract,
                record_phase=record_phase,
            )
            .execute(record_phase=record_phase)
        )
