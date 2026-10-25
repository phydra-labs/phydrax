# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from fractions import Fraction
from time import monotonic
from typing import final

import numpy as np

from ..._host_storage import declared_integer_storage_bytes
from ..._meshcore import (
    current_native_execution_scope,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
    NativeParallelExecution,
)
from ..._strict import StrictModule
from ...discretization import CellGeometrySpec, CellMesh
from ...discretization._cell_geometry import (
    cell_geometry_preparation_bounds,
    cell_geometry_preparation_query_upper,
    coordinate_lagrange_element,
    owned_coordinate_preparation,
    require_published_affine_carrier,
)
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    reserve_polynomial,
)
from ...discretization._coordinate_enclosure import (
    affine_simplex_corner_images,
    affine_simplex_source,
    CoordinateSourceBank,
    prepared_coordinate_source_bank,
    prepared_coordinate_vertex_images,
)
from ...discretization._coordinate_operands import (
    CoordinateOperandCollector,
    PreparedCoordinateOperands,
)
from ...discretization._exact_algebraic_geometry import (
    rational_coefficient,
    rounded_rational_square_root,
)
from ...typing import checked
from .._association import PlcAssociationTransfer
from .._contracts import MeshingLimits, VolumeMeshingSpec
from .._failures import MeshingFailure, MeshingFailureCategory
from .._measurements import NativeExecutionRecord, NativeMeshingPhaseRecorder
from .._native_preparation import (
    continue_native_completed_source_issue,
    native_preparation_ledger,
    stage_native_completed_source_issue,
)
from .._result import CellMeshingResult
from .._trace import MeshingStageKind
from .._volume_generation import (
    native_volume_source_execution_budget,
    prepare_plc_source,
    PreparedPlcSource,
)
from ..providers._native import NativeMeshingPlan, NativeMeshingProvider
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
    budget: CoordinateEnclosureBudget | None = None,
) -> np.ndarray:
    """Evaluate affine-source lengths with authenticated carrier publication.

    Source preparation verifies bounded coefficient representatives. Restricted
    affine roots separately verify retained bytes and once-rounded descendants.
    Neither publication law requires retained carriers to round the source map.
    """
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if geometry.exact_source is None and geometry.periodic_source is None:
        elements, routes, _ = geometry.resolve(mesh, budget=budget)
        if np.array_equal(np.asarray(geometry.coordinates), points) and all(
            element.element_id
            == coordinate_lagrange_element(block.cell_kind, 1).element_id
            and np.array_equal(np.asarray(route), np.asarray(block.vertices))
            for block, element, route in zip(mesh.blocks, elements, routes, strict=True)
        ):
            if budget is None:
                return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
            with budget.temporary_scope():
                budget.reserve(
                    edges.shape[0],
                    8 * edges.shape[0] * (2 * mesh.ambient_dimension + 1),
                )
                lengths = np.linalg.norm(
                    points[edges[:, 1]] - points[edges[:, 0]], axis=1
                )
                budget.retain_owner(lengths)
                return lengths
    if budget is None:
        work, memory = cell_geometry_preparation_bounds(geometry)
        visits = sum(block.cell_count for block in mesh.blocks) + edges.shape[0] + 1
        context = owned_coordinate_preparation(
            work + 512 * visits,
            memory + 4096 * visits,
            maximum_geometry_queries=cell_geometry_preparation_query_upper(geometry),
        )
    else:
        context = nullcontext(budget)
    with context as active_budget, active_budget.temporary_scope():
        # Source operands already belong to their prepared bank. This traversal
        # adds one vertex dictionary and rational-reference tuple per vertex,
        # not a new polynomial graph for every incident cell.
        active_budget.reserve(
            0, 64 + points.shape[0] * (128 + 128 * mesh.ambient_dimension)
        )
        vertices = _authenticated_affine_corners(
            mesh, geometry, points, active_budget, failure_factory
        )
        lengths = _once_rounded_edge_lengths(
            mesh, edges, vertices, failure_factory, active_budget
        )
        active_budget.retain_owner(lengths)
        return lengths


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
    coefficients = prepared_coordinate_source_bank(geometry, budget=budget)
    representatives = np.asarray(geometry.coordinates, dtype=np.float64)
    restriction = geometry.restriction_source
    published_roots = (
        restriction is not None
        and restriction.carrier_publication == "published_affine_roots"
    )
    if published_roots:
        require_published_affine_carrier(geometry, mesh, budget=budget)
    vertices: dict[int, tuple[Fraction, ...]] = {}
    if all(affine_simplex_source(element, budget=budget) for element in elements):
        images = prepared_coordinate_vertex_images(
            mesh,
            geometry,
            mismatch="native_lifecycle.source_edge_map",
            budget=budget,
        )
        if images is not None:
            for vertex, image in enumerate(images):
                if image is None:
                    continue
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


def _squared_edge_components(
    first: tuple[Fraction, ...],
    last: tuple[Fraction, ...],
    cache: dict[tuple[Fraction, Fraction], Fraction],
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, ...]:
    """Reuse actual scalar differences, never rounded coordinate differences."""
    components: list[Fraction] = []
    for a, b in zip(first, last, strict=True):
        key = (a, b)
        squared = cache.get(key)
        if squared is None:
            squared = cache.get((b, a))
        if squared is None:
            bits = (
                4
                * max(
                    a.numerator.bit_length(),
                    a.denominator.bit_length(),
                    b.numerator.bit_length(),
                    b.denominator.bit_length(),
                    1,
                )
                + 2
            )
            budget.reserve(0, 224 + 2 * declared_integer_storage_bytes(bits))
            with budget.temporary_scope():
                reserve_polynomial(2, 3, 0, bits, budget=budget)
                difference = b - a
                squared = difference * difference
                del difference
            cache[key] = squared
        components.append(squared)
    return tuple(components)


def _once_rounded_edge_lengths(
    mesh: CellMesh,
    edges: np.ndarray,
    vertices: dict[int, tuple[Fraction, ...]],
    failure_factory: QualificationFailureFactory,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Publish exact rational norms with one final binary64 rounding."""
    budget.reserve(
        edges.shape[0],
        256
        + 256 * mesh.ambient_dimension
        + np.dtype(np.float64).itemsize * edges.shape[0],
    )
    if any(int(vertex) not in vertices for vertex in edges.ravel()):
        raise failure_factory(
            "native_lifecycle.source_edge_oracle",
            "native_lifecycle.source_edge_map must cover the declared edge endpoints; got an unbound vertex",
            {},
        )
    differences: dict[tuple[Fraction, Fraction], Fraction] = {}
    norms: dict[tuple[Fraction, ...], float] = {}
    lengths = np.empty(edges.shape[0], dtype=np.float64)
    for index, (first, last) in enumerate(edges):
        components = _squared_edge_components(
            vertices[int(first)], vertices[int(last)], differences, budget
        )
        # Exact squared components determine the radicand as a multiset.
        # Canonical integer-pair ordering avoids rational comparison products;
        # the first evaluation still sums in the original coordinate order.
        norm_key = tuple(
            sorted(components, key=lambda value: (value.numerator, value.denominator))
        )
        length = norms.get(norm_key)
        if length is None:
            budget.reserve(0, 128 + 16 * mesh.ambient_dimension)
            bits = mesh.ambient_dimension * (
                max(
                    max(value.numerator.bit_length(), value.denominator.bit_length())
                    for value in components
                )
                + 1
            )
            with budget.temporary_scope():
                # Keep the original sixteen-unit norm charge for every actual
                # evaluation. The binary64 midpoint proof also needs at most
                # 2,200 exponent-shift bits beyond its rational operands.
                reserve_polynomial(
                    16, 2 * mesh.ambient_dimension + 4, 0, 2 * bits + 2200, budget=budget
                )
                length = rounded_rational_square_root(sum(components, Fraction(0)))
            norms[norm_key] = length
        lengths[index] = length
    return lengths


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
    call: Callable[[CoordinateEnclosureBudget], T],
    /,
    *,
    failure_factory: QualificationFailureFactory,
    target: Callable[[T], CellMeshingResult] | None = None,
    prepared_operands: PreparedCoordinateOperands | None = None,
    observe_managed_admission: bool = False,
) -> NativeConsumerResult[T]:
    """Continue the request that produced ``result`` without a fresh allowance.

    A consumer of a published result (acceptance, solve, cold continuation) is
    charged to that same request: the native scope imports its ended chain,
    and the published phase excludes the matched overlap because the retained
    preparation evidence already contributes the same work, queries, and
    elapsed time. A new adaptation request uses ``execute_native_request``.
    Explicit prepared operands must belong to this result and an exact earning
    receipt in its paid preparation chain; cold/archive consumers re-earn them.
    Managed-admission observation is explicit opt-in on this actual scope,
    before importing its paid chain. It does not renew any request allowance;
    unsupported or incomplete observation remains absent evidence.
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
    from ...lifecycle._meshing_sources import meshing_source_owner_reconstruction_scope

    with NativeExecutionScope(limits.native_limits()) as scope:
        if observe_managed_admission:
            scope.observe_managed_admission()
        with native_volume_source_execution_budget(
            limits, receipt, receipt.owner_id, scope=scope
        ):
            overlap = native_preparation_ledger(scope).overlap(scope, receipt)
            initial = scope.batch(0)
            initial.settle()
            budget = CoordinateEnclosureBudget(
                initial,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            # A view charges actual host work/live bytes without reserving the
            # remaining request. Nested native owners therefore retain their real
            # work/query/cavity/scratch headroom and the same nonrenewing deadline.
            with (
                budget.bind_ticket(scope.ticket_view()),
                meshing_source_owner_reconstruction_scope(budget=budget),
            ):
                if prepared_operands is not None:
                    prepared_operands.require(
                        result.mesh, result.geometry, receipt, budget,
                        coordinate_contract=result.coordinate_contract,
                    )
                value = call(budget)
                produced = result if target is None else target(value)
                if produced is result:
                    issue = continue_native_completed_source_issue(
                        scope, produced, receipt, overlap=overlap, budget=budget,
                    )
                else:
                    issue = stage_native_completed_source_issue(
                        scope, produced, owner_id=receipt.owner_id,
                        overlap=overlap, budget=budget,
                    )
    if scope.evidence is None:
        raise failure_factory(
            "_corpus_native_consumer",
            "Native corpus consumer did not retain ended scope evidence.",
            {"scope": scope},
        )
    bound = bind_native_execution_result(
        produced,
        overlap.normalize(scope.evidence),
        limits,
        started,
        preparation_evidence=receipt,
    )
    issue.complete(bound)
    return NativeConsumerResult(result=bound, value=value)


@checked
def execute_native_request[T](
    source: CellMeshingResult | NativeExecutionRecord,
    limits: MeshingLimits,
    call: Callable[[CoordinateEnclosureBudget], T],
    /,
    *,
    target: Callable[[T], CellMeshingResult],
    failure_factory: QualificationFailureFactory,
    workflow_remaining_seconds: float | None = None,
    observe_managed_admission: bool = False,
) -> NativeConsumerResult[T]:
    """Run one new complete request under its original allowance.

    ``call`` performs the whole request (including scientific archive restoration,
    preparation, execution and consumers) inside one root with the original
    ``limits``. An authenticated ended receipt may bootstrap a cold request before
    its source owners exist. That receipt remains history only: its earlier work,
    queries and elapsed time are never charged again. The absolute workflow clock
    intersects this request's own wall budget and is never reset.
    ``observe_managed_admission`` opts this same root into observation before
    recording history or opening its first batch; it creates no extra scope,
    allocation pool or resource allowance.
    """
    receipt = (
        source if isinstance(source, NativeExecutionRecord) else source.execution_evidence
    )
    if receipt is None or receipt.owner_id is None:
        raise failure_factory(
            "_corpus_native_request",
            "Native corpus request requires owner-bound ended source execution evidence.",
            {"receipt": receipt},
        )
    if current_native_execution_scope() is not None:
        raise failure_factory(
            "_corpus_native_request",
            "Native corpus request must open its own root allowance; got an active request scope.",
            {"scope": current_native_execution_scope()},
        )
    receipt.require_valid()
    native_limits = limits.native_limits()
    if workflow_remaining_seconds is not None:
        if (
            not np.isfinite(workflow_remaining_seconds)
            or workflow_remaining_seconds <= 0.0
        ):
            raise failure_factory(
                "_corpus_native_request",
                "Native corpus request must start before the workflow deadline.",
                {"workflow_remaining_seconds": workflow_remaining_seconds},
            )
        native_limits = replace(
            native_limits,
            maximum_seconds=min(
                native_limits.maximum_seconds, workflow_remaining_seconds
            ),
        )
    # The decoder's earned owners must remain available to this same request's
    # physical consumers; they never escape into a process-global proof cache.
    from ...lifecycle._meshing_sources import meshing_source_owner_reconstruction_scope

    with NativeExecutionScope(native_limits) as scope, ExitStack() as admission:
        if observe_managed_admission:
            scope.observe_managed_admission()
            view = admission.enter_context(scope.ticket_view())
            admission.enter_context(view.native_target())
        native_preparation_ledger(scope).record_history(
            scope, receipt,
            completed_source=None if isinstance(source, NativeExecutionRecord) else source,
        )
        started = scope.operation_started
        initial = scope.batch(0)
        initial.settle()
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with (
            budget.bind_ticket(scope.ticket_view()),
            meshing_source_owner_reconstruction_scope(budget=budget),
        ):
            value = call(budget)
            produced = target(value)
            request = produced.execution_evidence
            if (
                request is None and not isinstance(source, NativeExecutionRecord)
            ) or (request is not None and request.owner_id is None):
                raise failure_factory(
                    "_corpus_native_request",
                    "Native corpus request target must bind its own ended request record.",
                    {"request": request},
                )
            issue = stage_native_completed_source_issue(
                scope, produced,
                owner_id=produced.result_id if request is None else request.owner_id,
                budget=budget,
            )
    if scope.evidence is None:
        raise failure_factory(
            "_corpus_native_request",
            "Native corpus request did not retain ended scope evidence.",
            {"scope": scope},
        )
    bound = bind_native_execution_result(
        produced,
        scope.evidence,
        limits,
        started,
        history_evidence=issue.history_evidence,
    )
    issue.complete(bound)
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
    prepared_operands: PreparedCoordinateOperands | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedPlcAssociations:
    """Prepare PLC strata within execution limits and validate association support."""
    if prepared_operands is None:
        with NativeExecutionScope(limits.native_limits()) as scope:
            with scope.ticket_view() as ticket:
                source_budget = budget if budget is not None else CoordinateEnclosureBudget(
                    ticket, maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                prepared = prepare_plc_source(
                    source.complex,
                    source.source_id,
                    source.source_revision,
                    result.coordinate_contract,
                    limits=limits,
                    scope=scope,
                    record_phase=record_phase,
                    source_result=result,
                    coordinate_budget=source_budget,
                )
                prepared.association_transfer.source_associations(result, budget=source_budget)
    else:
        receipt = result.execution_evidence
        if budget is None or receipt is None:
            raise ValueError("Prepared PLC operands require the current paid request ledger.")
        prepared_operands.require(
            result.mesh, result.geometry, receipt, budget,
            coordinate_contract=result.coordinate_contract,
        )
        bank = prepared_operands._plc_source
        certificate = result.certification
        if (
            bank is None or bank.source is not source
            or bank.specification.limits is not limits
            or bank.coordinate_contract is not result.coordinate_contract
            or bank.domain is None or certificate is None
            or bank.domain is not certificate.request.domain
        ):
            raise ValueError("Prepared PLC source operands must retain the exact source, declared domain, frame and generating policy owners.")
        with budget.live_storage() as storage:
            storage.set_bound(
                65536 + 4096 * (
                    bank.constraints.points.shape[0]
                    + bank.constraints.input_triangles.shape[0]
                    + bank.constraints.plc_edges.shape[0]
                    + source.complex.polygon_facets.size
                )
            )
            prepared = PreparedPlcSource(
                source.complex, source.source_id, source.source_revision,
                result.coordinate_contract, bank.constraints,
                maximum_support_queries=limits.maximum_geometry_queries,
                source_domain=bank.domain, source_result=result, budget=budget,
            )
            budget.retain_owner(prepared, temporary_storage_credit=storage)
        prepared.association_transfer.source_associations(result, budget=budget)
    return PreparedPlcAssociations(
        association_transfer=prepared.association_transfer,
        prepared_id=prepared.prepared_id,
        preparation_counters=prepared.preparation_counters,
        memory_counters=prepared.memory_counters,
        prepared_source=prepared,
    )


@checked
def execute_native_capacity_trial(
    source: NativePlcSource | NativePolyhedralSource,
    request: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    result: CellMeshingResult,
    /,
    *,
    operation_started: float,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    coordinate_operands: CoordinateOperandCollector | None = None,
    prepared_plan: NativeMeshingPlan | None = None,
    root_maximum_work_units: int | None = None,
) -> CellMeshingResult:
    """Apply the workflow deadline without changing request resource ownership."""
    if (prepared_plan is None) != (root_maximum_work_units is None):
        raise ValueError("A mechanical work trial requires its original prepared plan.")
    if prepared_plan is not None and (
        prepared_plan.source is not source
        or prepared_plan.specification is not request
        or prepared_plan.options is not options
        or prepared_plan.preparation_evidence is None
        or result.execution_evidence is None
        or result.execution_evidence.owner_id != prepared_plan.plan_id
        or result.execution_evidence.preparation_evidence is not prepared_plan.preparation_evidence
    ):
        raise ValueError("Mechanical work admission must retain the original source, specification, options and publication plan.")
    def remaining_seconds(policy: NativeParallelExecution | None = None) -> float:
        elapsed = max(0.0, monotonic() - operation_started)
        remaining = request.limits.maximum_wall_seconds - elapsed
        if policy is not None:
            remaining = min(remaining, policy.remaining_seconds())
        if remaining <= 0.0:
            raise MeshingFailure(
                MeshingFailureCategory.TIMED_OUT,
                "capacity_trial.workflow_deadline must admit the new request; got an expired allowance",
                stage=MeshingStageKind.VOLUME_FILL.value,
                requested=(("maximum_wall_seconds", request.limits.maximum_wall_seconds),),
                achieved=(
                    ("elapsed_wall_seconds", elapsed),
                    ("remaining_workflow_seconds", max(0.0, remaining)),
                ),
            )
        return remaining

    remaining = remaining_seconds()
    try:
        # Match ordinary generation: preparation and execution retain their
        # actual paid imports, not an additional resource-parent pool. The
        # existing parallel policy carries the earlier absolute expiry through
        # both scopes, including final publication and any enclosing policy.
        workflow = NativeParallelExecution(None, remaining)
        with workflow:
            remaining_seconds(workflow)
            if prepared_plan is None:
                plan = NativeMeshingProvider(options).plan(
                    source,
                    request,
                    coordinate_contract=result.coordinate_contract,
                    record_phase=record_phase,
                )
                remaining_seconds(workflow)
                admitted = plan.execute(
                    record_phase=record_phase, coordinate_operands=coordinate_operands
                )
            else:
                assert root_maximum_work_units is not None
                admitted = prepared_plan._execute_with_root_work_admission(
                    root_maximum_work_units,
                    record_phase=record_phase, coordinate_operands=coordinate_operands,
                )
            remaining_seconds(workflow)
    except MeshcoreError as error:
        if error.status not in (
            MeshcoreStatus.TIMEOUT,
            MeshcoreStatus.CAPACITY_EXCEEDED,
        ):
            raise
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT
            if error.status is MeshcoreStatus.TIMEOUT
            else MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(error),
            stage=MeshingStageKind.VOLUME_FILL.value,
            requested=(("maximum_wall_seconds", request.limits.maximum_wall_seconds),),
            achieved=(("elapsed_wall_seconds", max(0.0, monotonic() - operation_started)),),
        ) from error
    # The native API accepts a relative duration. Recheck the original absolute
    # clock after context exit, so host-to-native entry latency cannot extend
    # the workflow's publication deadline.
    remaining_seconds()
    return admitted
