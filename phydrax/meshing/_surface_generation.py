#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native curved-surface generation over a compiled meshing domain.

Feature curves are discretized once (the native arc-length curve route) and
their vertices shared by every patch that uses them, so shared curves,
periodic seams and closed surfaces are conforming by construction rather than
welded by proximity. Each patch is triangulated in an affine execution chart
conditioned by RMS first-order physical speeds and one common length. The
source UVs stay unchanged for evaluation and certification. Constrained
Delaunay triangulation of its boundary loops precedes batched physical-space
refinement. A round measures every triangle on the physical
surface (effective edge length through the chart midpoint, minimum physical
angle, sampled deviation from the surface, agreement with the oriented source
normal), places one chart point per violating triangle, evaluates the batch
through the source query and inserts it with the native bounded
reconnection kernel (predicate chart validity, physical Delaunay flips). Chart
orientation is revalidated with exact predicates after every batch.

A declared pole keeps one physical corner identity and its full chart
chain. Source-bound ring depths and shared adjacent seam nodes prepare its
fan before CDT. Further pole refinement splits legal radial edges, never
inserts a centroid into a physically collapsed triangle. A terminal certified
rational pole split opens into the physical fan by exactly convex flips, so
its root binding is published rather than left in zero-width children.
Continuous source interpolation, certified trim-homotopy composition and
normal-turn bounds remain independent of sampled checks and share the
source physical allowance.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax.discretization._cell_geometry_validity as _owner_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._domain_chart_chain as domain_chart_chain
import phydrax.geometry._domain_numerics as domain_numerics
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._meshing_domain as _dependency_phydrax_geometry__meshing_domain
import phydrax.geometry._source_interpolation as _dependency_phydrax_geometry__source_interpolation
import phydrax.geometry.brep._patches as _dependency_phydrax_geometry_brep__patches
import phydrax.geometry.brep._placed as _dependency_phydrax_geometry_brep__placed
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._domain as _dependency_phydrax_meshing__domain
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._metric as _dependency_phydrax_meshing__metric
import phydrax.meshing._sizing as _dependency_phydrax_meshing__sizing
import phydrax.meshing._surface_construction as surface_construction_owner
import phydrax.meshing.providers._native_curve as _dependency_phydrax_meshing_providers__native_curve
import phydrax.meshing.providers._native_planar_domain as _dependency_phydrax_meshing_providers__native_planar
import phydrax.meshing.providers._native_route_checks as _dependency_phydrax_meshing_providers__native_publication
import phydrax.meshing.providers._native_schedules as _dependency_phydrax_meshing_providers__native_schedules
import phydrax.meshing.providers._native_sources as _dependency_phydrax_meshing_providers__native_sources

from .._meshcore import (
    exact_orient2d,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionScope,
    NativeResourceFailure,
    surface_reconnect,
)
from .._strict import StrictModule
from ..geometry import ConstrainedDelaunayTriangulation
from ..geometry._atlas import AbstractTrimCurve, CurveTrimLoop
from ..geometry._chart_restriction import (
    empty_chart_restrictions,
    ExactRationalChartRestriction,
    validate_chart_restrictions,
)
from ..geometry._rational import fraction_down, fraction_up
from ..geometry.brep._request_preparations import TrimRequestPreparations
from ..typing import checked
from ._measurements import (
    measure_phase,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from ._trace import MeshingStageKind


# Tangent-fan angle represented by one collapsed pole segment.
_POLE_FAN_ANGLE = math.pi / 3.0
_POLE_SAMPLES = 17
# Smallest barycentric weight of a clamped refinement point.
_CLAMP = 0.02
# Relative slack of the size criterion before a triangle is refined.
_SIZE_SLACK = 1.0e-9
# Construction stays inside the hard metric bound despite interpolation rounding.
_METRIC_AIM = 1.0 - 256 * np.finfo(np.float64).eps
# Chart probes per axis estimating chart speeds, and the seed-grid cap per axis.
_SPEED_PROBES = 9
_SEED_AXIS = 64


@final
@dataclass(slots=True)
class RefinementTotals:
    """Accumulate committed surface refinement work in patch order."""

    rounds: int = 0
    inserted: int = 0
    refused: int = 0
    flips: int = 0
    incomplete: int = 0


@final
class RestrictionBatch(StrictModule):
    """Carry one insertion batch with aligned rational edge restrictions."""

    charts: np.ndarray
    spacing: np.ndarray
    hints: np.ndarray
    parents: np.ndarray
    hard: np.ndarray
    unresolved: np.ndarray
    insert_edges: np.ndarray
    required: np.ndarray
    edges: np.ndarray
    parameters: np.ndarray

    def __init__(
        self,
        charts: np.ndarray,
        spacing: np.ndarray,
        hints: np.ndarray,
        parents: np.ndarray,
        hard: np.ndarray,
        unresolved: np.ndarray,
        insert_edges: np.ndarray,
        required: np.ndarray,
        edges: np.ndarray,
        parameters: np.ndarray,
        /,
    ) -> None:
        count = hints.size
        if (
            required.shape != (count,)
            or required.dtype != np.bool_
            or edges.shape != (count, 2)
            or edges.dtype != np.int64
            or parameters.shape != (count, 2)
            or parameters.dtype != np.int64
            or not np.array_equal(edges[required], insert_edges[required])
            or np.any(edges[~required] != -1)
            or np.any(parameters[~required] != 0)
        ):
            raise ValueError(
                "RestrictionBatch.restrictions must align with rational edge insertions; "
                f"got required={required.shape}/{required.dtype}, "
                f"edges={edges.shape}/{edges.dtype}, "
                f"parameters={parameters.shape}/{parameters.dtype}."
            )
        self.charts = charts
        self.spacing = spacing
        self.hints = hints
        self.parents = parents
        self.hard = hard
        self.unresolved = unresolved
        self.insert_edges = insert_edges
        self.required = required
        self.edges = edges
        self.parameters = parameters


@final
class _Patch:
    """Mutable chart triangulation of one patch during refinement."""

    __slots__ = (
        "charts",
        "constrained",
        "identities",
        "pole_ids",
        "index",
        "normals",
        "points",
        "triangles",
        "boundary_edges",
        "boundary_uses",
        "trim_deviation_bounds",
        "restriction_required",
        "restriction_edges",
        "restriction_parameters",
    )

    def __init__(
        self,
        index: int,
        charts: np.ndarray,
        points: np.ndarray,
        normals: np.ndarray,
        identities: np.ndarray,
        pole_ids: np.ndarray,
        triangles: np.ndarray,
        constrained: np.ndarray,
        boundary_edges: np.ndarray,
        boundary_uses: np.ndarray,
        trim_deviation_bounds: np.ndarray,
        restriction_required: np.ndarray,
        restriction_edges: np.ndarray,
        restriction_parameters: np.ndarray,
        /,
    ) -> None:
        self.index = index
        self.charts = charts
        self.points = points
        self.normals = normals
        self.identities = identities
        self.pole_ids = pole_ids
        self.triangles = triangles
        self.constrained = constrained
        self.boundary_edges = boundary_edges
        self.boundary_uses = boundary_uses
        self.trim_deviation_bounds = trim_deviation_bounds
        self.restriction_required = restriction_required
        self.restriction_edges = restriction_edges
        self.restriction_parameters = restriction_parameters


def _patch_restrictions(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain, state: _Patch, /
) -> tuple[np.ndarray, tuple[ExactRationalChartRestriction, ...], np.ndarray]:
    vertices = np.flatnonzero(state.restriction_required).astype(np.int64)
    edges = state.restriction_edges[vertices]
    if (
        edges.shape != (vertices.size, 2)
        or np.any(edges < 0)
        or np.any(edges >= state.charts.shape[0])
    ):
        raise ValueError(
            f"SurfaceConstruction.restriction_edges must index live patch vertices; got shape={edges.shape}, vertices={state.charts.shape[0]}."
        )
    endpoint_charts = (
        state.charts[edges] if vertices.size else np.empty((0, 2, 2), dtype=np.float64)
    )
    return validate_chart_restrictions(
        state.charts,
        state.restriction_required,
        vertices,
        edges,
        endpoint_charts,
        state.restriction_parameters[vertices],
        domain.source_revision,
        state.index,
    )


def _restriction_orientation_signs(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    exact, _, _ = _patch_restrictions(domain, state)
    triangles = state.triangles
    affected = np.any(state.restriction_required[triangles], axis=1)
    signs = exact_orient2d(
        state.charts[triangles[:, 0]],
        state.charts[triangles[:, 1]],
        state.charts[triangles[:, 2]],
    )
    for row in np.flatnonzero(affected):
        first, second, third = triangles[row]
        ax, ay = exact[first]
        bx, by = exact[second]
        cx, cy = exact[third]
        determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        signs[row] = (determinant > 0) - (determinant < 0)
    return signs


def _fraction_bounds(value: Fraction, /) -> tuple[float, float]:
    if not np.isfinite(float(value)):
        raise ValueError(
            f"SurfaceConstruction.restriction_coordinate must fit binary64 range; got {value}."
        )
    return fraction_down(value), fraction_up(value)


def _restriction_physical_errors(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    exact, _, coordinate_errors = _patch_restrictions(domain, state)
    result = np.zeros((state.charts.shape[0],), dtype=np.float64)
    rows = np.flatnonzero(state.restriction_required)
    if not rows.size:
        return result
    boxes = []
    for row in rows:
        bounds = [_fraction_bounds(exact[row, axis]) for axis in range(2)]
        boxes.append(
            np.stack(
                (
                    np.minimum(
                        np.asarray([bound[0] for bound in bounds]),
                        state.charts[row],
                    ),
                    np.maximum(
                        np.asarray([bound[1] for bound in bounds]),
                        state.charts[row],
                    ),
                )
            )
        )
    lower, upper = domain.patches[state.index].surface.derivative_bounds_batch(
        np.asarray(boxes), order=1, budget=budget
    )
    jacobian = np.linalg.norm(np.maximum(np.abs(lower), np.abs(upper)), axis=1)
    result[rows] = np.nextafter(
        np.sum(jacobian * coordinate_errors[rows], axis=1)
        + 256 * np.finfo(np.float64).eps * domain.scale,
        np.inf,
    )
    return result


def _vertex_normals(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    charts: np.ndarray,
    center: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Chart-order source normals, with limits at collapsed sides.

    Native triangles are counterclockwise in the parameter chart even when the
    authored face orientation is reversed. The reconnection kernel therefore
    consumes the raw ``du x dv`` sense; publication applies the authored
    orientation afterward. Only declared poles admit singular limits.
    """

    rows = np.full((charts.shape[0],), patch)
    normals, regular = domain.oriented_normals(rows, charts, budget=budget)
    if not np.all(regular):
        limits, found = domain.oriented_pole_limits(
            patch, charts[~regular], center, budget=budget
        )
        if not np.all(found):
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                f"SurfaceConstruction.source_normals must resolve singular charts through a declared limit; got patch {patch} without a source normal limit.",
                entity_ids=(patch,),
            )
        normals[~regular] = limits
    return domain.patches[patch].orientation * normals


def _failure(
    category: _leaf_phydrax_meshing__failures.MeshingFailureCategory,
    message: str,
    /,
    *,
    entity_ids: tuple[int, ...] = (),
    requested: tuple[
        tuple[str, _leaf_phydrax_meshing__failures.FailureQuantity], ...
    ] = (),
    achieved: tuple[
        tuple[str, _leaf_phydrax_meshing__failures.FailureQuantity], ...
    ] = (),
    provider_code: str = "",
) -> _leaf_phydrax_meshing__failures.MeshingFailure:
    return _leaf_phydrax_meshing__failures.MeshingFailure(
        category,
        message,
        stage=MeshingStageKind.SURFACE_MESHING.value,
        entity_ids=entity_ids,
        requested=requested,
        achieved=achieved,
        provider_code=provider_code,
    )


@contextmanager
def _interpolation_bounds(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    charts: np.ndarray,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    scratch_bytes: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    restriction_bank: _dependency_phydrax_geometry_brep__patches.BernsteinRestrictionBank
    | None = None,
    meridian_jets: list[_dependency_phydrax_geometry_brep__patches.MeridianOverlaps]
    | None = None,
) -> Iterator[np.ndarray]:
    """Borrow the source-bound allowance beneath the surface ledger."""
    owner = budget
    start = owner.work_units

    def queries(count: int) -> None:
        ledger.observe_native(count, 0, MeshingStageKind.SURFACE_MESHING)

    try:
        with owner.measurement() as phase:
            with _source_bound_scope(
                ledger,
                scratch_bytes,
                stage=MeshingStageKind.SURFACE_MESHING,
                scope=scope,
                budget=budget,
            ):
                result = domain.interpolation_bounds(
                    patch,
                    charts,
                    budget=owner,
                    reserve_queries=queries,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )

                yield result
                del result
    except _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError as error:
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"SurfaceConstruction.source_interpolation must fit the source allowance; got exhausted patch {patch}, resource={error.resource}.",
            entity_ids=(patch,),
            requested=((error.resource, error.limit),),
            achieved=(
                (error.resource, error.requested),
                ("completed_source_work", error.completed),
                ("geometry_queries", ledger.queries),
                ("work_units", ledger.work + owner.work_units - start),
                ("source_peak_scratch_bytes", phase.peak_bytes_upper),
            ),
        ) from error
    except (
        _dependency_phydrax_geometry__source_interpolation.SourceInterpolationFailure
    ) as error:
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            f"SurfaceConstruction.source_interpolation must satisfy the source bound check; got patch {patch}, check={error.check}.",
            entity_ids=(patch,),
            provider_code=error.check,
            achieved=(
                ("source_row", error.row),
                ("source_spans", error.spans),
                ("source_span_visits", error.visits),
                ("source_queries", error.queries),
                ("geometry_queries", ledger.queries),
                ("work_units", ledger.work + owner.work_units - start),
            ),
        ) from error


@contextmanager
def _source_bound_scope(
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    scratch_bytes: int,
    /,
    *,
    stage: MeshingStageKind = MeshingStageKind.CURVE_MESHING,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Iterator[None]:
    """Bind source bounds to the realization's coordinate budget."""
    maximum_work = ledger.limits.maximum_work_units - ledger.work
    maximum_scratch = max(0, ledger.limits.maximum_scratch_bytes - scratch_bytes)
    if not budget.ticket.settled:
        with (
            budget.measurement(),
            budget.bound_stage(
                maximum_work,
                maximum_scratch,
                starting_work_units=budget.work_units,
            ),
            budget.temporary_scope(),
        ):
            yield
        return
    remaining = scope.remaining()
    owner = budget
    start = owner.work_units
    ticket = scope.batch(
        min(maximum_work, remaining.remaining_work_units),
        queries_upper=remaining.remaining_geometry_queries,
        bytes_upper=min(maximum_scratch, remaining.remaining_managed_bytes),
    )
    try:
        with (
            owner.measurement() as phase,
            owner.bind_ticket(ticket),
            budget.ticket.native_target(),
            owner.bound_stage(
                maximum_work,
                maximum_scratch,
                starting_work_units=start,
            ),
            owner.temporary_scope(),
        ):
            yield
    except _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError as error:
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"SurfaceConstruction.source_bounds must fit the source allowance; got exhausted resource={error.resource}.",
            requested=((error.resource, error.limit),),
            achieved=(
                (error.resource, error.requested),
                ("completed_source_work", error.completed),
                ("geometry_queries", ledger.queries),
                ("work_units", ledger.work + owner.work_units - start),
                ("source_peak_scratch_bytes", phase.peak_bytes_upper),
            ),
        ) from error
    finally:
        ledger.observe_exact(owner.work_units - start, stage)


def _bisection_rows(split: np.ndarray, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """New rows of the old values and retained intervals, and the fresh intervals.

    Bisecting interval ``i`` inserts one value after old value ``i``; every
    unsplit interval keeps its endpoints and therefore its measured bounds.
    """
    shift = np.concatenate(((0,), np.cumsum(split)))
    value_rows = np.arange(split.size + 1) + shift
    interval_rows = np.arange(split.size) + shift[:-1]
    fresh = np.ones((split.size + int(np.sum(split)),), dtype=np.bool_)
    fresh[interval_rows[~split]] = False
    return value_rows, interval_rows[~split], fresh


@final
class _CurveConstraints(StrictModule):
    owner: _dependency_phydrax_geometry__meshing_domain.PatchCurveUse
    angular_uses: tuple[
        tuple[int, _dependency_phydrax_geometry__meshing_domain.PatchCurveUse, float], ...
    ]


@final
class _CurveIntervalState(StrictModule):
    """Retain measured intervals and stalled-ribbon evidence across bisection."""

    ribbons: list[np.ndarray]
    turns: list[np.ndarray]
    points: np.ndarray
    known: np.ndarray
    previous_ribbons: dict[int, float] = eqx.field(static=True)
    stagnant_ribbons: dict[int, int] = eqx.field(static=True)


@final
class _CurveSplit(StrictModule):
    split: np.ndarray


def _prepare_curve_constraints(
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    curve: int,
    angular_required: bool,
    maximum_normal_angle: float,
    /,
) -> _CurveConstraints:
    """Prepare incident angular targets without changing shared curve ownership."""
    domain = compiled.domain
    owner_patch, owner_loop, owner_position = domain.curve_owners[curve]
    owner = domain.patches[owner_patch].loops[owner_loop][owner_position]
    match owner:
        case _dependency_phydrax_geometry__meshing_domain.PatchCurveUse():
            pass
        case _:
            raise TypeError(
                f"MeshingDomain.curve_owners must select a PatchCurveUse; got {type(owner).__name__}."
            )
    angular_uses: list[
        tuple[int, _dependency_phydrax_geometry__meshing_domain.PatchCurveUse, float]
    ] = []
    if angular_required:
        for patch, use in domain.curve_uses(curve):
            row = np.searchsorted(compiled.patches, patch)
            target_angle = maximum_normal_angle
            if row < compiled.patches.size and compiled.patches[row] == patch:
                target_angle = min(target_angle, float(compiled.patch_normal_angles[row]))
            angular_uses.append((patch, use, target_angle))
    return _CurveConstraints(owner, tuple(angular_uses))


def _measure_curve_ribbons(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    prepared: _dependency_phydrax_meshing_providers__native_curve.PreparedCurveNetwork,
    curve_row: int,
    curve: int,
    values: np.ndarray,
    constraints: _CurveConstraints,
    ribbon_targets: list[
        tuple[int, _dependency_phydrax_geometry__meshing_domain.PatchCurveUse, float]
    ],
    measured: np.ndarray,
    state: _CurveIntervalState,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    split: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> _CurveSplit:
    """Measure new trim ribbons and refuse a repeatedly stagnant local bound.

    ``preparations`` is the curve request's mapping: endpoint-root contractions
    and interval programs are deterministic functions of immutable sources, so
    every round and incident use reuses them instead of re-deriving them.
    """
    owner = constraints.owner
    points, known, ribbons = state.points, state.known, state.ribbons
    previous_ribbons, stagnant_ribbons = state.previous_ribbons, state.stagnant_ribbons
    missing = np.flatnonzero(~known)
    ledger.reserve(missing.size, 0, MeshingStageKind.CURVE_MESHING)
    if missing.size:
        points[missing] = (
            _dependency_phydrax_meshing_providers__native_curve.curve_points(
                prepared.charts, curve_row, values[missing]
            )
        )
    known[:] = True
    for use_row, (
        incident_patch,
        incident_use,
        ribbon_target,
    ) in enumerate(ribbon_targets):
        order = np.arange(values.size)
        reversed_use = incident_use.first != owner.first
        if reversed_use:
            order = order[::-1]
        actual = owner.first + values[order] * (owner.last - owner.first)
        actual[0], actual[-1] = incident_use.first, incident_use.last
        ledger.reserve(2 * measured.size, 0, MeshingStageKind.CURVE_MESHING)
        if measured.size:
            ribbons[use_row][measured] = domain.trim_ribbon_bounds(
                incident_patch,
                incident_use,
                actual,
                points[order],
                intervals=values.size - 2 - measured if reversed_use else measured,
                budget=budget,
                preparations=preparations,
            )
        local = ribbons[use_row]
        local_split = local > ribbon_target
        split |= local_split
        maximum = float(np.max(local[local_split], initial=0.0))
        if maximum == 0.0:
            previous_ribbons.pop(use_row, None)
            stagnant_ribbons.pop(use_row, None)
            continue
        previous = previous_ribbons.get(use_row, np.inf)
        # An unbounded interval enclosure that bisection turns
        # finite is progress; only a finite bound can stagnate.
        # Persistently unbounded intervals still terminate
        # through the ledger and deadline.
        stagnant = (
            stagnant_ribbons.get(use_row, 0) + 1
            if np.isfinite(maximum) and maximum >= previous
            else 0
        )
        previous_ribbons[use_row] = maximum
        stagnant_ribbons[use_row] = stagnant
        if stagnant >= 2:
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                f"SurfaceConstruction.curve_trim_ribbon must decrease under interval bisection; got unrecoverable local trim ribbon on curve {curve}.",
                entity_ids=(curve, incident_patch),
                requested=(("maximum_deviation", ribbon_target),),
                achieved=(("local_trim_ribbon", maximum),),
            )
    return _CurveSplit(split)


def _measure_curve_turns(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    values: np.ndarray,
    constraints: _CurveConstraints,
    measured: np.ndarray,
    state: _CurveIntervalState,
    split: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CurveSplit:
    """Measure new source-normal spans while retaining corner angular headroom."""
    owner, angular_uses, turns = constraints.owner, constraints.angular_uses, state.turns
    actual = owner.first + values * (owner.last - owner.first)
    actual[0], actual[-1] = owner.first, owner.last
    for turn, (patch, use, target_angle) in zip(turns, angular_uses, strict=True):
        if measured.size:
            boxes = []
            for first, last in zip(actual[measured], actual[measured + 1], strict=True):
                first_, last_ = sorted((float(first), float(last)))
                box = use.pcurve.bounding_box(first_, last_)
                boxes.append(np.stack((box[0], box[1], box[0])))
            turn[measured] = domain.normal_turn_bounds(
                patch, np.asarray(boxes), budget=budget
            )
        # A corner triangle retains two constrained source
        # edges. Leave angular room for its interior span;
        # neither edge can be split by patch-local recovery.
        split |= turn > 0.5 * target_angle
    return _CurveSplit(split)


def _bisect_curve(
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    prepared: _dependency_phydrax_meshing_providers__native_curve.PreparedCurveNetwork,
    curve_row: int,
    curve: int,
    values: np.ndarray,
    target: float,
    ribbon_targets: list[
        tuple[int, _dependency_phydrax_geometry__meshing_domain.PatchCurveUse, float]
    ],
    angular_required: bool,
    maximum_normal_angle: float,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    started: float,
    reserved_bytes: int,
    count: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> np.ndarray:
    """Bisect one shared curve until every incident source bound is met.

    Shared source edges cannot be changed by patch-local interior refinement.
    Each offending source interval is bisected here, before every incident
    patch reconnects through its CDT, so all uses retain one conforming curve
    vertex and interval ledger. Interpolation, trim-ribbon and normal-turn
    bounds of unsplit intervals are retained; only bisected children are
    measured again.
    """
    domain = compiled.domain
    constraints = _prepare_curve_constraints(
        compiled, curve, angular_required, maximum_normal_angle
    )
    fresh = np.ones((values.size - 1,), dtype=np.bool_)
    interpolation = np.zeros((values.size - 1,), dtype=np.float64)
    ribbons = [np.zeros((values.size - 1,), dtype=np.float64) for _ in ribbon_targets]
    turns = [
        np.zeros((values.size - 1,), dtype=np.float64) for _ in constraints.angular_uses
    ]
    points = np.empty((values.size, 3), dtype=np.float64)
    known = np.zeros((values.size,), dtype=np.bool_)
    previous_ribbons: dict[int, float] = {}
    stagnant_ribbons: dict[int, int] = {}
    state = _CurveIntervalState(
        ribbons, turns, points, known, previous_ribbons, stagnant_ribbons
    )
    while True:
        _dependency_phydrax_meshing_providers__native_publication.check_deadline(
            started, ledger.limits, MeshingStageKind.CURVE_MESHING
        )
        measured = np.flatnonzero(fresh)
        with _source_bound_scope(
            ledger,
            reserved_bytes + values.nbytes + interpolation.nbytes,
            scope=scope,
            budget=budget,
        ):
            if measured.size:
                interpolation[measured] = domain.curve_interpolation_bounds(
                    curve, values, intervals=measured, budget=budget
                )
            if np.isfinite(target) and np.any(~np.isfinite(interpolation)):
                raise _failure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    f"SurfaceConstruction.curve_interpolation must provide a continuous bound; got unbounded curve {curve}.",
                    entity_ids=(curve,),
                )
            split = interpolation > target
            if ribbon_targets:
                split = _measure_curve_ribbons(
                    domain,
                    prepared,
                    curve_row,
                    curve,
                    values,
                    constraints,
                    ribbon_targets,
                    measured,
                    state,
                    ledger,
                    split,
                    budget=budget,
                    preparations=preparations,
                ).split
            if constraints.angular_uses:
                split = _measure_curve_turns(
                    domain,
                    values,
                    constraints,
                    measured,
                    state,
                    split,
                    budget=budget,
                ).split
        if not np.any(split):
            return values
        del state
        ledger.reserve(0, values.size, MeshingStageKind.CURVE_MESHING)
        if count + values.size + int(np.sum(split)) > ledger.limits.maximum_vertices:
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "SurfaceConstruction.curve_vertices must fit the vertex budget; got oversized curve fidelity refinement.",
            )
        middle = 0.5 * (values[:-1][split] + values[1:][split])
        if np.any((middle == values[:-1][split]) | (middle == values[1:][split])):
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                f"SurfaceConstruction.curve_parameter must distinguish bisected interval children; got no distinct execution parameter on curve {curve}.",
                entity_ids=(curve,),
            )
        value_rows, retained, fresh = _bisection_rows(split)
        values = np.sort(np.concatenate((values, middle)))
        carried = np.empty((fresh.size,), dtype=np.float64)
        carried[retained] = interpolation[~split]
        interpolation = carried
        for banks in (ribbons, turns):
            for index, bank in enumerate(banks):
                carried = np.empty((fresh.size,), dtype=np.float64)
                carried[retained] = bank[~split]
                banks[index] = carried
        carried_points = np.empty((values.size, 3), dtype=np.float64)
        carried_points[value_rows] = points
        points = carried_points
        carried_known = np.zeros((values.size,), dtype=np.bool_)
        carried_known[value_rows] = known
        known = carried_known
        state = _CurveIntervalState(
            ribbons, turns, points, known, previous_ribbons, stagnant_ribbons
        )


def _split_source_loop_parameters(
    uses: tuple[_dependency_phydrax_geometry__meshing_domain.PatchCurveUse, ...],
    partitions: list[np.ndarray],
    reversed_uses: tuple[bool, ...],
    parameters: dict[int, np.ndarray],
    split: set[int],
    patch: int,
    /,
) -> None:
    """Publish distinct named source children in the shared owner orientation."""
    offset = 0
    for use, values, reversed_use in zip(uses, partitions, reversed_uses, strict=True):
        selected = [row for row in range(values.size - 1) if offset + row in split]
        offset += values.size - 1
        if not selected:
            continue
        middle = 0.5 * (values[:-1][selected] + values[1:][selected])
        if np.any((middle == values[:-1][selected]) | (middle == values[1:][selected])):
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "SurfaceConstruction.source_loop_parameter must distinguish topology children; got no distinct source parameter.",
                entity_ids=(use.curve, patch),
            )
        normalized = 1.0 - middle if reversed_use else middle
        parameters[use.curve] = np.unique(
            np.concatenate((parameters[use.curve], normalized))
        )


def _source_loop_chord_chain(
    uses: tuple[_dependency_phydrax_geometry__meshing_domain.PatchCurveUse, ...],
    partitions: list[np.ndarray],
    patch: int,
    remaining_pairs: int,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
) -> tuple[np.ndarray, int, int]:
    """Build a noncrossing source chord chain within arc and pair allowances."""
    count = sum(values.size - 1 for values in partitions)
    if count > 65_536 or remaining_pairs < 1:
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "SurfaceConstruction.source_loop_topology must fit the trim arc and pair allowances; got exhausted source partition.",
            entity_ids=(patch,),
            requested=(
                ("source_trim_arcs", 65_536),
                ("source_trim_pairs", 200_000),
            ),
            achieved=(
                ("source_trim_arcs", count),
                ("source_trim_pairs", 200_000 - remaining_pairs),
            ),
        )
    ledger.reserve(4 * count, count, MeshingStageKind.CURVE_MESHING)
    chords = np.concatenate(
        [
            use.charts(use.first + values[:-1] * (use.last - use.first))
            for use, values in zip(uses, partitions, strict=True)
        ]
    )
    vertices = np.arange(count)
    boundary = np.column_stack((vertices, np.roll(vertices, -1)))
    simple, pair_work, exhausted = domain_chart_chain._simple_boundary(
        chords, boundary, remaining_pairs
    )
    remaining_pairs -= pair_work
    ledger.reserve(0, pair_work, MeshingStageKind.CURVE_MESHING)
    if exhausted or remaining_pairs < 1:
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "SurfaceConstruction.source_loop_topology must fit its pair allowance; got exhausted chord checks.",
            entity_ids=(patch,),
            achieved=(("source_trim_pairs", 200_000 - remaining_pairs),),
        )
    if not simple:
        raise ValueError(
            "SurfaceConstruction.source_loop_chords must not cross; got a crossing source-bound chord chain."
        )
    return chords, count, remaining_pairs


def _certify_source_loop_parameters(
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    patch: int,
    uses: tuple[_dependency_phydrax_geometry__meshing_domain.PatchCurveUse, ...],
    curves: tuple[AbstractTrimCurve, ...],
    reversed_uses: tuple[bool, ...],
    parameters: dict[int, np.ndarray],
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    started: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Earn one loop's source/chord certificate within its existing allowances."""
    domain = compiled.domain
    boxes = [curve.enclosure(0.0, 1.0, budget=budget) for curve in curves]
    tolerance = max(
        float(
            np.nextafter(
                max(np.linalg.norm(upper - lower) for lower, upper in boxes),
                np.inf,
            )
        ),
        np.finfo(np.float64).tiny,
    )
    remaining_pairs = 200_000
    for depth in range(33):
        _dependency_phydrax_meshing_providers__native_publication.check_deadline(
            started, ledger.limits, MeshingStageKind.CURVE_MESHING
        )
        partitions = [
            1.0 - parameters[use.curve][::-1] if reversed_use else parameters[use.curve]
            for use, reversed_use in zip(uses, reversed_uses, strict=True)
        ]
        chords, count, remaining_pairs = _source_loop_chord_chain(
            uses, partitions, patch, remaining_pairs, ledger
        )
        cover = CurveTrimLoop(
            curves,
            tolerance=tolerance,
            maximum_arcs=65_536,
            arc_parameters=partitions,
            chord_vertices=chords,
            relative_closure_tolerance=domain.tolerance,
            budget=budget,
        )
        topology = cover.certify_topology(maximum_pairs=remaining_pairs, budget=budget)
        remaining_pairs -= topology.pairs_checked
        ledger.reserve(0, topology.pairs_checked, MeshingStageKind.CURVE_MESHING)
        if topology.certified:
            return
        if topology.budget_exhausted or depth == 32:
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
                if topology.budget_exhausted
                else _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "SurfaceConstruction.source_loop_topology must certify its shared source partition; got unresolved trim topology.",
                entity_ids=(patch,),
                achieved=(
                    ("source_trim_arcs", count),
                    ("source_trim_pairs", 200_000 - remaining_pairs),
                ),
            )
        split = domain_chart_chain._trim_proof_splits(chords, topology.unresolved_pairs)
        vertices = compiled.corners.size + sum(
            values.size - 2 for values in parameters.values()
        )
        if (
            count + len(split) > 65_536
            or vertices + len(split) > ledger.limits.maximum_vertices
        ):
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "SurfaceConstruction.source_loop_vertices must fit the shared boundary capacity; got oversized topology refinement.",
                entity_ids=(patch,),
            )
        _split_source_loop_parameters(
            uses, partitions, reversed_uses, parameters, split, patch
        )


def _refine_source_loop_parameters(
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    parameters: dict[int, np.ndarray],
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    started: float,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Refine only named curved source/chord relations before shared publication.

    Density alone can prescribe one chord for a curved source use. Its isotopy
    need not have disjoint homotopy boxes, even when the source loop is simple.
    Refine the shared source partition, not the target or the proof's affine
    chord. Straight loops and the separately prepared pole fans are unchanged.
    """
    domain = compiled.domain
    for patch in compiled.patches.tolist():
        source = domain.patches[patch]
        for loop in source.loops:
            if any(
                not isinstance(
                    use, _dependency_phydrax_geometry__meshing_domain.PatchCurveUse
                )
                for use in loop
            ):
                continue
            uses = tuple(
                use
                for use in loop
                if isinstance(
                    use, _dependency_phydrax_geometry__meshing_domain.PatchCurveUse
                )
            )
            missing = tuple(use.curve for use in uses if use.curve not in parameters)
            if missing:
                raise _failure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "SurfaceConstruction.source_loop_parameters must bind every named curve; got an incomplete chart boundary.",
                    entity_ids=(patch, *missing),
                )
            # Existing multi-chord source uses keep their proof-only
            # cover. Only a newly admitted single curved chord needs topology
            # to earn source subdivisions independently of the density target.
            if all(
                parameters[use.curve].size > 2
                or (
                    isinstance(
                        use.pcurve, _dependency_phydrax_geometry_brep__patches.LineCurve
                    )
                    and use.first_root is None
                    and use.last_root is None
                )
                for use in uses
            ):
                continue
            curves = tuple(
                _dependency_phydrax_geometry__meshing_domain.domain_trim_curve(
                    source, use
                )
                for use in uses
            )
            reversed_uses = tuple(
                use.first
                != _prepare_curve_constraints(
                    compiled, use.curve, False, math.inf
                ).owner.first
                for use in uses
            )
            _certify_source_loop_parameters(
                compiled,
                patch,
                uses,
                curves,
                reversed_uses,
                parameters,
                ledger,
                started,
                budget=budget,
            )


@checked
def curve_vertices(
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    schedule: _dependency_phydrax_meshing_providers__native_schedules.NativeSurfaceSchedule,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    started: float,
    maximum_normal_angle: float,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    """Shared vertices: corners, then curve interiors; curve parameters and ids."""

    with _source_bound_scope(
        ledger, 0, stage=MeshingStageKind.SURFACE_MESHING, scope=scope, budget=budget
    ):
        domain = compiled.domain
        if compiled.corners.size > ledger.limits.maximum_vertices:
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "SurfaceConstruction.corner_vertices must fit the vertex capacity before allocation; got oversized declared-corner bank.",
                requested=(("maximum_vertices", ledger.limits.maximum_vertices),),
                achieved=(("corner_vertices", int(compiled.corners.size)),),
            )
        request = compiled.curve_request
        preparation_work, preparation_queries = (
            _dependency_phydrax_meshing_providers__native_curve.curve_preparation_upper(
                request
            )
        )
        with budget.ticket.batch(
            preparation_work, queries_upper=preparation_queries
        ) as preparation_ticket:
            prepared = _dependency_phydrax_meshing_providers__native_curve.PreparedCurveNetwork(
                _dependency_phydrax_meshing_providers__native_sources.NativeCurveSource(
                    domain.curve_atlas, domain.source_revision, gap_domain=domain
                ),
                request,
                schedule.curve_schedule,
                ticket=preparation_ticket,
                gap_budget=budget,
            )
        rows, parameters, _, _ = (
            _dependency_phydrax_meshing_providers__native_curve.discretize(
                prepared, request, ledger, started
            )
        )
        pole_parameters = _pole_curve_parameters(
            compiled, maximum_normal_angle, ledger, started, scope=scope, budget=budget
        )
        corner_ids = {int(corner): row for row, corner in enumerate(compiled.corners)}
        coordinates = [domain.corner_points[compiled.corners]]
        count = compiled.corners.size
        curve_parameters: dict[int, np.ndarray] = {}
        curve_ids: dict[int, np.ndarray] = {}
        angular_required = math.isfinite(maximum_normal_angle) or bool(
            np.any(np.isfinite(compiled.patch_normal_angles))
        )
        for curve_row, curve in enumerate(prepared.curves.tolist()):
            values = parameters[rows == curve_row]
            values = np.unique(
                np.concatenate(
                    (
                        values,
                        domain.curve_breakpoints(curve),
                        pole_parameters.get(curve, np.empty((0,), dtype=np.float64)),
                    )
                )
            )
            curve_parameters[curve] = values
        _refine_source_loop_parameters(
            compiled, curve_parameters, ledger, started, budget=budget
        )
        # One mapping for this curve request: shared endpoint roots are
        # contracted once and admitted once to this request's budget.
        preparations = TrimRequestPreparations()
        for curve_row, curve in enumerate(prepared.curves.tolist()):
            values = curve_parameters[curve]
            # Shared source edges cannot be changed by patch-local interior
            # refinement. Bisect each offending source interval here, before every
            # incident patch reconnects through its CDT, so all uses retain one
            # conforming curve vertex and interval ledger.
            target = (
                0.25 * compiled.curve_deviations[np.searchsorted(compiled.curves, curve)]
            )
            ribbon_targets = []
            for incident_patch, incident_use in domain.curve_uses(curve):
                patch_row = int(np.searchsorted(compiled.patches, incident_patch))
                if (
                    patch_row < compiled.patches.size
                    and compiled.patches[patch_row] == incident_patch
                    and np.isfinite(compiled.patch_deviations[patch_row])
                ):
                    ribbon_targets.append(
                        (
                            incident_patch,
                            incident_use,
                            float(compiled.patch_deviations[patch_row]),
                        )
                    )
            if np.isfinite(target) or angular_required or ribbon_targets:
                values = _bisect_curve(
                    compiled,
                    prepared,
                    curve_row,
                    curve,
                    values,
                    target,
                    ribbon_targets,
                    angular_required,
                    maximum_normal_angle,
                    ledger,
                    started,
                    parameters.nbytes + rows.nbytes,
                    count,
                    scope=scope,
                    budget=budget,
                    preparations=preparations,
                )
            interior = values[1:-1]
            if count + interior.size > ledger.limits.maximum_vertices:
                raise _failure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "SurfaceConstruction.curve_vertices must fit capacity before coordinate allocation; got oversized shared curve bank.",
                    entity_ids=(curve,),
                    requested=(("maximum_vertices", ledger.limits.maximum_vertices),),
                    achieved=(("boundary_vertices", int(count + interior.size)),),
                )
            ledger.reserve(interior.size, 0, MeshingStageKind.CURVE_MESHING)
            points = (
                _dependency_phydrax_meshing_providers__native_curve.curve_points(
                    prepared.charts, curve_row, interior
                )
                if interior.size
                else np.empty((0, 3), dtype=np.float64)
            )
            ids = np.concatenate(
                (
                    (corner_ids[domain.curves[curve].start],),
                    count + np.arange(interior.size),
                    (corner_ids[domain.curves[curve].end],),
                )
            ).astype(np.int64)
            coordinates.append(points)
            count += interior.size
            curve_parameters[curve] = values
            curve_ids[curve] = ids
        return (
            np.concatenate(coordinates, axis=0),
            curve_parameters,
            curve_ids,
            corner_ids,
        )


def _pole_segments(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    use: _dependency_phydrax_geometry__meshing_domain.PatchPoleUse,
    maximum_normal_angle: float,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> int:
    """Resolve the tangent fan and the source's nonvanishing pole Gauss map."""

    start = np.asarray(use.start, dtype=np.float64)
    end = np.asarray(use.end, dtype=np.float64)
    along = end - start
    inward = 1.0e-4 * np.asarray((-along[1], along[0]))
    fractions = np.linspace(0.0, 1.0, _POLE_SAMPLES)
    charts = use.charts(fractions) + inward
    rays = domain.evaluate(np.full((fractions.size,), patch), charts, budget=budget)
    rays -= domain.corner_points[use.corner]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    cosines = np.clip(np.sum(rays[1:] * rays[:-1], axis=1), -1.0, 1.0)
    fan = float(np.sum(np.arccos(cosines)))
    pieces = max(1, math.ceil(fan / _POLE_FAN_ANGLE - 1.0e-9))
    chart = np.stack((start, end, 0.5 * (start + end)))[None]
    turning = float(domain.normal_turn_bounds(patch, chart, budget=budget)[0])
    if np.isfinite(turning):
        angle = min(maximum_normal_angle, _POLE_FAN_ANGLE)
        pieces = max(pieces, math.ceil(turning / angle))
    return pieces


def _pole_ring(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    use: _dependency_phydrax_geometry__meshing_domain.PatchPoleUse,
    pieces: int,
    size: float,
    deviation: float,
    maximum_normal_angle: float,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    reach: float,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Fan seeds whose entire collapsed strip satisfies the source bounds."""

    start = np.asarray(use.start, dtype=np.float64)
    along = np.asarray(use.end, dtype=np.float64) - start
    inward = np.asarray((-along[1], along[0])) / np.linalg.norm(along)
    depths = reach * np.geomspace(1.0e-3, 1.0, 32)
    probes = start + 0.5 * along + depths[:, None] * inward
    ledger.reserve(depths.size, depths.size, MeshingStageKind.SURFACE_MESHING)
    distance = np.linalg.norm(
        domain.evaluate(np.full((depths.size,), patch), probes, budget=budget)
        - domain.corner_points[use.corner],
        axis=1,
    )
    depth = float(depths[int(np.argmin(np.abs(distance - 0.9 * size)))])
    fractions = (np.arange(pieces, dtype=np.float64) + 0.5) / pieces
    pole = use.charts(np.linspace(0.0, 1.0, pieces + 1))
    sphere_frame = domain_numerics.full_sphere_frame(domain, patch, budget=budget)
    # Collapsed strips are prepared before the measured trim-homotopy remainder
    # is available. Leave room for that source-bound contribution: a later
    # interior insertion cannot refine a physically collapsed chart triangle.
    strip_deviation = 0.25 * deviation
    for _ in range(64):
        ring = use.charts(fractions) + depth * inward
        # Include both radial children as well as the collapsed chart wedges.
        strips = np.concatenate(
            (
                np.stack((pole[:-1], pole[1:], ring), axis=1),
                np.stack((pole[1:-1], ring[:-1], ring[1:]), axis=1),
            ),
            axis=0,
        )
        ledger.reserve(0, 2 * strips.shape[0], MeshingStageKind.SURFACE_MESHING)
        with _interpolation_bounds(
            domain,
            patch,
            strips,
            ledger,
            sum(array.nbytes for array in (pole, ring, strips)),
            scope=scope,
            budget=budget,
        ) as position_bounds:
            if sphere_frame is not None:
                ledger.reserve(3 * strips.shape[0], 0, MeshingStageKind.SURFACE_MESHING)
                physical = domain.evaluate(
                    np.full((3 * strips.shape[0],), patch),
                    strips.reshape((-1, 2)),
                    budget=budget,
                ).reshape((-1, 3, 3))
                physical[:pieces, :2] = domain.corner_points[use.corner]
                position_bounds = np.minimum(
                    position_bounds,
                    domain_numerics.sphere_triangle_distance_bounds(
                        sphere_frame, physical
                    ),
                )
            normal_bounds = domain.normal_turn_bounds(patch, strips, budget=budget)
            if sphere_frame is not None:
                normal_bounds = np.minimum(
                    normal_bounds,
                    domain_numerics.sphere_triangle_normal_bounds(sphere_frame, physical),
                )
            passed = np.all(position_bounds <= strip_deviation) and np.all(
                normal_bounds <= maximum_normal_angle
            )
            del position_bounds
            if passed:
                return ring
        depth *= 0.5
    raise _failure(
        _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
        f"SurfaceConstruction.pole_strip must meet finite source bounds; got pole {use.corner} without a finite preparation strip.",
        entity_ids=(patch, use.corner),
    )


def _pole_curve_parameters(
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    maximum_normal_angle: float,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    started: float,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> dict[int, np.ndarray]:
    """Share straight seam nodes at every prepared pole-ring endpoint."""
    domain = compiled.domain
    additions: dict[int, list[tuple[float, float]]] = {}
    for row, patch in enumerate(compiled.patches.tolist()):
        source = domain.patches[patch]
        outer = np.concatenate([_use_charts(use) for use in source.loops[0]])
        reach = 0.4 * float(np.max(np.ptp(outer, axis=0)))
        for loop in source.loops:
            for position, use in enumerate(loop):
                if not isinstance(
                    use, _dependency_phydrax_geometry__meshing_domain.PatchPoleUse
                ):
                    continue
                _dependency_phydrax_meshing_providers__native_publication.check_deadline(
                    started, ledger.limits, MeshingStageKind.SURFACE_MESHING
                )
                ledger.reserve(
                    _POLE_SAMPLES, _POLE_SAMPLES, MeshingStageKind.SURFACE_MESHING
                )
                pieces = _pole_segments(
                    domain,
                    patch,
                    use,
                    min(maximum_normal_angle, float(compiled.patch_normal_angles[row])),
                    scope=scope,
                    budget=budget,
                )
                ring = _pole_ring(
                    domain,
                    patch,
                    use,
                    pieces,
                    float(compiled.patch_sizes[row]),
                    float(compiled.patch_deviations[row]),
                    min(maximum_normal_angle, float(compiled.patch_normal_angles[row])),
                    ledger,
                    reach,
                    scope=scope,
                    budget=budget,
                )
                along = np.asarray(use.end) - np.asarray(use.start)
                offset = ring[0] - use.charts(np.asarray((0.5 / pieces,)))[0]
                for neighbor, endpoint in (
                    (loop[(position - 1) % len(loop)], np.asarray(use.start) + offset),
                    (loop[(position + 1) % len(loop)], np.asarray(use.end) + offset),
                ):
                    if not isinstance(
                        neighbor,
                        _dependency_phydrax_geometry__meshing_domain.PatchCurveUse,
                    ):
                        continue
                    # Any p-curve affine in its parameter (a line, or a line in
                    # a native period gauge) is inverted by its chart chord;
                    # the certified source evaluation must reproduce the endpoint.
                    first, last = neighbor.charts(
                        np.asarray((neighbor.first, neighbor.last))
                    )
                    chord = last - first
                    fraction = float((endpoint - first) @ chord / (chord @ chord))
                    if not 0.0 < fraction < 1.0:
                        continue
                    parameter = neighbor.first + fraction * (
                        neighbor.last - neighbor.first
                    )
                    reconstructed = neighbor.charts(np.asarray((parameter,)))[0]
                    tolerance = domain.tolerance * max(1.0, float(np.linalg.norm(along)))
                    if np.linalg.norm(reconstructed - endpoint) > tolerance:
                        continue
                    owner_patch, owner_loop, owner_position = domain.curve_owners[
                        neighbor.curve
                    ]
                    owner = domain.patches[owner_patch].loops[owner_loop][owner_position]
                    match owner:
                        case _dependency_phydrax_geometry__meshing_domain.PatchCurveUse():
                            pass
                        case _:
                            raise TypeError(
                                f"MeshingDomain.curve_owners must select a PatchCurveUse; got {type(owner).__name__}."
                            )
                    value = (parameter - owner.first) / (owner.last - owner.first)
                    additions.setdefault(neighbor.curve, []).append(
                        (value, tolerance / float(np.linalg.norm(chord)))
                    )
    # Both chart copies of a periodic seam reach one physical ring endpoint;
    # their rounded parameters are one shared node, never a sliver pair.
    merged: dict[int, np.ndarray] = {}
    for curve, entries in additions.items():
        kept: list[float] = []
        for value, tolerance in sorted(entries):
            if not kept or value - kept[-1] > tolerance:
                kept.append(value)
        merged[curve] = np.asarray(kept, dtype=np.float64)
    return merged


def _boundary(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    size: float,
    curve_parameters: dict[int, np.ndarray],
    curve_ids: dict[int, np.ndarray],
    corner_ids: dict[int, int],
    deviation: float,
    maximum_normal_angle: float,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    local_limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Source chart boundary, shared ids, holes, pole seeds, rays and strips."""

    outer = np.concatenate(
        [_use_charts(use) for use in domain.patches[patch].loops[0]], axis=0
    )
    extent = float(np.max(np.ptp(outer, axis=0)))
    rings: list[np.ndarray] = [np.zeros((0, 2), dtype=np.float64)]
    pole_rays: list[np.ndarray] = []
    ring_chords: list[np.ndarray] = []
    pole_strips: list[np.ndarray] = []

    charts: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    segments: list[np.ndarray] = []
    holes: list[np.ndarray] = []
    provenance: list[np.ndarray] = []
    offset = 0
    for position, loop in enumerate(domain.patches[patch].loops):
        first_ray = len(pole_rays)
        loop_charts: list[np.ndarray] = []
        loop_ids: list[np.ndarray] = []
        for use_index, use in enumerate(loop):
            match use:
                case _dependency_phydrax_geometry__meshing_domain.PatchCurveUse():
                    owner_patch, owner_loop, owner_position = domain.curve_owners[
                        use.curve
                    ]
                    owner = domain.patches[owner_patch].loops[owner_loop][owner_position]
                    match owner:
                        case _dependency_phydrax_geometry__meshing_domain.PatchCurveUse():
                            pass
                        case _:
                            raise TypeError(
                                f"MeshingDomain.curve_owners must select a PatchCurveUse; got {type(owner).__name__}."
                            )
                    values = curve_parameters[use.curve]
                    order = np.arange(values.size)
                    if use.first != owner.first:
                        order = order[::-1]
                    parameters = owner.first + values[order] * (owner.last - owner.first)
                    # Preserve the authored interval endpoints exactly;
                    # affine arithmetic is only a realization of its interior.
                    parameters[0], parameters[-1] = use.first, use.last
                    loop_charts.append(use.charts(parameters)[:-1])
                    loop_ids.append(curve_ids[use.curve][order][:-1])
                    provenance.append(
                        np.column_stack(
                            (
                                np.full((parameters.size - 1,), position),
                                np.full((parameters.size - 1,), use_index),
                                parameters[:-1],
                                parameters[1:],
                            )
                        )
                    )
                case _dependency_phydrax_geometry__meshing_domain.PatchPoleUse():
                    ledger.reserve(
                        _POLE_SAMPLES, _POLE_SAMPLES, MeshingStageKind.SURFACE_MESHING
                    )
                    pieces = _pole_segments(
                        domain,
                        patch,
                        use,
                        maximum_normal_angle,
                        scope=scope,
                        budget=budget,
                    )
                    # A chart copy at each ring ray supplies an exactly
                    # axis-aligned radial split, without a second physical pole.
                    rays = (np.arange(pieces, dtype=np.float64) + 0.5) / pieces
                    full_fractions = np.sort(
                        np.concatenate((np.linspace(0.0, 1.0, pieces + 1), rays))
                    )
                    fractions = full_fractions[:-1]
                    segments_count = fractions.size
                    provenance.append(
                        np.column_stack(
                            (
                                np.full((segments_count,), position),
                                np.full((segments_count,), use_index),
                                full_fractions[:-1],
                                full_fractions[1:],
                            )
                        )
                    )
                    alias_offset = offset + sum(part.shape[0] for part in loop_charts)
                    ring_offset = sum(part.shape[0] for part in rings)
                    ring_ids = ring_offset + np.arange(pieces, dtype=np.int64)
                    base_ids = alias_offset + np.searchsorted(
                        full_fractions,
                        np.linspace(0.0, 1.0, pieces + 1),
                    )
                    pole_rays.append(
                        np.concatenate(
                            (
                                np.column_stack(
                                    (
                                        alias_offset
                                        + np.searchsorted(full_fractions, rays),
                                        ring_ids,
                                    )
                                ),
                                np.column_stack((base_ids[:-1], ring_ids)),
                                np.column_stack((base_ids[1:], ring_ids)),
                            ),
                            axis=0,
                        )
                    )
                    loop_charts.append(use.charts(fractions))
                    loop_ids.append(np.full((segments_count,), corner_ids[use.corner]))
                    # The certified strip also owns its wedges between adjacent
                    # rays. Without their ring chords the chart CDT may join a
                    # pole gauge to a deeper seed past the next ray; the chart
                    # cell stays positive while its physical apex cell inverts.
                    ring_chords.append(np.column_stack((ring_ids[:-1], ring_ids[1:])))
                    ring = _pole_ring(
                        domain,
                        patch,
                        use,
                        pieces,
                        size,
                        deviation,
                        maximum_normal_angle,
                        ledger,
                        0.4 * extent,
                        scope=scope,
                        budget=budget,
                    )
                    rings.append(ring)
                    pole_strips.append(
                        np.stack(
                            (
                                np.asarray(use.start, dtype=np.float64),
                                np.asarray(use.end, dtype=np.float64),
                                ring[0] - use.charts(rays[:1])[0],
                            )
                        )
                    )
                case _:
                    raise TypeError(
                        f"MeshingDomain.patch_boundary must contain curve or pole uses; got {type(use).__name__}."
                    )
        points = np.concatenate(loop_charts, axis=0)
        for prepared in pole_rays[first_ray:]:
            prepared[prepared[:, 0] == offset + points.shape[0], 0] = offset
        local = np.arange(points.shape[0], dtype=np.int64)
        charts.append(points)
        identities.append(np.concatenate(loop_ids).astype(np.int64))
        segments.append(np.stack((local, np.roll(local, -1)), axis=1) + offset)
        if position > 0:
            holes.append(
                _dependency_phydrax_meshing_providers__native_planar.interior_point(
                    points, local, local_limits, scope=scope
                )
            )
        offset += points.shape[0]
    radial = np.concatenate(pole_rays) if pole_rays else np.empty((0, 2), dtype=np.int64)
    radial[:, 1] += offset
    if ring_chords:
        radial = np.concatenate((radial, np.concatenate(ring_chords) + offset))
    return (
        np.concatenate(charts, axis=0),
        np.concatenate(identities),
        np.concatenate(segments, axis=0),
        np.asarray(holes, dtype=np.float64).reshape((-1, 2)),
        np.concatenate(rings, axis=0),
        np.concatenate(provenance, axis=0),
        radial,
        np.asarray(pole_strips, dtype=np.float64).reshape((-1, 3, 2)),
    )


def _use_charts(
    use: _dependency_phydrax_geometry__meshing_domain.PatchCurveUse
    | _dependency_phydrax_geometry__meshing_domain.PatchPoleUse,
    /,
) -> np.ndarray:
    match use:
        case _dependency_phydrax_geometry__meshing_domain.PatchCurveUse():
            return use.charts(np.linspace(use.first, use.last, _POLE_SAMPLES))
        case _dependency_phydrax_geometry__meshing_domain.PatchPoleUse():
            return use.charts(np.asarray((0.0, 1.0)))
        case _:
            raise TypeError(
                f"MeshingDomain.patch_boundary must contain curve or pole uses; got {type(use).__name__}."
            )


def _fidelity_seed_counts(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    lower: np.ndarray,
    upper: np.ndarray,
    probes: np.ndarray,
    counts: np.ndarray,
    deviation: float,
    normal_angle: float,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Seed hard fidelity at chart scale before irregular reconnects are needed."""
    position_requested = np.isfinite(deviation)
    normal_requested = np.isfinite(normal_angle)
    if not position_requested and not normal_requested:
        return counts
    while np.any(counts < _SEED_AXIS):
        steps = (upper - lower) / counts
        origins = np.minimum(probes, upper - steps)
        cells = np.stack(
            (
                origins,
                origins + steps,
                origins + steps * np.asarray((1.0, 0.0)),
            ),
            axis=1,
        )
        ledger.reserve(
            0,
            (int(position_requested) + int(normal_requested)) * cells.shape[0],
            MeshingStageKind.SURFACE_MESHING,
        )
        position_ok = True
        if position_requested:
            with _interpolation_bounds(
                domain,
                patch,
                cells,
                ledger,
                cells.nbytes + origins.nbytes + probes.nbytes,
                scope=scope,
                budget=budget,
            ) as positional:
                frame = domain_numerics.full_sphere_frame(domain, patch, budget=budget)
                if frame is not None:
                    ledger.reserve(
                        3 * cells.shape[0], 0, MeshingStageKind.SURFACE_MESHING
                    )
                    physical = domain.evaluate(
                        np.full((3 * cells.shape[0],), patch),
                        cells.reshape((-1, 2)),
                        budget=budget,
                    ).reshape((-1, 3, 3))
                    positional = np.minimum(
                        positional,
                        domain_numerics.sphere_triangle_distance_bounds(frame, physical),
                    )
                position_ok = np.max(positional) <= 0.25 * deviation
                del positional
        normal_ok = not normal_requested or (
            np.max(domain.normal_turn_bounds(patch, cells, budget=budget))
            <= 0.5 * normal_angle
        )
        if position_ok and normal_ok:
            break
        counts = np.minimum(2 * counts, _SEED_AXIS)
    return counts


@final
class _InteriorSeeds(StrictModule):
    """Chart seeds and RMS physical speeds of the two parameter axes."""

    charts: np.ndarray
    axis_speeds: np.ndarray


def _interior_seeds(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    size: float,
    boundary: np.ndarray,
    segments: np.ndarray,
    fixed: np.ndarray,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    local_limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    deviation: float,
    maximum_normal_angle: float,
    /,
    *,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    pole_strips: np.ndarray,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _InteriorSeeds:
    """A coarse chart grid of about twice the target size inside the loops.

    Seeds keep the initial triangulation local, so no initial triangle joins
    two chart copies of one physical vertex across a periodic chart. Grid steps
    follow the largest chart speeds; seeds within 0.75 steps (in step units)
    of a boundary segment or a fixed seed are dropped. Certified pole strips
    retain their prepared fan rather than admitting unrelated interior seeds.
    Soft density retains at least four grid steps per axis for quality recovery;
    hard uniform density starts from its target-sized grid.
    """

    lower = np.min(boundary, axis=0)
    upper = np.max(boundary, axis=0)
    probes = np.stack(
        np.meshgrid(*(np.linspace(lower[k], upper[k], _SPEED_PROBES) for k in range(2))),
        axis=-1,
    ).reshape((-1, 2))
    ledger.reserve(probes.shape[0], probes.shape[0], MeshingStageKind.SURFACE_MESHING)
    differential = domain_numerics.host_surface_differential(
        domain.patches[patch].surface, probes
    )
    axis_norms = np.linalg.norm(differential, axis=1)
    speeds = np.max(axis_norms, axis=0)
    axis_speeds = np.sqrt(np.mean(axis_norms**2, axis=0))
    # Resolve the authored physical controls, not only the coarse patch
    # envelope. An empty coarse grid can connect antipodal poles before local
    # recovery has any legal source-bound children to refine.
    ledger.reserve(probes.shape[0], probes.shape[0], MeshingStageKind.SURFACE_MESHING)
    images = domain.evaluate(np.full(probes.shape[0], patch), probes, budget=budget)
    seed_size = min(
        size,
        float(
            np.min(
                compiled.source_sizing.evaluate(
                    domain, patch, probes, images, budget=budget
                )
            )
        ),
    )
    counts = np.clip(
        np.ceil((upper - lower) * speeds / (2.0 * seed_size)), 1, _SEED_AXIS
    ).astype(np.int64)
    # Nonperiodic charts need no interior row when the target spans the patch.
    # Periodic charts still need retained neighbors strictly below half a
    # period, so CDT cannot join two chart copies of the same physical point.
    for axis, period in enumerate(domain.patches[patch].surface.periods):
        if period is not None:
            counts[axis] = min(
                _SEED_AXIS,
                max(
                    int(counts[axis]),
                    int(np.ceil(8 * (upper[axis] - lower[axis]) / period)),
                ),
            )
    source_face = domain.source_indices[2][patch]
    hard_density = any(
        isinstance(control, _dependency_phydrax_meshing__sizing.UniformSizeControl)
        and control.strength
        is _dependency_phydrax_meshing__sizing.SizeControlStrength.HARD
        and source_face in np.asarray(control.scope.global_entity_ids)
        for control in compiled.source_sizing.controls
    )
    # A hard density request starts from its own chart-sized grid; additional
    # accuracy is recovered locally, not by pre-filling an excessively fine grid.
    if not hard_density:
        # Soft density leaves quality recovery in charge of the final mesh.
        # Retain its local chart scaffold rather than forcing quality recovery
        # to grow from a coarse target's empty interior.
        counts = np.maximum(counts, 4)
        counts = _fidelity_seed_counts(
            domain,
            patch,
            lower,
            upper,
            probes,
            counts,
            deviation,
            maximum_normal_angle,
            ledger,
            scope=scope,
            budget=budget,
        )
    count = int(np.prod(counts))
    pairs = count * (segments.shape[0] + fixed.shape[0] + pole_strips.shape[0])
    scratch = 96 * pairs + 128 * (boundary.shape[0] + count)
    if scratch > local_limits.maximum_scratch_bytes:
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "SurfaceConstruction.seed_scratch must fit the resident scratch limit; got oversized pole/seam chart seeds.",
            requested=(("maximum_scratch_bytes", local_limits.maximum_scratch_bytes),),
            achieved=(("scratch_bytes", scratch),),
        )
    ledger.reserve(0, pairs, MeshingStageKind.SURFACE_MESHING)
    _surface_capacity(
        boundary.shape[0] + fixed.shape[0] + count,
        2 * (boundary.shape[0] + fixed.shape[0] + count),
        0,
        local_limits,
    )
    steps = (upper - lower) / counts
    axes = [lower[k] + (np.arange(counts[k]) + 0.5) * steps[k] for k in range(2)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 2))
    scaled = grid / steps
    start = boundary[segments[:, 0]] / steps
    end = boundary[segments[:, 1]] / steps
    # Even-odd ray casting along +u decides membership in the loops.
    crossing = (start[None, :, 1] > scaled[:, None, 1]) != (
        end[None, :, 1] > scaled[:, None, 1]
    )
    along = start[None, :, 0] + (scaled[:, None, 1] - start[None, :, 1]) * (
        end[None, :, 0] - start[None, :, 0]
    ) / np.where(crossing, end[None, :, 1] - start[None, :, 1], 1.0)
    inside = np.sum(crossing & (along > scaled[:, None, 0]), axis=1) % 2 == 1
    for pole_start, pole_end, inward in pole_strips:
        tangent = pole_end - pole_start
        relative = grid - pole_start
        along_pole = relative @ tangent
        depth = relative @ inward
        inside &= ~(
            (along_pole >= 0.0)
            & (along_pole <= tangent @ tangent)
            & (depth >= 0.0)
            & (depth <= inward @ inward)
        )
    direction = end - start
    offset = scaled[:, None, :] - start[None]
    squared = np.maximum(np.sum(direction * direction, axis=1), 1e-300)
    fraction = np.clip(np.sum(offset * direction[None], axis=2) / squared, 0.0, 1.0)
    clearance = np.min(
        np.linalg.norm(offset - fraction[..., None] * direction[None], axis=2), axis=1
    )
    if fixed.shape[0]:
        clearance = np.minimum(
            clearance,
            np.min(
                np.linalg.norm(scaled[:, None] - (fixed / steps)[None], axis=2), axis=1
            ),
        )
    return _InteriorSeeds(grid[inside & (clearance >= 0.75)], axis_speeds)


@final
class _InitialChart(StrictModule):
    """Retain prepared chart buffers and the simultaneous CDT scratch allowance."""

    charts: np.ndarray
    topology_charts: np.ndarray
    topology_holes: np.ndarray
    points: np.ndarray
    identities: np.ndarray
    boundary: np.ndarray
    segments: np.ndarray
    holes: np.ndarray
    provenance: np.ndarray
    chart_segments: np.ndarray
    retained_buffers: tuple[np.ndarray, ...]
    triangle_capacity: int = eqx.field(static=True)
    host_scratch: int = eqx.field(static=True)
    result_scratch: int = eqx.field(static=True)
    native_scratch: int = eqx.field(static=True)


@final
class _InitialTopology(StrictModule):
    triangles: np.ndarray
    constrained: np.ndarray
    triangulation: ConstrainedDelaunayTriangulation


@final
class _InitialTrim(StrictModule):
    ribbon: np.ndarray
    normals: np.ndarray


def _prepare_initial_chart(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    size: float,
    vertices: np.ndarray,
    curve_parameters: dict[int, np.ndarray],
    curve_ids: dict[int, np.ndarray],
    corner_ids: dict[int, int],
    deviation: float,
    maximum_normal_angle: float,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    local_limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _InitialChart:
    preparation_start = phase_started(record_phase)
    with _source_bound_scope(
        ledger, 0, stage=MeshingStageKind.SURFACE_MESHING, scope=scope, budget=budget
    ):
        (
            boundary,
            shared,
            segments,
            holes,
            rings,
            provenance,
            pole_rays,
            pole_strips,
        ) = _boundary(
            domain,
            patch,
            size,
            curve_parameters,
            curve_ids,
            corner_ids,
            deviation,
            maximum_normal_angle,
            ledger,
            local_limits,
            scope=scope,
            budget=budget,
        )
        seeds = _interior_seeds(
            domain,
            patch,
            size,
            boundary,
            segments,
            rings,
            ledger,
            local_limits,
            deviation,
            maximum_normal_angle,
            compiled=compiled,
            pole_strips=pole_strips,
            scope=scope,
            budget=budget,
        )
        rings = np.concatenate((rings, seeds.charts), axis=0)
        axis_speeds = seeds.axis_speeds
        del seeds
        del pole_strips
        # These are preparation edges, not authored protected features. Chart CDT
        # alone can connect the collapsed side directly to deeper grid points and
        # discard the prepared source strip. Its prepared radial rays and ring chords
        # keep that strip present initially; subsequent physical reconnection may
        # split them.
        chart_segments = np.concatenate((segments, pole_rays), axis=0)
        chart_count = boundary.shape[0] + rings.shape[0]
        native_capacity = 2 * chart_count + 16
        _surface_capacity(
            boundary.shape[0] + rings.shape[0], native_capacity, 0, local_limits
        )
        ledger.reserve(
            boundary.shape[0] + 2 * rings.shape[0], 0, MeshingStageKind.SURFACE_MESHING
        )
        charts = np.concatenate((boundary, rings), axis=0)
        identities = np.concatenate(
            (shared, np.full((rings.shape[0],), -1, dtype=np.int64))
        )
        points = np.concatenate(
            (
                vertices[shared],
                domain.evaluate(np.full((rings.shape[0],), patch), rings, budget=budget),
            ),
            axis=0,
        )
    chart_lower = np.min(charts, axis=0)
    chart_extent = np.ptp(charts, axis=0)
    if np.any(chart_extent <= 0.0) or not np.all(
        np.isfinite(axis_speeds) & (axis_speeds > 0.0)
    ):
        raise ValueError(
            "SurfaceConstruction.chart_metric must span two finite positive physical axes; "
            f"got extent={chart_extent.tolist()}, speeds={axis_speeds.tolist()}."
        )
    physical_extent = chart_extent * axis_speeds
    characteristic_length = np.max(physical_extent)
    relative_extent = physical_extent / characteristic_length
    # RMS physical axis speeds condition CDT's affine execution chart.
    # Source UVs remain unchanged for evaluation, restrictions, and proofs.
    topology_charts = (charts - chart_lower) / chart_extent * relative_extent
    topology_holes = (holes - chart_lower) / chart_extent * relative_extent
    triangle_capacity = min(local_limits.maximum_faces, native_capacity)
    host_scratch = sum(
        array.nbytes
        for array in (
            charts,
            topology_charts,
            topology_holes,
            axis_speeds,
            chart_lower,
            chart_extent,
            physical_extent,
            relative_extent,
            points,
            identities,
            boundary,
            shared,
            segments,
            holes,
            rings,
            provenance,
            pole_rays,
            chart_segments,
        )
    )
    result_scratch = 20 * chart_count + 24 * triangle_capacity + 120
    # The input segment conversion and result extraction coexist with the PMR
    # working set; the native allowance is the remaining simultaneous budget.
    native_scratch = (
        local_limits.maximum_scratch_bytes
        - host_scratch
        - result_scratch
        - 8 * chart_segments.shape[0]
    )
    if native_scratch <= 0:
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "SurfaceConstruction.initial_scratch must leave native CDT storage; got exhausted chart-buffer allowance.",
            entity_ids=(patch,),
            requested=(("maximum_scratch_bytes", local_limits.maximum_scratch_bytes),),
            achieved=(
                (
                    "scratch_bytes",
                    host_scratch + result_scratch + 8 * chart_segments.shape[0],
                ),
            ),
        )
    record_elapsed(record_phase, "native_preparation", preparation_start)
    return _InitialChart(
        charts,
        topology_charts,
        topology_holes,
        points,
        identities,
        boundary,
        segments,
        holes,
        provenance,
        chart_segments,
        (shared, rings, pole_rays),
        triangle_capacity,
        host_scratch,
        result_scratch,
        native_scratch,
    )


def _triangulate_initial_chart(
    prepared: _InitialChart,
    patch: int,
    local_limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    record_phase: NativeMeshingPhaseRecorder | None,
    /,
    *,
    scope: NativeExecutionScope,
) -> _InitialTopology:
    """Execute bounded CDT and retain its native refusal evidence."""
    charts, chart_segments, holes = (
        prepared.topology_charts,
        prepared.chart_segments,
        prepared.topology_holes,
    )
    segments = prepared.segments
    triangle_capacity = prepared.triangle_capacity
    host_scratch, result_scratch, native_scratch = (
        prepared.host_scratch,
        prepared.result_scratch,
        prepared.native_scratch,
    )
    topology_start = phase_started(record_phase)
    incoming = scope.ticket_view()
    work_allowance = min(
        ledger.limits.maximum_work_units - ledger.work, incoming.remaining_work_units
    )
    storage_allowance = min(
        native_scratch + result_scratch + 8 * chart_segments.shape[0],
        incoming.bytes_upper,
    )
    # CDT and extraction share the declared simultaneous bound. Admission of
    # this mapped child target remains outside the CDT-specific refusal handler.
    with (
        scope.batch(
            work_allowance,
            bytes_upper=storage_allowance,
            cavity_upper=local_limits.maximum_cavity_cells,
        ) as ticket,
        ticket.native_target(),
    ):
        try:
            triangulation = ConstrainedDelaunayTriangulation(
                charts,
                chart_segments,
                holes=holes if holes.shape[0] else None,
                keep_convex_hull=False,
                max_triangles=triangle_capacity,
                max_cavity_cells=local_limits.maximum_cavity_cells,
                maximum_work=work_allowance,
                max_scratch_bytes=native_scratch,
            )
        except MeshcoreError as error:
            if error.status != MeshcoreStatus.CAPACITY_EXCEEDED:
                raise
            work = error.work_evidence if isinstance(error, NativeResourceFailure) else None
            memory = error.memory_counters
            record_elapsed(
                record_phase,
                "topology_construction",
                topology_start,
                work_units=None if work is None else int(work[0]),
            )
            if work is None or memory is None:
                raise RuntimeError(
                    f"ConstrainedDelaunayTriangulation.refusal must carry native resource evidence; got work={work}, memory={memory}."
                ) from error
            ledger.observe_native(0, int(work[0]), MeshingStageKind.SURFACE_MESHING)
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "SurfaceConstruction.initial_triangulation must fit the native resource budget; got exhausted constrained chart CDT.",
                entity_ids=(patch,),
                achieved=(
                    ("work_units", ledger.work),
                    ("peak_cavity_cells", int(work[6])),
                    ("native_peak_scratch_bytes", memory.peak_bytes),
                    (
                        "simultaneous_scratch_bound",
                        host_scratch
                        + result_scratch
                        + 8 * segments.shape[0]
                        + memory.peak_bytes,
                    ),
                ),
            ) from error
    ledger.observe_native(
        0, int(triangulation.work_evidence[0]), MeshingStageKind.SURFACE_MESHING
    )
    record_elapsed(
        record_phase,
        "topology_construction",
        topology_start,
        work_units=int(triangulation.work_evidence[0]),
    )
    if np.asarray(triangulation.points).shape[0] != charts.shape[0] or (
        np.unique(charts, axis=0).shape[0] != charts.shape[0]
    ):
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SOURCE,
            f"SurfaceConstruction.boundary_loops must form a simple chart polygon; got invalid loops on patch {patch}.",
            entity_ids=(patch,),
        )
    triangles = np.asarray(triangulation.triangles, dtype=np.int64)
    segment_ids = np.asarray(triangulation.segment_ids, dtype=np.int64)
    constrained = (segment_ids >= 0) & (segment_ids < segments.shape[0])
    return _InitialTopology(triangles, constrained, triangulation)


def _certify_initial_trim(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    prepared: _InitialChart,
    topology: _InitialTopology,
    patch: int,
    deviation: float,
    restriction_required: np.ndarray,
    local_limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _InitialTrim:
    """Certify the boundary chain and evaluate normals before refinement starts."""
    charts, points, boundary = prepared.charts, prepared.points, prepared.boundary
    segments, provenance = prepared.segments, prepared.provenance
    triangles = topology.triangles
    # The carrier interpolation and trim homotopy are one continuous physical
    # request. Reserving their measured composition avoids a successful nodal
    # mesh that the independent two-sided cover must subsequently refuse.
    _surface_capacity(
        charts.shape[0], triangles.shape[0], segments.shape[0], local_limits
    )

    def reserve_queries(count: int) -> None:
        ledger.observe_native(count, 0, MeshingStageKind.SURFACE_MESHING)

    findings: list[
        _dependency_phydrax_geometry__embedding_state.MeshCertificateFinding
    ] = []
    resources: list[tuple[str, int, int]] = []
    with _source_bound_scope(
        ledger,
        charts.nbytes
        + points.nbytes
        + triangles.nbytes
        + segments.nbytes
        + provenance.nbytes,
        stage=MeshingStageKind.SURFACE_MESHING,
        scope=scope,
        budget=budget,
    ):
        valid, _, ribbon, _ = domain_chart_chain.verify_chart_chain(
            domain,
            patch,
            charts,
            points,
            triangles,
            segments,
            provenance,
            restriction_required,
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            findings,
            resources,
            reserve_queries=reserve_queries,
            budget=budget,
        )
    if not valid:
        exhausted = any("capacity" in finding.check for finding in findings)
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
            if exhausted
            else _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            f"SurfaceConstruction.trim_chain must certify its source-bound chain before refinement; got unresolved patch {patch}.",
            entity_ids=(patch,),
            requested=tuple((name, limit) for name, _, limit in resources)
            + (
                ("maximum_geometry_queries", ledger.limits.maximum_geometry_queries),
                ("maximum_work_units", ledger.limits.maximum_work_units),
            ),
            achieved=tuple((name, observed) for name, observed, _ in resources)
            + (
                ("geometry_queries", ledger.queries),
                ("work_units", ledger.work),
            ),
        )
    if np.any(ribbon > deviation):
        row = int(np.argmax(ribbon))
        interval = provenance[row, 2:]
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            f"SurfaceConstruction.trim_ribbon must meet the deviation bound after boundary refinement; got unrecoverable local trim ribbon on patch {patch}.",
            entity_ids=(patch,),
            requested=(("maximum_deviation", deviation),),
            achieved=(
                ("local_trim_ribbon", float(ribbon[row])),
                ("source_parameter_first", float(interval[0])),
                ("source_parameter_last", float(interval[1])),
            ),
        )
    with _source_bound_scope(
        ledger,
        0,
        stage=MeshingStageKind.SURFACE_MESHING,
        scope=scope,
        budget=budget,
    ):
        normals = _vertex_normals(
            domain,
            patch,
            charts,
            np.mean(boundary, axis=0),
            scope=scope,
            budget=budget,
        )
    return _InitialTrim(ribbon, normals)


def _initial_patch(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    patch: int,
    size: float,
    vertices: np.ndarray,
    curve_parameters: dict[int, np.ndarray],
    curve_ids: dict[int, np.ndarray],
    corner_ids: dict[int, int],
    deviation: float,
    maximum_normal_angle: float,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    local_limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _Patch:
    prepared = _prepare_initial_chart(
        domain,
        patch,
        size,
        vertices,
        curve_parameters,
        curve_ids,
        corner_ids,
        deviation,
        maximum_normal_angle,
        ledger,
        local_limits,
        compiled=compiled,
        record_phase=record_phase,
        scope=scope,
        budget=budget,
    )
    topology = _triangulate_initial_chart(
        prepared, patch, local_limits, ledger, record_phase, scope=scope
    )
    charts, identities = prepared.charts, prepared.identities
    declared_poles = [
        corner_ids[use.corner]
        for loop in domain.patches[patch].loops
        for use in loop
        if isinstance(use, _dependency_phydrax_geometry__meshing_domain.PatchPoleUse)
    ]
    pole_ids = np.where(np.isin(identities, declared_poles), identities, -1).astype(
        np.int64
    )
    restriction_required = np.zeros((charts.shape[0],), dtype=np.bool_)
    restriction_edges = np.full((charts.shape[0], 2), -1, dtype=np.int64)
    restriction_parameters = np.zeros((charts.shape[0], 2), dtype=np.int64)
    trim = _certify_initial_trim(
        domain,
        prepared,
        topology,
        patch,
        deviation,
        restriction_required,
        local_limits,
        ledger,
        scope=scope,
        budget=budget,
    )
    return _Patch(
        patch,
        prepared.charts,
        prepared.points,
        trim.normals,
        identities,
        pole_ids,
        topology.triangles,
        topology.constrained,
        prepared.segments,
        prepared.provenance,
        trim.ribbon,
        restriction_required,
        restriction_edges,
        restriction_parameters,
    )


def _angles(corners: np.ndarray, /) -> np.ndarray:
    """Interior angles ``(m, 3)`` of physical triangles ``(m, 3, 3)``."""

    first = np.roll(corners, -1, axis=1) - corners
    second = np.roll(corners, -2, axis=1) - corners
    lengths = np.linalg.norm(first, axis=2) * np.linalg.norm(second, axis=2)
    cosine = np.sum(first * second, axis=2) / np.maximum(
        lengths, np.finfo(np.float64).tiny
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _surface_capacity(
    vertices: int,
    triangles: int,
    candidates: int,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
) -> int:
    """Validate the simultaneous surface envelope and return its byte bound."""
    scratch = 2048 * (vertices + candidates) + 4096 * (triangles + 2 * candidates)
    capacities = (
        ("vertices", vertices + candidates, limits.maximum_vertices),
        ("faces", triangles + 2 * candidates, limits.maximum_faces),
        ("cells", triangles + 2 * candidates, limits.maximum_cells),
        ("scratch_bytes", scratch, limits.maximum_scratch_bytes),
        ("cavity_cells", min(triangles, 4), limits.maximum_cavity_cells),
    )
    for name, required, maximum in capacities:
        if required > maximum:
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"SurfaceConstruction.capacity must fit {name} limit {maximum} before allocation; got {required}.",
                requested=((f"maximum_{name}", maximum),),
                achieved=((name, required),),
            )
    return scratch


def _restriction_collapsed_pole_cells(state: _Patch, /) -> np.ndarray:
    """Identify cells confined to a zero-width rational pole-edge restriction."""
    triangles = state.triangles
    collapsed = np.zeros((triangles.shape[0],), dtype=np.bool_)
    for vertex in np.flatnonzero(state.restriction_required):
        first, second = (int(value) for value in state.restriction_edges[vertex])
        for pole_endpoint, radial_endpoint in ((first, second), (second, first)):
            pole = int(state.pole_ids[pole_endpoint])
            if pole < 0:
                continue
            collapsed |= (
                np.any(triangles == vertex, axis=1)
                & np.any(triangles == radial_endpoint, axis=1)
                & np.any(state.pole_ids[triangles] == pole, axis=1)
            )
    return collapsed


# One ordered vertex-row triple per cell; equal records are the same cell.
_CELL_KEY = np.dtype([("first", np.int64), ("second", np.int64), ("third", np.int64)])


@final
class _MetricsCache:
    """Chart-only source measurements of the cells of one refining patch.

    Native reconnection appends vertex rows and never moves one, so an certified
    ordered vertex triple fixes a cell's chart corners and, with them, its
    sampled source images, centroid normal and continuous interpolation and
    normal-turn bounds. Vertex nodal errors are likewise fixed per row. A
    changed retained vertex row invalidates the whole bank; cells that leave
    the triangulation are dropped when the next measurement is stored.
    """

    __slots__ = (
        "charts",
        "points",
        "keys",
        "surface",
        "normals",
        "regular",
        "interpolation",
        "normal_turn",
        "node_errors",
    )

    def __init__(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self.charts = np.empty((0, 2), dtype=np.float64)
        self.points = np.empty((0, 3), dtype=np.float64)
        self.keys = np.empty((0,), dtype=_CELL_KEY)
        self.surface = np.empty((0, 4, 3), dtype=np.float64)
        self.normals = np.empty((0, 3), dtype=np.float64)
        self.regular = np.empty((0,), dtype=np.bool_)
        self.interpolation = np.empty((0,), dtype=np.float64)
        self.normal_turn = np.empty((0,), dtype=np.float64)
        self.node_errors = np.empty((0,), dtype=np.float64)

    def _retained(self, state: _Patch, /) -> bool:
        known = self.charts.shape[0]
        return (
            state.charts.shape[0] >= known
            and np.array_equal(state.charts[:known], self.charts)
            and np.array_equal(state.points[:known], self.points)
        )

    def measure(
        self,
        domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
        state: _Patch,
        ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
        samples: np.ndarray,
        centroid_charts: np.ndarray,
        /,
        *,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
        restriction_bank: _dependency_phydrax_geometry_brep__patches.BernsteinRestrictionBank,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples, centroid normals/regularity, interpolation, turn, node errors."""
        if not self._retained(state):
            self._clear()
        triangles = state.triangles
        count = triangles.shape[0]
        keys = np.ascontiguousarray(triangles, dtype=np.int64).view(_CELL_KEY)[:, 0]
        order = np.argsort(self.keys, kind="stable")
        position = np.minimum(
            np.searchsorted(self.keys[order], keys), max(self.keys.size - 1, 0)
        )
        found = (
            self.keys[order][position] == keys
            if self.keys.size
            else np.zeros((count,), dtype=np.bool_)
        )
        rows = order[position[found]]
        fresh = np.flatnonzero(~found)
        surface = np.empty((count, 4, 3), dtype=np.float64)
        normals = np.empty((count, 3), dtype=np.float64)
        regular = np.empty((count,), dtype=np.bool_)
        interpolation = np.empty((count,), dtype=np.float64)
        normal_turn = np.empty((count,), dtype=np.float64)
        surface[found] = self.surface[rows]
        normals[found] = self.normals[rows]
        regular[found] = self.regular[rows]
        interpolation[found] = self.interpolation[rows]
        normal_turn[found] = self.normal_turn[rows]
        scratch = 2048 * state.points.shape[0] + 4096 * count
        if fresh.size:
            ledger.reserve(5 * fresh.size, 0, MeshingStageKind.SURFACE_MESHING)
            surface[fresh] = domain.evaluate(
                np.full((4 * fresh.size,), state.index),
                samples[fresh].reshape((-1, 2)),
                budget=budget,
            ).reshape((fresh.size, 4, 3))
            normals[fresh], regular[fresh] = domain.oriented_normals(
                np.full((fresh.size,), state.index), centroid_charts[fresh], budget=budget
            )
            charts = state.charts[triangles[fresh]]
            # Both bounds of the same fresh cells restrict one set of meridian
            # span overlaps; shared jets are enclosed and charged once.
            with (
                _dependency_phydrax_geometry_brep__patches.shared_meridian_jets() as jets
            ):
                with _interpolation_bounds(
                    domain,
                    state.index,
                    charts,
                    ledger,
                    scratch,
                    scope=scope,
                    budget=budget,
                    restriction_bank=restriction_bank,
                    meridian_jets=jets,
                ) as bounds:
                    interpolation[fresh] = bounds
                    del bounds
                with _source_bound_scope(
                    ledger,
                    scratch,
                    stage=MeshingStageKind.SURFACE_MESHING,
                    scope=scope,
                    budget=budget,
                ):
                    normal_turn[fresh] = domain.normal_turn_bounds(
                        state.index,
                        charts,
                        budget=budget,
                        restriction_bank=restriction_bank,
                        meridian_jets=jets,
                    )
        known = self.node_errors.size
        node_errors = self.node_errors
        if state.charts.shape[0] > known:
            added = state.charts.shape[0] - known
            ledger.reserve(added, 0, MeshingStageKind.SURFACE_MESHING)
            nodal = domain.evaluate(
                np.full((added,), state.index), state.charts[known:], budget=budget
            )
            node_errors = np.concatenate(
                (node_errors, np.linalg.norm(nodal - state.points[known:], axis=1))
            )
        self.charts = state.charts.copy()
        self.points = state.points.copy()
        self.keys = keys
        self.surface = surface
        self.normals = normals
        self.regular = regular
        self.interpolation = interpolation
        self.normal_turn = normal_turn.copy()
        self.node_errors = node_errors
        return surface, normals, regular, interpolation, normal_turn, node_errors


@final
class _Metrics:
    """Physical measurements of every triangle of one patch.

    Chart-only measurements of cells retained from an earlier round of the
    same refinement are reused from its ``_MetricsCache``.
    """

    __slots__ = (
        "angles",
        "collapsed",
        "deviation",
        "deviation_bound",
        "normal_bound",
        "invalid",
        "lengths",
        "size_limits",
        "vertex_metrics",
        "metric_lengths",
        "metric_angles",
        "metric_squared",
        "midpoint_charts",
        "misoriented",
        "worst_sample",
    )

    def __init__(
        self,
        domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
        state: _Patch,
        ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
        compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
        cache: _MetricsCache,
        /,
        *,
        sphere_frame: tuple[np.ndarray, float, float] | None = None,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> None:
        with (
            _source_bound_scope(
                ledger,
                0,
                stage=MeshingStageKind.SURFACE_MESHING,
                scope=scope,
                budget=budget,
            ),
            _dependency_phydrax_geometry_brep__patches.source_bernstein_restriction_scope(
                budget=budget
            ) as restriction_bank,
        ):
            self._initialize(
                domain,
                state,
                ledger,
                compiled,
                cache,
                sphere_frame=sphere_frame,
                scope=scope,
                budget=budget,
                restriction_bank=restriction_bank,
            )

    def _initialize(
        self,
        domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
        state: _Patch,
        ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
        compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
        cache: _MetricsCache,
        /,
        *,
        sphere_frame: tuple[np.ndarray, float, float] | None = None,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
        restriction_bank: _dependency_phydrax_geometry_brep__patches.BernsteinRestrictionBank,
    ) -> None:
        triangles = state.triangles
        _surface_capacity(state.points.shape[0], triangles.shape[0], 0, ledger.limits)
        corners = state.points[triangles]
        charts = state.charts[triangles]
        ids = state.identities[triangles]
        first = np.roll(ids, -1, axis=1)
        second = np.roll(ids, -2, axis=1)
        # Edge k joins the vertices after k; equal shared ids collapse it.
        same = (first == second) & (first >= 0)
        self.collapsed = np.any(same & state.constrained, axis=1)
        self.collapsed |= _restriction_collapsed_pole_cells(state)
        duplicated = np.any(same & ~state.constrained, axis=1)
        midpoint_charts = 0.5 * (
            np.roll(charts, -1, axis=1) + np.roll(charts, -2, axis=1)
        )
        centroid_charts = np.mean(charts, axis=1)
        samples = np.concatenate((centroid_charts[:, None], midpoint_charts), axis=1)
        count = triangles.shape[0]
        surface, normals, regular, interpolation, normal_turn, node_errors = (
            cache.measure(
                domain,
                state,
                ledger,
                samples,
                centroid_charts,
                scope=scope,
                budget=budget,
                restriction_bank=restriction_bank,
            )
        )
        ledger.observe_native(count * 3, 0, MeshingStageKind.SURFACE_MESHING)
        self.size_limits = compiled.source_sizing.evaluate(
            domain, state.index, centroid_charts, surface[:, 0], budget=budget
        )
        flat_mid = 0.5 * (np.roll(corners, -1, axis=1) + np.roll(corners, -2, axis=1))
        flat = np.concatenate((np.mean(corners, axis=1)[:, None], flat_mid), axis=1)
        distances = np.linalg.norm(surface - flat, axis=2)
        # Constrained edges follow feature curves bounded by their own route.
        distances[:, 1:][state.constrained] = 0.0
        orientation = domain.patches[state.index].orientation
        flat_normal = orientation * np.cross(
            corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        )
        active = ~self.collapsed
        if np.any(active & ~regular):
            rows = np.flatnonzero(active & ~regular)
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SOURCE,
                f"SurfaceConstruction.source_differential must be regular inside the chart; got singular patch {state.index}.",
                entity_ids=(state.index,),
                achieved=(("singular_samples", float(rows.size)),),
            )
        mid_surface = surface[:, 1:]
        ends_first = np.roll(corners, -1, axis=1)
        ends_second = np.roll(corners, -2, axis=1)
        self.lengths = np.linalg.norm(mid_surface - ends_first, axis=2) + np.linalg.norm(
            ends_second - mid_surface, axis=2
        )
        self.angles = _angles(corners)
        metric = compiled.source_sizing.metric
        self.vertex_metrics = None
        self.metric_lengths = np.zeros_like(self.lengths)
        self.metric_angles = self.angles
        self.metric_squared = np.sum((ends_first - ends_second) ** 2, axis=2)
        if metric is not None:
            ledger.reserve(state.points.shape[0], 0, MeshingStageKind.SURFACE_MESHING)
            values, work = metric.sample(
                state.points,
                ledger.limits.maximum_work_units - ledger.work,
                budget=budget,
            )
            ledger.reserve(0, work, MeshingStageKind.SURFACE_MESHING)
            self.vertex_metrics = values
            edges = np.stack(
                (np.roll(triangles, -1, axis=1), np.roll(triangles, -2, axis=1)), axis=-1
            ).reshape((-1, 2))
            bound = metric.control.maximum_metric_edge_length or 1.0
            self.metric_lengths = (
                np.asarray(
                    _dependency_phydrax_meshing__metric.metric_edge_lengths(
                        values, state.points, edges
                    ),
                    dtype=np.float64,
                ).reshape((-1, 3))
                / bound
            )
            mean = np.asarray(
                _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
                    values[triangles], np.full(triangles.shape, 1 / 3, dtype=np.float64)
                ),
                dtype=np.float64,
            )
            edge = ends_first - ends_second
            self.metric_squared = np.sum(
                edge * (mean[:, None] @ edge[..., None])[..., 0], axis=2
            )
            first = np.roll(corners, -1, axis=1) - corners
            second = np.roll(corners, -2, axis=1) - corners
            product = np.sum(first * (mean[:, None] @ second[..., None])[..., 0], axis=2)
            first_norm = np.sum(
                first * (mean[:, None] @ first[..., None])[..., 0], axis=2
            )
            second_norm = np.sum(
                second * (mean[:, None] @ second[..., None])[..., 0], axis=2
            )
            cosine = product / np.sqrt(
                np.maximum(first_norm * second_norm, np.finfo(np.float64).tiny)
            )
            self.metric_angles = np.arccos(np.clip(cosine, -1, 1))
        self.deviation = np.max(distances, axis=1)
        restriction_error = np.max(
            _restriction_physical_errors(domain, state, scope=scope, budget=budget)[
                triangles
            ],
            axis=1,
        )
        node_error = np.max(node_errors[triangles], axis=1) + restriction_error
        # Each source-boundary segment owns its curve, parameter interval,
        # and independently certified ribbon. Only its incident simplex is
        # charged; unrelated boundary arcs and interior cells receive no
        # patch-global maximum.
        trim_bound = domain_chart_chain.boundary_cell_ribbon_bounds(
            triangles,
            state.boundary_edges,
            state.trim_deviation_bounds,
        )
        self.deviation_bound = (
            interpolation
            + node_error
            + trim_bound
            + 256 * np.finfo(np.float64).eps * domain.scale
        )
        if sphere_frame is not None:
            analytic = (
                domain_numerics.sphere_triangle_distance_bounds(sphere_frame, corners)
                + restriction_error
                + trim_bound
            )
            self.deviation_bound = np.minimum(self.deviation_bound, analytic)
        self.normal_bound = normal_turn
        if sphere_frame is not None:
            self.normal_bound = np.minimum(
                self.normal_bound,
                domain_numerics.sphere_triangle_normal_bounds(sphere_frame, corners),
            )
        self.worst_sample = samples[np.arange(count), np.argmax(distances, axis=1)]
        self.midpoint_charts = midpoint_charts

        relative_floor = _owner_phydrax_discretization__cell_geometry_validity.CellValidityPolicy().relative_determinant_floor
        edge_scale = np.max(
            np.sum((ends_first - ends_second) ** 2, axis=2),
            axis=1,
        )
        nearly_singular = (
            np.linalg.norm(flat_normal, axis=1) <= relative_floor * edge_scale
        )
        self.invalid = active & (duplicated | nearly_singular)
        self.misoriented = (
            active & ~duplicated & (np.sum(flat_normal * normals, axis=1) <= 0.0)
        )


def _circumcenter_weights(squared: np.ndarray, /) -> np.ndarray:
    """Barycentric weights ``(m, 3)`` of circumcenters from squared opposite edge lengths."""

    total = np.sum(squared, axis=1, keepdims=True)
    weights = squared * (total - 2.0 * squared)
    return weights / np.sum(weights, axis=1, keepdims=True)


def _clamped_circumcenters(
    corners: np.ndarray,
    charts: np.ndarray,
    constrained: np.ndarray,
    squared: np.ndarray,
    /,
) -> np.ndarray:
    """Chart images of physical circumcenters, pulled into their triangles.

    Feature curves are never split, so a point is kept at least a third of
    the apex height away from a constrained edge (the centroid when two edges
    are constrained); a point near a fixed edge would only create a sliver.
    """

    weights = _circumcenter_weights(squared)
    centroid = np.full_like(weights, 1.0 / 3.0)
    shortfall = np.where(
        weights < _CLAMP,
        (centroid - _CLAMP) / np.maximum(centroid - weights, 1e-300),
        1.0,
    )
    step = np.clip(np.min(shortfall, axis=1, keepdims=True), 0.0, 1.0)
    clamped = centroid + step * (weights - centroid)
    single = np.sum(constrained, axis=1) == 1
    edge = np.argmax(constrained, axis=1)
    apex = clamped[np.arange(edge.size), edge]
    lift = np.where(single & (apex < 1.0 / 3.0), (1.0 / 3.0 - apex) / (1.0 - apex), 0.0)
    toward = np.eye(3, dtype=np.float64)[edge]
    clamped += lift[:, None] * (toward - clamped)
    clamped = np.where((np.sum(constrained, axis=1) >= 2)[:, None], centroid, clamped)
    return np.sum(clamped[..., None] * charts, axis=1)


def _patch_edge_lengths(state: _Patch, metrics: _Metrics, /) -> np.ndarray:
    """Physical lengths in the same unique scientific-edge bank as compliance."""
    active = ~metrics.collapsed
    triangles = state.triangles[active]
    if not triangles.shape[0]:
        return np.empty((0,), dtype=np.float64)
    local_edges = np.asarray(((1, 2), (2, 0), (0, 1)), dtype=np.int32)
    records = np.sort(triangles[:, local_edges].reshape((-1, 2)), axis=1)
    _, first = np.unique(records, axis=0, return_index=True)
    return metrics.lengths[active].reshape(-1)[first]


def _required_size_refinement(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    metrics: _Metrics,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    policy: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy | None,
    /,
) -> np.ndarray:
    """Select only size growth that can satisfy the authored hard obligations."""
    active = ~metrics.collapsed
    local = active & (
        (np.max(metrics.lengths, axis=1) > metrics.size_limits * (1 + _SIZE_SLACK))
        | (np.max(metrics.metric_lengths, axis=1) > _METRIC_AIM)
    )
    if policy is None:
        return local
    source_face = int(domain.source_indices[2][state.index])
    applicable = []
    for control in compiled.source_sizing.controls:
        match control:
            case _dependency_phydrax_meshing__sizing.ProximitySizeControl():
                scopes = (control.source_scope, control.target_scope)
            case (
                _dependency_phydrax_meshing__sizing.UniformSizeControl()
                | _dependency_phydrax_meshing__sizing.CurvatureSizeControl()
            ):
                scopes = (control.scope,)
            case _:
                raise TypeError(
                    f"CompiledSurfaceDomain.source_sizing.controls must contain a supported size control; got {type(control).__name__}."
                )
        if any(
            scope.entity_dimension == 2
            and source_face in np.asarray(scope.global_entity_ids)
            for scope in scopes
        ):
            applicable.append(control)
    hard_uniform = tuple(
        control
        for control in applicable
        if isinstance(control, _dependency_phydrax_meshing__sizing.UniformSizeControl)
        and control.strength
        is _dependency_phydrax_meshing__sizing.SizeControlStrength.HARD
    )
    if not hard_uniform or any(
        not isinstance(control, _dependency_phydrax_meshing__sizing.UniformSizeControl)
        for control in applicable
    ):
        return local
    required = active & (np.max(metrics.metric_lengths, axis=1) > _METRIC_AIM)
    lengths = _patch_edge_lengths(state, metrics)
    if not lengths.size:
        return required
    target_growth = False
    for control in hard_uniform:
        if control.maximum_size is not None:
            maximum = control.maximum_size + policy.tolerance(control.maximum_size)
            required |= active & (np.max(metrics.lengths, axis=1) > maximum)
        for statistic in policy.target_statistics:
            quantile = _dependency_phydrax_meshing__sizing.size_statistic_quantile(
                statistic
            )
            if float(np.quantile(lengths, quantile)) > (
                control.target_size + policy.tolerance(control.target_size)
            ):
                target_growth = True
    if target_growth:
        required |= local
    return required


def _physical_refinement_mask(
    metrics: _Metrics,
    deviation: float,
    maximum_normal_angle: float,
    required_minimum_angle: float,
    /,
    *,
    required_size: np.ndarray | None = None,
) -> np.ndarray:
    active = ~metrics.collapsed
    size = (
        active
        & (
            (np.max(metrics.lengths, axis=1) > metrics.size_limits * (1 + _SIZE_SLACK))
            | (np.max(metrics.metric_lengths, axis=1) > _METRIC_AIM)
        )
        if required_size is None
        else required_size
    )
    return (
        metrics.invalid
        | metrics.misoriented
        | size
        | (active & (np.min(metrics.angles, axis=1) < required_minimum_angle))
        | (np.isfinite(deviation) & (metrics.deviation_bound > deviation))
        | (
            np.isfinite(maximum_normal_angle)
            & (metrics.normal_bound > maximum_normal_angle)
        )
    )


def _candidates(
    state: _Patch,
    metrics: _Metrics,
    size: float,
    deviation: float,
    aim: float,
    spacing_fraction: float,
    maximum_normal_angle: float,
    /,
    *,
    required_minimum_angle: float = 0.0,
    planar_centroids: bool = False,
    required_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chart points, interior fallbacks, spacings, hints, unresolved mask and selected edges."""

    active = ~metrics.collapsed
    longest = np.max(metrics.lengths, axis=1)
    metric_longest = np.max(metrics.metric_lengths, axis=1)
    smallest = np.min(metrics.metric_angles, axis=1)
    corner = np.argmin(metrics.metric_angles, axis=1)
    # An angle between two constrained edges is fixed by the feature curves.
    fixed = np.all(
        state.constrained[
            np.arange(corner.size)[:, None],
            np.stack(((corner + 1) % 3, (corner + 2) % 3), axis=1),
        ],
        axis=1,
    )
    local_size = metrics.size_limits
    long = active & (
        (longest > local_size * (1.0 + _SIZE_SLACK)) | (metric_longest > _METRIC_AIM)
    )
    far = (metrics.deviation_bound > deviation) & np.isfinite(deviation)
    turning = (metrics.normal_bound > maximum_normal_angle) & np.isfinite(
        maximum_normal_angle
    )
    poor = active & (smallest < aim) & ~fixed
    poor |= active & (np.min(metrics.angles, axis=1) < required_minimum_angle) & ~fixed
    broken = metrics.invalid | metrics.misoriented
    bad = broken | long | far | poor | turning
    score = np.where(
        broken,
        np.inf,
        np.maximum.reduce(
            (
                longest / local_size,
                metric_longest,
                np.where(
                    np.isfinite(deviation), metrics.deviation_bound / deviation, 0.0
                ),
                np.where(poor, aim / np.maximum(smallest, 1e-300), 0.0),
            )
        ),
    )
    rows = np.flatnonzero(bad if required_mask is None else bad & required_mask)
    rows = rows[np.argsort(-score[rows], kind="stable")]
    charts = state.charts[state.triangles[rows]]
    corners = state.points[state.triangles[rows]]
    centroid = np.mean(charts, axis=1)
    points = centroid.copy()
    regular = active[rows] & ~broken[rows]
    points[regular] = _clamped_circumcenters(
        corners[regular],
        charts[regular],
        state.constrained[rows][regular],
        metrics.metric_squared[rows][regular],
    )
    if planar_centroids:
        size_only = long[rows] & ~far[rows] & ~turning[rows] & ~poor[rows] & ~broken[rows]
        # Affine planar circumcenters can align across neighboring Delaunay
        # cells and create an almost-zero simplex. Centroids still reduce every
        # size-only parent edge; native reconnection retains quality ownership.
        points[size_only] = centroid[size_only]
    interior = points.copy()
    insert_edges = np.full((rows.size, 2), -1, dtype=np.int32)
    # A required angle is a Ruppert/Chew condition: an obtuse triangle's
    # circumcenter lies across its longest edge, and a clamped point next to
    # that edge only shortens the edge or is refused. Insert the measured
    # circumcenter, located by the predicate chart walk; a walk blocked by a
    # constrained edge or the chart domain retries at the clamped interior point.
    hard = np.flatnonzero(
        active[rows]
        & ~broken[rows]
        & ~fixed[rows]
        & (np.min(metrics.angles[rows], axis=1) < required_minimum_angle)
    )
    weights = _circumcenter_weights(metrics.metric_squared[rows[hard]])
    outside = np.any(weights < 0.0, axis=1)
    points[hard[outside]] = np.sum(
        weights[outside, :, None] * charts[hard[outside]], axis=1
    )
    # A soft shape aim cannot override a required source-geometry split. Its
    # circumcenter can be arbitrarily close to a retained parent edge without
    # reducing that edge's continuous error.
    deviating = (far[rows] | turning[rows]) & ~broken[rows]
    # Hard geometry needs an edge split when the retained parent edge controls
    # its deviation; centroid splitting cannot reduce that edge's error floor.
    pending = np.flatnonzero(deviating)
    rank = np.argsort(
        -np.where(
            state.constrained[rows[pending]],
            -np.inf,
            metrics.metric_squared[rows[pending]],
        ),
        axis=1,
        kind="stable",
    )
    for choice in range(3):
        if not pending.size:
            break
        edge = rank[:, choice]
        eligible = ~state.constrained[rows[pending], edge]
        selected = np.flatnonzero(eligible)
        if not selected.size:
            continue
        triangles = state.triangles[rows[pending[selected]]]
        first = state.charts[
            triangles[np.arange(selected.size), (edge[selected] + 1) % 3]
        ]
        last = state.charts[triangles[np.arange(selected.size), (edge[selected] + 2) % 3]]
        midpoint = 0.5 * (first + last)
        # Endpoint identity, not rounded chart collinearity, selects the
        # full native paired-edge cavity.
        represented = np.any(midpoint != first, axis=1) & np.any(midpoint != last, axis=1)
        points[pending[selected[represented]]] = midpoint[represented]
        insert_edges[pending[selected[represented]]] = np.stack(
            (
                triangles[np.arange(selected.size), (edge[selected] + 1) % 3],
                triangles[np.arange(selected.size), (edge[selected] + 2) % 3],
            ),
            axis=1,
        )[represented]
        keep = np.ones(pending.size, dtype=np.bool_)
        keep[selected[represented]] = False
        pending, rank = pending[keep], rank[keep]
    spacing = spacing_fraction * np.minimum(local_size[rows], longest[rows])
    if metrics.vertex_metrics is not None:
        spacing = spacing_fraction * np.minimum(1.0, metric_longest[rows])
    if np.isfinite(deviation):
        spacing *= np.minimum(
            1.0,
            np.sqrt(
                deviation
                / np.maximum(metrics.deviation_bound[rows], np.finfo(np.float64).tiny)
            ),
        )
    if np.isfinite(maximum_normal_angle):
        spacing *= np.minimum(
            1.0,
            maximum_normal_angle
            / np.maximum(metrics.normal_bound[rows], np.finfo(np.float64).tiny),
        )
    unresolved = broken | long | far | poor | turning
    return points, interior, spacing, rows.astype(np.int32), unresolved, insert_edges


def _validate_charts(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    try:
        signs = _restriction_orientation_signs(domain, state, scope=scope, budget=budget)
    except (TypeError, ValueError) as error:
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SOURCE,
            f"SurfaceConstruction.chart_restrictions must retain rational source roots; got patch {state.index}: {error}.",
            entity_ids=(state.index,),
        ) from error
    if np.any(signs <= 0):
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"SurfaceConstruction.chart_orientation must stay positive after reconnection; got inverted chart triangle on patch {state.index}.",
            entity_ids=(state.index,),
        )


@final
class _RationalRestrictions(StrictModule):
    """Bind rounded insertion coordinates to their rational source edge roots."""

    charts: np.ndarray
    required: np.ndarray
    edges: np.ndarray
    parameters: np.ndarray


def _rational_edge_candidates(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    charts: np.ndarray,
    hints: np.ndarray,
    insert_edges: np.ndarray,
    /,
) -> _RationalRestrictions:
    """Bind collapsed-pole radial splits to unique rational roots."""
    result = charts.copy()
    required = np.zeros((charts.shape[0],), dtype=np.bool_)
    authority_edges = np.full((charts.shape[0], 2), -1, dtype=np.int64)
    parameters = np.zeros((charts.shape[0], 2), dtype=np.int64)
    for row, hint in enumerate(hints):
        triangle = state.triangles[hint]
        identities = state.pole_ids[triangle]
        for edge in range(3):
            first, last = (edge + 1) % 3, (edge + 2) % 3
            if not (
                state.constrained[hint, edge]
                and identities[first] >= 0
                and identities[first] == identities[last]
                and np.array_equal(
                    state.points[triangle[first]], state.points[triangle[last]]
                )
            ):
                continue
            selected_edge: np.ndarray | None = None
            for selected in (first, last):
                if not state.constrained[hint, selected]:
                    selected_edge = triangle[[(selected + 1) % 3, (selected + 2) % 3]]
                    break
            if selected_edge is None:
                raise _failure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    "SurfaceConstruction.pole_triangle must have an unconstrained radial source edge; got no restrictable radial edge.",
                    entity_ids=(state.index,),
                )
            insert_edges[row] = selected_edge
            authority_edges[row] = selected_edge
            required[row] = True
            # The rational affine midpoint defines the source root even when neither of its
            # rational UV coordinates has a binary64 representation.
            parameters[row] = np.asarray((1, 2), dtype=np.int64)
            break
    exact, _, _ = _patch_restrictions(domain, state)
    for row in np.flatnonzero(required):
        first, last = (int(value) for value in authority_edges[row])
        numerator, denominator = (int(value) for value in parameters[row])
        parameter = Fraction(numerator, denominator)
        coordinate = (
            (Fraction(1) - parameter) * exact[first, 0] + parameter * exact[last, 0],
            (Fraction(1) - parameter) * exact[first, 1] + parameter * exact[last, 1],
        )
        result[row] = np.asarray(
            (float(coordinate[0]), float(coordinate[1])), dtype=np.float64
        )
        if np.array_equal(result[row], state.charts[first]) or np.array_equal(
            result[row], state.charts[last]
        ):
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "SurfaceConstruction.restriction_coordinate must distinguish its edge endpoints after rounding; got no distinct binary64 representative.",
                entity_ids=(state.index,),
            )
    return _RationalRestrictions(result, required, authority_edges, parameters)


def _missing_pole_split_hints(state: _Patch, /) -> np.ndarray:
    """Return the first collapsed triangle per pole lacking a rational root binding."""
    covered = {
        int(state.pole_ids[endpoint])
        for vertex in np.flatnonzero(state.restriction_required)
        for endpoint in state.restriction_edges[vertex]
        if state.pole_ids[endpoint] >= 0
    }
    hints: dict[int, int] = {}
    for hint, triangle in enumerate(state.triangles):
        identities = state.pole_ids[triangle]
        for edge in range(3):
            first, last = (edge + 1) % 3, (edge + 2) % 3
            pole = int(identities[first])
            if (
                pole >= 0
                and pole not in covered
                and pole not in hints
                and state.constrained[hint, edge]
                and pole == identities[last]
                and np.array_equal(
                    state.points[triangle[first]], state.points[triangle[last]]
                )
            ):
                hints[pole] = hint
                break
    return np.asarray([hints[pole] for pole in sorted(hints)], dtype=np.int32)


@final
class _InsertionGeometry(StrictModule):
    """Bind insertion coordinates, source normals, and scheduling spacing."""

    points: np.ndarray
    normals: np.ndarray
    spacing: np.ndarray
    vertex_metrics: np.ndarray | None


def _prepare_insertion_geometry(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    metrics: _Metrics,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    record_phase: NativeMeshingPhaseRecorder | None,
    batch: RestrictionBatch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _InsertionGeometry:
    """Evaluate a bounded insertion batch and retain hard-criterion spacing."""
    charts, hints, spacing = batch.charts, batch.hints, batch.spacing
    parents, hard = batch.parents, batch.hard
    with _source_bound_scope(
        ledger, 0, stage=MeshingStageKind.SURFACE_MESHING, scope=scope, budget=budget
    ):
        _surface_capacity(
            state.points.shape[0], state.triangles.shape[0], hints.size, limits
        )
        ledger.reserve(hints.size, 0, MeshingStageKind.SURFACE_MESHING)
        with measure_phase(record_phase, "geometry_evaluation"):
            points = domain.evaluate(
                np.full((hints.size,), state.index), charts, budget=budget
            )
            normals = _vertex_normals(
                domain,
                state.index,
                charts,
                np.mean(state.charts, axis=0),
                scope=scope,
                budget=budget,
            )
        insert_metrics = None
        if compiled.source_sizing.metric is not None:
            ledger.reserve(points.shape[0], 0, MeshingStageKind.SURFACE_MESHING)
            insert_metrics, metric_work = compiled.source_sizing.metric.sample(
                points, limits.maximum_work_units - ledger.work, budget=budget
            )
            ledger.reserve(0, metric_work, MeshingStageKind.SURFACE_MESHING)
    # Scheduling separation must not refuse a scientifically necessary
    # insertion merely because its parent has a long edge and small altitude.
    # The native queue retains its certified chart and source-normal legality.
    spacing = spacing.copy()
    for row in np.flatnonzero(hard):
        offset = points[row] - state.points[parents[row]]
        if metrics.vertex_metrics is None:
            distance = np.linalg.norm(offset, axis=1)
        else:
            tensor = np.mean(metrics.vertex_metrics[parents[row]], axis=0)
            distance = np.sqrt(
                np.maximum(
                    np.sum(offset * (tensor @ offset[..., None])[..., 0], axis=1), 0.0
                )
            )
        spacing[row] = min(spacing[row], 0.25 * float(np.min(distance)))
    return _InsertionGeometry(points, normals, spacing, insert_metrics)


def _reconnect_batch(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    metrics: _Metrics,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    totals: RefinementTotals,
    record_phase: NativeMeshingPhaseRecorder | None,
    batch: RestrictionBatch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Insert one batch of chart points and commit the reconnected patch.

    ``parents`` are the vertex triples of the refined triangles at candidate
    selection; returns the accepted mask of the batch.
    """
    charts = batch.charts
    hints = batch.hints
    unresolved = batch.unresolved
    insert_edges = batch.insert_edges
    restriction_required = batch.required
    restriction_edges = batch.edges
    restriction_parameters = batch.parameters
    prepared = _prepare_insertion_geometry(
        domain,
        state,
        metrics,
        compiled,
        limits,
        ledger,
        record_phase,
        batch,
        scope=scope,
        budget=budget,
    )
    points, normals, spacing, insert_metrics = (
        prepared.points,
        prepared.normals,
        prepared.spacing,
        prepared.vertex_metrics,
    )
    remaining = limits.maximum_work_units - ledger.work
    capacity = min(limits.maximum_faces, state.triangles.shape[0] + 2 * hints.size)
    scratch = _surface_capacity(
        state.points.shape[0], state.triangles.shape[0], hints.size, limits
    )
    incoming = scope.ticket_view()
    remaining = min(remaining, incoming.remaining_work_units)
    reconnect_start = phase_started(record_phase)
    with (
        scope.batch(
            max(remaining, 0),
            bytes_upper=scratch,
            cavity_upper=min(state.triangles.shape[0], 4),
        ) as ticket,
        ticket.native_target(),
    ):
        try:
            native = surface_reconnect(
                state.charts,
                state.points,
                state.normals,
                state.triangles,
                state.constrained,
                charts,
                points,
                normals,
                spacing,
                hints,
                sweep=not np.any(restriction_required),
                max_triangles=max(capacity, state.triangles.shape[0]),
                work_limit=max(remaining, 0),
                vertex_metrics=metrics.vertex_metrics,
                insert_metrics=insert_metrics,
                vertex_pole_ids=state.pole_ids,
                insert_edges=insert_edges,
            )
        except MeshcoreError as error:
            # The remaining allowance cannot pay mandatory source validation;
            # the refused call spent no work and published nothing.
            if error.status != MeshcoreStatus.CAPACITY_EXCEEDED:
                raise
            record_elapsed(record_phase, "refinement", reconnect_start, work_units=0)
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"SurfaceConstruction.refinement must fit its work and face budgets; got exhausted source validation on patch {state.index}.",
                entity_ids=(state.index,),
                requested=(
                    ("maximum_work_units", limits.maximum_work_units),
                    ("maximum_faces", limits.maximum_faces),
                ),
                achieved=(
                    ("work_units", ledger.work),
                    ("faces", state.triangles.shape[0]),
                    ("unresolved_triangles", int(np.sum(unresolved))),
                ),
            ) from error
    triangles = native.triangles
    constrained = native.constrained
    ids = native.vertex_ids
    status = native.item_status
    counters = native.counters
    record_elapsed(record_phase, "refinement", reconnect_start, work_units=counters.work)
    ledger.observe_native(0, counters.work, MeshingStageKind.SURFACE_MESHING)
    accepted = ids >= 0
    previous_count = state.charts.shape[0]
    expected_ids = previous_count + np.arange(int(np.sum(accepted)), dtype=np.int64)
    if not np.array_equal(ids[accepted], expected_ids):
        raise RuntimeError(
            f"surface_reconnect.vertex_ids must append contiguous vertex rows; got {ids[accepted].tolist()}, expected {expected_ids.tolist()}."
        )
    state.charts = np.concatenate((state.charts, charts[accepted]), axis=0)
    state.points = np.concatenate((state.points, points[accepted]), axis=0)
    state.normals = np.concatenate((state.normals, normals[accepted]), axis=0)
    state.identities = np.concatenate(
        (state.identities, np.full((int(np.sum(accepted)),), -1, dtype=np.int64))
    )
    state.pole_ids = np.concatenate(
        (state.pole_ids, np.full((int(np.sum(accepted)),), -1, dtype=np.int64))
    )
    state.restriction_required = np.concatenate(
        (state.restriction_required, restriction_required[accepted])
    )
    state.restriction_edges = np.concatenate(
        (state.restriction_edges, restriction_edges[accepted]), axis=0
    )
    state.restriction_parameters = np.concatenate(
        (state.restriction_parameters, restriction_parameters[accepted]), axis=0
    )
    if metrics.vertex_metrics is not None and insert_metrics is not None:
        # A same-round retry reconnects against the committed vertex set.
        metrics.vertex_metrics = np.concatenate(
            (metrics.vertex_metrics, insert_metrics[accepted]), axis=0
        )
    state.triangles = triangles.astype(np.int64)
    state.constrained = constrained
    totals.inserted += counters.inserted
    totals.refused += counters.refused
    totals.flips += counters.flips
    _validate_charts(domain, state, scope=scope, budget=budget)
    if counters.exhausted or np.any(status == MeshcoreStatus.CAPACITY_EXCEEDED):
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"SurfaceConstruction.refinement must fit its work and face budgets; got exhausted refinement on patch {state.index}.",
            entity_ids=(state.index,),
            requested=(
                ("maximum_work_units", limits.maximum_work_units),
                ("maximum_faces", limits.maximum_faces),
            ),
            achieved=(
                ("work_units", ledger.work),
                ("faces", state.triangles.shape[0]),
                ("unresolved_triangles", int(np.sum(unresolved))),
            ),
        )
    return accepted


def _requires_exact_pole_restrictions(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain, patch: int, /
) -> bool:
    """Whether a pole patch is cut by a curve owned by another source face."""
    source = domain.patches[patch]
    if not any(
        isinstance(use, _dependency_phydrax_geometry__meshing_domain.PatchPoleUse)
        for loop in source.loops
        for use in loop
    ):
        return False
    return any(
        incident_patch != patch
        for loop in source.loops
        for use in loop
        if isinstance(use, _dependency_phydrax_geometry__meshing_domain.PatchCurveUse)
        for incident_patch, _ in domain.curve_uses(use.curve)
    )


def _open_pole_restriction_fans(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    vertices: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> int:
    """Rotate zero-width children of rational pole splits into the physical fan.

    Splitting a declared radial edge ``(P, R)`` at ``m`` leaves children
    ``(Q, m, R)`` whose pole copy ``Q`` gives them zero width in the pole gauge
    (``_restriction_collapsed_pole_cells``). A split radial edge between two
    such children therefore leaves ``m`` outside every physical cell, so its
    rational root binding is never published. Each child flips its edge ``(Q, R)``
    toward the opposite apex ``X`` into ``(m, Q, X)`` and ``(m, X, R)`` until
    ``m`` splits the physical cells on both sides of the pole segment. Flip
    legality uses the rational chart coordinates; any impossible flip
    refuses. Returns the number of flips.
    """
    exact, _, _ = _patch_restrictions(domain, state)
    flips = 0
    for vertex in vertices.tolist():
        first, second = (int(value) for value in state.restriction_edges[vertex])
        pole, radial = (
            (int(state.pole_ids[first]), second)
            if state.pole_ids[first] >= 0
            else (int(state.pole_ids[second]), first)
        )
        # Each flip advances one zero-width child around ``R``; the patch
        # cell count bounds that walk.
        for _ in range(state.triangles.shape[0] + 1):
            triangles = state.triangles
            children = np.flatnonzero(
                np.any(triangles == vertex, axis=1)
                & np.any(triangles == radial, axis=1)
                & np.any(state.pole_ids[triangles] == pole, axis=1)
            )
            if not children.size:
                break
            cell = int(children[0])
            index = int(np.flatnonzero(triangles[cell] == vertex)[0])
            # The child (m, a, b) and its neighbor (d, b, a) span the convex
            # quadrilateral (m, a, d, b) exactly when both new cells are positive.
            a = int(triangles[cell, (index + 1) % 3])
            b = int(triangles[cell, (index + 2) % 3])
            neighbors = np.flatnonzero(
                np.any(triangles == a, axis=1) & np.any(triangles == b, axis=1)
            )
            neighbors = neighbors[neighbors != cell]
            if neighbors.size != 1 or state.constrained[cell, index]:
                raise _failure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    f"SurfaceConstruction.pole_fan must expose a free radial edge for the split; got no free radial edge on patch {state.index}.",
                    entity_ids=(state.index,),
                )
            other = int(neighbors[0])
            apex = int(
                np.flatnonzero((triangles[other] != a) & (triangles[other] != b))[0]
            )
            d = int(triangles[other, apex])
            ax, ay = exact[a]
            bx, by = exact[b]
            dx, dy = exact[d]
            mx, my = exact[vertex]
            if (
                triangles[other, (apex + 1) % 3] != b
                or state.constrained[other, apex]
                or (ax - mx) * (dy - my) - (ay - my) * (dx - mx) <= 0
                or (dx - mx) * (by - my) - (dy - my) * (bx - mx) <= 0
            ):
                raise _failure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    f"SurfaceConstruction.pole_fan must permit a convex rational radial flip; got illegal pole split on patch {state.index}.",
                    entity_ids=(state.index,),
                )
            # Edge flags are indexed by their opposite vertex.
            flags_bm = state.constrained[cell, (index + 1) % 3]
            flags_ma = state.constrained[cell, (index + 2) % 3]
            flags_ad = state.constrained[other, (apex + 1) % 3]
            flags_db = state.constrained[other, (apex + 2) % 3]
            state.triangles[cell] = (vertex, a, d)
            state.constrained[cell] = (flags_ad, False, flags_ma)
            state.triangles[other] = (vertex, d, b)
            state.constrained[other] = (flags_db, flags_bm, False)
            flips += 1
        else:
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"SurfaceConstruction.pole_restriction must belong to the physical fan; got split outside patch {state.index}.",
                entity_ids=(state.index,),
            )
    _validate_charts(domain, state, scope=scope, budget=budget)
    return flips


def _insert_missing_pole_restrictions(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    metrics: _Metrics,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    totals: RefinementTotals,
    record_phase: NativeMeshingPhaseRecorder | None,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    """Commit all missing pole carriers in one terminal native reconnect."""
    hints = _missing_pole_split_hints(state)
    if not hints.size:
        return False
    restrictions = _rational_edge_candidates(
        domain,
        state,
        np.mean(state.charts[state.triangles[hints]], axis=1),
        hints,
        np.full((hints.size, 2), -1, dtype=np.int32),
    )
    parents = state.triangles[hints]
    previous_count = state.charts.shape[0]
    accepted = _reconnect_batch(
        domain,
        state,
        metrics,
        compiled,
        limits,
        ledger,
        totals,
        record_phase,
        RestrictionBatch(
            restrictions.charts,
            np.zeros((hints.size,), dtype=np.float64),
            hints,
            parents,
            np.ones((hints.size,), dtype=np.bool_),
            np.zeros((state.triangles.shape[0],), dtype=np.bool_),
            restrictions.edges,
            restrictions.required,
            restrictions.edges,
            restrictions.parameters,
        ),
        scope=scope,
        budget=budget,
    )
    if not np.all(accepted):
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"SurfaceConstruction.pole_restriction must accept a rational radial split; got refused split on patch {state.index}.",
            entity_ids=(state.index,),
        )
    totals.flips += _open_pole_restriction_fans(
        domain,
        state,
        previous_count + np.flatnonzero(state.restriction_required[previous_count:]),
        scope=scope,
        budget=budget,
    )
    totals.rounds += 1
    return True


@final
class _RefinementPolicy(StrictModule):
    """Keep one patch's physical bounds and scheduling policy together."""

    size: float = eqx.field(static=True)
    deviation: float = eqx.field(static=True)
    aim: float = eqx.field(static=True)
    maximum_normal_angle: float = eqx.field(static=True)
    schedule: _dependency_phydrax_meshing_providers__native_schedules.NativeSurfaceSchedule = eqx.field(
        static=True
    )
    required_minimum_angle: float = eqx.field(static=True)
    size_compliance: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy | None = (
        eqx.field(static=True)
    )


@final
class _RefinementPreparation(StrictModule):
    sphere_frame: tuple[np.ndarray, float, float] | None
    planar_centroids: bool = eqx.field(static=True)
    exact_pole_restrictions: bool = eqx.field(static=True)


@final
class _RefinementCandidates(StrictModule):
    physical: np.ndarray
    charts: np.ndarray
    interior: np.ndarray
    spacing: np.ndarray
    hints: np.ndarray
    unresolved: np.ndarray
    insert_edges: np.ndarray


@final
class _RefinementInsertion(StrictModule):
    batch: RestrictionBatch | None
    interior: np.ndarray


@final
class _RefinementCommit(StrictModule):
    accepted: np.ndarray


@final
class _RefinementCompletion(StrictModule):
    metrics: _Metrics


def _prepare_refinement(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    policy: _RefinementPolicy,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _RefinementPreparation:
    """Prepare sphere bounds, planar scheduling, and cross-face pole obligations."""
    with _source_bound_scope(
        ledger,
        0,
        stage=MeshingStageKind.SURFACE_MESHING,
        scope=scope,
        budget=budget,
    ):
        sphere_frame = domain_numerics.full_sphere_frame(
            domain, state.index, budget=budget
        )
    surface = domain.patches[state.index].surface
    match surface:
        case _dependency_phydrax_geometry_brep__patches.PlanePatch():
            planar_centroids = True
        case _dependency_phydrax_geometry_brep__placed.PlacedSurface(
            definition=_dependency_phydrax_geometry_brep__patches.PlanePatch()
        ):
            planar_centroids = True
        case _:
            planar_centroids = False
    exact_pole_restrictions = np.isfinite(
        policy.maximum_normal_angle
    ) and _requires_exact_pole_restrictions(domain, state.index)
    return _RefinementPreparation(
        sphere_frame, planar_centroids, bool(exact_pole_restrictions)
    )


def _select_refinement_candidates(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    metrics: _Metrics,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    policy: _RefinementPolicy,
    prepared: _RefinementPreparation,
    /,
) -> _RefinementCandidates:
    """Select hard physical violations separately from optional shape work."""
    size, deviation, aim = policy.size, policy.deviation, policy.aim
    maximum_normal_angle, required_minimum_angle = (
        policy.maximum_normal_angle,
        policy.required_minimum_angle,
    )
    schedule, size_compliance, planar_centroids = (
        policy.schedule,
        policy.size_compliance,
        prepared.planar_centroids,
    )
    required_size = _required_size_refinement(
        domain, state, metrics, compiled, size_compliance
    )
    physical = _physical_refinement_mask(
        metrics,
        deviation,
        maximum_normal_angle,
        required_minimum_angle,
        required_size=required_size,
    )
    charts, interior, spacing, hints, unresolved, insert_edges = _candidates(
        state,
        metrics,
        size,
        deviation,
        aim,
        schedule.spacing_fraction,
        maximum_normal_angle,
        required_minimum_angle=required_minimum_angle,
        planar_centroids=planar_centroids,
        required_mask=physical,
    )
    return _RefinementCandidates(
        physical, charts, interior, spacing, hints, unresolved, insert_edges
    )


def _prepare_refinement_insertion(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    metrics: _Metrics,
    selected: _RefinementCandidates,
    policy: _RefinementPolicy,
    totals: RefinementTotals,
    /,
) -> _RefinementInsertion:
    """Defer rational pole requests and bind ordinary insertions to parent triangles."""
    charts, interior, spacing, hints = (
        selected.charts,
        selected.interior,
        selected.spacing,
        selected.hints,
    )
    insert_edges, unresolved = selected.insert_edges, selected.unresolved
    deviation, maximum_normal_angle, required_minimum_angle = (
        policy.deviation,
        policy.maximum_normal_angle,
        policy.required_minimum_angle,
    )
    if hints.size == 0:
        return _RefinementInsertion(None, interior)
    restrictions = _rational_edge_candidates(domain, state, charts, hints, insert_edges)
    charts = restrictions.charts
    restriction_required = restrictions.required
    restriction_edges = restrictions.edges
    restriction_parameters = restrictions.parameters
    del restrictions
    if np.any(restriction_required):
        ordinary = ~restriction_required
        if not np.any(ordinary):
            return _RefinementInsertion(None, interior)
        charts = charts[ordinary]
        interior = interior[ordinary]
        spacing = spacing[ordinary]
        hints = hints[ordinary]
        insert_edges = insert_edges[ordinary]
        restriction_required = restriction_required[ordinary]
        restriction_edges = restriction_edges[ordinary]
        restriction_parameters = restriction_parameters[ordinary]
    totals.rounds += 1
    hard_geometry_or_angle = (
        (metrics.deviation_bound[hints] > deviation)
        | (metrics.normal_bound[hints] > maximum_normal_angle)
        | (
            (~metrics.collapsed[hints])
            & (np.min(metrics.angles[hints], axis=1) < required_minimum_angle)
        )
    )
    parents = state.triangles[hints]
    return _RefinementInsertion(
        RestrictionBatch(
            charts,
            spacing,
            hints,
            parents,
            hard_geometry_or_angle,
            unresolved,
            insert_edges,
            restriction_required,
            restriction_edges,
            restriction_parameters,
        ),
        interior,
    )


def _execute_refinement_round(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    metrics: _Metrics,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    totals: RefinementTotals,
    record_phase: NativeMeshingPhaseRecorder | None,
    batch: RestrictionBatch,
    interior: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _RefinementCommit:
    """Reconnect a batch, then schedule distinct interior requests for refused hard edges."""
    charts, hints, parents = batch.charts, batch.hints, batch.parents
    hard_geometry_or_angle, unresolved = batch.hard, batch.unresolved
    accepted = _reconnect_batch(
        domain,
        state,
        metrics,
        compiled,
        limits,
        ledger,
        totals,
        record_phase,
        batch,
        scope=scope,
        budget=budget,
    )
    # A refused hard-geometry edge request may schedule a distinct generic
    # interior-point request. The native refusal itself never changes intent.
    retry = np.flatnonzero(
        ~accepted
        & hard_geometry_or_angle
        & ~metrics.collapsed[hints]
        & np.any(interior != charts, axis=1)
    )
    if retry.size:
        accepted[retry] = _reconnect_batch(
            domain,
            state,
            metrics,
            compiled,
            limits,
            ledger,
            totals,
            record_phase,
            RestrictionBatch(
                interior[retry],
                np.zeros((retry.size,), dtype=np.float64),
                hints[retry],
                parents[retry],
                hard_geometry_or_angle[retry],
                unresolved,
                np.full((retry.size, 2), -1, dtype=np.int32),
                np.zeros((retry.size,), dtype=np.bool_),
                np.full((retry.size, 2), -1, dtype=np.int64),
                np.zeros((retry.size, 2), dtype=np.int64),
            ),
            scope=scope,
            budget=budget,
        )
    return _RefinementCommit(accepted)


def _certify_refinement_completion(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    metrics: _Metrics,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    totals: RefinementTotals,
    policy: _RefinementPolicy,
    prepared: _RefinementPreparation,
    /,
) -> _RefinementCompletion:
    """Refuse unresolved hard criteria and retain soft scheduling evidence."""
    size, deviation, aim = policy.size, policy.deviation, policy.aim
    maximum_normal_angle, required_minimum_angle = (
        policy.maximum_normal_angle,
        policy.required_minimum_angle,
    )
    schedule, size_compliance, planar_centroids = (
        policy.schedule,
        policy.size_compliance,
        prepared.planar_centroids,
    )
    _, _, _, hints, unresolved, _ = _candidates(
        state,
        metrics,
        size,
        deviation,
        aim,
        schedule.spacing_fraction,
        maximum_normal_angle,
        required_minimum_angle=required_minimum_angle,
        planar_centroids=planar_centroids,
    )
    physical = _physical_refinement_mask(
        metrics,
        deviation,
        maximum_normal_angle,
        required_minimum_angle,
        required_size=_required_size_refinement(
            domain, state, metrics, compiled, size_compliance
        ),
    )
    if np.any(physical):
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            f"SurfaceConstruction.physical_criteria must be resolved after refinement; got patch {state.index} with invalid={int(np.sum(metrics.invalid))}, misoriented={int(np.sum(metrics.misoriented))}, deviating={int(np.sum(metrics.deviation_bound > deviation))}, turning={int(np.sum(metrics.normal_bound > maximum_normal_angle))}, max_deviation={float(np.max(metrics.deviation_bound)):.17g}, max_normal={float(np.max(metrics.normal_bound)):.17g}.",
            entity_ids=(state.index,),
            provider_code="surface_refinement_incomplete",
            requested=(("maximum_rounds", schedule.maximum_rounds),),
            achieved=(
                ("rounds", totals.rounds),
                ("unresolved_triangles", int(np.sum(unresolved))),
                ("geometry_queries", ledger.queries),
                ("work_units", ledger.work),
                ("invalid_triangles", int(np.sum(metrics.invalid))),
                ("misoriented_triangles", int(np.sum(metrics.misoriented))),
                (
                    "oversized_triangles",
                    int(
                        np.sum(
                            (~metrics.collapsed)
                            & (
                                np.max(metrics.lengths, axis=1)
                                > metrics.size_limits * (1 + _SIZE_SLACK)
                            )
                        )
                    ),
                ),
                ("deviating_triangles", int(np.sum(metrics.deviation_bound > deviation))),
                (
                    "turning_triangles",
                    int(np.sum(metrics.normal_bound > maximum_normal_angle)),
                ),
                ("maximum_deviation_bound", float(np.max(metrics.deviation_bound))),
                ("maximum_normal_bound", float(np.max(metrics.normal_bound))),
                (
                    "minimum_physical_angle",
                    float(np.min(metrics.angles[~metrics.collapsed], initial=np.inf)),
                ),
                ("required_minimum_angle", required_minimum_angle),
            ),
        )
    if hints.size:
        totals.incomplete += 1
    return _RefinementCompletion(metrics)


def _refine(
    domain: _dependency_phydrax_geometry__meshing_domain.MeshingDomain,
    state: _Patch,
    policy: _RefinementPolicy,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    started: float,
    totals: RefinementTotals,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _Metrics:
    """Refine one patch by measured, committed, and certified rounds."""
    prepared = _prepare_refinement(
        domain, state, ledger, policy, scope=scope, budget=budget
    )
    cache = _MetricsCache()
    for _ in range(policy.schedule.maximum_rounds):
        _dependency_phydrax_meshing_providers__native_publication.check_deadline(
            started, limits, MeshingStageKind.SURFACE_MESHING
        )
        with measure_phase(record_phase, "feature_certification"):
            metrics = _Metrics(
                domain,
                state,
                ledger,
                compiled,
                cache,
                sphere_frame=prepared.sphere_frame,
                scope=scope,
                budget=budget,
            )
        selected = _select_refinement_candidates(
            domain, state, metrics, compiled, policy, prepared
        )
        # Soft shape work must not destroy a hard lower-size statistic.
        if not np.any(selected.physical):
            if prepared.exact_pole_restrictions and _insert_missing_pole_restrictions(
                domain,
                state,
                metrics,
                compiled,
                limits,
                ledger,
                totals,
                record_phase,
                scope=scope,
                budget=budget,
            ):
                break
            if np.any(selected.unresolved):
                totals.incomplete += 1
            return metrics
        insertion = _prepare_refinement_insertion(
            domain, state, metrics, selected, policy, totals
        )
        del selected
        if insertion.batch is None:
            break
        committed = _execute_refinement_round(
            domain,
            state,
            metrics,
            compiled,
            limits,
            ledger,
            totals,
            record_phase,
            insertion.batch,
            insertion.interior,
            scope=scope,
            budget=budget,
        )
        if not np.any(committed.accepted):
            break
    if prepared.exact_pole_restrictions:
        _dependency_phydrax_meshing_providers__native_publication.check_deadline(
            started, limits, MeshingStageKind.SURFACE_MESHING
        )
        _insert_missing_pole_restrictions(
            domain,
            state,
            metrics,
            compiled,
            limits,
            ledger,
            totals,
            record_phase,
            scope=scope,
            budget=budget,
        )
    with measure_phase(record_phase, "feature_certification"):
        metrics = _Metrics(
            domain,
            state,
            ledger,
            compiled,
            cache,
            sphere_frame=prepared.sphere_frame,
            scope=scope,
            budget=budget,
        )
    return _certify_refinement_completion(
        domain,
        state,
        metrics,
        compiled,
        ledger,
        totals,
        policy,
        prepared,
    ).metrics


def _remaining_surface_limits(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    blocks: list[tuple[_Patch, _Metrics]],
    patch: int,
    vertices: np.ndarray,
    curve_ids: dict[int, np.ndarray],
    corner_ids: dict[int, int],
    /,
) -> _dependency_phydrax_meshing__contracts.MeshingLimits:
    shared = {
        identifier
        for curve in compiled.domain.patch_curves(patch).tolist()
        for identifier in curve_ids[curve].tolist()
    }
    shared.update(
        corner_ids[use.corner]
        for loop in compiled.domain.patches[patch].loops
        for use in loop
        if isinstance(use, _dependency_phydrax_geometry__meshing_domain.PatchPoleUse)
    )
    used_vertices = (
        vertices.shape[0]
        - len(shared)
        + sum(int(np.count_nonzero(state.identities < 0)) for state, _ in blocks)
    )
    used_faces = sum(int(np.count_nonzero(~metrics.collapsed)) for _, metrics in blocks)
    scratch = vertices.nbytes + sum(
        state.points.nbytes
        + state.charts.nbytes
        + state.normals.nbytes
        + state.identities.nbytes
        + state.triangles.nbytes
        + state.constrained.nbytes
        + state.boundary_edges.nbytes
        + state.boundary_uses.nbytes
        + state.trim_deviation_bounds.nbytes
        + state.restriction_required.nbytes
        + state.restriction_edges.nbytes
        + state.restriction_parameters.nbytes
        + metrics.lengths.nbytes
        + metrics.angles.nbytes
        + metrics.deviation_bound.nbytes
        + metrics.normal_bound.nbytes
        + metrics.size_limits.nbytes
        + metrics.metric_lengths.nbytes
        + metrics.metric_squared.nbytes
        + (0 if metrics.vertex_metrics is None else metrics.vertex_metrics.nbytes)
        for state, metrics in blocks
    )
    remaining_vertices = limits.maximum_vertices - used_vertices
    remaining_faces = limits.maximum_faces - used_faces
    remaining_cells = limits.maximum_cells - used_faces
    remaining_scratch = limits.maximum_scratch_bytes - scratch
    if min(remaining_vertices, remaining_faces, remaining_cells, remaining_scratch) <= 0:
        raise _failure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "SurfaceConstruction.patch_capacity must accommodate the next patch; got exhausted retained-source capacity.",
            entity_ids=(patch,),
            achieved=(
                ("retained_vertices", used_vertices),
                ("retained_faces", used_faces),
                ("retained_scratch_bytes", scratch),
            ),
        )
    return _dependency_phydrax_meshing__contracts.MeshingLimits(
        maximum_vertices=remaining_vertices,
        maximum_edges=limits.maximum_edges,
        maximum_faces=remaining_faces,
        maximum_cells=remaining_cells,
        maximum_connectivity_entries=limits.maximum_connectivity_entries,
        maximum_data_bytes=limits.maximum_data_bytes,
        maximum_work_units=limits.maximum_work_units,
        maximum_cavity_cells=limits.maximum_cavity_cells,
        maximum_geometry_queries=limits.maximum_geometry_queries,
        maximum_scratch_bytes=remaining_scratch,
        maximum_wall_seconds=limits.maximum_wall_seconds,
    )


@checked
def generate_surface(
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    schedule: _dependency_phydrax_meshing_providers__native_schedules.NativeSurfaceSchedule,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    minimum_angle: float,
    /,
    *,
    maximum_normal_angle: float = math.inf,
    required_minimum_angle: float = 0.0,
    size_compliance: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy
    | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    boundary_compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain
    | None = None,
    prepared_boundary: tuple[
        np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]
    ]
    | None = None,
    boundary_queries: int = 0,
    boundary_work: int = 0,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> surface_construction_owner.SurfaceConstruction:
    """Discretize shared curves, triangulate and refine every selected patch.

    ``minimum_angle`` (radians) is the scheduling aim. A separately declared
    ``required_minimum_angle`` is a physical hard condition; a remaining soft
    shape aim is reported rather than driving uncontrolled node growth.
    Invalid triangles or triangles opposed to the source normal refuse construction.
    ``prepared_boundary`` is admitted only after the distributed owner has
    validated its certified source evaluations, corner/curve incidence and source
    size/fidelity bounds. Its measured prior query/work counts are charged here.
    """

    if not budget.ticket.settled:
        raise RuntimeError(
            "generate_surface.budget must follow source ticket settlement; got live ticket."
        )
    started_ = scope.operation_started
    if maximum_normal_angle <= 0 or math.isnan(maximum_normal_angle):
        raise ValueError(
            f"generate_surface.maximum_normal_angle must be positive or infinity; got {maximum_normal_angle}."
        )
    domain = compiled.domain
    boundary = compiled if boundary_compiled is None else boundary_compiled
    if (
        boundary.domain.domain_id != domain.domain_id
        or boundary.specification_id != compiled.specification_id
        or not np.array_equal(boundary.curves, compiled.curves)
        or not np.array_equal(boundary.corners, compiled.corners)
        or not np.all(np.isin(compiled.patches, boundary.patches))
    ):
        raise ValueError(
            f"generate_surface.boundary_compiled must preserve source, controls, and incident entity identities; got domain={boundary.domain.domain_id}, specification={boundary.specification_id}."
        )
    ledger = _dependency_phydrax_meshing_providers__native_curve.QueryLedger(
        limits, budget=budget
    )
    ledger.observe_native(
        compiled.source_sizing.geometry_queries, 0, MeshingStageKind.SOURCE_INSPECTION
    )
    boundary_start = phase_started(record_phase)
    if prepared_boundary is None:
        if boundary_queries or boundary_work:
            raise ValueError(
                f"generate_surface.boundary_accounting must accompany a prepared boundary; got queries={boundary_queries}, work={boundary_work}, prepared_boundary=None."
            )
        vertices, curve_parameters, curve_ids, corner_ids = curve_vertices(
            boundary,
            schedule,
            ledger,
            started_,
            maximum_normal_angle,
            scope=scope,
            budget=budget,
        )
    else:
        if (
            isinstance(boundary_queries, bool)
            or not isinstance(boundary_queries, int)
            or isinstance(boundary_work, bool)
            or (not isinstance(boundary_work, int))
            or (min(boundary_queries, boundary_work) < 0)
        ):
            raise ValueError(
                f"generate_surface.boundary_accounting must contain nonnegative integer counts; got queries={boundary_queries!r}, work={boundary_work!r}."
            )
        vertices, curve_parameters, curve_ids, corner_ids = prepared_boundary
        if (
            set(curve_parameters) != set(compiled.curves.tolist())
            or set(curve_ids) != set(curve_parameters)
            or set(corner_ids) != set(compiled.corners.tolist())
            or (vertices.dtype != np.float64)
            or (vertices.ndim != 2)
            or (vertices.shape[1] != 3)
            or (not np.all(np.isfinite(vertices)))
        ):
            raise ValueError(
                f"generate_surface.prepared_boundary must match the authored source closure; got curve_rows={tuple(sorted(curve_parameters))}, corner_rows={tuple(sorted(corner_ids))}, vertices={vertices.shape}."
            )
        ledger.observe_native(
            boundary_queries, boundary_work, MeshingStageKind.CURVE_MESHING
        )
    record_elapsed(
        record_phase, "boundary_recovery", boundary_start, work_units=ledger.work
    )
    totals = RefinementTotals()
    blocks = []
    for row, patch in enumerate(compiled.patches.tolist()):
        local_limits = _remaining_surface_limits(
            limits, compiled, blocks, patch, vertices, curve_ids, corner_ids
        )
        patch_normal_angle = min(
            maximum_normal_angle, float(compiled.patch_normal_angles[row])
        )
        state = _initial_patch(
            domain,
            patch,
            float(compiled.patch_sizes[row]),
            vertices,
            curve_parameters,
            curve_ids,
            corner_ids,
            float(compiled.patch_deviations[row]),
            patch_normal_angle,
            ledger,
            local_limits,
            compiled=compiled,
            record_phase=record_phase,
            scope=scope,
            budget=budget,
        )
        metrics = _refine(
            domain,
            state,
            _RefinementPolicy(
                float(compiled.patch_sizes[row]),
                float(compiled.patch_deviations[row]),
                minimum_angle,
                patch_normal_angle,
                schedule,
                required_minimum_angle,
                size_compliance,
            ),
            local_limits,
            ledger,
            started_,
            totals,
            compiled,
            record_phase=record_phase,
            scope=scope,
            budget=budget,
        )
        broken = metrics.invalid | metrics.misoriented
        if np.any(broken):
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"SurfaceConstruction.triangles must stay within the seam and source-normal orientation; got invalid patch {patch} after refinement.",
                entity_ids=(patch,),
                achieved=(("invalid_triangles", int(np.sum(broken))),),
            )
        if np.isfinite(maximum_normal_angle) and np.any(
            metrics.normal_bound > maximum_normal_angle
        ):
            raise _failure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
                f"SurfaceConstruction.normal_turn must meet its continuous bound; got unresolved normal-turn bound on patch {patch}.",
                entity_ids=(patch,),
                requested=(("maximum_normal_angle", maximum_normal_angle),),
                achieved=(("maximum_normal_bound", float(np.max(metrics.normal_bound))),),
            )
        blocks.append((state, metrics))
    with measure_phase(record_phase, "native_publication"):
        return _assemble(
            compiled,
            vertices,
            curve_parameters,
            curve_ids,
            blocks,
            totals,
            ledger,
            scope=scope,
            budget=budget,
        )


def _assemble(
    compiled: _dependency_phydrax_meshing__domain.CompiledSurfaceDomain,
    vertices: np.ndarray,
    curve_parameters: dict[int, np.ndarray],
    curve_ids: dict[int, np.ndarray],
    blocks: list[tuple[_Patch, _Metrics]],
    totals: RefinementTotals,
    ledger: _dependency_phydrax_meshing_providers__native_curve.QueryLedger,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> surface_construction_owner.SurfaceConstruction:
    with _source_bound_scope(
        ledger, 0, stage=MeshingStageKind.SURFACE_MESHING, scope=scope, budget=budget
    ):
        domain = compiled.domain
        coordinates = [vertices]
        count = vertices.shape[0]
        dimensions = np.full((count,), 1, dtype=np.int8)
        source_indices = np.full((count,), -1, dtype=np.int64)
        parameters = np.full((count, 2), np.nan)
        dimensions[: compiled.corners.size] = 0
        source_indices[: compiled.corners.size] = compiled.corners
        for curve in compiled.curves.tolist():
            ids = curve_ids[curve][1:-1]
            source_indices[ids] = curve
            patch, loop, position = domain.curve_owners[curve]
            use = domain.patches[patch].loops[loop][position]
            parameters[ids, 0] = use.first + curve_parameters[curve][1:-1] * (
                use.last - use.first
            )
        vertex_dimensions = [dimensions]
        vertex_indices = [source_indices]
        vertex_parameters = [parameters]
        (
            triangles,
            patches,
            charts,
            chart_corners,
            deviations,
            deviation_bounds,
            unresolved,
            angles,
        ) = [], [], [], [], [], [], [], []
        normal_bounds = []
        boundary_occurrences: list[np.ndarray] = []
        restriction_vertices: list[np.ndarray] = []
        restriction_edges: list[np.ndarray] = []
        restriction_endpoint_parameters: list[np.ndarray] = []
        restriction_parameters: list[np.ndarray] = []
        for state, metrics in blocks:
            interior = state.identities < 0
            identities = state.identities.copy()
            identities[interior] = count + np.arange(int(np.sum(interior)))
            count += int(np.sum(interior))
            coordinates.append(state.points[interior])
            vertex_dimensions.append(np.full((int(np.sum(interior)),), 2, dtype=np.int8))
            vertex_indices.append(
                np.full((int(np.sum(interior)),), state.index, dtype=np.int64)
            )
            vertex_parameters.append(state.charts[interior])
            restricted = np.flatnonzero(state.restriction_required)
            if restricted.size:
                local_edges = state.restriction_edges[restricted]
                restriction_vertices.append(identities[restricted])
                restriction_edges.append(identities[local_edges])
                restriction_endpoint_parameters.append(state.charts[local_edges])
                restriction_parameters.append(state.restriction_parameters[restricted])
            keep = ~metrics.collapsed
            cells = identities[state.triangles[keep]]
            if domain.patches[state.index].reversed:
                cells = cells[:, [0, 2, 1]]
            local_charts = state.charts[state.triangles[keep]]
            if domain.patches[state.index].reversed:
                local_charts = local_charts[:, [0, 2, 1]]
            chart_vertices = np.arange(state.charts.shape[0], dtype=np.int64)
            for loop_ in domain.patches[state.index].loops:
                for use_ in loop_:
                    if isinstance(use_, _dependency_phydrax_geometry__meshing_domain.PatchPoleUse):
                        pole_vertices = np.flatnonzero(state.identities == use_.corner)
                        if pole_vertices.size:
                            chart_vertices[pole_vertices] = pole_vertices[0]
            local_cells = chart_vertices[state.triangles[keep]]
            if domain.patches[state.index].reversed:
                local_cells = local_cells[:, [0, 2, 1]]
            bank = np.full((cells.shape[0], 3, 3), -1, dtype=np.int32)
            owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for cell, vertices_ in enumerate(local_cells):
                for side, (first, last) in enumerate(((0, 1), (1, 2), (2, 0))):
                    a, b = int(vertices_[first]), int(vertices_[last])
                    key = (min(a, b), max(a, b))
                    owners.setdefault(key, []).append((cell, side))
            seen: set[tuple[int, int]] = set()
            for edge, provenance in zip(
                state.boundary_edges, state.boundary_uses, strict=True
            ):
                loop, ordinal = int(provenance[0]), int(provenance[1])
                use = domain.patches[state.index].loops[loop][ordinal]
                a, b = (int(value) for value in chart_vertices[edge])
                if isinstance(use, _dependency_phydrax_geometry__meshing_domain.PatchPoleUse):
                    if a != b:
                        raise ValueError("Authored pole boundary must collapse to its declared corner")
                    continue
                key = (min(a, b), max(a, b))
                matches = owners.get(key, [])
                if len(matches) != 1 or key in seen:
                    raise ValueError("Authored chart boundary requires one distinct active cell side")
                seen.add(key)
                if use.last == use.first:
                    raise ValueError("Authored chart boundary use requires nonzero traversal")
                cell, side = matches[0]
                bank[cell, side] = (loop, ordinal, 1 if use.last > use.first else -1)
            if any(len(value) == 1 and key not in seen for key, value in owners.items()):
                raise ValueError("Active chart boundary side is missing authored occurrence identity")
            boundary_occurrences.append(bank)
            chart_corners.append(local_charts)
            triangles.append(cells)
            patches.append(np.full((cells.shape[0],), state.index, dtype=np.int64))
            charts.append(np.mean(state.charts[state.triangles[keep]], axis=1))
            deviations.append(metrics.deviation[keep])
            deviation_bounds.append(metrics.deviation_bound[keep])
            normal_bounds.append(metrics.normal_bound[keep])
            angles.append(np.min(metrics.angles[keep], axis=1))
            unresolved.append(
                (
                    np.max(metrics.lengths[keep], axis=1)
                    > metrics.size_limits[keep] * (1 + _SIZE_SLACK)
                )
                | (np.max(metrics.metric_lengths[keep], axis=1) > _METRIC_AIM)
            )
        edges, edge_curves, edge_deviations, edge_bounds = [], [], [], []
        curve_charts = _dependency_phydrax_meshing_providers__native_curve.CurveCharts(
            domain.curve_atlas, jnp.asarray(compiled.curves)
        )
        # Curve vertices are the shared prefix of the assembled vertex array.
        for curve_row, curve in enumerate(compiled.curves.tolist()):
            ids = curve_ids[curve]
            values = curve_parameters[curve]
            middle = 0.5 * (values[:-1] + values[1:])
            ledger.reserve(middle.size, 0, MeshingStageKind.CURVE_MESHING)
            on_curve = _dependency_phydrax_meshing_providers__native_curve.curve_points(
                curve_charts, curve_row, middle
            )
            chord = 0.5 * (vertices[ids[:-1]] + vertices[ids[1:]])
            edges.append(np.stack((ids[:-1], ids[1:]), axis=1))
            edge_curves.append(np.full((middle.size,), curve, dtype=np.int64))
            edge_deviations.append(np.linalg.norm(on_curve - chord, axis=1))
            with _source_bound_scope(
                ledger,
                2048 * count
                + 4096 * sum(state.triangles.shape[0] for state, _ in blocks),
                stage=MeshingStageKind.CURVE_MESHING,
                scope=scope,
                budget=budget,
            ):
                edge_bounds.append(
                    domain.curve_interpolation_bounds(curve, values, budget=budget)
                )
        if restriction_vertices:
            assembled_restriction_vertices = np.concatenate(restriction_vertices)
            assembled_restriction_edges = np.concatenate(restriction_edges, axis=0)
            assembled_restriction_endpoint_parameters = np.concatenate(
                restriction_endpoint_parameters, axis=0
            )
            assembled_restriction_parameters = np.concatenate(
                restriction_parameters, axis=0
            )
        else:
            (
                _,
                assembled_restriction_vertices,
                assembled_restriction_edges,
                assembled_restriction_endpoint_parameters,
                assembled_restriction_parameters,
            ) = empty_chart_restrictions(count)
        all_angles = np.concatenate(angles)
        return surface_construction_owner.SurfaceConstruction(
            np.concatenate(coordinates, axis=0),
            np.concatenate(vertex_dimensions),
            np.concatenate(vertex_indices),
            np.concatenate(vertex_parameters, axis=0),
            assembled_restriction_vertices,
            assembled_restriction_edges,
            assembled_restriction_endpoint_parameters,
            assembled_restriction_parameters,
            np.concatenate(triangles, axis=0),
            np.concatenate(patches),
            np.concatenate(charts, axis=0),
            np.concatenate(chart_corners, axis=0),
            np.concatenate(deviations),
            np.concatenate(deviation_bounds),
            np.concatenate(normal_bounds),
            np.concatenate(edges, axis=0),
            np.concatenate(edge_curves),
            np.concatenate(edge_deviations),
            np.concatenate(edge_bounds),
            np.concatenate(unresolved),
            np.concatenate(boundary_occurrences, axis=0),
            tuple(
                (
                    state.index,
                    state.charts,
                    state.points,
                    state.triangles,
                    state.boundary_edges,
                    state.boundary_uses,
                    state.restriction_required,
                    np.flatnonzero(state.restriction_required).astype(np.int64),
                    state.restriction_edges[state.restriction_required],
                    state.restriction_parameters[state.restriction_required],
                )
                for state, _ in blocks
            ),
            tuple((curve, curve_parameters[curve]) for curve in compiled.curves.tolist()),
            float(np.min(all_angles)),
            totals.rounds,
            totals.inserted,
            totals.refused,
            totals.flips,
            ledger.queries,
            ledger.work,
            bool(totals.incomplete),
        )


__all__ = ["generate_surface"]
