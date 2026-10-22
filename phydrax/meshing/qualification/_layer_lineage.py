#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Original periodic two-material layer/core source and its durable lineage.

The source is a planar periodic wall (flat, or the ridge profile of the accepted
31-cell case) with two 0.01-thick ADVANCING layers, a 0.03 core offset above the
immutable cap, a wall/core material interface, explicit seam polygon orbit
pairs, and one unit ``x`` translation. ``produce_layer_core_lineage_archive``
generates the ridge source natively, authors and solves the initial accepted
state, and for every round refines one physical layer column, continues the
accepted state warm, durably reopens that accepted history, applies the inverse
coarsening and continues the accepted state warm again. Every accepted state
is checkpointed in the authenticated closure consumed by
``continue_layer_core_lifecycle``.
Each staged reopen and inverse continuation owns one current native request;
archived preparation receipts are provenance, not a debit to that request.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import final

import equinox as eqx
import numpy as np

import phydrax.meshing as M
from phydrax import StrictModule
from phydrax._meshcore import NativeExecutionScope
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    PeriodicIsometryGroup,
    PeriodicMeshTopology,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.lifecycle import (
    MeshingAcceptedEpoch,
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    write_meshing_source_closure,
)
from phydrax.lifecycle._meshing_sources import meshing_source_owner_reconstruction_scope
from phydrax.meshing._layer_core import validate_layer_index_attributes
from phydrax.meshing.providers import NativeLayerCoreSource
from phydrax.typing import checked

from ._hybrid import prepare_hybrid_adaptation_policy
from ._layer_continuation import (
    continue_layer_core_state,
    layer_core_accepted_epoch,
    layer_core_continuation_execution,
    layer_core_epoch_identity,
    layer_core_lineage_records,
    layer_core_material_declaration,
    layer_core_restoration_defects,
    LayerCoreAcceptedState,
    LayerCoreEpochIdentity,
    LayerCoreLifecyclePlans,
    LAYER_CORE_REGION_DENSITIES,
    LayerCoreSolveEpoch,
    LayerCoreStepContinuation,
    restore_layer_core_state,
    warm_layer_core_initial_state,
    warm_layer_core_step,
)


PERIODIC_MATERIAL_LAYER_SOURCE_ID = "periodic-layered-gap"
PERIODIC_MATERIAL_LAYER_SOURCE_REVISION = "source-authored"
PERIODIC_MATERIAL_CORE_OFFSET = 0.03
PERIODIC_MATERIAL_REGION_IDS = ("wall-material", "core-material")


def _periodic_wall(*, ridge: bool) -> CellMesh:
    """Author the wall rows and their ``x``-period vertex orbits."""
    if ridge:
        points = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.5, 0.1),
                (1.0, 0.5, 0.1),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 0.0),
            )
        )
        rows = np.asarray(((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)), dtype=np.int64)
    else:
        points = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0))
        )
        rows = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
    block = CellBlock("wall", "triangle", rows)
    generator = np.eye(4, dtype=np.float64)[None]
    generator[0, 0, 3] = 1.0
    roots = np.repeat(np.arange(0, points.shape[0], 2, dtype=np.int64), 2)
    shifts = np.tile(np.asarray(((0,), (1,)), dtype=np.int64), (points.shape[0] // 2, 1))
    topology = PeriodicMeshTopology(
        CellMesh(points, (block,)), PeriodicIsometryGroup(generator), roots, shifts
    )
    return CellMesh(points, (block,), periodic_topology=topology)


def _periodic_material_source(*, ridge: bool) -> NativeLayerCoreSource:
    wall = _periodic_wall(ridge=ridge)
    entities = wall.entity_set(2)
    scope = M.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        entities.entity_set_id,
        entities.entity_ids,
    )
    control = M.BoundaryLayerControl(
        scope,
        M.LayerSchedule.geometric(2, 0.01, growth_rate=1.0),
        route=M.BoundaryLayerRoute.ADVANCING,
        feature_angle=0.1,
        corner=M.BoundaryLayerCornerPolicy.SMOOTH,
    )
    layers = M.prepare_boundary_layers(wall, control)
    cap = layers.cap
    if cap is None or cap.periodic_topology is None:
        raise RuntimeError(
            "BoundaryLayerMesh.cap must retain periodic topology for the wall; got a missing cap or topology."
        )
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks]
    )
    cap_points = np.asarray(cap.coordinates, dtype=np.float64)
    count = cap_points.shape[0]
    upper = cap_points + np.asarray((0.0, 0.0, PERIODIC_MATERIAL_CORE_OFFSET))
    cap_roots = np.asarray(cap.periodic_topology.vertex_representatives, dtype=np.int64)
    cap_shifts = np.asarray(cap.periodic_topology.vertex_shifts, dtype=np.int64)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row in cap_rows:
        for a, b in zip(row, np.roll(row, -1), strict=True):
            first, second = int(a), int(b)
            edges.setdefault((min(first, second), max(first, second)), []).append(
                (first, second)
            )
    polygons = cap_rows.tolist() + (cap_rows + count).tolist()
    seam_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for incidents in edges.values():
        if len(incidents) != 1:
            continue
        a, b = incidents[0]
        forward = (int(cap_roots[a]), a) <= (int(cap_roots[b]), b)
        first, second = (a, b) if forward else (b, a)
        triangles = (
            (first, second, second + count),
            (first, second + count, first + count),
        )
        start = len(polygons)
        polygons.extend(
            triangles if forward else tuple(tuple(reversed(row)) for row in triangles)
        )
        if cap_roots[a] != cap_roots[b] and cap_shifts[a, 0] == cap_shifts[b, 0]:
            first_root, second_root = int(cap_roots[a]), int(cap_roots[b])
            seam_edges.setdefault(
                (min(first_root, second_root), max(first_root, second_root)), []
            ).append((start, start + 1))
    pairs = np.asarray(
        [
            (copies[0][triangle], copies[1][triangle])
            for copies in seam_edges.values()
            for triangle in range(2)
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    polygon_rows = np.asarray(polygons, dtype=np.int64)
    cap_count = cap_rows.shape[0]
    incidence = np.tile(np.asarray((-1, 0), dtype=np.int64), (polygon_rows.shape[0], 1))
    incidence[:cap_count] = np.asarray((0, -1))
    core = M.PiecewiseLinearComplex(
        np.concatenate((cap_points, upper)),
        tuple(polygon_rows),
        np.arange(polygon_rows.shape[0], dtype=np.int64),
        incidence,
        (PERIODIC_MATERIAL_REGION_IDS[1],),
        boundary="fixed",
    )
    return NativeLayerCoreSource(
        layers,
        core,
        PERIODIC_MATERIAL_LAYER_SOURCE_ID,
        PERIODIC_MATERIAL_LAYER_SOURCE_REVISION,
        vertex_layer_ids=np.concatenate(
            (
                np.asarray(layers.cap_vertices, dtype=np.int64),
                np.full(count, -1, dtype=np.int64),
            )
        ),
        cap_polygon_ids=np.arange(cap_count, dtype=np.int64),
        layer_regions=np.zeros(
            sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64
        ),
        region_ids=PERIODIC_MATERIAL_REGION_IDS,
        core_region_map=np.asarray([1], dtype=np.int64),
        core_vertex_representatives=np.concatenate((cap_roots, cap_roots + count)),
        core_vertex_shifts=np.concatenate((cap_shifts, cap_shifts)),
        core_seam_polygon_pairs=pairs,
    )


def _periodic_material_specification(
    source: NativeLayerCoreSource, /, *, limits: M.MeshingLimits | None = None
) -> M.VolumeMeshingSpec:
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "authored-core-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    pairs = source.core_seam_polygon_pairs
    roots = source.core_vertex_representatives
    shifts = source.core_vertex_shifts
    if pairs is None or roots is None or shifts is None:
        raise RuntimeError(
            "NativeLayerCoreSource must retain its authored seam pairs, roots and shifts; got a missing seam declaration."
        )
    rows = source.complex.polygon_vertices.reshape(-1, 3)
    first, second = pairs[:, 0], pairs[:, 1]
    reverse = shifts[rows[first, 0], 0] > shifts[rows[second, 0], 0]
    first, second = np.where(reverse, second, first), np.where(reverse, first, second)
    order = np.argsort(second, kind="stable")
    first, second = first[order], second[order]
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 1.0
    constraint = M.PeriodicConstraint(
        M.MeshingScope(
            scope.source_id,
            scope.source_revision,
            scope.entity_kind,
            2,
            scope.entity_set_id,
            first,
        ),
        M.MeshingScope(
            scope.source_id,
            scope.source_revision,
            scope.entity_kind,
            2,
            scope.entity_set_id,
            second,
        ),
        transform,
        source_entity_ids=first,
        orientations=np.full(first.size, -1, dtype=np.int64),
    )
    regions = tuple(
        M.RegionControl(
            M.MeshingScope(
                scope.source_id,
                scope.source_revision,
                scope.entity_kind,
                3,
                "authored-regions",
                np.asarray([index], dtype=np.int64),
            ),
            name,
            f"material:{index}",
            M.RegionRole.FLUID,
        )
        for index, name in enumerate(source.region_ids)
    )
    cap_scope = M.MeshingScope(
        scope.source_id,
        scope.source_revision,
        scope.entity_kind,
        2,
        scope.entity_set_id,
        source.cap_polygon_ids,
    )
    return M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3,
            3,
            M.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allowed_transitions=("hexahedron", "pyramid"),
                allow_mixed=True,
            ),
        ),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
        region_controls=regions,
        patch_controls=(
            M.PatchControl("material-contact", cap_scope, source.region_ids),
        ),
        periodic_constraints=(constraint,),
        limits=limits,
    )


@checked
def prepare_periodic_material_layer_source(
    *, ridge: bool = False, limits: M.MeshingLimits | None = None
) -> tuple[NativeLayerCoreSource, M.VolumeMeshingSpec]:
    """Author the original periodic two-material layer/core source and request."""
    source = _periodic_material_source(ridge=ridge)
    return source, _periodic_material_specification(source, limits=limits)


@checked
def authored_layer_columns(result: M.CellMeshingResult, /) -> M.MixedLayerColumns:
    """Return the published physical layer columns with their fixed schedule."""
    validate_layer_index_attributes(result)
    attributes = {
        attribute.name: attribute
        for attribute in result.attributes
        if attribute.name in ("layer_index", "layer_column")
    }
    ids = np.asarray(attributes["layer_index"].scope.entity_ids, dtype=np.int64)
    intervals = np.asarray(attributes["layer_index"].values, dtype=np.int64)
    columns = np.asarray(attributes["layer_column"].values, dtype=np.int64)
    layered = intervals >= 0
    return M.MixedLayerColumns(
        ids[layered],
        columns[layered],
        intervals[layered],
        hard_first_thickness=True,
        axial_refinement=False,
        allow_schedule_change=False,
    )


LAYER_CORE_LINEAGE_PART_NAME = "periodic-layer-core"


@final
class LayerCoreLineageArchive(StrictModule):
    """A natively produced, authenticated layer/core accepted-state closure."""

    generated: M.CellMeshingResult
    specification: M.VolumeMeshingSpec
    columns: M.MixedLayerColumns
    archive: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    validation_id: str | None = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)
    adaptation_ids: tuple[str, ...] = eqx.field(static=True)
    cells: tuple[int, ...] = eqx.field(static=True)
    accepted_epochs: tuple[LayerCoreEpochIdentity, ...] = eqx.field(static=True)
    initial_solve: LayerCoreSolveEpoch = eqx.field(static=True)
    continuations: tuple[LayerCoreStepContinuation, ...] = eqx.field(static=True)
    restoration_defects: tuple[tuple[tuple[str, float], ...], ...] = eqx.field(
        static=True
    )
    stages_seconds: tuple[tuple[str, float], ...] = eqx.field(static=True)
    warm_execution: tuple[M.NativeExecutionRecord, ...]

    def to_record(self) -> dict[str, object]:
        """Return the producer receipt a separate consumer process authenticates."""
        return {
            "archive": self.archive,
            "content_id": self.content_id,
            "validation_id": self.validation_id,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "source_binding_id": self.source_binding_id,
            "specification_id": self.specification_id,
            "limits_id": self.limits_id,
            "generation_result_id": self.generated.result_id,
            "adaptation_ids": list(self.adaptation_ids),
            "cells": list(self.cells),
            "accepted_epochs": list(self.accepted_epochs),
            "accepted_epoch_ids": [epoch["epoch_id"] for epoch in self.accepted_epochs],
            "warm_initial_solve": self.initial_solve,
            "warm_continuations": list(self.continuations),
            "warm_restoration_defects": [
                dict(value) for value in self.restoration_defects
            ],
            "warm_execution_evidence": [
                record.to_record() for record in self.warm_execution
            ],
            "stages_seconds": dict(self.stages_seconds),
        }


def _cell_count(result: M.CellMeshingResult, /) -> int:
    return int(sum(block.cell_count for block in result.mesh.blocks))


@checked
def produce_layer_core_lineage_archive(
    destination: str | Path,
    /,
    *,
    rounds: int = 1,
    solver_repeats: int = 1,
    limits: M.MeshingLimits | None = None,
    region_densities: tuple[float, ...] = LAYER_CORE_REGION_DENSITIES,
    material_site: str = "two-material-density",
) -> LayerCoreLineageArchive:
    """Generate, adapt and checkpoint every accepted state of the original lineage.

    The initial accepted state (H1/DG/H(curl)/H(div) coefficients, declared
    material density, harmonic solution and damage history) is authored on
    the native generation. Each of ``rounds`` refines one physical layer
    column, continues the accepted state warm through the refinement, stages
    and durably reopens that accepted history, coarsens the reopened
    refinement and continues the accepted state warm through the inverse.
    ``destination`` receives each staged ``refined-<round>`` closure and the
    durable ``lifecycle`` closure whose content identity is re-authenticated
    from the written manifest before it is reported.

    Authenticated source controls bound each staged reopen, inverse adaptation
    and accepted-state continuation to one current request allowance.
    """
    if isinstance(rounds, bool) or rounds < 1:
        raise ValueError(f"lineage.rounds must be a positive integer; got {rounds!r}")
    stages: dict[str, float] = {}

    def timed[T](name: str, call: Callable[[], T], /) -> T:
        started = perf_counter()
        try:
            return call()
        finally:
            stages[name] = perf_counter() - started

    root = Path(destination)
    source, specification = prepare_periodic_material_layer_source(
        ridge=True, limits=limits
    )
    plan = timed(
        "native_generation_plan",
        lambda: M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core")).plan(
            source, specification, coordinate_contract=SpatialCoordinateContract.si()
        ),
    )
    generated = timed("native_generation", plan.execute)
    report = generated.certification
    previous = generated.execution_evidence
    if report is None or not report.passed or previous is None:
        raise ValueError(
            f"lineage.generated.certification must pass with an executed receipt for the original source; got {(report, previous)!r}"
        )
    columns = authored_layer_columns(generated)
    policy = prepare_hybrid_adaptation_policy(
        source, specification, generated, limits=specification.limits
    )
    material = layer_core_material_declaration(
        generated, densities=region_densities, site_id=material_site
    )
    plans = LayerCoreLifecyclePlans()
    limits = specification.limits
    state, epoch, initial_solve, record = timed(
        "warm_initial_state",
        lambda: warm_layer_core_initial_state(
            generated,
            material=material,
            plans=plans,
            limits=limits,
            part_name=LAYER_CORE_LINEAGE_PART_NAME,
            solver_repeats=solver_repeats,
        ),
    )
    epochs: tuple[MeshingAcceptedEpoch, ...] = (epoch,)
    continuations: list[LayerCoreStepContinuation] = []
    restorations: list[tuple[tuple[str, float], ...]] = []
    warm_execution = [record]
    for round_ in range(rounds):
        before, reference = state, epochs[-1]
        layered = authored_layer_columns(before.result)
        refinement = timed(
            f"native_refinement:{round_}",
            lambda: M.execute_mesh_adaptation(
                M.prepare_mesh_adaptation(
                    before.result,
                    M.MarkedMeshAdaptation(
                        np.asarray(
                            (int(np.min(np.asarray(layered.cell_ids))),), dtype=np.int64
                        ),
                        layer_columns=layered,
                    ),
                    policy=policy,
                )
            ),
        )
        if refinement.status is not M.MeshAdaptationStatus.COMPLETE:
            raise ValueError(
                f"lineage.refinement.status must be COMPLETE; got {refinement.status!r}"
            )
        state, epoch, step, record = timed(
            f"warm_refinement_continuation:{round_}",
            lambda: warm_layer_core_step(
                before,
                refinement,
                material=material,
                plans=plans,
                limits=limits,
                part_name=LAYER_CORE_LINEAGE_PART_NAME,
                solver_repeats=solver_repeats,
            ),
        )
        continuations.append(step)
        warm_execution.append(record)
        staged_epochs = (*epochs, epoch)
        staged = timed(
            f"staged_refinement_archive:{round_}",
            lambda: write_meshing_source_closure(
                root / f"refined-{round_}",
                layer_core_lineage_records(
                    source, specification, plan.options, staged_epochs
                ),
                retained_layer_core_plan=plan,
            ),
        )
        staged_controls = timed(
            f"staged_refinement_controls:{round_}",
            lambda: read_meshing_source_execution_controls(
                staged.path, expected_content_id=staged.content_id
            ),
        )
        if (
            staged_controls.source_id,
            staged_controls.source_revision,
            staged_controls.source_binding_id,
            staged_controls.specification_id,
            staged_controls.limits.limits_id,
        ) != (
            source.source_id,
            source.source_revision,
            source.binding_id,
            specification.specification_id,
            limits.limits_id,
        ):
            raise ValueError(
                f"lineage.staged_controls must authenticate the produced source and limits; got {staged_controls!r}"
            )

        def reopen_and_continue(execution: NativeExecutionScope) -> tuple[
            tuple[MeshingAcceptedEpoch, ...],
            LayerCoreAcceptedState,
            MeshingAcceptedEpoch,
            LayerCoreStepContinuation,
            tuple[tuple[str, float], ...],
        ]:
            with execution.ticket_view() as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=staged_controls.limits.maximum_work_units,
                    maximum_memory_bytes=staged_controls.limits.maximum_scratch_bytes,
                )
                with budget.activate(), meshing_source_owner_reconstruction_scope(budget=budget):
                    return continue_reopened(execution, budget)

        def continue_reopened(
            execution: NativeExecutionScope, budget: CoordinateEnclosureBudget
        ) -> tuple[
            tuple[MeshingAcceptedEpoch, ...],
            LayerCoreAcceptedState,
            MeshingAcceptedEpoch,
            LayerCoreStepContinuation,
            tuple[tuple[str, float], ...],
        ]:
            reopened_epochs = timed(
                f"staged_refinement_reopen:{round_}",
                lambda: read_meshing_source_closure(
                    staged.path,
                    expected_content_id=staged.content_id,
                    budget=budget,
                )["accepted_data"]["accepted_epochs"],
            )
            if tuple(value.epoch_id for value in reopened_epochs) != tuple(
                value.epoch_id for value in staged_epochs
            ):
                raise ValueError(
                    f"lineage.reopened_refinement must restore accepted epochs {tuple(value.epoch_id for value in staged_epochs)!r}; got {reopened_epochs!r}"
                )
            reopened = reopened_epochs[-1].transition
            if not isinstance(reopened, M.MeshAdaptationResult) or (
                reopened.result_id != refinement.result_id
            ):
                raise ValueError(
                    f"lineage.reopened_refinement must restore result {refinement.result_id!r}; got {reopened!r}"
                )
            hierarchy = reopened.hierarchy
            if not isinstance(hierarchy, M.MixedAdaptationHierarchy):
                raise TypeError(
                    f"lineage.reopened_refinement.hierarchy must be a MixedAdaptationHierarchy; got {type(hierarchy).__name__}"
                )
            active = set(
                np.concatenate(
                    [np.asarray(block.global_ids) for block in reopened.target.mesh.blocks]
                ).tolist()
            )
            children = sorted(set(hierarchy.siblings.all_child_ids().tolist()) & active)
            coarsening = timed(
                f"native_coarsening:{round_}",
                lambda: M.execute_mesh_adaptation(
                    M.prepare_mesh_adaptation(
                        reopened.target,
                        M.MarkedMeshAdaptation(
                            (), np.asarray(children, dtype=np.int64), hierarchy=hierarchy
                        ),
                        policy=policy,
                    )
                ),
            )
            if coarsening.status is not M.MeshAdaptationStatus.COMPLETE:
                raise ValueError(
                    f"lineage.coarsening.status must be COMPLETE; got {coarsening.status!r}"
                )
            reopened_plans = LayerCoreLifecyclePlans(budget=budget)
            refined = restore_layer_core_state(
                reopened_epochs[-1], material=material, plans=reopened_plans
            )
            continued, continued_step = timed(
                f"warm_coarsening_continuation:{round_}",
                lambda: continue_layer_core_state(
                    refined,
                    coarsening,
                    material=material,
                    plans=reopened_plans,
                    execution=execution,
                    solver_repeats=solver_repeats,
                ),
            )
            continued_epoch = layer_core_accepted_epoch(
                continued,
                part_name=LAYER_CORE_LINEAGE_PART_NAME,
                plans=reopened_plans,
                transition=coarsening,
            )
            defects = tuple(
                sorted(
                    layer_core_restoration_defects(
                        continued, reference, plans=reopened_plans
                    ).items()
                )
            )
            return reopened_epochs, continued, continued_epoch, continued_step, defects

        (reopened_epochs, state, epoch, step, defects), record = (
            layer_core_continuation_execution(
                staged_controls.limits,
                staged_controls.execution_evidence,
                reopen_and_continue,
                import_preparation=False,
            )
        )
        continuations.append(step)
        warm_execution.append(record)
        restorations.append(defects)
        epochs = (*reopened_epochs, epoch)
    durable = timed(
        "durable_lifecycle_archive",
        lambda: write_meshing_source_closure(
            root / "lifecycle",
            layer_core_lineage_records(source, specification, plan.options, epochs),
            retained_layer_core_plan=plan,
        ),
    )
    controls = timed(
        "durable_lifecycle_controls",
        lambda: read_meshing_source_execution_controls(
            durable.path, expected_content_id=durable.content_id
        ),
    )
    if (
        controls.source_id,
        controls.source_revision,
        controls.source_binding_id,
        controls.specification_id,
    ) != (
        source.source_id,
        source.source_revision,
        source.binding_id,
        specification.specification_id,
    ):
        raise ValueError(
            f"lineage.archive_controls must authenticate the produced source; got {controls!r}"
        )
    return LayerCoreLineageArchive(
        generated=generated,
        specification=specification,
        columns=columns,
        archive=str(durable.path),
        content_id=durable.content_id,
        validation_id=durable.validation_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_binding_id=source.binding_id,
        specification_id=specification.specification_id,
        limits_id=limits.limits_id,
        adaptation_ids=tuple(
            value.transition.result_id
            for value in epochs[1:]
            if value.transition is not None
        ),
        cells=tuple(_cell_count(value.carrier) for value in epochs),
        accepted_epochs=tuple(layer_core_epoch_identity(value) for value in epochs),
        initial_solve=initial_solve,
        continuations=tuple(continuations),
        restoration_defects=tuple(restorations),
        stages_seconds=tuple(sorted(stages.items())),
        warm_execution=tuple(warm_execution),
    )
