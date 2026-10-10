#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Actual UV carrier operations under one original periodic metric ledger."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..discretization import CellMesh
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import CoordinateEnclosureBudget
from ..discretization._surface_chart_deformation import SurfaceChartWitness
from ..geometry import MeshingDomain
from ..geometry._surface_source_support import SurfaceSourceRootAtlas
from ._adaptation import MeshAdaptationPolicy
from ._association import SurfaceAssociationTransfer
from ._metric import interpolate_mesh_metric
from ._periodic import (
    ConstructionPointKey,
    metric_target_lineage,
    PeriodicMetricCandidateStage,
    PeriodicMetricGeometryStage,
    PeriodicMetricOperation,
    PeriodicMetricOrbitOutcome,
    PeriodicMetricOrbitSelection,
    synchronize_periodic_metric_edit,
)
from ._result import CellMeshingResult
from ._surface_metric import (
    _assemble_surface,
    _chart_exact_weights,
    _chart_stencil,
    _collapse,
    _flip,
    _relocate,
    _split,
    _surface_metric_source_cover,
    _surface_vertex_strata,
    _SurfaceMetricGeometryWitness,
    _SurfaceSource,
    _SurfaceState,
    prepare_surface_metric_geometry_transition,
)
from ._topology_edit import CellTopologyEdit, key_rows


@dataclass(slots=True)
class _SurfaceMetricController:
    """Private UV continuation retaining the original scientific source."""

    source: CellMeshingResult
    transfer: SurfaceAssociationTransfer
    policy: MeshAdaptationPolicy
    budget: CoordinateEnclosureBudget
    witness: SurfaceChartWitness
    maximum_fidelity: float
    material_root_witness: SurfaceChartWitness | None = None
    background: _SurfaceSource | None = None
    prior_stage: PeriodicMetricOrbitOutcome | None = None

    @property
    def operation_count(self) -> int:
        if self.prior_stage is None:
            raise ValueError(
                "A UV operation count requires its actual completed orbit stage."
            )
        return self.prior_stage.evidence.orbit_copy_count

    def stage(
        self,
        state: _SurfaceState,
        operation: PeriodicMetricOperation,
        vertices: tuple[int, ...],
        fixed: np.ndarray,
        features: set[tuple[int, int]],
        feature_codes: dict[tuple[int, int], int],
        blocked: set[tuple[int, int]],
        /,
    ) -> _SurfaceState | None:
        background, certification = self.background, self.source.certification
        if background is None or certification is None:
            raise ValueError(
                "UV orbit staging requires its original background and certificates."
            )
        if self.material_root_witness is None:
            raise ValueError(
                "UV orbit staging requires its original admitted material root witness."
            )
        domain = self.transfer.support.domain
        with self.budget.activate(), self.budget.temporary_scope():
            self.budget.reserve(
                0,
                16
                * sum(
                    value.nbytes
                    for value in (
                        state.points,
                        state.metric,
                        state.cells,
                        state.charts,
                        state.vertex_ids,
                        state.cell_ids,
                    )
                ),
            )
            if state.vertex_strata is not None:
                self.budget.reserve(0, 2048 * state.curve_workspace_vertices)
            self.budget.reserve(0, 2 * state.material_workspace_bytes_upper)
            seed, aggregate = deepcopy(state), deepcopy(state)
            trial_features, trial_codes = features.copy(), feature_codes.copy()
            if not _apply_operation(
                seed,
                operation,
                vertices,
                background,
                domain,
                fixed,
                features.copy(),
                feature_codes.copy(),
                blocked,
                self.policy,
                self.maximum_fidelity,
            ):
                return None
            candidate, exact = _exact_edit(
                self.source.mesh, seed, background, self.policy, self.budget, operation
            )

            def build_candidate(
                edit: CellTopologyEdit,
                stencils: Mapping[int, ConstructionPointKey],
                selection: PeriodicMetricOrbitSelection,
            ) -> PeriodicMetricCandidateStage:
                current = (
                    self.source.mesh
                    if self.prior_stage is None
                    else self.prior_stage.target_mesh
                )
                geometry = (
                    self.source.geometry
                    if self.prior_stage is None
                    else self.prior_stage.geometry_stage.target_geometry
                )
                if (
                    selection.original_result_id,
                    selection.source_topology_id,
                    selection.source_geometry_id,
                    selection.source_numeric_version,
                    selection.source_frame_id,
                    selection.coordinate_contract_id,
                ) != (
                    self.source.result_id,
                    current.topology_id,
                    cell_geometry_id(geometry),
                    current.numeric_version,
                    canonical_fingerprint(array_tree_fingerprint(current.coordinates)),
                    self.source.coordinate_contract.spatial_id,
                ):
                    raise ValueError(
                        "UV orbit selection is stale for the actual private scientific frame."
                    )
                executed = []
                for carrier in selection.carriers:
                    if carrier.protected_entity or carrier.protected_vertex_global_ids:
                        raise ValueError(
                            "A UV orbit operation intersects protected scientific strata."
                        )
                    identifiers = np.asarray(
                        [
                            carrier.vertex_global_ids[index]
                            for index in carrier.vertex_permutation
                        ],
                        dtype=np.int64,
                    )
                    rows = key_rows(aggregate.vertex_ids[:, None], identifiers[:, None])
                    if np.any(rows < 0):
                        raise ValueError(
                            "UV orbit carrier omits its actual source vertex identities."
                        )
                    if not _apply_operation(
                        aggregate,
                        operation,
                        tuple(map(int, rows)),
                        background,
                        domain,
                        fixed,
                        trial_features,
                        trial_codes,
                        blocked,
                        self.policy,
                        self.maximum_fidelity,
                    ):
                        raise ValueError(
                            "A required UV carrier fails its actual chart/source constraints."
                        )
                    executed.append(carrier.entity_global_id)
                result, support = _exact_edit(
                    self.source.mesh,
                    aggregate,
                    background,
                    self.policy,
                    self.budget,
                    operation,
                )
                if edit.operation != result.operation or not stencils:
                    raise ValueError(
                        "UV orbit callback changed the actual operation or source support."
                    )
                return PeriodicMetricCandidateStage(result, support, tuple(executed))

            def build_geometry(
                edit: CellTopologyEdit, target: CellMesh
            ) -> PeriodicMetricGeometryStage:
                _, _, patches, charts, material_charts = _assemble_surface(
                    self.source.mesh,
                    aggregate,
                    background.cells,
                    background.patches,
                    background.charts,
                    self.policy.limits.maximum_geometry_queries,
                )
                vertex_strata = _surface_vertex_strata(
                    aggregate, np.unique(aggregate.cells)
                )
                witness = _SurfaceMetricGeometryWitness(
                    edit,
                    patches,
                    charts,
                    material_charts,
                    background.cell_ids,
                    background.charts,
                    *vertex_strata,
                    domain.source_id,
                    domain.source_revision,
                    domain.domain_id,
                    tuple(domain.entity_id(2, int(patch)) for patch in patches),
                    tuple(domain.source_occurrences[2][int(patch)] for patch in patches),
                )
                actual = prepare_surface_metric_geometry_transition(
                    self.source.mesh,
                    self.source.geometry,
                    target,
                    witness,
                    domain,
                    self.witness,
                    maximum_fidelity=self.maximum_fidelity,
                    policy=self.policy.geometry_transition,
                    maximum_candidate_pairs=self.policy.limits.maximum_geometry_queries,
                    maximum_memory_bytes=self.policy.limits.maximum_scratch_bytes,
                    certificate_limits=certification.request.limits,
                    validity_policy=self.policy.audit_policy.validity_policy,
                    source_chart_cover=_surface_metric_source_cover(self.source),
                    source_atlas=self.transfer.support.original
                    if isinstance(self.transfer.support.original, SurfaceSourceRootAtlas)
                    else None,
                    curve_witness=aggregate.curve_witness,
                    material_root_witness=self.material_root_witness,
                )
                target = actual.target_mesh
                lineage = metric_target_lineage(self.source.mesh, edit, target)
                associations = self.transfer.propagate(
                    self.source,
                    lineage,
                    target,
                    geometry=actual.transition.geometry,
                    embedding=actual.deformation.target_embedding,
                    deformation=actual.deformation,
                )
                return PeriodicMetricGeometryStage(
                    target,
                    actual.transition.geometry,
                    associations,
                    actual.deformation,
                    actual.transition,
                )

            retained = (
                ()
                if self.prior_stage is None
                else self.prior_stage.periodic_witness.quotient_entities
            )
            result = synchronize_periodic_metric_edit(
                self.source,
                candidate,
                exact,
                operation=operation,
                limits=self.policy.limits,
                policy=self.policy,
                coordinate_budget=self.budget,
                certificate_limits=certification.request.limits,
                prior_stage=self.prior_stage,
                build_candidate=build_candidate,
                build_geometry=build_geometry,
                retained_quotient_entities=retained,
            )
            self.prior_stage = result
            features.clear()
            features.update(trial_features)
            feature_codes.clear()
            feature_codes.update(trial_codes)
        if aggregate.vertex_strata is not None:
            self.budget.reserve(
                0,
                1024
                * max(
                    0, aggregate.curve_workspace_vertices - state.curve_workspace_vertices
                ),
            )
        self.budget.reserve(
            0,
            max(
                0,
                aggregate.material_workspace_bytes_upper
                - state.material_workspace_bytes_upper,
            ),
        )
        return aggregate


def _apply_operation(
    state: _SurfaceState,
    operation: PeriodicMetricOperation,
    vertices: tuple[int, ...],
    background: _SurfaceSource,
    domain: MeshingDomain,
    fixed: np.ndarray,
    features: set[tuple[int, int]],
    feature_codes: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
    policy: MeshAdaptationPolicy,
    maximum_fidelity: float,
    /,
) -> bool:
    if len(vertices) != (1 if operation == "relocate" else 2) or any(
        isinstance(vertex, bool) or not isinstance(vertex, int) or vertex < 0
        for vertex in vertices
    ):
        raise ValueError("A UV operation requires its actual vertex or edge rows.")
    match operation:
        case "split":
            first, second = sorted((vertices[0], vertices[1]))
            key = first, second
            edge = np.asarray(key, dtype=np.int32)
            if key in blocked:
                return False
            rows = np.flatnonzero(np.sum(np.isin(state.cells, edge), axis=1) == 2)
            if not rows.size:
                return False
            owner = rows[np.argmin(state.cell_ids[rows])]
            material = (
                state.charts if state.material_charts is None else state.material_charts
            )
            chart = np.mean(material[owner, np.isin(state.cells[owner], edge)], axis=0)
            sampling = _chart_stencil(
                background.cells,
                background.charts,
                background.patches,
                background.cell_ids,
                background.vertex_ids,
                state.patches[owner : owner + 1],
                chart[None],
                policy.limits.maximum_geometry_queries,
            )
            metric_rows = key_rows(
                background.vertex_ids[:, None], sampling[0].reshape((-1, 1))
            ).reshape(sampling[0].shape)
            metric = np.asarray(
                interpolate_mesh_metric(background.metric[metric_rows], sampling[1]),
                dtype=np.float64,
            )[0]
            vertex = state.points.shape[0]
            if not _split(domain, state, edge, metric, source_curve=key in features):
                return False
            if key in features:
                code = feature_codes.pop(key)
                features.remove(key)
                children = (
                    (min(first, vertex), max(first, vertex)),
                    (min(second, vertex), max(second, vertex)),
                )
                features.update(children)
                feature_codes.update((child, code) for child in children)
            return True
        case "collapse":
            return _collapse(
                domain,
                state,
                vertices[0],
                vertices[1],
                fixed,
                features,
                0.05,
                maximum_fidelity,
            )
        case "flip":
            return _flip(
                domain,
                state,
                np.asarray(sorted(vertices), dtype=np.int32),
                features,
                maximum_fidelity,
            )
        case "relocate":
            changes, _ = _relocate(
                domain,
                state,
                fixed,
                features,
                maximum_fidelity,
                1,
                policy.limits.maximum_geometry_queries,
                background,
                feature_codes,
                blocked,
                0.05,
                selected_vertices=frozenset(vertices),
            )
            return changes == 1
        case _:
            raise ValueError("Unknown UV metric operation.")


def _exact_edit(
    mesh: CellMesh,
    state: _SurfaceState,
    background: _SurfaceSource,
    policy: MeshAdaptationPolicy,
    budget: CoordinateEnclosureBudget,
    operation: PeriodicMetricOperation,
    /,
) -> tuple[CellTopologyEdit, Mapping[int, ConstructionPointKey]]:
    edit, _, _, _, _ = _assemble_surface(
        mesh,
        state,
        background.cells,
        background.patches,
        background.charts,
        policy.limits.maximum_geometry_queries,
    )
    queries = []
    for vertex in np.unique(state.cells):
        uses = np.flatnonzero(np.any(state.cells == vertex, axis=1))
        owner = uses[np.argmin(state.cell_ids[uses])]
        material = (
            state.charts if state.material_charts is None else state.material_charts
        )
        queries.append(material[owner, np.flatnonzero(state.cells[owner] == vertex)[0]])
    with budget.activate():
        keys = _chart_exact_weights(
            background.cells,
            background.charts,
            background.vertex_ids,
            edit.stencil_sources,
            np.asarray(
                queries, dtype=object if state.material_charts is not None else np.float64
            ),
        )
    exact: dict[int, ConstructionPointKey] = {}
    sources = np.zeros_like(edit.stencil_sources)
    coefficients = np.zeros_like(edit.stencil_weights)
    valid = np.zeros_like(edit.stencil_valid)
    for index, (identifier, key) in enumerate(
        zip(edit.vertex_global_ids, keys, strict=True)
    ):
        exact[int(identifier)] = key
        for slot, (parent, weight) in enumerate(key):
            sources[index, slot], coefficients[index, slot], valid[index, slot] = (
                parent,
                float(weight),
                True,
            )
    return edit._replace(
        operation="relocation" if operation == "relocate" else "local_reconnection",
        stencil_sources=sources,
        stencil_weights=coefficients,
        stencil_valid=valid,
    ), MappingProxyType(exact)
