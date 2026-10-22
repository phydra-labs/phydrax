#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Stage UV carrier operations under one source periodic metric ledger."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType

import numpy as np

import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._surface_chart_records as _dependency_phydrax_discretization__surface_chart_deformation
import phydrax.meshing._adaptation as _dependency_phydrax_meshing__adaptation
import phydrax.meshing._association as _dependency_phydrax_meshing__association
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._metric as _dependency_phydrax_meshing__metric
import phydrax.meshing._periodic as _dependency_phydrax_meshing__periodic
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._surface_metric as _dependency_phydrax_meshing__surface_metric
import phydrax.meshing._topology_edit as _dependency_phydrax_meshing__topology_edit

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeExecutionScope
from .._strict import StrictModule
from ..discretization import CellMesh
from ..geometry import MeshingDomain
from ..typing import checked


class _SurfaceOrbitPreparation(StrictModule):
    aggregate: _dependency_phydrax_meshing__surface_metric.SurfaceState
    background: _dependency_phydrax_meshing__surface_metric.SurfaceSource
    domain: MeshingDomain
    operation: _dependency_phydrax_meshing__periodic.PeriodicMetricOperation
    fixed: np.ndarray
    trial_features: set[tuple[int, int]]
    trial_codes: dict[tuple[int, int], int]
    blocked: set[tuple[int, int]]


@dataclass(slots=True)
class SurfaceMetricController:
    """Stage UV continuation against a fixed scientific source."""

    source: _dependency_phydrax_meshing__result.CellMeshingResult
    transfer: _dependency_phydrax_meshing__association.SurfaceAssociationTransfer
    policy: _dependency_phydrax_meshing__adaptation.MeshAdaptationPolicy
    budget: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    )
    scope: NativeExecutionScope = field(kw_only=True)
    witness: (
        _dependency_phydrax_discretization__surface_chart_deformation.SurfaceChartWitness
    )
    maximum_fidelity: float
    material_root_witness: (
        _dependency_phydrax_discretization__surface_chart_deformation.SurfaceChartWitness
        | None
    ) = None
    background: _dependency_phydrax_meshing__surface_metric.SurfaceSource | None = None
    prior_stage: (
        _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitOutcome | None
    ) = None
    minimum_metric_quality: float = 0.05
    material_routing: (
        _dependency_phydrax_meshing__surface_metric.PreparedMaterialRouting | None
    ) = None

    @property
    def operation_count(self) -> int:
        if self.prior_stage is None:
            raise ValueError(
                "surface_metric.controller must retain source-bound orbit support; got a UV operation count requires its completed orbit stage"
            )
        return self.prior_stage.evidence.orbit_copy_count

    @checked
    def stage(
        self,
        state: _dependency_phydrax_meshing__surface_metric.SurfaceState,
        operation: _dependency_phydrax_meshing__periodic.PeriodicMetricOperation,
        vertices: tuple[int, ...],
        fixed: np.ndarray,
        features: set[tuple[int, int]],
        feature_codes: dict[tuple[int, int], int],
        blocked: set[tuple[int, int]],
        /,
    ) -> _dependency_phydrax_meshing__surface_metric.SurfaceState | None:
        background, certification = self.background, self.source.certification
        if background is None or certification is None:
            raise ValueError(
                "surface_metric.controller must retain source-bound orbit support; got uV orbit staging requires its background and certificates"
            )
        if self.material_root_witness is None:
            raise ValueError(
                "surface_metric.controller.material_root_binding must retain admitted root charts; got a missing material root binding"
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
                        *(() if state.cell_coedges is None else (state.cell_coedges,)),
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
                minimum_metric_quality=self.minimum_metric_quality,
                material_routing=self.material_routing,
                budget=self.budget,
            ):
                return None
            candidate, exact = _exact_edit(
                self.source.mesh,
                seed,
                background,
                self.policy,
                operation,
                material_routing=self.material_routing,
                budget=self.budget,
            )

            preparation = _SurfaceOrbitPreparation(
                aggregate,
                background,
                domain,
                operation,
                fixed,
                trial_features,
                trial_codes,
                blocked,
            )
            build_candidate = partial(self._build_candidate, preparation)
            build_geometry = partial(self._build_geometry, preparation)

            retained = (
                ()
                if self.prior_stage is None
                else self.prior_stage.periodic_witness.quotient_entities
            )
            result = (
                _dependency_phydrax_meshing__periodic.synchronize_periodic_metric_edit(
                    _dependency_phydrax_meshing__periodic.PeriodicMetricEditRequest(
                        self.source,
                        candidate,
                        exact,
                        scope=self.scope,
                        operation=operation,
                        limits=self.policy.limits,
                        policy=self.policy,
                        budget=self.budget,
                        certificate_limits=certification.request.limits,
                        prior_stage=self.prior_stage,
                        build_candidate=build_candidate,
                        build_geometry=build_geometry,
                        retained_quotient_entities=retained,
                    ),
                )
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

    def _build_candidate(
        self,
        preparation: _SurfaceOrbitPreparation,
        edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
        stencils: Mapping[
            int, _dependency_phydrax_meshing__periodic.ConstructionPointKey
        ],
        selection: _dependency_phydrax_meshing__periodic.PeriodicMetricOrbitSelection,
    ) -> _dependency_phydrax_meshing__periodic.PeriodicMetricCandidateStage:
        aggregate = preparation.aggregate
        background = preparation.background
        domain = preparation.domain
        operation = preparation.operation
        fixed = preparation.fixed
        trial_features = preparation.trial_features
        trial_codes = preparation.trial_codes
        blocked = preparation.blocked
        current = (
            self.source.mesh if self.prior_stage is None else self.prior_stage.target_mesh
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
            _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                geometry
            ),
            current.numeric_version,
            canonical_fingerprint(array_tree_fingerprint(current.coordinates)),
            self.source.coordinate_contract.spatial_id,
        ):
            raise ValueError(
                "surface_metric.controller must bind the current source mesh and domain; got uV orbit selection is stale for the private scientific frame"
            )
        executed = []
        for carrier in selection.carriers:
            if carrier.protected_entity or carrier.protected_vertex_global_ids:
                raise ValueError(
                    "surface_metric.controller must preserve protected source strata; got a UV orbit operation intersects protected scientific strata"
                )
            identifiers = np.asarray(
                [
                    carrier.vertex_global_ids[index]
                    for index in carrier.vertex_permutation
                ],
                dtype=np.int64,
            )
            rows = entity_keys_owner.key_rows(
                aggregate.vertex_ids[:, None], identifiers[:, None]
            )
            if np.any(rows < 0):
                raise ValueError(
                    "surface_metric.controller must retain valid source vertex rows; got uV orbit carrier omits its source vertex identities"
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
                minimum_metric_quality=self.minimum_metric_quality,
                material_routing=self.material_routing,
                budget=self.budget,
            ):
                raise ValueError(
                    "surface_metric.controller must retain source-bound orbit support; got a required UV carrier fails its chart/source constraints"
                )
            executed.append(carrier.entity_global_id)
        result, support = _exact_edit(
            self.source.mesh,
            aggregate,
            background,
            self.policy,
            operation,
            material_routing=self.material_routing,
            budget=self.budget,
        )
        if edit.operation != result.operation or not stencils:
            raise ValueError(
                "surface_metric.controller must preserve operation and source support; got uV orbit callback changed the operation or source support"
            )
        return _dependency_phydrax_meshing__periodic.PeriodicMetricCandidateStage(
            result, support, tuple(executed)
        )

    def _build_geometry(
        self,
        preparation: _SurfaceOrbitPreparation,
        edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
        target: CellMesh,
    ) -> _dependency_phydrax_meshing__periodic.PeriodicMetricGeometryStage:
        aggregate = preparation.aggregate
        background = preparation.background
        domain = preparation.domain
        _, _, patches, charts, material_charts = (
            _dependency_phydrax_meshing__surface_metric.assemble_surface(
                self.source.mesh,
                aggregate,
                background.cells,
                background.patches,
                background.charts,
                self.policy.limits.maximum_geometry_queries,
                material_routing=self.material_routing,
                budget=self.budget,
            )
        )
        vertex_strata = _dependency_phydrax_meshing__surface_metric.surface_vertex_strata(
            aggregate, np.unique(aggregate.cells)
        )
        witness = (
            _dependency_phydrax_meshing__surface_metric.SurfaceMetricGeometryWitness(
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
                _dependency_phydrax_meshing__surface_metric.surface_cell_coedges(
                    aggregate, edit
                ),
            )
        )
        actual = _dependency_phydrax_meshing__surface_metric.prepare_policy_surface_metric_transition(
            self.source,
            target,
            witness,
            self.transfer,
            self.policy,
            self.witness,
            maximum_fidelity=self.maximum_fidelity,
            curve_witness=aggregate.curve_witness,
            material_root_witness=self.material_root_witness,
            scope=self.scope,
            budget=self.budget,
        )
        target = actual.target_mesh
        lineage = _dependency_phydrax_meshing__periodic.metric_target_lineage(
            self.source.mesh, edit, target, scope=self.scope, budget=self.budget
        )
        associations = self.transfer.propagate(
            self.source,
            lineage,
            target,
            geometry=actual.transition.geometry,
            embedding=actual.deformation.target_embedding,
            deformation=actual.deformation,
            budget=self.budget,
        )
        return _dependency_phydrax_meshing__periodic.PeriodicMetricGeometryStage(
            target,
            actual.transition.geometry,
            associations,
            actual.deformation,
            actual.transition,
        )


def _apply_operation(
    state: _dependency_phydrax_meshing__surface_metric.SurfaceState,
    operation: _dependency_phydrax_meshing__periodic.PeriodicMetricOperation,
    vertices: tuple[int, ...],
    background: _dependency_phydrax_meshing__surface_metric.SurfaceSource,
    domain: MeshingDomain,
    fixed: np.ndarray,
    features: set[tuple[int, int]],
    feature_codes: dict[tuple[int, int], int],
    blocked: set[tuple[int, int]],
    policy: _dependency_phydrax_meshing__adaptation.MeshAdaptationPolicy,
    maximum_fidelity: float,
    /,
    *,
    minimum_metric_quality: float,
    material_routing: _dependency_phydrax_meshing__surface_metric.PreparedMaterialRouting
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    if len(vertices) != (1 if operation == "relocate" else 2) or any(
        isinstance(vertex, bool) or not isinstance(vertex, int) or vertex < 0
        for vertex in vertices
    ):
        raise ValueError(
            "surface_metric.controller must retain valid source vertex rows; got a UV operation requires its vertex or edge rows"
        )
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
            sampling = _dependency_phydrax_meshing__surface_metric.chart_stencil(
                background.cells,
                background.charts,
                background.patches,
                background.cell_ids,
                background.vertex_ids,
                state.patches[owner : owner + 1],
                chart[None],
                policy.limits.maximum_geometry_queries,
                material_cache=material_routing,
                budget=budget,
            )
            metric_rows = entity_keys_owner.key_rows(
                background.vertex_ids[:, None], sampling[0].reshape((-1, 1))
            ).reshape(sampling[0].shape)
            metric = np.asarray(
                _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
                    background.metric[metric_rows], sampling[1]
                ),
                dtype=np.float64,
            )[0]
            vertex = state.points.shape[0]
            if not _dependency_phydrax_meshing__surface_metric.split(
                domain, state, edge, metric, source_curve=key in features, budget=budget
            ):
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
            return _dependency_phydrax_meshing__surface_metric.collapse(
                domain,
                state,
                vertices[0],
                vertices[1],
                fixed,
                features,
                minimum_metric_quality,
                maximum_fidelity,
                budget=budget,
            )
        case "flip":
            return _dependency_phydrax_meshing__surface_metric.flip(
                domain,
                state,
                np.asarray(sorted(vertices), dtype=np.int32),
                features,
                maximum_fidelity,
                budget=budget,
            )
        case "relocate":
            changes, _ = _dependency_phydrax_meshing__surface_metric.relocate(
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
                minimum_metric_quality,
                selected_vertices=frozenset(vertices),
                material_routing=material_routing,
                budget=budget,
            )
            return changes == 1
        case _:
            raise ValueError(
                "surface_metric.controller must select a supported UV operation; got unknown UV metric operation"
            )


def _exact_edit(
    mesh: CellMesh,
    state: _dependency_phydrax_meshing__surface_metric.SurfaceState,
    background: _dependency_phydrax_meshing__surface_metric.SurfaceSource,
    policy: _dependency_phydrax_meshing__adaptation.MeshAdaptationPolicy,
    operation: _dependency_phydrax_meshing__periodic.PeriodicMetricOperation,
    /,
    *,
    material_routing: _dependency_phydrax_meshing__surface_metric.PreparedMaterialRouting
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    Mapping[int, _dependency_phydrax_meshing__periodic.ConstructionPointKey],
]:
    edit, _, _, _, _ = _dependency_phydrax_meshing__surface_metric.assemble_surface(
        mesh,
        state,
        background.cells,
        background.patches,
        background.charts,
        policy.limits.maximum_geometry_queries,
        material_routing=material_routing,
        budget=budget,
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
        keys = _dependency_phydrax_meshing__surface_metric.chart_exact_weights(
            background.cells,
            background.charts,
            background.vertex_ids,
            edit.stencil_sources,
            np.asarray(
                queries, dtype=object if state.material_charts is not None else np.float64
            ),
            budget=budget,
        )
    exact: dict[int, _dependency_phydrax_meshing__periodic.ConstructionPointKey] = {}
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
