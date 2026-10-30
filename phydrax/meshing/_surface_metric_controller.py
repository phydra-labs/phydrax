#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-source controller for periodic source-chart metric edits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import final

from jax.typing import ArrayLike

from .._geometry_predicates import PredicateMode
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryQuery,
)
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ._result import CellMeshingResult
from ._surface_metric import (
    execute_surface_metric_adaptation,
    prepare_implicit_surface_metric_source,
    prepare_surface_metric_source,
    SurfaceMetricOutcome,
    SurfaceMetricSource,
)


@final
@dataclass(slots=True)
class SurfaceMetricController:
    """Stage all UV edits against one immutable accepted source and metric.

    The prepared UV carrier owns a periodic quotient whenever the retained source
    chart has periodic parameters. The local metric executor synchronizes every
    split, collapse, flip, and relocation orbit before this controller publishes
    the physical-coordinate edit.
    """

    source: CellMeshingResult
    metric: ArrayLike
    predicate_mode: PredicateMode
    topology_operations: bool = True
    relocation: bool = True
    cell_classes: ArrayLike | None = None
    edge_classes: ArrayLike | None = None
    protected_edges: ArrayLike | None = None
    fixed_vertices: ArrayLike | None = None
    implicit_profile: AnalyticImplicitProfile | None = None
    prepared: SurfaceMetricSource = field(init=False, repr=False)
    _source_mesh_id: str = field(init=False, repr=False)
    _source_chart_id: str = field(init=False, repr=False)
    _implicit_witness_id: str | None = field(init=False, default=None, repr=False)
    _staged: SurfaceMetricOutcome | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        if not isinstance(self.predicate_mode, PredicateMode):
            raise TypeError("predicate_mode must be PredicateMode.")
        if self.predicate_mode is PredicateMode.FILTERED_DEVICE:
            raise ValueError("Embedded surface metric adaptation is a host route.")
        if not isinstance(self.topology_operations, bool) or not isinstance(
            self.relocation, bool
        ):
            raise TypeError("topology_operations and relocation must be bool values.")
        charts = self.source.surface_source
        if isinstance(charts, SurfaceSourceCharts):
            if self.implicit_profile is not None:
                raise ValueError(
                    "BRep source charts cannot be paired with an implicit profile."
                )
            charts.require_root(self.source.mesh, self.source.geometry)
            self.prepared = prepare_surface_metric_source(
                self.source.mesh,
                charts,
                self.metric,
                cell_classes=self.cell_classes,
                edge_classes=self.edge_classes,
                protected_edges=self.protected_edges,
                fixed_vertices=self.fixed_vertices,
            )
            self._source_chart_id = charts.source_chart_id
        elif charts is None:
            certification = self.source.certification
            profile = self.implicit_profile
            if certification is None:
                raise ValueError(
                    "Implicit surface metric adaptation requires source certification."
                )
            witness = certification.inputs.fidelity_witness
            fidelity_source = certification.inputs.source
            if (
                not isinstance(profile, AnalyticImplicitProfile)
                or not isinstance(witness, SourceBoundaryChartWitness)
                or not isinstance(fidelity_source, SourceBoundaryQuery)
            ):
                raise ValueError(
                    "Implicit surface metric adaptation requires an authoring profile, exact source chart witness, and bound source query."
                )
            profile.require_bound(profile.geometry, self.source.coordinate_contract)
            self.prepared = prepare_implicit_surface_metric_source(
                self.source.mesh,
                self.source.geometry,
                self.source.coordinate_contract,
                profile,
                witness,
                fidelity_source,
                self.metric,
                cell_classes=self.cell_classes,
                edge_classes=self.edge_classes,
                protected_edges=self.protected_edges,
                fixed_vertices=self.fixed_vertices,
            )
            self._source_chart_id = witness.witness_id
            self._implicit_witness_id = witness.witness_id
        else:
            raise TypeError("Surface metric source charts have an unsupported owner.")
        self._source_mesh_id = self.source.mesh.mesh_id

    @property
    def staged(self) -> SurfaceMetricOutcome:
        """Return the completed candidate or refuse access before staging."""
        if self._staged is None:
            raise ValueError("No source-chart metric edit has been staged.")
        return self._staged

    def stage(self, /) -> SurfaceMetricOutcome:
        """Execute one complete, orbit-synchronized edit from the fixed source."""
        charts = self.source.surface_source
        if self.source.mesh.mesh_id != self._source_mesh_id:
            raise ValueError("The fixed surface metric source changed after preparation.")
        if isinstance(charts, SurfaceSourceCharts):
            if charts.source_chart_id != self._source_chart_id:
                raise ValueError(
                    "The fixed surface metric source charts changed after preparation."
                )
            charts.require_root(self.source.mesh, self.source.geometry)
        else:
            certification = self.source.certification
            profile = self.implicit_profile
            if certification is None or not isinstance(profile, AnalyticImplicitProfile):
                raise ValueError(
                    "Implicit source certification was removed after preparation."
                )
            witness = certification.inputs.fidelity_witness
            fidelity_source = certification.inputs.source
            if (
                not isinstance(witness, SourceBoundaryChartWitness)
                or witness.witness_id != self._implicit_witness_id
                or not isinstance(fidelity_source, SourceBoundaryQuery)
            ):
                raise ValueError(
                    "The fixed implicit metric source charts changed after preparation."
                )
            witness.require_current(
                self.source.mesh, self.source.geometry, fidelity_source
            )
        result = execute_surface_metric_adaptation(
            self.prepared,
            predicate_mode=self.predicate_mode,
            topology_operations=self.topology_operations,
            relocation=self.relocation,
        )
        if self.prepared.periodic_rank and result.geometry.periodic_topology_id is None:
            raise RuntimeError(
                "Periodic UV adaptation lost its synchronized target orbit."
            )
        self._staged = result
        return result


__all__ = ["SurfaceMetricController"]
