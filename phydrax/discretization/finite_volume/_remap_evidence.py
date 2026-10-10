#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Actual finite-volume remap acceptance and ended-execution evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._sphere_chart_deformation import PreparedSphereChartDeformation
from .._surface_chart_deformation import PreparedSurfaceChartDeformation
from ._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
    UnstructuredRemapReport,
)


if TYPE_CHECKING:
    from ...geometry._supermesh import (
        CommonRefinementEvidence,
        CommonRefinementStatus,
        PreparedCommonRefinement,
    )
    from ...meshing._measurements import NativeExecutionRecord
    from ..fem._surface_chart_transfer import PreparedSurfaceChartFiniteVolumeContents


class MappedNestedRemapEvidence(StrictModule, NonTrainableState):
    """Exact reference tiling plus certified physical overlap measure evidence."""

    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    reference_witness_id: str = eqx.field(static=True)
    geometry_error_bound: float = eqx.field(static=True)
    measure_exact: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    report: UnstructuredRemapReport

    @property
    def passed(self) -> bool:
        """Admission result of the certified tiling/map and physical ledger."""
        return (
            bool(np.asarray(self.report.coverage_complete))
            and np.isfinite(self.geometry_error_bound)
            and self.geometry_error_bound >= 0.0
        )


class MappedSurfaceChartRemapEvidence(StrictModule, NonTrainableState):
    """Actual old physical contents and new physical areas on certified material cells."""

    deformation: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation
    physical_contents: PreparedSurfaceChartFiniteVolumeContents
    report: UnstructuredRemapReport
    execution_evidence: NativeExecutionRecord | None
    logical_exact_work_units: Array
    host_storage_upper_bytes: Array
    reason: str = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(np.asarray(self.report.coverage_complete)) and (
            self.deformation.source_domain_coverage == "certified"
            and self.deformation.target_domain_coverage == "certified"
        )


class RemapPreparationFailure(StrictModule, NonTrainableState):
    """An actual refused preparation; no overlap or successful plan is invented."""

    source_discretization_id: str = eqx.field(static=True)
    target_discretization_id: str = eqx.field(static=True)
    execution_evidence: NativeExecutionRecord
    logical_exact_work_units: Array
    host_storage_upper_bytes: Array
    reason: str = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return False


class PreparedUnstructuredConservativeRemap(StrictModule, NonTrainableState):
    """Prepared overlap ledger with its actual geometric construction evidence.

    ``plan`` is present exactly when ``status`` is ``SUCCESS``. ``refinement``
    exists only for geometrically intersected corner cells. ``nested_evidence``
    carries exact reference tiling; ``surface_chart_evidence`` retains actual
    native chart deformation and its certified old physical contents. Neither
    route fabricates geometrically intersected common-refinement cells.
    """

    refinement: PreparedCommonRefinement | None
    nested_evidence: MappedNestedRemapEvidence | None
    surface_chart_evidence: MappedSurfaceChartRemapEvidence | None
    preparation_failure: RemapPreparationFailure | None
    execution_evidence: NativeExecutionRecord | None
    route: str = eqx.field(static=True)
    plan: UnstructuredConservativeRemapPlan | None
    status: CommonRefinementStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    remap_id: str = eqx.field(static=True)

    def __init__(
        self,
        refinement: PreparedCommonRefinement | None,
        plan: UnstructuredConservativeRemapPlan | None,
        /,
        *,
        status: CommonRefinementStatus,
        reason: str,
        nested_evidence: MappedNestedRemapEvidence | None = None,
        surface_chart_evidence: MappedSurfaceChartRemapEvidence | None = None,
        preparation_failure: RemapPreparationFailure | None = None,
        execution_evidence: NativeExecutionRecord | None = None,
    ) -> None:
        from ...geometry._supermesh import (
            CommonRefinementStatus,
            PreparedCommonRefinement,
        )

        if refinement is not None and not isinstance(
            refinement, PreparedCommonRefinement
        ):
            raise TypeError("refinement must be a PreparedCommonRefinement or None.")
        if (
            sum(
                item is not None
                for item in (
                    refinement,
                    nested_evidence,
                    surface_chart_evidence,
                    preparation_failure,
                )
            )
            != 1
        ):
            raise ValueError(
                "Exactly one actual refinement, nested proof, chart evidence, or preparation refusal is required."
            )
        if plan is not None and not isinstance(plan, UnstructuredConservativeRemapPlan):
            raise TypeError("plan must be an UnstructuredConservativeRemapPlan or None.")
        if not isinstance(status, CommonRefinementStatus):
            raise TypeError("status must be a CommonRefinementStatus.")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string.")
        succeeded = status is CommonRefinementStatus.SUCCESS
        if succeeded != (plan is not None):
            raise ValueError("A remap plan is present exactly for a successful status.")
        if succeeded and refinement is not None and not refinement.succeeded:
            raise ValueError("A successful remap requires a successful refinement.")
        if (
            succeeded
            and surface_chart_evidence is not None
            and not surface_chart_evidence.passed
        ):
            raise ValueError(
                "A successful remap requires its actual complete material content certificate."
            )
        if succeeded and nested_evidence is not None and not nested_evidence.passed:
            raise ValueError(
                "A successful nested remap requires its actual complete tiling and measure certificate."
            )
        if preparation_failure is not None and (succeeded or plan is not None):
            raise ValueError(
                "A refused preparation cannot publish a successful remap plan."
            )
        if execution_evidence is not None:
            execution_evidence.require_valid()
            if succeeded and int(np.asarray(execution_evidence.status)) != 0:
                raise ValueError(
                    "A successful remap cannot carry a failed native execution."
                )
        self.refinement = refinement
        self.nested_evidence = nested_evidence
        self.surface_chart_evidence = surface_chart_evidence
        self.preparation_failure = preparation_failure
        self.execution_evidence = execution_evidence
        self.route = (
            "resource-limit"
            if preparation_failure is not None
            else "mapped-surface-chart"
            if surface_chart_evidence is not None
            else "mapped-nested"
            if nested_evidence is not None
            else "common-refinement"
        )
        self.plan = plan
        self.status = status
        self.reason = reason
        self.remap_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-conservative-remap",
                "refinement": None if refinement is None else refinement.refinement_id,
                "nested_witness": None
                if nested_evidence is None
                else nested_evidence.reference_witness_id,
                "surface_chart": None
                if surface_chart_evidence is None
                else array_tree_fingerprint(surface_chart_evidence),
                "preparation_failure": None
                if preparation_failure is None
                else array_tree_fingerprint(preparation_failure),
                "execution": None
                if execution_evidence is None
                else array_tree_fingerprint(execution_evidence),
                "plan": None if plan is None else plan.plan_id,
                "status": status.name,
                "reason": reason,
            }
        )

    @property
    def succeeded(self) -> bool:
        return self.plan is not None

    @property
    def evidence(
        self,
    ) -> (
        CommonRefinementEvidence
        | MappedNestedRemapEvidence
        | MappedSurfaceChartRemapEvidence
        | RemapPreparationFailure
    ):
        if self.preparation_failure is not None:
            return self.preparation_failure
        if self.surface_chart_evidence is not None:
            return self.surface_chart_evidence
        if self.nested_evidence is not None:
            return self.nested_evidence
        if self.refinement is not None:
            return self.refinement.evidence
        raise ValueError("Prepared remap has no actual route evidence.")


__all__ = [
    "MappedNestedRemapEvidence",
    "MappedSurfaceChartRemapEvidence",
    "PreparedUnstructuredConservativeRemap",
    "RemapPreparationFailure",
]
