#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Finite-volume remap acceptance and ended-execution evidence."""

from __future__ import annotations

from typing import Self

import equinox as eqx
import numpy as np
from jax import Array

from phydrax.discretization._sphere_chart_records import PreparedSphereChartDeformation
from phydrax.discretization._surface_chart_records import PreparedSurfaceChartDeformation

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import _supermesh as supermesh_owner
from ...geometry._volume_material_map import VolumeMaterialChain
from ...meshing._measurements import NativeExecutionRecord
from ...typing import checked, parse
from ..fem._surface_chart_transfer import PreparedSurfaceChartFiniteVolumeContents
from ._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
    UnstructuredRemapReport,
)


class MappedNestedRemapEvidence(StrictModule, NonTrainableState):
    """Retain reference tiling and certified physical overlap measures."""

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
        """Return whether the tiling, map, and physical ledger are admitted."""
        return (
            bool(np.asarray(self.report.coverage_complete))
            and np.isfinite(self.geometry_error_bound)
            and self.geometry_error_bound >= 0.0
        )


class MappedSurfaceChartRemapEvidence(StrictModule, NonTrainableState):
    """Old physical contents and new physical areas on certified material cells."""

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
    """Retain refused preparation and its ended native execution."""

    source_discretization_id: str = eqx.field(static=True)
    target_discretization_id: str = eqx.field(static=True)
    execution_evidence: NativeExecutionRecord
    logical_exact_work_units: Array
    host_storage_upper_bytes: Array
    reason: str = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return False


type UnstructuredRemapRoute = (
    supermesh_owner.PreparedCommonRefinement
    | MappedNestedRemapEvidence
    | MappedSurfaceChartRemapEvidence
    | VolumeMaterialChain
    | RemapPreparationFailure
)


class PreparedUnstructuredConservativeRemap(StrictModule, NonTrainableState):
    """Retain one geometric route and its conservative overlap plan.

    A plan is present exactly for ``SUCCESS``. The route retains common
    refinement, nested tiling, chart contents, or preparation-refusal evidence.
    """

    route: UnstructuredRemapRoute
    execution_evidence: NativeExecutionRecord | None
    plan: UnstructuredConservativeRemapPlan | None
    status: supermesh_owner.CommonRefinementStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    remap_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        route: UnstructuredRemapRoute,
        plan: UnstructuredConservativeRemapPlan | None,
        /,
        *,
        status: supermesh_owner.CommonRefinementStatus,
        reason: str,
        execution_evidence: NativeExecutionRecord | None = None,
    ) -> None:
        status = parse(status, supermesh_owner.CommonRefinementStatus, "remap.status")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"remap.reason must be a non-empty string; got {reason!r}.")
        succeeded = status is supermesh_owner.CommonRefinementStatus.SUCCESS
        if succeeded != (plan is not None):
            raise ValueError(
                f"remap.plan must be present exactly for SUCCESS; got {status.name}."
            )
        refinement_id = None
        nested_id = None
        chart_id = None
        failure_id = None
        volume_chain_id = None
        match route:
            case VolumeMaterialChain():
                if plan is None or plan.volume_material_chain is None or plan.volume_material_chain.chain_id != route.chain_id:
                    raise ValueError("remap.route must bind the volume material chain plan")
                volume_chain_id = route.chain_id
            case supermesh_owner.PreparedCommonRefinement():
                if succeeded and not route.succeeded:
                    raise ValueError(
                        "remap.route must contain successful refinement for SUCCESS; "
                        f"got {route.status.name}."
                    )
                refinement_id = route.refinement_id
            case MappedNestedRemapEvidence():
                if succeeded and not route.passed:
                    raise ValueError(
                        "remap.route must certify nested tiling and measures for SUCCESS; "
                        "got failed evidence."
                    )
                nested_id = route.reference_witness_id
            case MappedSurfaceChartRemapEvidence():
                if succeeded and not route.passed:
                    raise ValueError(
                        "remap.route must certify material contents for SUCCESS; "
                        "got failed evidence."
                    )
                chart_id = array_tree_fingerprint(route)
            case RemapPreparationFailure():
                if succeeded:
                    raise ValueError(
                        "remap.status must refuse a failed preparation; got SUCCESS."
                    )
                failure_id = array_tree_fingerprint(route)
            case _:
                raise TypeError(
                    "remap.route must contain refinement, tiling, chart contents, "
                    f"or preparation refusal; got {type(route).__name__}."
                )
        if execution_evidence is not None:
            execution_evidence.require_valid()
            if succeeded and int(np.asarray(execution_evidence.status)) != 0:
                raise ValueError(
                    "remap.execution_evidence must report native success for SUCCESS; "
                    f"got {np.asarray(execution_evidence.status)}."
                )
        self.route = route
        self.execution_evidence = execution_evidence
        self.plan = plan
        self.status = status
        self.reason = reason
        self.remap_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-conservative-remap",
                "refinement": refinement_id,
                "nested_witness": nested_id,
                "surface_chart": chart_id,
                "volume_material_chain": volume_chain_id,
                "preparation_failure": failure_id,
                "execution": None
                if execution_evidence is None
                else array_tree_fingerprint(execution_evidence),
                "plan": None if plan is None else plan.plan_id,
                "status": status.name,
                "reason": reason,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(cls, value: PreparedUnstructuredConservativeRemap, /) -> Self:
        """Rebuild a stored remap and validate its preparation identity."""
        rebuilt = cls(
            value.route,
            value.plan,
            status=value.status,
            reason=value.reason,
            execution_evidence=value.execution_evidence,
        )
        if rebuilt.remap_id != value.remap_id or rebuilt.succeeded != value.succeeded:
            raise ValueError(
                "remap.remap_id must preserve restored preparation and acceptance; "
                f"got {rebuilt.remap_id!r}, expected {value.remap_id!r}."
            )
        return rebuilt

    @property
    def succeeded(self) -> bool:
        return self.plan is not None

    @property
    def evidence(
        self,
    ) -> (
        supermesh_owner.CommonRefinementEvidence
        | MappedNestedRemapEvidence
        | MappedSurfaceChartRemapEvidence
        | VolumeMaterialChain
        | RemapPreparationFailure
    ):
        match self.route:
            case supermesh_owner.PreparedCommonRefinement() as route:
                return route.evidence
            case (
                MappedNestedRemapEvidence()
                | MappedSurfaceChartRemapEvidence()
                | VolumeMaterialChain()
                | RemapPreparationFailure()
            ) as route:
                return route
            case _:
                raise TypeError(
                    "remap.route must contain supported geometric evidence; "
                    f"got {type(self.route).__name__}."
                )


__all__ = [
    "MappedNestedRemapEvidence",
    "MappedSurfaceChartRemapEvidence",
    "PreparedUnstructuredConservativeRemap",
    "RemapPreparationFailure",
    "UnstructuredRemapRoute",
]
