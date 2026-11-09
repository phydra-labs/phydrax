#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative finite-volume remap construction on the canonical common refinement.

Candidate discovery, clipping, and coverage certification are owned by
:func:`phydrax.geometry.prepare_common_refinement`.  This module binds its CSR
ledger to two prepared finite-volume discretizations and fails closed with the
refinement's own status vocabulary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_remap import _coverage_ledger, UnstructuredConservativeRemapPlan


if TYPE_CHECKING:
    from ...geometry._global_embedding import GlobalEmbeddingCertificate
    from ...geometry._supermesh import (
        CommonRefinementEvidence,
        CommonRefinementPolicy,
        CommonRefinementStatus,
        PreparedCommonRefinement,
    )
    from .._cell_geometry_transition_types import CellGeometryTransition


class PreparedUnstructuredConservativeRemap(StrictModule, NonTrainableState):
    """Common refinement of two finite-volume discretizations and its remap plan.

    ``plan`` is present exactly when ``status`` is ``SUCCESS``.  ``status`` is the
    refinement status, except that a successful complete-coverage refinement whose
    certified overlap measures do not cover the finite-volume cell volumes within
    the policy tolerance reports ``COVERAGE_GAP`` or ``DOUBLE_COVERAGE``.
    ``reason`` explains every status and ``evidence`` is the refinement evidence.
    """

    refinement: PreparedCommonRefinement
    plan: UnstructuredConservativeRemapPlan | None
    status: CommonRefinementStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    remap_id: str = eqx.field(static=True)

    def __init__(
        self,
        refinement: PreparedCommonRefinement,
        plan: UnstructuredConservativeRemapPlan | None,
        /,
        *,
        status: CommonRefinementStatus,
        reason: str,
    ) -> None:
        from ...geometry._supermesh import (
            CommonRefinementStatus,
            PreparedCommonRefinement,
        )

        if not isinstance(refinement, PreparedCommonRefinement):
            raise TypeError("refinement must be a PreparedCommonRefinement.")
        if plan is not None and not isinstance(plan, UnstructuredConservativeRemapPlan):
            raise TypeError("plan must be an UnstructuredConservativeRemapPlan or None.")
        if not isinstance(status, CommonRefinementStatus):
            raise TypeError("status must be a CommonRefinementStatus.")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string.")
        succeeded = status is CommonRefinementStatus.SUCCESS
        if succeeded != (plan is not None):
            raise ValueError("A remap plan is present exactly for a successful status.")
        if succeeded and not refinement.succeeded:
            raise ValueError("A successful remap requires a successful refinement.")
        self.refinement = refinement
        self.plan = plan
        self.status = status
        self.reason = reason
        self.remap_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-conservative-remap",
                "refinement": refinement.refinement_id,
                "plan": None if plan is None else plan.plan_id,
                "status": status.name,
                "reason": reason,
            }
        )

    @property
    def succeeded(self) -> bool:
        return self.plan is not None

    @property
    def evidence(self) -> CommonRefinementEvidence:
        return self.refinement.evidence


def prepare_unstructured_conservative_remap(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    /,
    *,
    provenance: str,
    policy: CommonRefinementPolicy | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    source_embedding: GlobalEmbeddingCertificate | None = None,
    target_embedding: GlobalEmbeddingCertificate | None = None,
) -> PreparedUnstructuredConservativeRemap:
    """Prepare the conservative remap of ``source`` cells onto ``target`` cells.

    The discretizations' cell meshes are intersected by
    :func:`phydrax.geometry.prepare_common_refinement` under ``policy`` (the
    default policy when ``None``).  ``COMPLETE`` coverage yields a plan requiring
    complete coverage with the policy's relative ``coverage_tolerance``; other
    coverage modes yield a plan that reports, but does not require, coverage.
    Geometric and resource failures are returned as a failed status without a
    plan; they never raise.
    """
    from ...geometry._supermesh import (
        CommonRefinementCoverage,
        CommonRefinementPolicy,
        CommonRefinementStatus,
        prepare_common_refinement,
    )

    if not isinstance(source, UnstructuredFiniteVolumeDiscretization) or not isinstance(
        target, UnstructuredFiniteVolumeDiscretization
    ):
        raise TypeError("Remap source and target must be unstructured FV geometry.")
    if not isinstance(provenance, str):
        raise TypeError("provenance must be a string.")
    if not provenance:
        raise ValueError("provenance must be non-empty.")
    policy_ = CommonRefinementPolicy() if policy is None else policy
    if not isinstance(policy_, CommonRefinementPolicy):
        raise TypeError("policy must be a CommonRefinementPolicy.")
    source_mesh, target_mesh = source.mesh, target.mesh
    if source_mesh is None or target_mesh is None:
        raise TypeError("Common refinement requires source and target CellMesh objects.")
    source_spec = source.coordinate_spec
    target_spec = target.coordinate_spec
    if geometry_transition is None:
        if source_spec is not None or target_spec is not None:
            raise ValueError(
                "Geometry-aware FV remapping requires a geometry transition."
            )
    else:
        from ...discretization._cell_geometry_transition_types import (
            CellGeometryTransition,
        )

        if not isinstance(geometry_transition, CellGeometryTransition):
            raise TypeError("geometry_transition must be a CellGeometryTransition.")
        if source_spec is None or target_spec is None:
            raise ValueError(
                "Geometry-aware remapping requires both FV coordinate specs."
            )
        if source_mesh.mesh_id == geometry_transition.source_mesh.mesh_id:
            geometry_transition.require_bound(
                source_mesh, source_spec, target_mesh, target_spec
            )
        else:
            geometry_transition.require_bound(
                target_mesh, target_spec, source_mesh, source_spec
            )
    refinement = prepare_common_refinement(
        source_mesh,
        target_mesh,
        policy=policy_,
        geometry_transition=geometry_transition,
        source_embedding=source_embedding,
        target_embedding=target_embedding,
    )
    if not refinement.succeeded:
        return PreparedUnstructuredConservativeRemap(
            refinement,
            None,
            status=refinement.status,
            reason=f"{refinement.status.name}: {refinement.evidence.reason}",
        )
    if geometry_transition is not None:
        relative_budget = min(1.0e-12, policy_.coverage_tolerance / 16.0)
        for endpoint in (source, target):
            errors = endpoint.cell_measure_errors
            if errors is None or np.any(
                np.asarray(errors, dtype=np.float64)
                > relative_budget * np.asarray(endpoint.cell_volumes, dtype=np.float64)
            ):
                return PreparedUnstructuredConservativeRemap(
                    refinement,
                    None,
                    status=CommonRefinementStatus.PREDICATE_UNCERTAIN,
                    reason="prepared FV cell measure errors exceed the requested relative budget",
                )
    require_complete = policy_.coverage is CommonRefinementCoverage.COMPLETE
    if require_complete:
        target_defect, source_defect, target_limit, source_limit = _coverage_ledger(
            np.asarray(refinement.target_cells),
            np.asarray(refinement.source_cells),
            np.asarray(refinement.volumes, dtype=np.float64),
            np.asarray(source.cell_volumes),
            np.asarray(target.cell_volumes),
            policy_.coverage_tolerance,
        )
        if np.any(target_defect > target_limit) or np.any(source_defect > source_limit):
            return PreparedUnstructuredConservativeRemap(
                refinement,
                None,
                status=CommonRefinementStatus.DOUBLE_COVERAGE,
                reason="certified overlap measures exceed finite-volume cell volumes",
            )
        if np.any(target_defect < -target_limit) or np.any(source_defect < -source_limit):
            return PreparedUnstructuredConservativeRemap(
                refinement,
                None,
                status=CommonRefinementStatus.COVERAGE_GAP,
                reason="certified overlap measures leave finite-volume cell volume uncovered",
            )
    plan = UnstructuredConservativeRemapPlan(
        source,
        target,
        refinement.target_offsets,
        refinement.source_cells,
        refinement.volumes,
        method="common-refinement",
        provenance=provenance,
        tolerance=policy_.coverage_tolerance,
        require_complete=require_complete,
    )
    return PreparedUnstructuredConservativeRemap(
        refinement,
        plan,
        status=CommonRefinementStatus.SUCCESS,
        reason="certified complete common refinement"
        if require_complete
        else f"certified {policy_.coverage.value} common refinement",
    )


__all__ = [
    "PreparedUnstructuredConservativeRemap",
    "prepare_unstructured_conservative_remap",
]
