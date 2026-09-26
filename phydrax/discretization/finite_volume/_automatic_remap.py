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
    from ...geometry._supermesh import (
        CommonRefinementEvidence,
        CommonRefinementPolicy,
        CommonRefinementStatus,
        PreparedCommonRefinement,
    )


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
    ):
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
    refinement = prepare_common_refinement(source.mesh, target.mesh, policy=policy_)
    if not refinement.succeeded:
        return PreparedUnstructuredConservativeRemap(
            refinement,
            None,
            status=refinement.status,
            reason=f"{refinement.status.name}: {refinement.evidence.reason}",
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
