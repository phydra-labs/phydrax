#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite, explicitly regulated semiclassical spinor electrodynamics."""

from ._finite_modes import (
    FiniteSpatialSpinorQEDPlan,
    FiniteSpatialSpinorQEDState,
    FiniteSpatialSpinorQEDVectorField,
    HomogeneousSpinorQED3DPlan,
    HomogeneousSpinorQED3DState,
    HomogeneousSpinorQED3DVectorField,
    negative_energy_spinor_modes_3d,
    PreparedFiniteSpatialSpinorQED,
    PreparedHomogeneousSpinorQED3D,
    ZeroExternalCurrent3D,
    ZeroSpatialExternalCurrent,
)
from ._homogeneous import (
    HomogeneousSpinorQEDEvidence,
    HomogeneousSpinorQEDPlan,
    HomogeneousSpinorQEDResult,
    HomogeneousSpinorQEDState,
    HomogeneousSpinorQEDVectorField,
    negative_energy_spinor_modes,
    PreparedHomogeneousSpinorQED,
    SemiclassicalQEDEvidenceStatus,
    solve_homogeneous_spinor_qed,
    TabulatedExternalCurrent,
    ZeroExternalCurrent,
)
from ._response import (
    HomogeneousSpinorQEDTangentEvidence,
    HomogeneousSpinorQEDTangentResult,
    HomogeneousSpinorQEDTangentState,
    HomogeneousSpinorQEDTangentVectorField,
    PreparedHomogeneousSpinorQEDTangent,
    RetardedVolterraEvidence,
    RetardedVolterraResponsePlan,
    RetardedVolterraResult,
    solve_homogeneous_spinor_qed_tangent,
    tangent_finite_difference_evidence,
    TangentFiniteDifferenceEvidence,
)


__all__ = [
    "FiniteSpatialSpinorQEDPlan",
    "FiniteSpatialSpinorQEDState",
    "FiniteSpatialSpinorQEDVectorField",
    "HomogeneousSpinorQED3DPlan",
    "HomogeneousSpinorQED3DState",
    "HomogeneousSpinorQED3DVectorField",
    "HomogeneousSpinorQEDEvidence",
    "HomogeneousSpinorQEDPlan",
    "HomogeneousSpinorQEDResult",
    "HomogeneousSpinorQEDState",
    "HomogeneousSpinorQEDTangentEvidence",
    "HomogeneousSpinorQEDTangentResult",
    "HomogeneousSpinorQEDTangentState",
    "HomogeneousSpinorQEDTangentVectorField",
    "HomogeneousSpinorQEDVectorField",
    "PreparedFiniteSpatialSpinorQED",
    "PreparedHomogeneousSpinorQED",
    "PreparedHomogeneousSpinorQED3D",
    "PreparedHomogeneousSpinorQEDTangent",
    "RetardedVolterraEvidence",
    "RetardedVolterraResponsePlan",
    "RetardedVolterraResult",
    "SemiclassicalQEDEvidenceStatus",
    "TabulatedExternalCurrent",
    "TangentFiniteDifferenceEvidence",
    "ZeroExternalCurrent",
    "ZeroExternalCurrent3D",
    "ZeroSpatialExternalCurrent",
    "negative_energy_spinor_modes",
    "negative_energy_spinor_modes_3d",
    "solve_homogeneous_spinor_qed",
    "solve_homogeneous_spinor_qed_tangent",
    "tangent_finite_difference_evidence",
]
