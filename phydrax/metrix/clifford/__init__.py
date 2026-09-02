#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Metric-dependent Clifford algebra plans over explicit blade layouts."""

from ._action import (
    audit_clifford_action,
    audit_clifford_actions,
    CliffordActionAuditReport,
    CliffordOutermorphismPlan,
)
from ._blades import CliffordBladeLayout
from ._families import (
    CliffordCochainProductPlan,
    CliffordInverseResult,
    CliffordMetricField,
    CliffordProjectorEvidence,
    ConformalCliffordModel,
    invert_multivector,
    MinimalLeftIdeal,
    PinElement,
    PreparedCauchyCliffordProjector,
    PreparedCliffordMetricProduct,
    ProjectiveCliffordModel,
    SpinElement,
)
from ._forms import CliffordMetricBridge
from ._involutions import (
    basis_blade,
    clifford_conjugate,
    embed_layout,
    extract_layout,
    grade_involution,
    grade_layout,
    project_grades,
    reverse,
    scalar_part,
)
from ._isometries import (
    FiniteMetricIsometryGroup,
    lorentz_boost_action,
    MetricIsometryAction,
    MetricIsometryAuditSet,
)
from ._product import (
    basis_blade_product,
    CliffordProductKind,
    CliffordProductPlan,
    prepare_product,
)
from ._provider import CliffordFiniteAlgebraProvider
from ._reports import CliffordProductEvidence
from ._resources import CliffordResourceBudget, CliffordResourceEvidence
from ._spec import CliffordAlgebraSpec


__all__ = [
    "CliffordCochainProductPlan",
    "CliffordInverseResult",
    "CliffordMetricField",
    "CliffordProjectorEvidence",
    "ConformalCliffordModel",
    "MinimalLeftIdeal",
    "PinElement",
    "PreparedCauchyCliffordProjector",
    "PreparedCliffordMetricProduct",
    "ProjectiveCliffordModel",
    "SpinElement",
    "invert_multivector",
    "audit_clifford_action",
    "audit_clifford_actions",
    "basis_blade",
    "basis_blade_product",
    "CliffordAlgebraSpec",
    "CliffordBladeLayout",
    "clifford_conjugate",
    "CliffordProductEvidence",
    "CliffordActionAuditReport",
    "CliffordMetricBridge",
    "CliffordOutermorphismPlan",
    "CliffordProductKind",
    "CliffordProductPlan",
    "CliffordResourceBudget",
    "CliffordResourceEvidence",
    "CliffordFiniteAlgebraProvider",
    "FiniteMetricIsometryGroup",
    "embed_layout",
    "extract_layout",
    "grade_involution",
    "grade_layout",
    "lorentz_boost_action",
    "MetricIsometryAction",
    "MetricIsometryAuditSet",
    "prepare_product",
    "project_grades",
    "reverse",
    "scalar_part",
]
