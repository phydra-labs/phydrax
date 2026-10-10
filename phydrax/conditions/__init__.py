#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scientific conditions independent of soft or hard numerical treatment."""

from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from . import (
        cfd,
        conservation,
        electromagnetics,
        free_boundary,
        solids,
        stochastic,
        thermal,
    )
    from ._functional import (
        EventLinearMap,
        linear_functional_condition,
        LinearFunctional,
        LinearReductionAction,
        MatrixLinearFunctional,
        PointJetAction,
    )
    from ._periodic import JetAction, Periodic, PeriodicTraceAction
    from .stochastic import StochasticBoundaryResidual

from ._base import (
    AbstractCondition,
    AbstractMomentCondition,
    AbstractResidualCondition,
    ConditionSupport,
    Moment,
    Observation,
    Residual,
)
from ._evidence import (
    AffineProjectionCertificate,
    ConditionCertificate,
    ConditionEvidence,
    ConditionRealizationStamp,
    FeasibilityCertificate,
    NonlinearRetractionCertificate,
    ProbabilisticConditioningEvidence,
)
from ._ir import (
    AbstractConditionOperator,
    ArrayCodomain,
    CallableConditionOperator,
    Condition,
    ConditionCodomain,
    ConditionQuantifier,
    FieldCodomain,
    FieldSpec,
    OperatorCapabilities,
    OperatorLinearization,
    ProductCodomain,
    ProductFieldSpec,
    ValueAxis,
)
from ._lowering import bind_condition, BoundCondition, lower_condition
from ._relations import (
    AbstractConditionRelation,
    Complementarity,
    ConditionRelation,
    ConeKind,
    ConeMembership,
    Equality,
    Inequality,
    NoisyObservation,
)
from ._subdomain import (
    localize_residual,
    LocalizedResidual,
    subdomain_overlap_consistency,
    SubdomainFluxJump,
    SubdomainTransmission,
    SubdomainValueJump,
)
from ._trace import (
    AbstractJetDeclaration,
    equal,
    field_jet,
    FieldJet,
    JetDeclaration,
    LinearTraceEquation,
    LinearTraceExpression,
    point_jet,
    PointJet,
    trace_jet,
    TraceJet,
)
from .boundary import Absorbing, ConditionValue, Dirichlet, Neumann, Robin
from .initial import Initial


_FUNCTIONAL_EXPORTS = frozenset(
    {
        "EventLinearMap",
        "linear_functional_condition",
        "LinearFunctional",
        "LinearReductionAction",
        "MatrixLinearFunctional",
        "PointJetAction",
    }
)
_PERIODIC_EXPORTS = frozenset({"JetAction", "Periodic", "PeriodicTraceAction"})


def __getattr__(name: str) -> object:
    if name in {
        "cfd",
        "conservation",
        "electromagnetics",
        "free_boundary",
        "solids",
        "stochastic",
        "thermal",
    }:
        value = import_module(f".{name}", __name__)
    elif name == "StochasticBoundaryResidual":
        value = import_module(".stochastic", __name__).StochasticBoundaryResidual
    elif name in _FUNCTIONAL_EXPORTS:
        value = getattr(import_module("._functional", __name__), name)
    elif name in _PERIODIC_EXPORTS:
        value = getattr(import_module("._periodic", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "AbstractCondition",
    "AbstractConditionOperator",
    "AbstractConditionRelation",
    "AbstractJetDeclaration",
    "AbstractMomentCondition",
    "AbstractResidualCondition",
    "Absorbing",
    "AffineProjectionCertificate",
    "ArrayCodomain",
    "bind_condition",
    "BoundCondition",
    "CallableConditionOperator",
    "cfd",
    "Complementarity",
    "Condition",
    "ConditionCertificate",
    "ConditionCodomain",
    "ConditionEvidence",
    "ConditionQuantifier",
    "ConditionRealizationStamp",
    "ConditionRelation",
    "ConditionSupport",
    "ConditionValue",
    "ConeKind",
    "ConeMembership",
    "conservation",
    "Dirichlet",
    "electromagnetics",
    "Equality",
    "equal",
    "FeasibilityCertificate",
    "FieldCodomain",
    "FieldJet",
    "field_jet",
    "FieldSpec",
    "free_boundary",
    "Inequality",
    "Initial",
    "JetDeclaration",
    "JetAction",
    "LinearTraceEquation",
    "LinearTraceExpression",
    "lower_condition",
    "Moment",
    "Neumann",
    "NoisyObservation",
    "NonlinearRetractionCertificate",
    "Observation",
    "OperatorCapabilities",
    "OperatorLinearization",
    "Periodic",
    "PeriodicTraceAction",
    "PointJet",
    "point_jet",
    "ProbabilisticConditioningEvidence",
    "ProductCodomain",
    "ProductFieldSpec",
    "Residual",
    "Robin",
    "LocalizedResidual",
    "localize_residual",
    "SubdomainFluxJump",
    "SubdomainTransmission",
    "SubdomainValueJump",
    "subdomain_overlap_consistency",
    "solids",
    "StochasticBoundaryResidual",
    "stochastic",
    "thermal",
    "TraceJet",
    "trace_jet",
    "ValueAxis",
    "EventLinearMap",
    "LinearFunctional",
    "LinearReductionAction",
    "MatrixLinearFunctional",
    "PointJetAction",
    "linear_functional_condition",
]
