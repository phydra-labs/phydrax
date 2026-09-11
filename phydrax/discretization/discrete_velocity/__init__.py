#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._energy_equilibrium import (
    EnergyEquilibriumEvidence,
    EnergyEquilibriumResult,
    EnergyEquilibriumStatus,
    PositiveEnergyEquilibriumPlan,
)
from ._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
    QuadratureMomentCertification,
    VelocityTransportKind,
)
from ._semi_lagrangian import (
    PreparedOffLatticeSemiLagrangianDVM,
    SemiLagrangianTransferRequirements,
    SemiLagrangianTransportEvidence,
)


_SMOOTH_COMPRESSIBLE_EXPORTS = frozenset(
    {
        "SmoothCompressibleCollisionEvidence",
        "SmoothCompressibleD2VKineticMethod",
        "SmoothCompressibleEquilibriumEvidence",
        "SmoothCompressibleKineticState",
        "SmoothCompressibleMoments",
        "SmoothCompressibleRealizabilityEvidence",
        "SmoothCompressibleLearnedCollisionResult",
        "SmoothCompressibleLearnedEquilibriumEvidence",
        "smooth_compressible_d2v17_method",
        "smooth_compressible_d2v37_off_lattice_method",
    }
)
_HYBRID_EXPORTS = frozenset(
    {
        "AtomicHybridUpdateEvidence",
        "AtomicHybridUpdateResult",
        "CommonFVKineticFluxEvidence",
        "ConformingFVKineticState",
        "FixedConformingFVKineticInterfacePlan",
        "KineticShockSensorEvidence",
        "KineticShockSensorPlan",
    }
)


def __getattr__(name: str) -> object:
    if name in _SMOOTH_COMPRESSIBLE_EXPORTS:
        from . import _smooth_compressible

        return getattr(_smooth_compressible, name)
    if name in _HYBRID_EXPORTS:
        from . import _hybrid

        return getattr(_hybrid, name)
    raise AttributeError(name)


__all__ = [
    "EnergyEquilibriumEvidence",
    "EnergyEquilibriumResult",
    "EnergyEquilibriumStatus",
    "CertifiedDiscreteVelocityQuadrature",
    "PreparedOffLatticeSemiLagrangianDVM",
    "QuadratureMomentCertification",
    "SemiLagrangianTransferRequirements",
    "SemiLagrangianTransportEvidence",
    "VelocityTransportKind",
    "PositiveEnergyEquilibriumPlan",
    "d2v17_quadrature",
    "d2v37_off_lattice_quadrature",
]
