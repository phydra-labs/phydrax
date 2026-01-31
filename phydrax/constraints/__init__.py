#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""
# Constraints

Constraints define objective terms for training or evaluation. They operate on
domain functions and typically return a scalar loss term $\\ell(u)$.

## Categories

- **Pointwise** constraints for PDE residuals and boundary conditions.
- **Integral** constraints for global conservation or averages.
- **Discrete** constraints for sensors and labeled data.
- **Enforced** constraints that build an ansatz satisfying boundary or initial data.

## Typical loss form

Given constraints $\\ell_i$, a solver builds
$L = \\sum_i w_i \\, \\ell_i$.

!!! example
    ```python
    import phydrax as phx

    geom = phx.domain.Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    structure = phx.domain.ProductStructure((("x",),))
    constraint = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "u",
        geom,
        operator=lambda f: phx.operators.laplacian(f, var="x"),
        num_points=64,
        structure=structure,
    )
    ```
"""

from ._bc_cfd import (
    ContinuousNoPenetrationBoundaryConstraint,
    ContinuousSlipWallBoundaryConstraint,
    ContinuousSymmetryVelocityBoundaryConstraint,
    DiscreteNoPenetrationBoundaryConstraint,
    DiscreteZeroNormalGradientVelocityBoundaryConstraint,
)
from ._bc_em import (
    ContinuousElectricSurfaceChargeBoundaryConstraint,
    ContinuousImpedanceBoundaryConstraint,
    ContinuousInterfaceNormalBContinuityConstraint,
    ContinuousInterfaceNormalDJumpConstraint,
    ContinuousInterfaceTangentialEContinuityConstraint,
    ContinuousInterfaceTangentialHJumpConstraint,
    ContinuousMagneticSurfaceCurrentBoundaryConstraint,
    ContinuousPECBoundaryConstraint,
    ContinuousPMCBoundaryConstraint,
    DiscreteElectricSurfaceChargeBoundaryConstraint,
    DiscreteInterfaceNormalBContinuityConstraint,
    DiscreteInterfaceNormalDJumpConstraint,
    DiscreteInterfaceTangentialEContinuityConstraint,
    DiscreteInterfaceTangentialHJumpConstraint,
    DiscreteMagneticSurfaceCurrentBoundaryConstraint,
    DiscretePECBoundaryConstraint,
    DiscretePMCBoundaryConstraint,
)
from ._bc_solid import (
    ContinuousElasticFoundationBoundaryConstraint,
    ContinuousElasticSymmetryBoundaryConstraint,
    ContinuousNormalDisplacementBoundaryConstraint,
    ContinuousTractionBoundaryConstraint,
    DiscreteDisplacementBoundaryConstraint,
    DiscreteNormalDisplacementBoundaryConstraint,
    DiscreteTractionBoundaryConstraint,
)
from ._bc_thermal import (
    ContinuousConvectionBoundaryConstraint,
    ContinuousHeatFluxBoundaryConstraint,
    DiscreteConvectionBoundaryConstraint,
    DiscreteHeatFluxBoundaryConstraint,
    DiscreteRobinBoundaryConstraint,
)
from ._continuous_interior import (
    ContinuousInitialFunctionConstraint,
    ContinuousPointwiseInteriorConstraint,
)
from ._discrete_interior import (
    DiscreteInteriorDataConstraint,
)
from ._enforced import (
    enforce_blend,
    enforce_dirichlet,
    enforce_initial,
    enforce_neumann,
    enforce_robin,
    enforce_sommerfeld,
    enforce_traction,
)
from ._functional import FunctionalConstraint
from ._functional_boundary import (
    AbsorbingBoundaryConstraint,
    ContinuousDirichletBoundaryConstraint,
    ContinuousNeumannBoundaryConstraint,
    ContinuousRobinBoundaryConstraint,
    DiscreteDirichletBoundaryConstraint,
    DiscreteNeumannBoundaryConstraint,
)
from ._functional_initial import (
    ContinuousInitialConstraint,
    DiscreteInitialConstraint,
)
from ._functional_integral import IntegralEqualityConstraint
from ._integral import (
    AveragePressureBoundaryConstraint,
    CFDBoundaryFlowRateConstraint,
    CFDKineticEnergyFluxBoundaryConstraint,
    ContinuousIntegralBoundaryConstraint,
    ContinuousIntegralInitialConstraint,
    ContinuousIntegralInteriorConstraint,
    EMBoundaryChargeConstraint,
    EMPoyntingFluxBoundaryConstraint,
    MagneticFluxZeroConstraint,
    SolidTotalReactionBoundaryConstraint,
)
from ._ode import (
    ContinuousODEConstraint,
    DiscreteODEConstraint,
    DiscreteTimeDataConstraint,
    InitialODEConstraint,
)
from ._pointset import PointSetConstraint


__all__ = [
    "FunctionalConstraint",
    "PointSetConstraint",
    "IntegralEqualityConstraint",
    "ContinuousPointwiseInteriorConstraint",
    "ContinuousInitialFunctionConstraint",
    "ContinuousIntegralInteriorConstraint",
    "ContinuousIntegralBoundaryConstraint",
    "ContinuousIntegralInitialConstraint",
    "EMBoundaryChargeConstraint",
    "MagneticFluxZeroConstraint",
    "CFDBoundaryFlowRateConstraint",
    "SolidTotalReactionBoundaryConstraint",
    "AveragePressureBoundaryConstraint",
    "EMPoyntingFluxBoundaryConstraint",
    "CFDKineticEnergyFluxBoundaryConstraint",
    "ContinuousODEConstraint",
    "DiscreteODEConstraint",
    "DiscreteTimeDataConstraint",
    "InitialODEConstraint",
    "ContinuousDirichletBoundaryConstraint",
    "ContinuousNeumannBoundaryConstraint",
    "ContinuousRobinBoundaryConstraint",
    "AbsorbingBoundaryConstraint",
    "DiscreteDirichletBoundaryConstraint",
    "DiscreteNeumannBoundaryConstraint",
    "ContinuousInitialConstraint",
    "DiscreteInitialConstraint",
    "ContinuousSymmetryVelocityBoundaryConstraint",
    "ContinuousNoPenetrationBoundaryConstraint",
    "ContinuousSlipWallBoundaryConstraint",
    "DiscreteNoPenetrationBoundaryConstraint",
    "DiscreteZeroNormalGradientVelocityBoundaryConstraint",
    "ContinuousTractionBoundaryConstraint",
    "ContinuousNormalDisplacementBoundaryConstraint",
    "ContinuousElasticFoundationBoundaryConstraint",
    "ContinuousElasticSymmetryBoundaryConstraint",
    "DiscreteDisplacementBoundaryConstraint",
    "DiscreteTractionBoundaryConstraint",
    "DiscreteNormalDisplacementBoundaryConstraint",
    "ContinuousHeatFluxBoundaryConstraint",
    "ContinuousConvectionBoundaryConstraint",
    "DiscreteRobinBoundaryConstraint",
    "DiscreteHeatFluxBoundaryConstraint",
    "DiscreteConvectionBoundaryConstraint",
    "ContinuousPECBoundaryConstraint",
    "ContinuousImpedanceBoundaryConstraint",
    "ContinuousPMCBoundaryConstraint",
    "ContinuousElectricSurfaceChargeBoundaryConstraint",
    "ContinuousMagneticSurfaceCurrentBoundaryConstraint",
    "ContinuousInterfaceTangentialEContinuityConstraint",
    "ContinuousInterfaceNormalDJumpConstraint",
    "ContinuousInterfaceTangentialHJumpConstraint",
    "ContinuousInterfaceNormalBContinuityConstraint",
    "DiscretePECBoundaryConstraint",
    "DiscretePMCBoundaryConstraint",
    "DiscreteElectricSurfaceChargeBoundaryConstraint",
    "DiscreteMagneticSurfaceCurrentBoundaryConstraint",
    "DiscreteInterfaceTangentialEContinuityConstraint",
    "DiscreteInterfaceNormalDJumpConstraint",
    "DiscreteInterfaceTangentialHJumpConstraint",
    "DiscreteInterfaceNormalBContinuityConstraint",
    "DiscreteInteriorDataConstraint",
    "enforce_blend",
    "enforce_dirichlet",
    "enforce_initial",
    "enforce_neumann",
    "enforce_robin",
    "enforce_sommerfeld",
    "enforce_traction",
]
