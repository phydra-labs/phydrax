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
- **Discrete** constraints for sensors, labeled data, and ragged trajectories.
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

from ._adaptive import (
    AbstractCollocationPolicy,
    CollocationPolicy,
    CollocationPopulation,
    PeriodicCollocation,
    R3,
    RARD,
    with_collocation_policy,
)
from ._adaptive_control import (
    AdaptationBudget,
    COLLOCATION_POLICY_SUPPORT,
    collocation_policy_support,
    CollocationDefaults,
    CollocationPolicySupport,
    controlled_collocation,
    ControlledCollocationPolicy,
    ControlledCollocationPopulation,
    CoverageAnchors,
    PolicySupportTier,
    RECOMMENDED_COLLOCATION_DEFAULTS,
    RefreshGuard,
    RefreshSchedule,
    ResidualMonitor,
)
from ._adaptive_separable import (
    HierarchicalAxisCollocation,
    HierarchicalAxisPolicy,
    PeriodicSeparableCollocation,
    SeparableCollocationPolicy,
    SeparableCollocationPopulation,
)
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
from ._cochain import cochain_residual_field, CochainResidualConstraint
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
from ._graph_data import (
    GraphSupervisedConstraint,
    GraphTarget,
    GraphTargetInterpolation,
    GraphTrajectorySignal,
    GraphTrajectorySupervisedConstraint,
)
from ._graph_enforced import enforce_cochain_values, enforce_graph_values
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
from ._likelihood import SupervisedLikelihoodConstraint
from ._ode import (
    ContinuousODEConstraint,
    DiscreteODEConstraint,
    DiscreteTimeDataConstraint,
    InitialODEConstraint,
)
from ._operator_dataset import (
    DifferentialPhysicsInformedOperatorConstraint,
    operator_constraint_suite,
    OperatorDatasetConstraint,
    PhysicsInformedOperatorConstraint,
)
from ._pointset import PointSetConstraint
from ._ragged_series import (
    RaggedSeriesSupervisedBatch,
    RaggedSeriesSupervisedConstraint,
)
from ._ragged_time_series import (
    RaggedTimeSeriesBatch,
    RaggedTimeSeriesDataConstraint,
)
from ._ragged_time_series_enforced import (
    enforce_ragged_time_series,
    RaggedTimeSeriesHardGate,
    RaggedTimeSeriesHardInterpolation,
)
from ._supervised_dataset import (
    SupervisedDatasetBatch,
    SupervisedDatasetConstraint,
)
from ._trajectory_data import (
    TrajectoryCaseDataBatch,
    TrajectoryCaseDataConstraint,
    TrajectoryCaseTime,
    TrajectorySignal,
    TrajectorySignalInterpolation,
)


__all__ = [
    "CochainResidualConstraint",
    "cochain_residual_field",
    "AbstractCollocationPolicy",
    "CollocationPolicy",
    "CollocationPopulation",
    "PeriodicCollocation",
    "R3",
    "RARD",
    "with_collocation_policy",
    "AdaptationBudget",
    "COLLOCATION_POLICY_SUPPORT",
    "CollocationDefaults",
    "CollocationPolicySupport",
    "ControlledCollocationPolicy",
    "ControlledCollocationPopulation",
    "CoverageAnchors",
    "PolicySupportTier",
    "RECOMMENDED_COLLOCATION_DEFAULTS",
    "RefreshGuard",
    "RefreshSchedule",
    "ResidualMonitor",
    "collocation_policy_support",
    "controlled_collocation",
    "HierarchicalAxisCollocation",
    "HierarchicalAxisPolicy",
    "PeriodicSeparableCollocation",
    "SeparableCollocationPolicy",
    "SeparableCollocationPopulation",
    "FunctionalConstraint",
    "PointSetConstraint",
    "RaggedTimeSeriesBatch",
    "RaggedTimeSeriesDataConstraint",
    "RaggedSeriesSupervisedBatch",
    "RaggedSeriesSupervisedConstraint",
    "RaggedTimeSeriesHardGate",
    "RaggedTimeSeriesHardInterpolation",
    "enforce_ragged_time_series",
    "TrajectoryCaseDataBatch",
    "TrajectoryCaseDataConstraint",
    "TrajectoryCaseTime",
    "TrajectorySignal",
    "TrajectorySignalInterpolation",
    "SupervisedDatasetBatch",
    "SupervisedDatasetConstraint",
    "DifferentialPhysicsInformedOperatorConstraint",
    "OperatorDatasetConstraint",
    "PhysicsInformedOperatorConstraint",
    "operator_constraint_suite",
    "GraphSupervisedConstraint",
    "GraphTarget",
    "GraphTargetInterpolation",
    "GraphTrajectorySignal",
    "GraphTrajectorySupervisedConstraint",
    "enforce_cochain_values",
    "enforce_graph_values",
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
    "SupervisedLikelihoodConstraint",
    "enforce_blend",
    "enforce_dirichlet",
    "enforce_initial",
    "enforce_neumann",
    "enforce_robin",
    "enforce_sommerfeld",
    "enforce_traction",
]
