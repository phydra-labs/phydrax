#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar numerical terms consumed uniformly by ``FunctionalSolver``.

Conditions state scientific requirements. Terms define how those requirements, data,
or signed functionals contribute a real scalar to optimization or evaluation.
"""

from .._term import (
    AbstractEvaluatedScalarTerm,
    AbstractSamplingTerm,
    AbstractScalarTerm,
    evaluate,
    TermEvaluation,
)
from ._bsde import BSDETerm
from ._cochain import cochain_residual_field, CochainResidualTerm
from ._deep_bsde import (
    deep_bsde_rollout,
    deep_bsde_shooting_diagnostics,
    DeepBSDEPredictor,
    DeepBSDERollout,
    DeepBSDESamplingMode,
    DeepBSDEShootingDiagnostics,
    DeepBSDEShootingTerm,
)
from ._deep_splitting import (
    deep_splitting_labels,
    DeepSplittingLabelBatch,
    DeepSplittingLabelProvider,
    DeepSplittingPredictor,
    DeepSplittingRegressionDiagnostics,
    DeepSplittingRegressionTerm,
)
from ._feynman_kac import (
    FeynmanKacRegressionDiagnostics,
    FeynmanKacRegressionTerm,
    LabelProvider,
)
from ._graph_data import (
    GraphSupervisedTerm,
    GraphTarget,
    GraphTargetInterpolation,
    GraphTrajectorySignal,
    GraphTrajectorySupervisedTerm,
)
from ._integral_functional import IntegralFunctional
from ._likelihood import SupervisedLikelihoodTerm
from ._moment import MomentPenalty
from ._observation import ObservationPenalty
from ._operator_dataset import (
    DifferentialPhysicsInformedOperatorTerm,
    operator_term_suite,
    OperatorDatasetTerm,
    PhysicsInformedOperatorTerm,
)
from ._ragged_series import RaggedSeriesSupervisedBatch, RaggedSeriesSupervisedTerm
from ._ragged_time_series import (
    RaggedTimeSeriesBatch,
    RaggedTimeSeriesDataTerm,
    RaggedTimeSeriesInterpolation,
)
from ._randomized_residual import (
    BatchSampler,
    RandomizedResidualBatch,
    RandomizedResidualDiagnostics,
    RandomizedResidualLossMode,
    RandomizedResidualSamples,
    RandomizedResidualSamplingMode,
    RandomizedResidualTerm,
    ResidualEvaluator,
)
from ._residual import ResidualPenalty
from ._score_matching import (
    ScoreMatchingBatch,
    ScoreMatchingDiagnostics,
    ScoreMatchingMethod,
    ScoreMatchingPolicy,
    ScoreMatchingSamplingMode,
    ScoreMatchingTerm,
    ScoreSampleProvider,
)
from ._supervised_dataset import SupervisedDatasetBatch, SupervisedDatasetTerm
from ._trajectory_data import (
    TrajectoryCaseDataBatch,
    TrajectoryCaseDataTerm,
    TrajectoryCaseTime,
    TrajectorySignal,
    TrajectorySignalInterpolation,
)
from ._transport import (
    BarycenterObjectiveTerm,
    EmpiricalSinkhornDivergenceTerm,
    SlicedWassersteinTerm,
    SoftQuantileFunctional,
    SpatialSinkhornDivergenceTerm,
)
from ._unbalanced_transport import SpatialUnbalancedSinkhornDivergenceTerm


__all__ = [
    "AbstractEvaluatedScalarTerm",
    "AbstractSamplingTerm",
    "AbstractScalarTerm",
    "BSDETerm",
    "BarycenterObjectiveTerm",
    "BatchSampler",
    "CochainResidualTerm",
    "DeepBSDEPredictor",
    "DeepBSDERollout",
    "DeepBSDESamplingMode",
    "DeepBSDEShootingDiagnostics",
    "DeepBSDEShootingTerm",
    "DeepSplittingLabelBatch",
    "DeepSplittingLabelProvider",
    "DeepSplittingPredictor",
    "DeepSplittingRegressionDiagnostics",
    "DeepSplittingRegressionTerm",
    "DifferentialPhysicsInformedOperatorTerm",
    "EmpiricalSinkhornDivergenceTerm",
    "FeynmanKacRegressionDiagnostics",
    "FeynmanKacRegressionTerm",
    "GraphSupervisedTerm",
    "GraphTarget",
    "GraphTargetInterpolation",
    "GraphTrajectorySignal",
    "GraphTrajectorySupervisedTerm",
    "IntegralFunctional",
    "LabelProvider",
    "MomentPenalty",
    "ObservationPenalty",
    "OperatorDatasetTerm",
    "PhysicsInformedOperatorTerm",
    "RaggedSeriesSupervisedBatch",
    "RaggedSeriesSupervisedTerm",
    "RaggedTimeSeriesBatch",
    "RaggedTimeSeriesDataTerm",
    "RaggedTimeSeriesInterpolation",
    "RandomizedResidualBatch",
    "RandomizedResidualDiagnostics",
    "RandomizedResidualLossMode",
    "RandomizedResidualSamples",
    "RandomizedResidualSamplingMode",
    "RandomizedResidualTerm",
    "ResidualEvaluator",
    "ResidualPenalty",
    "ScoreMatchingBatch",
    "ScoreMatchingDiagnostics",
    "ScoreMatchingMethod",
    "ScoreMatchingPolicy",
    "ScoreMatchingSamplingMode",
    "ScoreMatchingTerm",
    "SlicedWassersteinTerm",
    "SoftQuantileFunctional",
    "SpatialSinkhornDivergenceTerm",
    "SpatialUnbalancedSinkhornDivergenceTerm",
    "ScoreSampleProvider",
    "SupervisedDatasetBatch",
    "SupervisedDatasetTerm",
    "SupervisedLikelihoodTerm",
    "TermEvaluation",
    "TrajectoryCaseDataBatch",
    "TrajectoryCaseDataTerm",
    "TrajectoryCaseTime",
    "TrajectorySignal",
    "TrajectorySignalInterpolation",
    "cochain_residual_field",
    "deep_bsde_rollout",
    "deep_bsde_shooting_diagnostics",
    "deep_splitting_labels",
    "evaluate",
    "operator_term_suite",
]
