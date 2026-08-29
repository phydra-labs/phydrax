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
from ._classification import (
    SupervisedClassificationTerm,
    SupervisedFocalClassificationTerm,
    SupervisedOrdinalClassificationTerm,
    SupervisedSoftClassificationTerm,
)
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
from ._dense_classification import (
    DenseOverlapClassificationTerm,
    DenseSiteClassificationBatch,
    DenseSiteClassificationTerm,
)
from ._factorized_variational_eigenspace import (
    factorized_variational_eigenspace,
    FactorizedVariationalEigenspaceResult,
)
from ._feynman_kac import (
    FeynmanKacRegressionDiagnostics,
    FeynmanKacRegressionTerm,
    LabelProvider,
)
from ._flow_matching import (
    AbstractFlowMatchingMetric,
    EuclideanFlowMatchingMetric,
    FlowEndpointProvider,
    FlowMatchingBatch,
    FlowMatchingDiagnostics,
    FlowMatchingPolicy,
    FlowMatchingSamplingMode,
    FlowMatchingTerm,
    ManifoldFlowMatchingMetric,
    RiemannianFlowMatchingMetric,
)
from ._graph_classification import (
    GraphClassificationReduction,
    GraphClassificationTarget,
    GraphClassificationTargetEncoding,
    GraphClassificationTerm,
    GraphTrajectoryClassificationSignal,
    GraphTrajectoryClassificationTerm,
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
from ._randomized_moment import (
    RandomizedMomentBatch,
    RandomizedMomentDiagnostics,
    RandomizedMomentPenalty,
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
from ._ricci_flat import ricci_flat_kahler_term
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
from ._trajectory_classification import (
    RaggedTimeSeriesClassificationBatch,
    RaggedTimeSeriesClassificationTerm,
    TrajectoryCaseClassificationBatch,
    TrajectoryCaseClassificationTerm,
    TrajectoryClassificationMeasure,
)
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
from ._variational_eigenspace import (
    FormDensity,
    VariationalEigenspace,
    VariationalEigenspaceEvaluation,
    VariationalEigenspaceResult,
)


__all__ = [
    "AbstractEvaluatedScalarTerm",
    "AbstractFlowMatchingMetric",
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
    "FactorizedVariationalEigenspaceResult",
    "factorized_variational_eigenspace",
    "EuclideanFlowMatchingMetric",
    "RiemannianFlowMatchingMetric",
    "ManifoldFlowMatchingMetric",
    "ricci_flat_kahler_term",
    "FeynmanKacRegressionDiagnostics",
    "FeynmanKacRegressionTerm",
    "FlowEndpointProvider",
    "DenseOverlapClassificationTerm",
    "DenseSiteClassificationBatch",
    "DenseSiteClassificationTerm",
    "FlowMatchingBatch",
    "FlowMatchingDiagnostics",
    "FlowMatchingPolicy",
    "FlowMatchingSamplingMode",
    "FlowMatchingTerm",
    "GraphSupervisedTerm",
    "GraphTarget",
    "GraphClassificationReduction",
    "GraphClassificationTarget",
    "GraphClassificationTargetEncoding",
    "GraphClassificationTerm",
    "GraphTrajectoryClassificationSignal",
    "GraphTrajectoryClassificationTerm",
    "GraphTargetInterpolation",
    "GraphTrajectorySignal",
    "GraphTrajectorySupervisedTerm",
    "IntegralFunctional",
    "FormDensity",
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
    "RandomizedMomentBatch",
    "RandomizedMomentDiagnostics",
    "RandomizedMomentPenalty",
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
    "VariationalEigenspace",
    "VariationalEigenspaceEvaluation",
    "VariationalEigenspaceResult",
    "SoftQuantileFunctional",
    "SpatialSinkhornDivergenceTerm",
    "SpatialUnbalancedSinkhornDivergenceTerm",
    "RaggedTimeSeriesClassificationBatch",
    "RaggedTimeSeriesClassificationTerm",
    "ScoreSampleProvider",
    "SupervisedClassificationTerm",
    "SupervisedFocalClassificationTerm",
    "SupervisedOrdinalClassificationTerm",
    "SupervisedSoftClassificationTerm",
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
    "TrajectoryCaseClassificationBatch",
    "TrajectoryCaseClassificationTerm",
    "TrajectoryClassificationMeasure",
    "operator_term_suite",
]
