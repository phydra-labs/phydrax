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
from ._adversarial import (
    AdversarialEvaluation,
    ImplicitGenerator,
    wasserstein_adversarial_evaluation,
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
from ._denoising_score_matching import (
    DenoisingScoreDataProvider,
    DenoisingScoreMatchingBatch,
    DenoisingScoreMatchingDiagnostics,
    DenoisingScoreMatchingTerm,
    DenoisingScoreSamplingMode,
    DenoisingScoreWeighting,
)
from ._dense_classification import (
    DenseOverlapClassificationTerm,
    DenseSiteClassificationBatch,
    DenseSiteClassificationTerm,
)
from ._diffusion_bridge import DiffusionBridgeControlDataset, DiffusionBridgeDriftTerm
from ._energy_model import (
    EnergyTarget,
    PersistentContrastiveDivergence,
    PersistentEnergyState,
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
from ._interface import (
    free_boundary_term_suite,
    implicit_interface_penalty,
    implicit_phase_penalty,
)
from ._likelihood import SupervisedLikelihoodTerm
from ._modal import (
    CompiledModalResidualTerm,
    ModalObservationTerm,
    ModalTimeProvider,
)
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
from ._residual_layout import ResidualBlockLayout, ResidualBlockRef
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
from ._target_consistency import TargetConsistencyTerm
from ._time_sampling import UniformTimeSamplingPolicy
from ._topology import FrozenTopologyTerm
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
from ._transport_learning import (
    audit_transport_map,
    LearnedTransportAudit,
    MongeMapTerm,
    NeuralDualTransportTerm,
)
from ._unbalanced_transport import SpatialUnbalancedSinkhornDivergenceTerm
from ._variational_eigenspace import (
    EigenspaceAction,
    FormDensity,
    InvariantSubspaceResidual,
    InvariantSubspaceResidualEvaluation,
    InvariantSubspaceResidualResult,
    VariationalEigenspace,
    VariationalEigenspaceEvaluation,
    VariationalEigenspaceResult,
)
from ._variational_functional import bind_functional


__all__ = [
    "bind_functional",
    "AbstractEvaluatedScalarTerm",
    "AbstractFlowMatchingMetric",
    "AbstractSamplingTerm",
    "AbstractScalarTerm",
    "AdversarialEvaluation",
    "ImplicitGenerator",
    "EnergyTarget",
    "PersistentContrastiveDivergence",
    "PersistentEnergyState",
    "wasserstein_adversarial_evaluation",
    "FrozenTopologyTerm",
    "BSDETerm",
    "BarycenterObjectiveTerm",
    "BatchSampler",
    "CochainResidualTerm",
    "CompiledModalResidualTerm",
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
    "DenoisingScoreDataProvider",
    "DenoisingScoreMatchingBatch",
    "DenoisingScoreMatchingDiagnostics",
    "DenoisingScoreMatchingTerm",
    "DenoisingScoreSamplingMode",
    "DenoisingScoreWeighting",
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
    "EigenspaceAction",
    "FlowEndpointProvider",
    "DenseOverlapClassificationTerm",
    "DenseSiteClassificationBatch",
    "DenseSiteClassificationTerm",
    "FlowMatchingBatch",
    "FlowMatchingDiagnostics",
    "FlowMatchingSamplingMode",
    "FlowMatchingTerm",
    "free_boundary_term_suite",
    "implicit_interface_penalty",
    "implicit_phase_penalty",
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
    "TargetConsistencyTerm",
    "FormDensity",
    "InvariantSubspaceResidual",
    "InvariantSubspaceResidualEvaluation",
    "InvariantSubspaceResidualResult",
    "LabelProvider",
    "ModalObservationTerm",
    "ModalTimeProvider",
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
    "ResidualBlockLayout",
    "ResidualBlockRef",
    "ResidualPenalty",
    "ScoreMatchingBatch",
    "ScoreMatchingDiagnostics",
    "ScoreMatchingMethod",
    "ScoreMatchingPolicy",
    "ScoreMatchingSamplingMode",
    "ScoreMatchingTerm",
    "UniformTimeSamplingPolicy",
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
    "DiffusionBridgeControlDataset",
    "DiffusionBridgeDriftTerm",
    "LearnedTransportAudit",
    "MongeMapTerm",
    "NeuralDualTransportTerm",
    "audit_transport_map",
]
