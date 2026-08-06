#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Raw scalar objective terms for functional optimization."""

from .._objective import AbstractObjectiveTerm, AbstractSamplingObjectiveTerm
from ._bsde import BSDEObjective
from ._deep_bsde import (
    deep_bsde_rollout,
    deep_bsde_shooting_diagnostics,
    DeepBSDEPredictor,
    DeepBSDERollout,
    DeepBSDESamplingMode,
    DeepBSDEShootingDiagnostics,
    DeepBSDEShootingObjective,
)
from ._deep_splitting import (
    deep_splitting_labels,
    DeepSplittingLabelBatch,
    DeepSplittingLabelProvider,
    DeepSplittingPredictor,
    DeepSplittingRegressionDiagnostics,
    DeepSplittingRegressionObjective,
)
from ._feynman_kac import (
    FeynmanKacRegressionDiagnostics,
    FeynmanKacRegressionObjective,
    LabelProvider,
)
from ._integral import IntegralFunctional
from ._randomized_residual import (
    BatchSampler,
    RandomizedResidualBatch,
    RandomizedResidualDiagnostics,
    RandomizedResidualLossMode,
    RandomizedResidualObjective,
    RandomizedResidualSamples,
    RandomizedResidualSamplingMode,
    ResidualEvaluator,
)
from ._score_matching import (
    ScoreMatchingBatch,
    ScoreMatchingDiagnostics,
    ScoreMatchingMethod,
    ScoreMatchingObjective,
    ScoreMatchingPolicy,
    ScoreMatchingSamplingMode,
    ScoreSampleProvider,
)


__all__ = [
    "BSDEObjective",
    "BatchSampler",
    "deep_bsde_rollout",
    "deep_bsde_shooting_diagnostics",
    "DeepBSDEPredictor",
    "DeepBSDERollout",
    "DeepBSDESamplingMode",
    "DeepBSDEShootingDiagnostics",
    "DeepBSDEShootingObjective",
    "deep_splitting_labels",
    "DeepSplittingLabelBatch",
    "DeepSplittingLabelProvider",
    "DeepSplittingPredictor",
    "DeepSplittingRegressionDiagnostics",
    "DeepSplittingRegressionObjective",
    "FeynmanKacRegressionDiagnostics",
    "FeynmanKacRegressionObjective",
    "LabelProvider",
    "RandomizedResidualBatch",
    "RandomizedResidualDiagnostics",
    "RandomizedResidualLossMode",
    "RandomizedResidualObjective",
    "RandomizedResidualSamples",
    "RandomizedResidualSamplingMode",
    "ResidualEvaluator",
    "ScoreMatchingBatch",
    "ScoreMatchingDiagnostics",
    "ScoreMatchingMethod",
    "ScoreMatchingObjective",
    "ScoreMatchingPolicy",
    "ScoreMatchingSamplingMode",
    "ScoreSampleProvider",
    "AbstractObjectiveTerm",
    "AbstractSamplingObjectiveTerm",
    "IntegralFunctional",
]
