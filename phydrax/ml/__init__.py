"""Differentiable classical machine learning for scientific JAX workflows."""

import importlib
from types import ModuleType

from . import _numerics as _numerics
from ._batch import MLBatch, WeightPolicy
from ._classification import ClassificationObjective, ClassificationObjectiveKind
from ._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    ML_CAPACITY_EXHAUSTED,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_RANK_DEFICIENT,
    ML_SUCCESS,
)
from ._fit import fit
from ._overlap import (
    dice_score,
    jaccard_score,
    overlap_score,
    OverlapClassReduction,
    OverlapEmptyPolicy,
    OverlapKind,
    OverlapScoreConfig,
    reduce_overlap_score,
    tversky_score,
)
from ._schema import FeatureKind, FeatureSchema, TargetKind, TargetSchema
from ._soft_discrete import (
    gumbel_softmax,
    masked_softmax,
    relaxed_bernoulli,
    relaxed_top_k,
    RelaxedDiscreteSample,
    soft_ranks,
    soft_topk_weights,
    temperature_sigmoid,
    temperature_softmax,
)
from ._sparse_features import SparseFeatures


_ML_SUBMODULES = frozenset(
    {
        "artifacts",
        "calibration",
        "clustering",
        "compose",
        "covariance",
        "decomposition",
        "discriminant",
        "ensemble",
        "feature_selection",
        "inspection",
        "interop",
        "kernel_methods",
        "linear",
        "manifold",
        "metrics",
        "mixture",
        "model_selection",
        "multiclass",
        "quantum",
        "naive_bayes",
        "optimization",
        "neighbors",
        "outliers",
        "preprocessing",
        "semi_supervised",
        "tree",
    }
)


def __getattr__(name: str) -> ModuleType:
    if name not in _ML_SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


__all__ = [
    "ML_CAPACITY_EXHAUSTED",
    "ML_INFEASIBLE",
    "ML_INSUFFICIENT_DATA",
    "ML_NONCONVERGED",
    "ML_NONFINITE",
    "ML_RANK_DEFICIENT",
    "ML_SUCCESS",
    "AbstractRecipe",
    "ClassificationObjective",
    "ClassificationObjectiveKind",
    "FeatureKind",
    "FeatureSchema",
    "FitDiagnostics",
    "FitResult",
    "MLBatch",
    "OverlapClassReduction",
    "OverlapEmptyPolicy",
    "OverlapKind",
    "OverlapScoreConfig",
    "RelaxedDiscreteSample",
    "SparseFeatures",
    "TargetKind",
    "TargetSchema",
    "WeightPolicy",
    "artifacts",
    "calibration",
    "clustering",
    "compose",
    "covariance",
    "decomposition",
    "dice_score",
    "discriminant",
    "ensemble",
    "feature_selection",
    "fit",
    "gumbel_softmax",
    "inspection",
    "interop",
    "jaccard_score",
    "kernel_methods",
    "linear",
    "manifold",
    "masked_softmax",
    "metrics",
    "mixture",
    "model_selection",
    "multiclass",
    "naive_bayes",
    "neighbors",
    "optimization",
    "outliers",
    "overlap_score",
    "preprocessing",
    "quantum",
    "reduce_overlap_score",
    "relaxed_bernoulli",
    "relaxed_top_k",
    "semi_supervised",
    "soft_ranks",
    "soft_topk_weights",
    "temperature_sigmoid",
    "temperature_softmax",
    "tree",
    "tversky_score",
]
