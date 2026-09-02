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
    GradientContract,
    ML_CAPACITY_EXHAUSTED,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_RANK_DEFICIENT,
    ML_SUCCESS,
    ML_UNSUPPORTED_GRADIENT,
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
    "AbstractRecipe",
    "artifacts",
    "calibration",
    "clustering",
    "compose",
    "covariance",
    "ClassificationObjective",
    "ClassificationObjectiveKind",
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
    "naive_bayes",
    "neighbors",
    "dice_score",
    "jaccard_score",
    "overlap_score",
    "OverlapClassReduction",
    "OverlapEmptyPolicy",
    "OverlapKind",
    "OverlapScoreConfig",
    "reduce_overlap_score",
    "tversky_score",
    "quantum",
    "outliers",
    "preprocessing",
    "semi_supervised",
    "tree",
    "FeatureKind",
    "FeatureSchema",
    "FitDiagnostics",
    "FitResult",
    "gumbel_softmax",
    "relaxed_bernoulli",
    "RelaxedDiscreteSample",
    "relaxed_top_k",
    "GradientContract",
    "MLBatch",
    "ML_CAPACITY_EXHAUSTED",
    "ML_INFEASIBLE",
    "ML_INSUFFICIENT_DATA",
    "ML_NONCONVERGED",
    "ML_NONFINITE",
    "ML_RANK_DEFICIENT",
    "ML_SUCCESS",
    "masked_softmax",
    "ML_UNSUPPORTED_GRADIENT",
    "SparseFeatures",
    "soft_ranks",
    "soft_topk_weights",
    "TargetKind",
    "TargetSchema",
    "WeightPolicy",
    "temperature_sigmoid",
    "temperature_softmax",
    "fit",
]
