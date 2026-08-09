"""Native multiclass and multilabel classifier compositions."""

from ._models import (
    ClassifierChainModel,
    ClassifierChainRecipe,
    CompositionDiagnostics,
    MultilabelModel,
    MultilabelRecipe,
    OneVsOneModel,
    OneVsOneRecipe,
    OneVsRestModel,
    OneVsRestRecipe,
    OutputCodeModel,
    OutputCodeRecipe,
    SmoothClassifierChainModel,
    SmoothClassifierChainRecipe,
)


__all__ = [
    "ClassifierChainModel",
    "ClassifierChainRecipe",
    "CompositionDiagnostics",
    "MultilabelModel",
    "MultilabelRecipe",
    "OneVsOneModel",
    "OneVsOneRecipe",
    "OneVsRestModel",
    "OneVsRestRecipe",
    "OutputCodeModel",
    "OutputCodeRecipe",
    "SmoothClassifierChainModel",
    "SmoothClassifierChainRecipe",
]
