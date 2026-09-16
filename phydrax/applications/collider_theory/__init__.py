"""Collider-theory provider semantics, uncertainty, and EFT morphing."""

from ._contracts import PerturbativeOrder, TheoryModel, TheoryModelKind, TheoryProcessPlan
from ._prediction import (
    covariance_from_variations,
    EFTMorphingPlan,
    EFTMorphingResult,
    evaluate_eft_morphing,
    TheoryPrediction,
    TheoryVariationMode,
)


__all__ = [
    "EFTMorphingPlan",
    "EFTMorphingResult",
    "PerturbativeOrder",
    "TheoryModel",
    "TheoryModelKind",
    "TheoryPrediction",
    "TheoryProcessPlan",
    "TheoryVariationMode",
    "covariance_from_variations",
    "evaluate_eft_morphing",
]
