"""Native weighted discriminant-analysis classifiers."""

from ._models import (
    DiscriminantDiagnostics,
    LinearDiscriminantModel,
    LinearDiscriminantRecipe,
    QuadraticDiscriminantModel,
    QuadraticDiscriminantRecipe,
    RegularizedDiscriminantRecipe,
    ShrinkageDiscriminantRecipe,
)


__all__ = [
    "DiscriminantDiagnostics",
    "LinearDiscriminantModel",
    "LinearDiscriminantRecipe",
    "QuadraticDiscriminantModel",
    "QuadraticDiscriminantRecipe",
    "RegularizedDiscriminantRecipe",
    "ShrinkageDiscriminantRecipe",
]
