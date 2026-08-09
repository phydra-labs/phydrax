#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native differentiable kernel estimators built on :mod:`phydrax.kernels`."""

from ._approximations import (
    KernelPCAModel,
    KernelPCARecipe,
    NystromModel,
    NystromRecipe,
    RandomFourierFeatureModel,
    RandomFourierFeaturesRecipe,
)
from ._estimators import (
    KernelRidgeModel,
    KernelRidgeRecipe,
    LeastSquaresSVMModel,
    LeastSquaresSVMRecipe,
    OneClassSVMModel,
    OneClassSVMRecipe,
    SupportVectorClassifierModel,
    SupportVectorClassifierRecipe,
    SupportVectorRegressorModel,
    SupportVectorRegressorRecipe,
)
from ._gp import (
    BernoulliGaussianProcessClassifierRecipe,
    CategoricalGaussianProcessClassifierRecipe,
    GaussianProcessClassifierModel,
    GaussianProcessClassifierRecipe,
)


__all__ = [
    "BernoulliGaussianProcessClassifierRecipe",
    "CategoricalGaussianProcessClassifierRecipe",
    "GaussianProcessClassifierModel",
    "GaussianProcessClassifierRecipe",
    "KernelRidgeModel",
    "KernelPCAModel",
    "KernelPCARecipe",
    "KernelRidgeRecipe",
    "LeastSquaresSVMModel",
    "LeastSquaresSVMRecipe",
    "NystromModel",
    "NystromRecipe",
    "OneClassSVMModel",
    "OneClassSVMRecipe",
    "RandomFourierFeatureModel",
    "RandomFourierFeaturesRecipe",
    "SupportVectorClassifierModel",
    "SupportVectorClassifierRecipe",
    "SupportVectorRegressorModel",
    "SupportVectorRegressorRecipe",
]
