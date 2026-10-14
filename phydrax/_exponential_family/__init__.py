from typing import TYPE_CHECKING

from .._deferred_exports import deferred_exports


#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "CategoricalFamily": "._categorical",
    "DirichletCategoricalConjugacy": "._conjugacy",
    "DirichletCategoricalStatistics": "._conjugacy",
    "DirichletCategoricalUpdate": "._conjugacy",
    "GammaPoissonConjugacy": "._conjugacy",
    "GammaPoissonStatistics": "._conjugacy",
    "GammaPoissonUpdate": "._conjugacy",
    "AbstractExponentialFamily": "._contracts",
    "EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT": "._contracts",
    "EXPONENTIAL_FAMILY_INVALID_EVENT": "._contracts",
    "EXPONENTIAL_FAMILY_MEAN_BOUNDARY": "._contracts",
    "EXPONENTIAL_FAMILY_NONCONVERGED": "._contracts",
    "EXPONENTIAL_FAMILY_NONFINITE": "._contracts",
    "EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN": "._contracts",
    "EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN": "._contracts",
    "exponential_family_status_name": "._contracts",
    "EXPONENTIAL_FAMILY_SUCCESS": "._contracts",
    "ExponentialFamilyConversionResult": "._contracts",
    "ExponentialFamilyDomainResult": "._contracts",
    "ExponentialFamilyLaw": "._contracts",
    "ExponentialFamilySignature": "._contracts",
    "ExponentialFamilyStatus": "._contracts",
    "MeanCoordinates": "._contracts",
    "NaturalCoordinates": "._contracts",
    "StatisticBatch": "._contracts",
    "DirichletFamily": "._dirichlet",
    "BernoulliFamily": "._elementary",
    "ExponentialRateFamily": "._elementary",
    "NormalFamily": "._elementary",
    "PoissonFamily": "._elementary",
    "FiniteSupportExponentialFamily": "._finite_support",
    "FiniteSupportNaturalSolvePlan": "._finite_support",
    "FiniteSupportSolveEvidence": "._finite_support",
    "FiniteSupportSolveResult": "._finite_support",
    "solve_finite_support_mean": "._finite_support",
    "GammaFamily": "._gamma",
    "MultivariateNormalFamily": "._multivariate_normal",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from ._categorical import CategoricalFamily
    from ._conjugacy import (
        DirichletCategoricalConjugacy,
        DirichletCategoricalStatistics,
        DirichletCategoricalUpdate,
        GammaPoissonConjugacy,
        GammaPoissonStatistics,
        GammaPoissonUpdate,
    )
    from ._contracts import (
        AbstractExponentialFamily,
        EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT,
        EXPONENTIAL_FAMILY_INVALID_EVENT,
        EXPONENTIAL_FAMILY_MEAN_BOUNDARY,
        EXPONENTIAL_FAMILY_NONCONVERGED,
        EXPONENTIAL_FAMILY_NONFINITE,
        EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN,
        EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN,
        exponential_family_status_name,
        EXPONENTIAL_FAMILY_SUCCESS,
        ExponentialFamilyConversionResult,
        ExponentialFamilyDomainResult,
        ExponentialFamilyLaw,
        ExponentialFamilySignature,
        ExponentialFamilyStatus,
        MeanCoordinates,
        NaturalCoordinates,
        StatisticBatch,
    )
    from ._dirichlet import DirichletFamily
    from ._elementary import (
        BernoulliFamily,
        ExponentialRateFamily,
        NormalFamily,
        PoissonFamily,
    )
    from ._finite_support import (
        FiniteSupportExponentialFamily,
        FiniteSupportNaturalSolvePlan,
        FiniteSupportSolveEvidence,
        FiniteSupportSolveResult,
        solve_finite_support_mean,
    )
    from ._gamma import GammaFamily
    from ._multivariate_normal import MultivariateNormalFamily


__all__ = [
    "AbstractExponentialFamily",
    "CategoricalFamily",
    "BernoulliFamily",
    "DirichletCategoricalConjugacy",
    "DirichletCategoricalStatistics",
    "DirichletCategoricalUpdate",
    "EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT",
    "EXPONENTIAL_FAMILY_INVALID_EVENT",
    "EXPONENTIAL_FAMILY_MEAN_BOUNDARY",
    "EXPONENTIAL_FAMILY_NONFINITE",
    "EXPONENTIAL_FAMILY_NONCONVERGED",
    "EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN",
    "DirichletFamily",
    "EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN",
    "EXPONENTIAL_FAMILY_SUCCESS",
    "ExponentialFamilyConversionResult",
    "ExponentialFamilyDomainResult",
    "ExponentialFamilyLaw",
    "GammaFamily",
    "GammaPoissonConjugacy",
    "GammaPoissonStatistics",
    "GammaPoissonUpdate",
    "ExponentialFamilySignature",
    "ExponentialFamilyStatus",
    "ExponentialRateFamily",
    "MeanCoordinates",
    "MultivariateNormalFamily",
    "NaturalCoordinates",
    "NormalFamily",
    "PoissonFamily",
    "StatisticBatch",
    "exponential_family_status_name",
    "FiniteSupportExponentialFamily",
    "FiniteSupportNaturalSolvePlan",
    "FiniteSupportSolveEvidence",
    "FiniteSupportSolveResult",
    "solve_finite_support_mean",
]
