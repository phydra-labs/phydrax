#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx

from .._strict import StrictModule


SPDESolutionConcept: TypeAlias = Literal[
    "strong",
    "mild",
    "variational",
    "martingale",
    "wiener_chaos",
]
SPDENoiseRegularization: TypeAlias = Literal[
    "none",
    "finite_rank",
    "trace_class",
    "space_time_white",
    "distributional",
]
SPDEFormulation: TypeAlias = Literal[
    "pointwise_strong",
    "mild",
    "variational",
    "martingale",
    "wiener_chaos",
]


class SPDESolutionSpec(StrictModule):
    """Declared SPDE solution concept, forcing regularity, and cutoff provenance."""

    concept: SPDESolutionConcept = eqx.field(static=True)
    noise_regularization: SPDENoiseRegularization = eqx.field(static=True)
    cutoff_id: str | None = eqx.field(static=True)
    renormalization: str | None = eqx.field(static=True)

    def __init__(
        self,
        concept: SPDESolutionConcept = "strong",
        /,
        *,
        noise_regularization: SPDENoiseRegularization = "finite_rank",
        cutoff_id: str | None = None,
        renormalization: str | None = None,
    ):
        if concept not in (
            "strong",
            "mild",
            "variational",
            "martingale",
            "wiener_chaos",
        ):
            raise ValueError(
                "concept must be 'strong', 'mild', 'variational', 'martingale', "
                "or 'wiener_chaos'."
            )
        if noise_regularization not in (
            "none",
            "finite_rank",
            "trace_class",
            "space_time_white",
            "distributional",
        ):
            raise ValueError(
                "noise_regularization must be 'none', 'finite_rank', 'trace_class', "
                "'space_time_white', or 'distributional'."
            )
        if cutoff_id is not None and (not isinstance(cutoff_id, str) or not cutoff_id):
            raise ValueError("cutoff_id must be a non-empty string or None.")
        if renormalization is not None and (
            not isinstance(renormalization, str) or not renormalization
        ):
            raise ValueError("renormalization must be a non-empty string or None.")
        self.concept = concept
        self.noise_regularization = noise_regularization
        self.cutoff_id = cutoff_id
        self.renormalization = renormalization

    @property
    def rough_forcing(self) -> bool:
        return self.noise_regularization in ("space_time_white", "distributional")

    @property
    def permits_pointwise_strong_residual(self) -> bool:
        return self.concept == "strong" and not self.rough_forcing

    def supports(self, formulation: SPDEFormulation, /) -> bool:
        if formulation == "pointwise_strong":
            return self.permits_pointwise_strong_residual
        if formulation == "mild":
            return self.concept in ("strong", "mild")
        if formulation == "variational":
            return self.concept in ("strong", "variational")
        if formulation == "martingale":
            return self.concept in ("strong", "martingale")
        if formulation == "wiener_chaos":
            return self.concept in ("strong", "wiener_chaos")
        raise ValueError(
            "formulation must be 'pointwise_strong', 'mild', 'variational', "
            "'martingale', or 'wiener_chaos'."
        )

    def assert_supports(self, formulation: SPDEFormulation, /) -> None:
        if self.supports(formulation):
            return
        if formulation == "pointwise_strong" and self.rough_forcing:
            raise ValueError(
                "A pointwise strong residual is unsupported for unregularized "
                f"{self.noise_regularization} forcing. Declare and construct a mild, "
                "variational, martingale, or Wiener-chaos formulation, or provide an "
                "explicit finite-rank/trace-class cutoff with provenance."
            )
        raise ValueError(
            f"Declared {self.concept!r} SPDE solutions do not support the "
            f"{formulation!r} formulation."
        )


def validate_spde_formulation(
    solution: SPDESolutionSpec,
    formulation: SPDEFormulation,
    /,
) -> SPDESolutionSpec:
    """Validate a residual/objective formulation and return the unchanged spec."""
    if not isinstance(solution, SPDESolutionSpec):
        raise TypeError("solution must be an SPDESolutionSpec.")
    solution.assert_supports(formulation)
    return solution


__all__ = [
    "SPDEFormulation",
    "SPDENoiseRegularization",
    "SPDESolutionConcept",
    "SPDESolutionSpec",
    "validate_spde_formulation",
]
