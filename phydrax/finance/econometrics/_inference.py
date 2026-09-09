#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...uq._forecast_comparison import (
    compare_forecasts as compare_forecast_arrays,
    ForecastComparisonResult as GenericForecastComparisonResult,
)
from ...uq._multiple_testing import (
    adjust_p_values,
    MultipleTestingResult as GenericMultipleTestingResult,
)
from ..core import FinanceEvidenceBinding
from ._experiments import WalkForwardResult


class HypothesisFamilyDefinition(StrictModule):
    """Predeclared test family; identifiers cannot be selected after observing p-values."""

    hypothesis_ids: tuple[str, ...] = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(self, hypothesis_ids: Sequence[str], /):
        identifiers = tuple(str(identifier).strip() for identifier in hypothesis_ids)
        if not identifiers or any(not identifier for identifier in identifiers):
            raise ValueError(
                "hypothesis_ids must be a nonempty sequence of nonempty strings."
            )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("hypothesis_ids must be unique.")
        self.hypothesis_ids = identifiers
        self.family_id = canonical_fingerprint(
            {"kind": "econometric-hypothesis-family", "hypotheses": identifiers}
        )


class MultipleTestingDefinition(StrictModule):
    method: Literal["bonferroni", "holm", "benjamini-hochberg"] = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: Literal["bonferroni", "holm", "benjamini-hochberg"] = "holm",
        alpha: float = 0.05,
    ):
        if method not in ("bonferroni", "holm", "benjamini-hochberg"):
            raise ValueError("unsupported multiple-testing method.")
        level = float(alpha)
        if not 0.0 < level < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")
        self.method = method
        self.alpha = level
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-multiple-testing-definition",
                "method": method,
                "alpha": level,
            }
        )


class MultipleTestingResult(StrictModule):
    result: GenericMultipleTestingResult
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    family_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    experiment_id: str = eqx.field(static=True)


class ForecastComparisonDefinition(StrictModule):
    hac_lags: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    alternative: Literal["two-sided", "less", "greater"] = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        hac_lags: int = 0,
        horizon: int = 1,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ):
        lags = int(hac_lags)
        horizon_ = int(horizon)
        if lags < 0 or horizon_ < 1 or lags < horizon_ - 1:
            raise ValueError("hac_lags must be nonnegative and cover horizon overlap.")
        if alternative not in ("two-sided", "less", "greater"):
            raise ValueError("unsupported forecast-comparison alternative.")
        self.hac_lags = lags
        self.horizon = horizon_
        self.alternative = alternative
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-forecast-comparison-definition",
                "hac_lags": lags,
                "horizon": horizon_,
                "alternative": alternative,
            }
        )


class ForecastComparisonResult(StrictModule):
    result: GenericForecastComparisonResult
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    first_experiment_id: str = eqx.field(static=True)
    second_experiment_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)


def _evidence(
    data_ids: tuple[str, ...], model_id: str, route: str
) -> FinanceEvidenceBinding:
    return FinanceEvidenceBinding(
        tuple(
            canonical_fingerprint({"kind": "inference-data-evidence", "data": data_id})
            for data_id in data_ids
        ),
        (canonical_fingerprint({"kind": "inference-model-evidence", "model": model_id}),),
        (
            canonical_fingerprint(
                {"kind": "inference-numerical-evidence", "route": route}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "inference-use-evidence", "trade_emission": False}
            ),
        ),
    )


def evaluate_multiple_testing(
    p_values: ArrayLike,
    family: HypothesisFamilyDefinition,
    definition: MultipleTestingDefinition,
    experiment_id: str,
    /,
    *,
    mask: ArrayLike | None = None,
) -> MultipleTestingResult:
    """Correct exactly the family declared before the supplied experiment results."""

    if not isinstance(family, HypothesisFamilyDefinition):
        raise TypeError("family must be a HypothesisFamilyDefinition.")
    if not isinstance(definition, MultipleTestingDefinition):
        raise TypeError("definition must be a MultipleTestingDefinition.")
    result = adjust_p_values(
        p_values,
        method=definition.method,
        alpha=definition.alpha,
        mask=mask,
    )
    if result.raw_p_values.shape[-1] != len(family.hypothesis_ids):
        raise ValueError("p-value family width must match the predeclared hypotheses.")
    return MultipleTestingResult(
        result=result,
        evidence=_evidence((str(experiment_id),), family.family_id, definition.method),
        family_id=family.family_id,
        definition_id=definition.definition_id,
        experiment_id=str(experiment_id),
    )


def compare_forecasts(
    first: WalkForwardResult,
    second: WalkForwardResult,
    definition: ForecastComparisonDefinition,
    /,
    *,
    use_test_rows: bool = True,
) -> ForecastComparisonResult:
    """Compare chronological out-of-sample losses from two compatible experiments."""

    if not isinstance(first, WalkForwardResult) or not isinstance(
        second, WalkForwardResult
    ):
        raise TypeError("first and second must be WalkForwardResult values.")
    if not isinstance(definition, ForecastComparisonDefinition):
        raise TypeError("definition must be a ForecastComparisonDefinition.")
    if first.dataset_id != second.dataset_id or first.loss != second.loss:
        raise ValueError(
            "forecast comparisons require the same dataset and loss functional."
        )
    first_mask = first.test_mask if use_test_rows else first.validation_mask
    second_mask = second.test_mask if use_test_rows else second.validation_mask
    mask = first_mask & second_mask
    if bool(jnp.any(jnp.sum(mask, axis=0) > 1)):
        raise ValueError(
            "overlapping folds repeat forecast rows; compare a nonoverlapping evaluation path."
        )
    first_loss = first.losses.reshape(-1)
    second_loss = second.losses.reshape(-1)
    result = compare_forecast_arrays(
        first_loss,
        second_loss,
        mask=mask.reshape(-1),
        hac_lags=definition.hac_lags,
        horizon=definition.horizon,
        alternative=definition.alternative,
    )
    return ForecastComparisonResult(
        result=result,
        evidence=_evidence(
            (first.result_id, second.result_id),
            definition.definition_id,
            "diebold-mariano-hac",
        ),
        first_experiment_id=first.result_id,
        second_experiment_id=second.result_id,
        definition_id=definition.definition_id,
    )


__all__ = [
    "ForecastComparisonDefinition",
    "ForecastComparisonResult",
    "HypothesisFamilyDefinition",
    "MultipleTestingDefinition",
    "MultipleTestingResult",
    "compare_forecasts",
    "evaluate_multiple_testing",
]
