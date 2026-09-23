#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx

from ..._strict import StrictModule


ScorerResponse: TypeAlias = Literal[
    "call", "predict", "predict_proba", "decision_function"
]


def _is_immutable_config(value: Any, /) -> bool:
    if value is None or isinstance(value, (str, bool, int, float, complex, bytes)):
        return True
    if isinstance(value, tuple):
        return all(_is_immutable_config(item) for item in value)
    if isinstance(value, frozenset):
        return all(_is_immutable_config(item) for item in value)
    return False


class AbstractScorer(StrictModule):
    """Immutable model-selection scoring protocol without string dispatch.

    Scorers receive predictions first to match model-selection execution, while the
    wrapped metric retains the familiar ``metric(targets, predictions)`` order.
    """

    name: eqx.AbstractVar[str]
    greater_is_better: eqx.AbstractVar[bool]
    response_method: eqx.AbstractVar[ScorerResponse]

    @abstractmethod
    def score(
        self,
        predictions: Any,
        targets: Any,
        /,
        *,
        sample_weight: Any = None,
        mask: Any = None,
    ) -> Any:
        raise NotImplementedError

    def __call__(
        self,
        predictions: Any,
        targets: Any,
        /,
        *,
        sample_weight: Any = None,
        mask: Any = None,
    ) -> Any:
        return self.score(
            predictions,
            targets,
            sample_weight=sample_weight,
            mask=mask,
        )


class FunctionScorer(AbstractScorer):
    """Frozen adapter from a metric callable to :class:`AbstractScorer`."""

    metric: Callable[..., Any] = eqx.field(static=True)
    name: str = eqx.field(static=True)
    greater_is_better: bool = eqx.field(static=True)
    response_method: ScorerResponse = eqx.field(static=True)
    metric_kwargs: tuple[tuple[str, Any], ...] = eqx.field(static=True)

    def __init__(
        self,
        metric: Callable[..., Any],
        /,
        *,
        name: str | None = None,
        greater_is_better: bool,
        response_method: ScorerResponse = "call",
        metric_kwargs: Mapping[str, Any] | None = None,
    ):
        if not callable(metric):
            raise TypeError("metric must be callable.")
        if response_method not in (
            "call",
            "predict",
            "predict_proba",
            "decision_function",
        ):
            raise ValueError(
                "response_method must be 'call', 'predict', 'predict_proba', or "
                "'decision_function'."
            )
        options = () if metric_kwargs is None else tuple(sorted(metric_kwargs.items()))
        for key, value in options:
            if not isinstance(key, str):
                raise TypeError("metric_kwargs keys must be strings.")
            if not _is_immutable_config(value):
                raise TypeError(
                    "metric_kwargs values must be immutable scalar, tuple, or frozenset scorer configuration."
                )
        self.metric = metric
        self.name = type(metric).__name__ if name is None else str(name)
        self.greater_is_better = bool(greater_is_better)
        self.response_method = response_method
        self.metric_kwargs = options

    def score(
        self,
        predictions: Any,
        targets: Any,
        /,
        *,
        sample_weight: Any = None,
        mask: Any = None,
    ) -> Any:
        return self.metric(
            targets,
            predictions,
            sample_weight=sample_weight,
            mask=mask,
            **dict(self.metric_kwargs),
        )


__all__ = ["AbstractScorer", "FunctionScorer", "ScorerResponse"]
