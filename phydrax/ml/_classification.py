#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import equinox as eqx
import numpy as np

from .._strict import StrictModule
from .._trainable import NonTrainableState


ClassificationObjectiveKind = Literal["nll", "soft_cross_entropy", "focal"]


class ClassificationObjective(StrictModule, NonTrainableState):
    """JSON-safe static configuration for a pointwise classification objective."""

    kind: ClassificationObjectiveKind = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    alpha: float | tuple[float, ...] | None = eqx.field(static=True)
    thresholds: tuple[float, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        kind: ClassificationObjectiveKind = "nll",
        /,
        *,
        gamma: float = 2.0,
        alpha: float | Sequence[float] | None = None,
        thresholds: Sequence[float] | None = None,
    ):
        if kind not in ("nll", "soft_cross_entropy", "focal"):
            raise ValueError(f"Unsupported classification objective {kind!r}.")
        gamma_value = float(gamma)
        if not np.isfinite(gamma_value) or gamma_value < 0.0:
            raise ValueError("gamma must be finite and nonnegative.")
        if alpha is None:
            alpha_value: float | tuple[float, ...] | None = None
        elif np.isscalar(alpha):
            alpha_value = float(alpha)
            if not np.isfinite(alpha_value):
                raise ValueError("alpha must be finite.")
        else:
            alpha_value = tuple(float(value) for value in alpha)
            if not alpha_value or not all(np.isfinite(value) for value in alpha_value):
                raise ValueError("alpha values must be finite and nonempty.")
        if thresholds is None:
            threshold_value = None
        else:
            threshold_value = tuple(float(value) for value in thresholds)
            if len(threshold_value) < 2 or not all(
                np.isfinite(value) for value in threshold_value
            ):
                raise ValueError(
                    "Ordinal thresholds must contain at least two finite values."
                )
            if any(right <= left for left, right in pairwise(threshold_value)):
                raise ValueError("Ordinal thresholds must be strictly increasing.")
        if kind != "focal" and alpha_value is not None:
            raise ValueError("alpha is only defined for focal objectives.")
        self.kind = kind
        self.gamma = gamma_value
        self.alpha = alpha_value
        self.thresholds = threshold_value

    @classmethod
    def nll(
        cls,
        *,
        thresholds: Sequence[float] | None = None,
    ) -> ClassificationObjective:
        return cls("nll", thresholds=thresholds)

    @classmethod
    def soft_cross_entropy(
        cls,
        *,
        thresholds: Sequence[float] | None = None,
    ) -> ClassificationObjective:
        return cls("soft_cross_entropy", thresholds=thresholds)

    @classmethod
    def focal(
        cls,
        *,
        gamma: float = 2.0,
        alpha: float | Sequence[float] | None = None,
    ) -> ClassificationObjective:
        return cls("focal", gamma=gamma, alpha=alpha)

    @property
    def target_encoding(self) -> Literal["hard", "soft"]:
        return "soft" if self.kind == "soft_cross_entropy" else "hard"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "thresholds": self.thresholds,
        }


__all__ = ["ClassificationObjective", "ClassificationObjectiveKind"]
