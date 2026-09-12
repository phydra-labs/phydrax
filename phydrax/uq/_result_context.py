#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


ResultNormalization = Literal["absolute", "noise-relative", "unnormalized"]


def _identifier(value: str, role: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{role} must be non-empty.")
    return identifier


class UQResultContext(StrictModule, NonTrainableState):
    """Portable scientific identities intentionally excluded from live result objects."""

    analysis_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    likelihood_id: str = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    data_ids: tuple[str, ...] = eqx.field(static=True)
    provider_ids: tuple[str, ...] = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    normalization: ResultNormalization = eqx.field(static=True)
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        analysis_id: str,
        problem_id: str,
        likelihood_id: str,
        parameterization_id: str,
        data_ids: tuple[str, ...],
        provider_ids: tuple[str, ...],
        approximation_id: str,
        normalization: ResultNormalization,
    ):
        if normalization not in ("absolute", "noise-relative", "unnormalized"):
            raise ValueError("Unknown result normalization semantics.")
        identifiers = tuple(
            _identifier(value, role)
            for value, role in (
                (analysis_id, "analysis ID"),
                (problem_id, "problem ID"),
                (likelihood_id, "likelihood ID"),
                (parameterization_id, "parameterization ID"),
                (approximation_id, "approximation ID"),
            )
        )
        data = tuple(_identifier(value, "data ID") for value in data_ids)
        providers = tuple(_identifier(value, "provider ID") for value in provider_ids)
        if (
            not data
            or not providers
            or len(set(data)) != len(data)
            or len(set(providers)) != len(providers)
        ):
            raise ValueError(
                "Result context requires unique data and provider identities."
            )
        (
            self.analysis_id,
            self.problem_id,
            self.likelihood_id,
            self.parameterization_id,
            self.approximation_id,
        ) = identifiers
        self.data_ids = data
        self.provider_ids = providers
        self.normalization = normalization
        self.context_id = canonical_fingerprint(
            {
                "kind": "uq-result-context",
                "analysis": self.analysis_id,
                "problem": self.problem_id,
                "likelihood": self.likelihood_id,
                "parameterization": self.parameterization_id,
                "data": list(data),
                "providers": list(providers),
                "approximation": self.approximation_id,
                "normalization": normalization,
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "analysis_id": self.analysis_id,
            "problem_id": self.problem_id,
            "likelihood_id": self.likelihood_id,
            "parameterization_id": self.parameterization_id,
            "data_ids": list(self.data_ids),
            "provider_ids": list(self.provider_ids),
            "approximation_id": self.approximation_id,
            "normalization": self.normalization,
            "context_id": self.context_id,
        }


__all__ = ["ResultNormalization", "UQResultContext"]
