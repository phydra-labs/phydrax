#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any, Protocol

from jaxtyping import Array

from ..._doc import DOC_KEY0
from .capabilities import ConfiguredOperatorContract
from .data import FunctionSamples, OperatorBatch, OperatorOutputSpec, OperatorPrediction


class OperatorPredictionBuilder(Protocol):
    """Construct a named prediction from one engine and canonical batch."""

    def __call__(
        self,
        model: Any,
        batch: OperatorBatch,
        key: Any,
        /,
    ) -> OperatorPrediction: ...


class OperatorModel(abc.ABC):
    """Named neural-operator engine contract over canonical operator batches."""

    @property
    @abc.abstractmethod
    def operator_contract(self) -> ConfiguredOperatorContract:
        """Return the configured capability contract for this engine."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def operator_output_specs(self) -> Mapping[str, OperatorOutputSpec]:
        """Return statically declared named output fields."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: Any = DOC_KEY0,
    ) -> OperatorPrediction:
        """Validate and evaluate one canonical batch into named outputs."""
        raise NotImplementedError


class EncodedOperatorModel(OperatorModel):
    """Operator engine with reusable encoded source state and query decoding."""

    @abc.abstractmethod
    def encode_inputs(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: Any = DOC_KEY0,
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def decode_query(
        self,
        state: Any,
        query: FunctionSamples,
        /,
        *,
        key: Any = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError


__all__ = [
    "EncodedOperatorModel",
    "OperatorModel",
    "OperatorPredictionBuilder",
]
