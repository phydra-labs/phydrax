#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from jaxtyping import Array

from ...._doc import DOC_KEY0
from ._base import _AbstractOperatorModel
from ._keys import EvalKey
from ._operator import FunctionSamples, OperatorBatch


class AbstractEncodedOperatorModel(_AbstractOperatorModel):
    """Operator with a reusable source state and independent query decoder."""

    @abstractmethod
    def encode_inputs(
        self,
        batch: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode_query(
        self,
        state: Any,
        query: FunctionSamples,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        state = self.encode_inputs(batch, key=key)
        return self.decode_query(state, batch.require_single_query(), key=key)


__all__ = ["AbstractEncodedOperatorModel"]
