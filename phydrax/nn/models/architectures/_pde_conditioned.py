#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

from jaxtyping import Array

from ...._strict import StrictModule
from ....equations import PDETokenBatch
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey
from ..core._operator import OperatorBatch, OperatorOutputSpec
from ._equation_conditioning import attach_pde_condition, PDEConditionEncoder


class PDEConditionedInput(StrictModule):
    """One operator task paired with its canonical PDE tokens."""

    batch: OperatorBatch
    tokens: PDETokenBatch

    def __init__(self, batch: OperatorBatch, tokens: PDETokenBatch, /):
        if not isinstance(batch, OperatorBatch):
            raise TypeError("PDEConditionedInput batch must be an OperatorBatch.")
        if not isinstance(tokens, PDETokenBatch):
            raise TypeError("PDEConditionedInput tokens must be a PDETokenBatch.")
        self.batch = batch
        self.tokens = tokens


class PDEConditionedOperator(_AbstractOperatorModel):
    """Task-specific composition of a PDE encoder and an existing operator.

    ``PDEConditionEncoder`` is deterministic at evaluation time, so the supplied
    evaluation key is delegated unchanged to ``operator``.
    """

    operator: _AbstractOperatorModel
    encoder: PDEConditionEncoder
    input_name: str
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(
        self,
        operator: _AbstractOperatorModel,
        encoder: PDEConditionEncoder,
        /,
        *,
        input_name: str = "equation",
    ):
        if not isinstance(operator, _AbstractOperatorModel):
            raise TypeError("PDEConditionedOperator requires an operator model.")
        if not isinstance(encoder, PDEConditionEncoder):
            raise TypeError("PDEConditionedOperator requires a PDEConditionEncoder.")
        name = str(input_name)
        if not name:
            raise ValueError("PDE condition input_name must be non-empty.")
        self.operator = operator
        self.encoder = encoder
        self.input_name = name
        self.in_size = operator.in_size
        self.out_size = operator.out_size

    @property
    def operator_output_specs(self) -> dict[str, OperatorOutputSpec]:
        """Preserve the wrapped operator's named output contracts."""
        return self.operator.operator_output_specs


    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(batch, OperatorBatch):
            raise TypeError(
                "PDEConditionedOperator.__call_operator_batch__ requires an "
                "OperatorBatch."
            )
        if self.input_name not in batch.inputs:
            raise ValueError(
                "PDEConditionedOperator requires an already-conditioned input "
                f"branch {self.input_name!r}; pass PDEConditionedInput to encode "
                "and attach PDE tokens automatically."
            )
        return self.operator.__call_operator_batch__(batch, key=key)

    def __call__(
        self,
        x: PDEConditionedInput,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, PDEConditionedInput):
            raise TypeError("PDEConditionedOperator requires a PDEConditionedInput.")
        if self.input_name in x.batch.inputs:
            raise ValueError(
                "Cannot attach PDE condition: OperatorBatch already contains input "
                f"{self.input_name!r}."
            )
        conditioned = attach_pde_condition(
            x.batch,
            x.tokens,
            self.encoder,
            input_name=self.input_name,
        )
        return self.__call_operator_batch__(conditioned, key=key)


__all__ = ["PDEConditionedInput", "PDEConditionedOperator"]
