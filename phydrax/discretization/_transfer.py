#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import AbstractLinearOperator
from ._core import PreparationReport, resolved_identifier
from ._spaces import DiscreteFieldSpace


class TransferProperties(StrictModule, NonTrainableState):
    """Explicitly claimed structural properties of one field transfer."""

    constant_preserving: bool = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    positivity_preserving: bool = eqx.field(static=True)
    nested: bool = eqx.field(static=True)
    adjoint_paired: bool = eqx.field(static=True)
    differentiable_geometry: bool = eqx.field(static=True)
    exact_on: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        constant_preserving: bool = False,
        conservative: bool = False,
        positivity_preserving: bool = False,
        nested: bool = False,
        adjoint_paired: bool = False,
        differentiable_geometry: bool = False,
        exact_on: Sequence[str] = (),
    ):
        exact = tuple(str(value) for value in exact_on)
        if any(not value for value in exact) or len(set(exact)) != len(exact):
            raise ValueError("exact_on entries must be unique non-empty strings.")
        self.constant_preserving = bool(constant_preserving)
        self.conservative = bool(conservative)
        self.positivity_preserving = bool(positivity_preserving)
        self.nested = bool(nested)
        self.adjoint_paired = bool(adjoint_paired)
        self.differentiable_geometry = bool(differentiable_geometry)
        self.exact_on = exact


class FieldTransfer(StrictModule, NonTrainableState):
    """Prepared map between exact source and target field spaces."""

    source: DiscreteFieldSpace
    target: DiscreteFieldSpace
    operator: AbstractLinearOperator
    adjoint_operator: AbstractLinearOperator | None
    properties: TransferProperties
    preparation: PreparationReport
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: DiscreteFieldSpace,
        target: DiscreteFieldSpace,
        operator: AbstractLinearOperator,
        /,
        *,
        adjoint_operator: AbstractLinearOperator | None = None,
        properties: TransferProperties | None = None,
        preparation: PreparationReport | None = None,
        transfer_id: str | None = None,
    ):
        if not isinstance(source, DiscreteFieldSpace) or not isinstance(
            target, DiscreteFieldSpace
        ):
            raise TypeError("source and target must be DiscreteFieldSpace values.")
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if not operator.source.compatible(
            source.vector_space
        ) or not operator.target.compatible(target.vector_space):
            raise ValueError(
                "Transfer operator spaces must match source and target fields."
            )
        if adjoint_operator is not None:
            if not isinstance(adjoint_operator, AbstractLinearOperator):
                raise TypeError(
                    "adjoint_operator must be an AbstractLinearOperator or None."
                )
            if not adjoint_operator.source.compatible(
                target.vector_space
            ) or not adjoint_operator.target.compatible(source.vector_space):
                raise ValueError(
                    "Adjoint transfer spaces must reverse source and target."
                )
        properties_ = TransferProperties() if properties is None else properties
        if not isinstance(properties_, TransferProperties):
            raise TypeError("properties must be TransferProperties.")
        if properties_.adjoint_paired and adjoint_operator is None:
            raise ValueError("adjoint_paired transfers require an adjoint_operator.")
        preparation_ = PreparationReport() if preparation is None else preparation
        if not isinstance(preparation_, PreparationReport):
            raise TypeError("preparation must be a PreparationReport.")
        self.source = source
        self.target = target
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.properties = properties_
        self.preparation = preparation_
        self.transfer_id = resolved_identifier(
            "transfer_id",
            transfer_id,
            {
                "kind": "field-transfer",
                "source": source.field_space_id,
                "target": target.field_space_id,
                "operator": operator.operator_id,
                "adjoint_operator": None
                if adjoint_operator is None
                else adjoint_operator.operator_id,
                "properties": {
                    "constant_preserving": properties_.constant_preserving,
                    "conservative": properties_.conservative,
                    "positivity_preserving": properties_.positivity_preserving,
                    "nested": properties_.nested,
                    "adjoint_paired": properties_.adjoint_paired,
                    "differentiable_geometry": properties_.differentiable_geometry,
                    "exact_on": list(properties_.exact_on),
                },
                "preparation": preparation_.report_id,
            },
        )


__all__ = ["FieldTransfer", "TransferProperties"]
