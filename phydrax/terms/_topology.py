#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._term import AbstractEvaluatedScalarTerm, TermEvaluation
from ..domain import DomainFunction
from ..topology import FrozenPersistencePairing, PreparedVertexFiltration
from ..topology._features import frozen_total_persistence


class FrozenTopologyTerm(AbstractEvaluatedScalarTerm):
    """Locally exact persistence penalty under one frozen filtration pairing."""

    field: str = eqx.field(static=True)
    vertex_points: Any
    filtration: PreparedVertexFiltration
    pairing: FrozenPersistencePairing
    degree: int = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    weight: Array
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        vertex_points: Any,
        filtration: PreparedVertexFiltration,
        pairing: FrozenPersistencePairing,
        /,
        *,
        degree: int,
        exponent: float = 1.0,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        name = str(field)
        if not name:
            raise ValueError("Frozen topology field name must be non-empty.")
        if int(degree) < 0 or float(exponent) <= 0.0:
            raise ValueError("Frozen topology degree and exponent are invalid.")
        coefficient = jnp.asarray(weight, dtype=float)
        if coefficient.shape != () or not bool(jnp.isfinite(coefficient)):
            raise ValueError("Frozen topology weight must be one finite scalar.")
        if float(coefficient) < 0.0:
            raise ValueError("Frozen topology weight must be non-negative.")
        if pairing.layout.layout_id != filtration.complex.layout.layout_id:
            raise ValueError("Frozen pairing and prepared filtration layouts differ.")
        self.field = name
        self.vertex_points = vertex_points
        self.filtration = filtration
        self.pairing = pairing
        self.degree = int(degree)
        self.exponent = float(exponent)
        self.weight = coefficient
        self.label = None if label is None else str(label)

    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> TermEvaluation:
        del iter_
        prediction = functions[self.field](self.vertex_points, key=key, **kwargs)
        if not isinstance(prediction, cx.Field):
            raise TypeError("Frozen topology field must evaluate to coordax.Field.")
        values = jnp.asarray(prediction.data)
        if values.ndim != 1:
            raise ValueError("Frozen topology currently requires one scalar per vertex.")
        cell_values = self.filtration.cell_values(values)
        evaluation = self.pairing.evaluate(cell_values)
        objective, valid = frozen_total_persistence(
            evaluation,
            degree=self.degree,
            exponent=self.exponent,
        )
        value = self.weight * jnp.asarray(objective, dtype=float).reshape(())
        return TermEvaluation(
            value,
            diagnostics=frozendict(
                {
                    "ordering_valid": valid,
                    "ordering_margin": evaluation.ordering_margin,
                    "pairing_id": self.pairing.pairing.pairing_id,
                }
            ),
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.term_evaluation(
            functions,
            key=key,
            iter_=iter_,
            **kwargs,
        ).value


__all__ = ["FrozenTopologyTerm"]
