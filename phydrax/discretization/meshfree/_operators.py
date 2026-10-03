# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Local functionals composed with the native sparse Hilbert-space owner."""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import ArraySpace
from ...sparse import linear_apply, linear_transpose_apply, SparseCoordinateOperator
from ...typing import checked
from ._neighbors import _integer
from ._stencils import LocalStencilEvidence, MeshfreeFunctional, PreparedLocalStencils


@final
class MeshfreeOperator(StrictModule):
    operator: SparseCoordinateOperator
    functional: MeshfreeFunctional
    evidence: LocalStencilEvidence
    operator_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        stencils: PreparedLocalStencils,
        functional_index: int = 0,
        *,
        source: ArraySpace | None = None,
        target: ArraySpace | None = None,
    ) -> None:
        index = _integer(functional_index, "functional_index", 0)
        if not index < len(stencils.functionals):
            raise ValueError("functional_index is outside the prepared functional tuple.")
        relation = stencils.neighborhood.relation
        weights = stencils.weights[index]
        # Default spaces carry the declared compute (source) and output
        # (target) roles; actions accumulate in the accumulation role.
        precision = stencils.neighborhood.precision
        if (source is not None and not isinstance(source, ArraySpace)) or (
            target is not None and not isinstance(target, ArraySpace)
        ):
            raise TypeError(
                "Meshfree source and target must be native ArraySpace values."
            )
        source_ = (
            ArraySpace(
                (relation.source_size,),
                dtype=precision.compute_dtype,
                space_id=f"{stencils.neighborhood.neighborhood_id}:source",
            )
            if source is None
            else source
        )
        target_ = (
            ArraySpace(
                (relation.targets_per_case,),
                dtype=precision.output_dtype,
                space_id=f"{stencils.neighborhood.neighborhood_id}:target",
            )
            if target is None
            else target
        )
        identifier = canonical_fingerprint(
            {
                "kind": "meshfree-operator",
                "stencils": stencils.prepared_id,
                "functional": index,
                "source": source_.space_id,
                "target": target_.space_id,
            }
        )
        operator = SparseCoordinateOperator(
            relation,
            weights,
            source=source_,
            target=target_,
            accumulation_dtype=precision.accumulation_dtype,
            operator_id=identifier,
        )
        self.operator = operator
        self.functional = stencils.functionals[index]
        self.evidence = stencils.evidence
        self.operator_id = self.operator.operator_id

    def apply(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim == 0 or value.shape[0] != self.operator.relation.source_size:
            raise ValueError("Meshfree values must begin with the source axis.")
        return linear_apply(self.operator.relation, self.operator.coefficients, value)

    __call__ = apply

    def transpose_apply(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim == 0 or value.shape[0] != self.operator.relation.output_shape[0]:
            raise ValueError("Meshfree cotangents must begin with the target axis.")
        return linear_transpose_apply(
            self.operator.relation, self.operator.coefficients, value
        )

    def mv(self, values: ArrayLike, /) -> Array:
        return self.operator.mv(values)

    def transpose_mv(self, values: ArrayLike, /) -> Array:
        return self.operator.transpose_mv(values)

    def adjoint_mv(self, values: ArrayLike, /) -> Array:
        return self.operator.adjoint_mv(values)
