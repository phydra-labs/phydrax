#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._operators import AbstractLinearOperator
from ._pairings import AbstractPairing
from ._prepared import PreparedLinearSolve
from ._problems import LinearSystem


class OperatorPairing(AbstractPairing):
    """Hilbert pairing induced by a certified positive-definite Riesz operator.

    With ``prepared_inverse``, ``inverse_riesz`` is the linear solve
    ``M x = b`` whose tangent and transpose solves reuse that prepared primal
    solve (the transpose of Hermitian ``M`` is ``conj(M)``), so its derivative
    is exact to the primal tolerance rather than to a separate derivative-solve
    tolerance; each solve raises if it is unsuccessful.
    """

    operator: AbstractLinearOperator
    inverse_action: Callable[[PyTree[Any]], PyTree[Array]] | None
    prepared_inverse: PreparedLinearSolve | None

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        inverse_action: Callable[[PyTree[Any]], PyTree[Array]] | None = None,
        prepared_inverse: PreparedLinearSolve | None = None,
        pairing_id: str | None = None,
    ) -> None:
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape:
            raise ValueError("A Riesz operator must be unbatched.")
        if not operator.source.compatible(operator.target):
            raise ValueError("A Riesz operator must be an endomorphism.")
        if not operator.properties.certifies(
            "self_adjoint"
        ) or not operator.properties.certifies("positive_definite"):
            raise ValueError(
                "A Riesz operator requires certified self-adjoint positive definiteness."
            )
        if (inverse_action is None) == (prepared_inverse is None):
            raise ValueError(
                "Supply exactly one explicit inverse_action or prepared_inverse."
            )
        if inverse_action is not None and not callable(inverse_action):
            raise TypeError("inverse_action must be callable or None.")
        if prepared_inverse is not None:
            if not isinstance(prepared_inverse, PreparedLinearSolve):
                raise TypeError("prepared_inverse must be a PreparedLinearSolve or None.")
            if (
                not isinstance(prepared_inverse.problem, LinearSystem)
                or prepared_inverse.problem.operator is not operator
            ):
                raise ValueError(
                    "prepared_inverse must solve this exact Riesz operator instance."
                )
        self.operator = operator
        self.inverse_action = inverse_action
        self.prepared_inverse = prepared_inverse
        self.pairing_id = (
            canonical_fingerprint(
                {
                    "kind": "operator-pairing",
                    "operator": operator.operator_id,
                    "state": array_tree_fingerprint(operator),
                }
            )
            if pairing_id is None
            else _nonempty(pairing_id)
        )

    def inner(self, left: PyTree[Any], right: PyTree[Any], /) -> Array:
        left_ = self.operator.source.validate(left)
        right_ = self.operator.source.validate(right)
        image = self.operator.target.validate(self.operator.mv(right_))
        return self.operator.source.inner(left_, image)

    def riesz(self, vector: PyTree[Any], /) -> PyTree[Array]:
        image = self.operator.target.validate(
            self.operator.mv(self.operator.source.validate(vector))
        )
        return self.operator.source.riesz(image)

    def inverse_riesz(self, covector: PyTree[Any], /) -> PyTree[Array]:
        value = self.operator.source.validate(covector)
        rhs = self.operator.source.inverse_riesz(value)
        if self.inverse_action is not None:
            return self.operator.source.validate(self.inverse_action(rhs))
        from ._runtime import solve

        prepared = self.prepared_inverse
        assert prepared is not None

        def prepared_solve(right: PyTree[Array], /) -> PyTree[Array]:
            result = solve(prepared, right)
            return eqx.error_if(
                result.value,
                jnp.any(~result.successful),
                "Prepared Riesz inverse solve failed.",
            )

        def transposed_solve(right: PyTree[Array], /) -> PyTree[Array]:
            return jax.tree.map(jnp.conj, prepared_solve(jax.tree.map(jnp.conj, right)))

        candidate = jax.lax.custom_linear_solve(
            self.operator.mv,
            rhs,
            solve=lambda _, right: prepared_solve(right),
            transpose_solve=lambda _, right: transposed_solve(right),
        )
        return self.operator.source.validate(candidate)


def _nonempty(value: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError("pairing_id must be non-empty.")
    return result


__all__ = ["OperatorPairing"]
