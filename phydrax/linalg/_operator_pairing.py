#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..typing import checked
from ._operators import AbstractLinearOperator
from ._pairings import AbstractPairing
from ._prepared import PreparedLinearSolve
from ._problems import LinearSystem


class OperatorPairingArchiveRecipe(NamedTuple):
    operator: object
    solve_policy: object
    pairing_id: str
    operator_id: str
    operator_content_fingerprint: str
    solve_policy_fingerprint: str

    def rebuild(
        self,
        operator: AbstractLinearOperator,
        solve_policy: object,
        /,
    ) -> OperatorPairing:
        from ._policies import LinearSolvePolicy
        from ._runtime import prepare

        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("Stored Riesz operator recipe lost its operator.")
        if operator.operator_id != self.operator_id:
            raise ValueError("Stored Riesz operator identity changed.")
        # Sparse operator IDs name structure, not coefficients; the archived
        # pairing binds the operator's exact array content separately.
        if _operator_content_fingerprint(operator) != self.operator_content_fingerprint:
            raise ValueError("Stored Riesz operator content changed.")
        if type(solve_policy) is not LinearSolvePolicy:
            raise TypeError("Stored Riesz recipe requires LinearSolvePolicy.")
        if (
            canonical_fingerprint(array_tree_fingerprint(solve_policy))
            != self.solve_policy_fingerprint
        ):
            raise ValueError("Stored Riesz solve policy identity changed.")
        prepared = prepare(LinearSystem(operator), solve_policy)
        pairing = OperatorPairing(
            operator,
            prepared_inverse=prepared,
            pairing_id=self.pairing_id,
        )
        if pairing.pairing_id != self.pairing_id:
            raise ValueError("Stored operator-pairing identity changed.")
        return pairing


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

    @checked
    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        inverse_action: Callable[[PyTree[Any]], PyTree[Array]] | None = None,
        prepared_inverse: PreparedLinearSolve | None = None,
        pairing_id: str | None = None,
    ) -> None:
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
        return self.operator.source.validate(_solve_prepared(self, rhs))

    def archive_recipe(self, /) -> OperatorPairingArchiveRecipe:
        if type(self) is not OperatorPairing:
            raise TypeError("Riesz archive requires the canonical OperatorPairing.")
        if self.inverse_action is not None or self.prepared_inverse is None:
            raise TypeError(
                "Portable Riesz archives require a re-preparable solver policy."
            )
        if type(self.prepared_inverse.problem) is not LinearSystem:
            raise TypeError("Portable Riesz archives require LinearSystem solves.")
        return OperatorPairingArchiveRecipe(
            self.operator,
            self.prepared_inverse.template.plan.policy,
            self.pairing_id,
            self.operator.operator_id,
            _operator_content_fingerprint(self.operator),
            canonical_fingerprint(
                array_tree_fingerprint(self.prepared_inverse.template.plan.policy)
            ),
        )


@eqx.filter_jit
def _solve_prepared(pairing: OperatorPairing, rhs: PyTree[Array], /) -> PyTree[Array]:
    """Stage primal and transpose closures with dynamic prepared coefficients."""
    from ._runtime import solve

    prepared = pairing.prepared_inverse
    assert prepared is not None

    def prepared_solve(_: Any, right: PyTree[Array], /) -> PyTree[Array]:
        result = solve(prepared, right)
        return eqx.error_if(
            result.value,
            jnp.any(~result.successful),
            "Prepared Riesz inverse solve failed.",
        )

    def transposed_solve(_: Any, right: PyTree[Array], /) -> PyTree[Array]:
        return jax.tree.map(jnp.conj, prepared_solve(None, jax.tree.map(jnp.conj, right)))

    return jax.lax.custom_linear_solve(
        pairing.operator.mv,
        rhs,
        solve=prepared_solve,
        transpose_solve=transposed_solve,
    )


def _operator_content_fingerprint(operator: AbstractLinearOperator, /) -> str:
    """Content of the operator's archive form, independent of query caches."""
    from .._model._structure import model_archive_projection

    return canonical_fingerprint(
        {
            "kind": "operator-pairing-content",
            "state": array_tree_fingerprint(model_archive_projection(operator)),
        }
    )


def _nonempty(value: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError("pairing_id must be non-empty.")
    return result


__all__ = ["OperatorPairing", "OperatorPairingArchiveRecipe"]
