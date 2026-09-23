#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from ._matrix_function_contracts import MatrixFunctionProvenance, MatrixFunctionResult
from ._operators import (
    AbstractLinearOperator,
    BlockLinearOperator,
    FunctionLinearOperator,
)
from ._spaces import _coordinate_dtype, ArraySpace, BlockSpace


if TYPE_CHECKING:
    from ._exponential_taylor import TaylorExponentialPolicy
    from ._matrix_functions import MatrixFunctionPolicy


def _augmented_exponential_operator(
    operator: AbstractLinearOperator,
    coefficients: tuple[PyTree[Array], ...],
    /,
) -> BlockLinearOperator:
    """Use J e1 = e2 so the kth column contributes t^k phi_k(tA)."""
    count = len(coefficients)
    auxiliary = ArraySpace((count,), dtype=_coordinate_dtype(operator.source))
    space = BlockSpace((operator.source, auxiliary), names=("physical", "phi-chain"))

    def coupling(coordinates: Array) -> PyTree[Array]:
        output = jax.tree.map(lambda leaf: coordinates[0] * leaf, coefficients[0])
        for index, column in enumerate(coefficients[1:], start=1):
            output = jax.tree.map(
                lambda total, leaf, index_=index: total + coordinates[index_] * leaf,
                output,
                column,
            )
        return output

    def shift(coordinates: Array) -> Array:
        return jnp.concatenate((jnp.zeros_like(coordinates[:1]), coordinates[:-1]))

    return BlockLinearOperator(
        (
            (
                operator,
                FunctionLinearOperator(
                    coupling,
                    source=auxiliary,
                    target=operator.source,
                    operator_id=canonical_fingerprint(
                        {
                            "kind": "augmented-phi-coupling",
                            "operator": operator.operator_id,
                            "coefficient_count": count,
                        }
                    ),
                ),
            ),
            (
                None,
                FunctionLinearOperator(
                    shift,
                    source=auxiliary,
                    target=auxiliary,
                    operator_id=canonical_fingerprint(
                        {"kind": "augmented-phi-shift", "size": count}
                    ),
                ),
            ),
        ),
        source=space,
        target=space,
    )


def matrix_exponential_phi_combination_action(
    operator: AbstractLinearOperator | Callable[[PyTree[Any]], PyTree[Array]],
    vectors: tuple[PyTree[Any], ...],
    scale: ArrayLike = 1.0,
    /,
    *,
    policy: MatrixFunctionPolicy | TaylorExponentialPolicy | None = None,
    key: Array | None = None,
) -> MatrixFunctionResult:
    """Apply exp(tA)v0 + sum(t^k phi_k(tA)vk, k=1..3) in one action.

    The auxiliary chain is scaled by the same t as A. Its initial coordinate
    is e1, so no factorial normalization or pre-scaling of the coefficients is
    needed; in particular the t=0 action is exactly v0.
    """
    if not isinstance(vectors, tuple):
        raise TypeError("vectors must be a tuple of physical-space vectors.")
    if not 1 <= len(vectors) <= 4:
        raise ValueError("vectors must contain v0 and at most three phi coefficients.")

    # Lazy resolution avoids a cycle with the public matrix-function facade.
    from ._exponential_taylor import (
        execute_taylor_exponential_action,
        TaylorExponentialPolicy,
    )
    from ._matrix_functions import (
        _coerce_matrix_operator,
        matrix_exponential_action,
        MatrixFunctionPolicy,
    )

    operator_ = _coerce_matrix_operator(operator, vectors[0])
    if not operator_.source.compatible(operator_.target):
        raise ValueError("The physical operator must be an endomorphism.")
    if operator_.batch_shape:
        raise ValueError("Augmented actions require an unbatched physical operator.")
    if policy is not None and not isinstance(
        policy, (MatrixFunctionPolicy, TaylorExponentialPolicy)
    ):
        raise TypeError(
            "policy must be a MatrixFunctionPolicy, TaylorExponentialPolicy, or None."
        )
    if key is not None and not isinstance(policy, TaylorExponentialPolicy):
        raise ValueError("key is available only for Taylor exponential policies.")
    validated = tuple(operator_.source.validate(vector) for vector in vectors)
    scalar = jnp.asarray(scale)
    if scalar.shape != ():
        raise ValueError("scale must be scalar.")

    if len(validated) == 1:
        augmented_operator = operator_
        augmented_vector = validated[0]
    else:
        augmented_operator = _augmented_exponential_operator(operator_, validated[1:])
        first_auxiliary = (
            jnp.zeros((len(validated) - 1,), dtype=_coordinate_dtype(operator_.source))
            .at[0]
            .set(1)
        )
        augmented_vector = (validated[0], first_auxiliary)

    if isinstance(policy, TaylorExponentialPolicy):
        result = execute_taylor_exponential_action(
            augmented_operator, augmented_vector, scalar, policy=policy, key=key
        )
    else:
        result = matrix_exponential_action(
            augmented_operator, augmented_vector, scalar, policy=policy
        )

    provenance = result.provenance
    return MatrixFunctionResult(
        value=result.value if len(validated) == 1 else result.value[0],
        status=result.status,
        diagnostics=result.diagnostics,
        provenance=MatrixFunctionProvenance(
            method=provenance.method,
            kind="exp-phi-combination",
            description=(
                provenance.description
                + "; matrix-free augmented exponential/phi action of physical operator "
                + operator_.operator_id
                if len(validated) > 1
                else provenance.description
                + "; direct exponential action of physical operator "
                + operator_.operator_id
            ),
            operator_id=provenance.operator_id,
            plan_id=provenance.plan_id,
            prepared_id=provenance.prepared_id,
            trace_source=provenance.trace_source,
            norm_source=provenance.norm_source,
            numeric_version=provenance.numeric_version,
        ),
    )


__all__ = ["matrix_exponential_phi_combination_action"]
