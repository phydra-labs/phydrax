#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike
from opt_einsum import contract

from ...domain._function import DomainFunction


def norm(u: DomainFunction, /, *, order: int = 2) -> DomainFunction:
    r"""Pointwise vector norm of a `DomainFunction`.

    Interprets the last axis of $u(z)$ as a vector and returns

    $$
    \|u(z)\|_p,
    $$

    where `order=p`. If $u(z)$ is scalar, this returns $|u(z)|$.

    **Arguments:**

    - `u`: Input function.
    - `order`: Norm order $p$ passed to `jax.numpy.linalg.norm` (`ord=p`).
    """
    order_i = int(order)

    def _op(*args, key=None, **kwargs):
        val = jnp.asarray(u.func(*args, key=key, **kwargs))
        if val.ndim == 0:
            return jnp.abs(val)
        return jnp.linalg.norm(val, ord=order_i, axis=-1)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_op, metadata=u.metadata)


def det(u: DomainFunction, /) -> DomainFunction:
    r"""Pointwise matrix determinant.

    Interprets the last two axes of $u(z)$ as a square matrix and returns
    $\det(u(z))$.

    **Arguments:**

    - `u`: Input `DomainFunction` whose values have trailing shape `(n, n)`.

    **Returns:**

    - A `DomainFunction` representing the scalar determinant field.
    """

    def _op(*args, key=None, **kwargs):
        val = jnp.asarray(u.func(*args, key=key, **kwargs))
        return jnp.linalg.det(val)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_op, metadata=u.metadata)


def trace(u: DomainFunction, /) -> DomainFunction:
    r"""Pointwise matrix trace.

    Interprets the last two axes of $u(z)$ as a matrix and returns
    $\text{tr}(u(z))$.

    **Arguments:**

    - `u`: Input `DomainFunction` whose values have at least two trailing axes.

    **Returns:**

    - A `DomainFunction` representing the scalar trace field.
    """

    def _op(*args, key=None, **kwargs):
        val = jnp.asarray(u.func(*args, key=key, **kwargs))
        return jnp.trace(val, axis1=-2, axis2=-1)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_op, metadata=u.metadata)


def einsum(subscript: str, /, *operands: DomainFunction | ArrayLike) -> DomainFunction:
    r"""Einstein summation of `DomainFunction` and/or constant array operands.

    Given operands $u^{(1)}(z),\dots,u^{(k)}(z)$ (each either a `DomainFunction`
    or a constant array-like), returns the pointwise contraction specified by
    `subscript`, i.e.

    $$
    w(z) = \texttt{einsum}(\texttt{subscript}, u^{(1)}(z), \dots, u^{(k)}(z)).
    $$

    Domains are joined across `DomainFunction` operands before evaluation.
    Constant operands are broadcast by `einsum` and do not contribute domain deps.

    **Arguments:**

    - `subscript`: Einsum subscript string (as in `opt_einsum.contract`).
    - `operands`: One or more operands (`DomainFunction` or array-like). At least
      one operand must be a `DomainFunction`.

    **Returns:**

    - A `DomainFunction` representing the contracted result.
    """
    if not operands:
        raise ValueError("einsum requires at least one operand.")

    fn_operands = tuple(op for op in operands if isinstance(op, DomainFunction))
    if not fn_operands:
        raise ValueError("einsum requires at least one DomainFunction operand.")

    joined = fn_operands[0].domain
    for op in fn_operands[1:]:
        joined = joined.join(op.domain)

    promoted = tuple(op.promote(joined) for op in fn_operands)

    deps = tuple(lbl for lbl in joined.labels if any(lbl in op.deps for op in promoted))

    idx = {lbl: i for i, lbl in enumerate(deps)}
    promoted_iter = iter(promoted)
    operand_specs = []
    for op in operands:
        if isinstance(op, DomainFunction):
            op_promoted = next(promoted_iter)
            pos = tuple(idx[lbl] for lbl in op_promoted.deps)
            operand_specs.append((op_promoted, pos, None))
        else:
            operand_specs.append((None, (), jnp.asarray(op)))

    meta = promoted[0].metadata
    for op in promoted[1:]:
        if op.metadata != meta:
            meta = {}
            break

    def _op(*args, key=None, **kwargs):
        arrays = []
        for op, pos, const in operand_specs:
            if op is None:
                arrays.append(const)
                continue
            op_args = [args[i] for i in pos]
            arrays.append(op.func(*op_args, key=key, **kwargs))
        return contract(subscript, *arrays)

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=meta)


__all__ = [
    "det",
    "einsum",
    "norm",
    "trace",
]
