#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact admission of JAX's stock high-level operations in staged programs.

JAX stages ordinary ``jax.numpy`` operations such as ``searchsorted`` and
``ldexp`` as ``call_hi_primitive`` equations carrying a Python operation
object. Only the stock operation types listed here are admitted; any other
high-level payload is user/library-defined and fails closed, because its
derivative, batching and effect rules are not visible in the program.
"""

from __future__ import annotations

import functools
from typing import Any, Literal, TypeAlias

import jax
from jax._src.hijax import call_hi_primitive_linearized_p, call_hi_primitive_p
from jax._src.lax.lax import _polynomial_lower
from jax._src.numpy.hijax import Nonzero, SearchSorted
from jax._src.numpy.ufuncs import Frexp, Ldexp
from jax.extend import core as jax_core


StockHighOperation: TypeAlias = Literal["searchsorted", "nonzero", "ldexp", "frexp"]

_STOCK_HIGH_OPERATIONS: dict[type, StockHighOperation] = {
    SearchSorted: "searchsorted",
    Nonzero: "nonzero",
    Ldexp: "ldexp",
    Frexp: "frexp",
}


def _admitted_kind(equation: jax_core.JaxprEqn, /) -> StockHighOperation | None:
    """Stock kind of a ``call_hi_primitive`` equation: exact type, no effects."""
    operation = equation.params["_prim"]
    if operation.effects:
        return None
    return _STOCK_HIGH_OPERATIONS.get(type(operation))


def is_opaque_high_operation(equation: jax_core.JaxprEqn, /) -> bool:
    """Whether an equation binds a high-level operation that is not admitted.

    Such an operation hides its program, derivative rules and effects in a
    Python object; walkers must refuse or isolate it, never treat it as a
    standard primitive. Linearized high-level applications are never admitted.
    """
    primitive = equation.primitive
    if primitive is call_hi_primitive_linearized_p:
        return True
    return primitive is call_hi_primitive_p and _admitted_kind(equation) is None


def stock_high_operation(equation: jax_core.JaxprEqn, /) -> StockHighOperation | None:
    """Return the stock operation a high-level equation binds.

    Returns ``None`` for equations that do not bind a high-level operation
    object. Any other operation type, effectful payload, or linearized
    high-level application raises ``TypeError``.
    """
    if is_opaque_high_operation(equation):
        if equation.primitive is call_hi_primitive_linearized_p:
            raise TypeError(
                "Linearized high-level JAX operation needs an explicit program owner."
            )
        operation_type = type(equation.params["_prim"])
        raise TypeError(
            "Unsupported or effectful high-level JAX operation "
            f"{operation_type.__module__}.{operation_type.__qualname__}."
        )
    if equation.primitive is not call_hi_primitive_p:
        return None
    return _admitted_kind(equation)


def stock_high_expansion(equation: jax_core.JaxprEqn, /) -> jax_core.ClosedJaxpr:
    """Trace an admitted stock operation's lowering over its flat input types.

    The expansion's flat inputs and outputs match the equation's. It is the
    executed low-level program, not a derivative rule: bitwise expansions
    such as ``ldexp`` have no useful derivatives of their own.
    """
    if stock_high_operation(equation) is None:
        raise TypeError(
            f"Equation {equation.primitive.name} is not a high-level JAX operation."
        )
    operation = equation.params["_prim"]

    def expanded(*flat_inputs: Any) -> list[Any]:
        inputs = jax.tree_util.tree_unflatten(operation.in_tree, flat_inputs)
        return jax.tree_util.tree_leaves(operation.expand(*inputs))

    return jax.make_jaxpr(expanded)(*operation.in_avals_flat)


def lowered_expansion(equation: jax_core.JaxprEqn, /) -> jax_core.ClosedJaxpr | None:
    """Executed low-level program of an operation whose lowering hides work.

    Stock high-level operations expand to their lowering; ``polynomial``
    expands to its Horner recurrence, which stacks broadcast coefficients and
    scans them once the term count exceeds the unroll threshold. ``None`` for
    every other equation; opaque high-level operations raise ``TypeError``.
    Resource estimates use this; it never replaces derivative or identity
    analysis.
    """
    if equation.primitive is jax.lax.polynomial_p:
        return jax.make_jaxpr(functools.partial(_polynomial_lower, **equation.params))(
            *(variable.aval for variable in equation.invars)
        )
    if stock_high_operation(equation) is None:
        return None
    return stock_high_expansion(equation)


__all__ = [
    "StockHighOperation",
    "is_opaque_high_operation",
    "lowered_expansion",
    "stock_high_expansion",
    "stock_high_operation",
]
