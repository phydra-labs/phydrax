"""Propagation rule for element-wise division."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_value_bounds,
    _clear_where_zero,
    _is_floating,
    _propagate_const_binary,
    _store_bounds,
    StateBounds,
    StateConsts,
    StateIndices,
)
from ._elementwise import _binary_elementwise, _integer_division_defined


def _integer_quotient(dividend: np.ndarray, divisor: np.ndarray) -> np.ndarray | None:
    """Integer ``lax.div``: the quotient is truncated toward zero.

    Floating quotients are never folded (see ``_store_folded_const``).
    """
    if not _integer_division_defined(dividend, divisor):
        return None
    quotient = np.floor_divide(dividend, divisor)
    inexact = np.remainder(dividend, divisor) != 0
    return np.where(inexact & ((dividend < 0) != (divisor < 0)), quotient + 1, quotient)


def _prop_div(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Division is element-wise with a special case for known zero numerators.

    Like other binary element-wise ops,
    each output depends on the corresponding elements from both inputs.
    However, since d(0 / y)/dy = 0,
    output positions where the numerator is a known constant zero
    have no dependency on the inputs.

    Example: z = [0, x] / [y, y]
        Input index sets: [{}, {1}], [{2}, {3}]
        Output index sets: [{}, {1, 3}]  (first cleared by known zero numerator)

    Jaxpr:
        invars[0]: numerator
        invars[1]: denominator
    """
    _binary_elementwise(eqn, state_indices)
    _propagate_const_binary(eqn, state_consts, _integer_quotient)
    _clear_where_zero(eqn, state_indices, state_consts, 0)
    _propagate_bounds_div(eqn, state_consts, state_bounds)


def _propagate_bounds_div(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``div``.

    Only propagates when divisor bounds have constant sign (no zero crossing).
    True and truncating division are monotone in each operand on such a box,
    so the extreme quotients are attained at its corners. Floating corners are
    widened to backend enclosures (see ``_backend_bounds``).
    """
    in1_bounds = _atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    in2_bounds = _atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if in1_bounds is None or in2_bounds is None:
        return

    lo1, hi1 = in1_bounds
    lo2, hi2 = in2_bounds

    # Skip if divisor bounds span zero.
    if not (np.all(lo2 > 0) or np.all(hi2 < 0)):
        return

    pairs = [(dividend, divisor) for dividend in (lo1, hi1) for divisor in (lo2, hi2)]
    if any(_is_floating(end) for end in (lo1, hi1, lo2, hi2)):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            corners = [np.true_divide(dividend, divisor) for dividend, divisor in pairs]
    else:
        quotients = [_integer_quotient(dividend, divisor) for dividend, divisor in pairs]
        if any(quotient is None for quotient in quotients):
            return
        corners = [quotient for quotient in quotients if quotient is not None]
    stacked = np.stack(np.broadcast_arrays(*corners))
    _store_bounds(eqn, state_bounds, np.min(stacked, axis=0), np.max(stacked, axis=0))
