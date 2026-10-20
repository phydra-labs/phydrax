"""Propagation rules for JAX's stock high-level ``jax.numpy`` operations.

JAX stages ``searchsorted``, ``nonzero``, ``ldexp`` and ``frexp`` as
``call_hi_primitive`` equations. Their derivative semantics are read from the
operation itself, never from the bitwise low-level expansion: ``ldexp`` and
``frexp`` lower to integer bit manipulation whose own derivatives vanish.
Only the stock operation types are admitted; any other payload fails closed.
"""

from typing import assert_never

import jax
import numpy as np
from jax._src.core import JaxprEqn
from jax.extend.core import jaxpr_as_fun

from ..._jax_program import stock_high_expansion, stock_high_operation
from ._common import (
    _atom_const_val,
    _atom_numel,
    _atom_shape,
    _atom_value_bounds,
    _copy_index_sets,
    _empty_index_sets,
    _index_sets,
    StateBounds,
    StateConsts,
    StateIndices,
)
from ._elementwise import _binary_elementwise, _zero_derivative


def _prop_stock_high(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Dispatch one admitted stock high-level operation.

    - ``searchsorted(sorted, query)``: integer result, zero derivative,
      values in ``[0, sorted.shape[dimension]]``.
    - ``nonzero(a, *fill)``: integer results, zero derivative,
      values in ``[0, a.shape[axis] - 1]`` or the fill value (default 0).
    - ``ldexp(x, e) = x * 2**e``: ``∂/∂x = 2**e`` never vanishes, so
      ``out[i]`` depends on ``x[i]``; the integer exponent contributes nothing.
    - ``frexp(x) = (m, e)`` with ``m = x * 2**-e``: ``∂m/∂x = 2**-e``, so
      ``m[i]`` depends on ``x[i]``; the integer exponent ``e`` has none.

    When every operand is statically known, the concrete results are recorded
    so downstream gather/scatter handlers resolve exact indices.
    """
    kind = stock_high_operation(eqn)
    if kind is None:
        msg = f"Primitive '{eqn.primitive.name}' is not a stock high-level JAX operation."
        raise NotImplementedError(msg)

    match kind:
        case "searchsorted":
            _zero_derivative(eqn, state_indices)
            _searchsorted_bounds(eqn, state_bounds)
        case "nonzero":
            _zero_derivative(eqn, state_indices)
            _nonzero_bounds(eqn, state_consts, state_bounds)
        case "ldexp":
            _binary_elementwise(eqn, state_indices, is_der2_zero_globally=True)
        case "frexp":
            mantissa, exponent = eqn.outvars
            state_indices[mantissa] = _copy_index_sets(
                _index_sets(state_indices, eqn.invars[0])
            )
            state_indices[exponent] = _empty_index_sets(_atom_numel(exponent))
        case _:
            assert_never(kind)

    _propagate_const_stock_high(eqn, state_consts)


def _propagate_const_stock_high(eqn: JaxprEqn, state_consts: StateConsts) -> None:
    """Evaluate the operation's executed program on statically known operands."""
    values = [_atom_const_val(invar, state_consts) for invar in eqn.invars]
    if any(value is None for value in values):
        return
    expansion = jaxpr_as_fun(stock_high_expansion(eqn))
    with jax.ensure_compile_time_eval():
        results = expansion(*values)
    for outvar, result in zip(eqn.outvars, results, strict=True):
        state_consts[outvar] = np.asarray(result)


def _searchsorted_bounds(eqn: JaxprEqn, state_bounds: StateBounds) -> None:
    """Insertion points lie in ``[0, n]`` for ``n`` sorted entries."""
    length = _atom_shape(eqn.invars[0])[eqn.params["_prim"].dimension]
    out_shape = _atom_shape(eqn.outvars[0])
    state_bounds[eqn.outvars[0]] = (
        np.zeros(out_shape, dtype=np.int64),
        np.full(out_shape, length, dtype=np.int64),
    )


def _nonzero_bounds(
    eqn: JaxprEqn, state_consts: StateConsts, state_bounds: StateBounds
) -> None:
    """Bound each coordinate output by its axis range and padding fill value.

    Outputs follow the sorted axes, matching JAX's implementation. An empty
    operand yields uninitialized padding, so no bound is recorded.
    """
    in_shape = _atom_shape(eqn.invars[0])
    if _atom_numel(eqn.invars[0]) == 0 or _atom_numel(eqn.outvars[0]) == 0:
        return
    axes = sorted(eqn.params["_prim"].axes)
    fills = eqn.invars[1:]
    for position, (axis, outvar) in enumerate(zip(axes, eqn.outvars, strict=True)):
        fill_lo, fill_hi = 0, 0
        if fills:
            fill_bounds = _atom_value_bounds(fills[position], state_consts, state_bounds)
            if fill_bounds is None:
                continue
            fill_lo, fill_hi = int(np.min(fill_bounds[0])), int(np.max(fill_bounds[1]))
        out_shape = _atom_shape(outvar)
        state_bounds[outvar] = (
            np.full(out_shape, min(0, fill_lo), dtype=np.int64),
            np.full(out_shape, max(in_shape[axis] - 1, fill_hi), dtype=np.int64),
        )
