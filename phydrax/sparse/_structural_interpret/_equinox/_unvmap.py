"""Propagation rule for unvmap_any and unvmap_max (Equinox primitives)."""

import numpy as np
from jax._src.core import JaxprEqn

from .._common import (
    _atom_const_val,
    _atom_numel,
    _conservative_indices,
    _index_sets,
    StateConsts,
    StateIndices,
)


_UNVMAP_REDUCTIONS = {"unvmap_any": np.any, "unvmap_max": np.max}


def _prop_unvmap(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """unvmap_any/unvmap_max reduce over every axis, ignoring vmap batch axes.

    Outside ``vmap`` they are ``jnp.any``/``jnp.max`` of the whole operand, so
    each output conservatively depends on every input. A known operand reduces
    to a known result: ``eqx.error_if`` bounds checks on static index arrays
    then resolve their ``cond`` to the non-raising branch, keeping the checked
    indices static for downstream gather/scatter.

    https://github.com/patrick-kidger/equinox/blob/main/equinox/_unvmap.py
    """
    operand = eqn.invars[0]
    out_var = eqn.outvars[0]
    state_indices[out_var] = _conservative_indices(
        _index_sets(state_indices, operand), _atom_numel(out_var)
    )
    value = _atom_const_val(operand, state_consts)
    if value is not None:
        state_consts[out_var] = np.asarray(_UNVMAP_REDUCTIONS[eqn.primitive.name](value))
