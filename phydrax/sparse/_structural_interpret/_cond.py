"""Propagation rule for cond (conditional branching)."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_const_val,
    _copy_index_sets,
    _export_scope,
    _index_sets,
    _nested_scope,
    IndexSet,
    PropJaxprFn,
    StateBounds,
    StateConsts,
    StateIndices,
)


def _prop_cond(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
    _prop_jaxpr: PropJaxprFn,
) -> None:
    """cond/switch selects one of several branches based on an integer index.

    When the index is statically known only that branch executes, so its
    dependencies and known result values pass through exactly. Otherwise the
    output state_indices are the union across all branches.

    Layout:
        invars: [index_scalar, operands...]
        outvars: [results...]
        params: branches (tuple of ClosedJaxpr)

    Example: cond(pred, true_fn, false_fn, x)
        true_fn:  out = x[:2]  → state_indices [{0}, {1}]
        false_fn: out = x[1:]  → state_indices [{1}, {2}]
        union:    [{0, 1}, {1, 2}]

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html
    """
    branches = eqn.params["branches"]
    operands = eqn.invars[1:]
    operand_indices: list[list[IndexSet]] = [
        _index_sets(state_indices, v) for v in operands
    ]
    index = _atom_const_val(eqn.invars[0], state_consts)
    if index is not None:
        # lax.switch semantics: out-of-range indices clamp to the end branches.
        branches = (branches[int(np.clip(index, 0, len(branches) - 1))],)

    # Each branch runs in its own scope; see ``_nested_scope``.
    branch_outputs: list[list[list[IndexSet]]] = []
    for branch in branches:
        inner_consts, inner_bounds = _nested_scope(
            branch.jaxpr.constvars,
            branch.consts,
            operands,
            branch.jaxpr.invars,
            state_consts,
            state_bounds,
        )
        branch_outputs.append(
            _prop_jaxpr(branch.jaxpr, operand_indices, inner_consts, inner_bounds)
        )
        if index is not None:
            _export_scope(
                eqn.outvars,
                branch.jaxpr.outvars,
                inner_consts,
                inner_bounds,
                state_consts,
                state_bounds,
            )

    # Union across branches for each output variable
    for i, outvar in enumerate(eqn.outvars):
        merged: list[IndexSet] = _copy_index_sets(branch_outputs[0][i])
        for branch_out in branch_outputs[1:]:
            for j in range(len(merged)):
                merged[j] |= branch_out[i][j]
        state_indices[outvar] = merged
