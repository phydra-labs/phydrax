#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Guarded provider initial states for nonlinear solves."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from ..linalg import AbstractInitialGuessProvider, InitialGuessDiagnostics, PyTreeSpace
from ..linalg._initial_guess import _select_proposal
from ._types import NonlinearSystemProblem
from ._updates import _space_norm


def select_initial_state(
    problem: NonlinearSystemProblem,
    baseline: PyTree[Any],
    provider: AbstractInitialGuessProvider,
    /,
    *,
    args: Any = None,
) -> tuple[PyTree[Array], InitialGuessDiagnostics]:
    """Return the initial state of a nonlinear solve and its branch evidence.

    `provider.propose(args, baseline)` is untrusted. The original residual is
    evaluated at the proposal and at the native `baseline`; the proposal is
    selected only when `problem.valid` accepts it with a strictly smaller
    residual norm, or when the baseline itself is invalid. The selected state is
    stopped, so any nonlinear method started from it, including an implicit
    root, carries no derivative through the proposal or the branch. Training
    differentiates the raw proposal instead.
    """
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be a NonlinearSystemProblem.")
    if not isinstance(provider, AbstractInitialGuessProvider):
        raise TypeError("provider must be an AbstractInitialGuessProvider.")
    native = problem.validate_state(baseline)
    proposal = jax.tree.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        problem.validate_state(provider.propose(args, native)),
    )
    native_residual, native_auxiliary = problem.evaluate(native, args)
    proposal_residual, proposal_auxiliary = problem.evaluate(proposal, args)
    space = (
        PyTreeSpace(native_residual)
        if problem.residual_space is None
        else problem.residual_space
    )
    return _select_proposal(
        proposal,
        native,
        _space_norm(space, proposal_residual),
        _space_norm(space, native_residual),
        problem.valid(proposal, proposal_residual, proposal_auxiliary, args),
        provider_id=provider.provider_id,
        baseline_valid=problem.valid(native, native_residual, native_auxiliary, args),
    )


__all__ = ["select_initial_state"]
