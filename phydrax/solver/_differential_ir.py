#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ..discretization import DiscretizationBundle
from ._differential import DifferentialProblem
from ._split_differential import SplitDifferentialProblem


class DeterministicDifferentialIR(StrictModule):
    """Private explicit/implicit decomposition shared by temporal backends."""

    explicit_rhs: Callable[[Array, Any, Any], Any]
    implicit_rhs: Callable[[Array, Any, Any], Any] | None
    implicit_residual: Callable[[Array, Any, Any, Any], Any] | None
    args: Any
    discretization_bundle: DiscretizationBundle | None
    state_structure: tuple[tuple[str, tuple[int, ...], str], ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    equation_form: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        explicit_rhs: Callable[[Array, Any, Any], Any],
        implicit_rhs: Callable[[Array, Any, Any], Any] | None,
        implicit_residual: Callable[[Array, Any, Any, Any], Any] | None,
        args: Any,
        discretization_bundle: DiscretizationBundle | None,
        state_structure: tuple[tuple[str, tuple[int, ...], str], ...],
        problem_id: str,
        equation_form: str,
    ):
        if not callable(explicit_rhs):
            raise TypeError("explicit_rhs must be callable.")
        if implicit_rhs is not None and not callable(implicit_rhs):
            raise TypeError("implicit_rhs must be callable or None.")
        if implicit_residual is not None and not callable(implicit_residual):
            raise TypeError("implicit_residual must be callable or None.")
        if implicit_rhs is not None and implicit_residual is not None:
            raise ValueError("An equation IR cannot carry two implicit formulations.")
        self.explicit_rhs = explicit_rhs
        self.implicit_rhs = implicit_rhs
        self.implicit_residual = implicit_residual
        self.args = args
        self.discretization_bundle = discretization_bundle
        self.state_structure = tuple(state_structure)
        self.problem_id = str(problem_id)
        self.equation_form = str(equation_form)


def _state_structure(state: Any, /) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (
            jax.tree_util.keystr(path) or "<root>",
            tuple(int(size) for size in jnp.asarray(leaf).shape),
            str(jnp.asarray(leaf).dtype),
        )
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]
    )


def lower_deterministic_problem(
    problem: DifferentialProblem | SplitDifferentialProblem, /
) -> DeterministicDifferentialIR:
    """Lower one public deterministic problem without erasing an additive split."""
    if isinstance(problem, SplitDifferentialProblem):
        return DeterministicDifferentialIR(
            explicit_rhs=problem.explicit_drift,
            implicit_rhs=problem.implicit_drift,
            implicit_residual=None,
            args=problem.args,
            discretization_bundle=problem.discretization_bundle,
            state_structure=_state_structure(problem.initial_state),
            problem_id=problem.problem_id,
            equation_form="additive-ode",
        )
    if not isinstance(problem, DifferentialProblem) or problem.stochastic:
        raise TypeError("Deterministic lowering requires a deterministic problem.")
    return DeterministicDifferentialIR(
        explicit_rhs=problem.drift,
        implicit_rhs=None,
        implicit_residual=None,
        args=problem.args,
        discretization_bundle=problem.discretization_bundle,
        state_structure=_state_structure(problem.initial_state),
        problem_id=problem.problem_id,
        equation_form="explicit-ode",
    )


__all__ = ["DeterministicDifferentialIR", "lower_deterministic_problem"]
