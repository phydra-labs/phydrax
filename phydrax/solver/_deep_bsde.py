#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.random as jr
import optax
from jaxtyping import ArrayLike

from .._strict import StrictModule
from ..stochastic._bsde import BSDEPathBatch, BSDEProblem
from ..terms._deep_bsde import (
    deep_bsde_rollout,
    deep_bsde_shooting_diagnostics,
    DeepBSDERollout,
    DeepBSDESamplingMode,
    DeepBSDEShootingDiagnostics,
    DeepBSDEShootingTerm,
)
from ._functional_solver import FunctionalSolver


class DeepBSDEResult(StrictModule):
    """Trained shooting solver and an independent validation rollout."""

    solver: FunctionalSolver
    rollout: DeepBSDERollout
    diagnostics: DeepBSDEShootingDiagnostics
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    initial_value_name: str = eqx.field(static=True)
    control_name: str = eqx.field(static=True)


def solve_deep_bsde(
    solver: FunctionalSolver,
    problem: BSDEProblem,
    /,
    *,
    initial_value_name: str,
    control_name: str,
    num_iter: int,
    gradient_accumulation: int = 1,
    optim: Any = None,
    terminal_weight: ArrayLike = 1.0,
    sampling_mode: DeepBSDESamplingMode = "resample",
    fixed_paths: BSDEPathBatch | None = None,
    validation_paths: BSDEPathBatch | None = None,
    seed: int = 0,
    jit: bool = True,
    keep_best: bool = True,
    log_every: int = 0,
) -> DeepBSDEResult:
    """Train an initial value and time-conditioned control by terminal shooting."""
    if not isinstance(solver, FunctionalSolver) or not isinstance(problem, BSDEProblem):
        raise TypeError("solver and problem must be FunctionalSolver and BSDEProblem.")
    steps = int(num_iter)
    if steps < 1:
        raise ValueError("num_iter must be positive.")
    functions = solver.ansatz_functions()
    if initial_value_name not in functions:
        raise KeyError(
            f"Missing Deep BSDE initial-value function {initial_value_name!r}."
        )
    if control_name not in functions:
        raise KeyError(f"Missing Deep BSDE control function {control_name!r}.")
    if optim is None:
        optim = optax.adam(1e-3)
    root_key = jr.key(int(seed))
    objective = DeepBSDEShootingTerm(
        problem,
        initial_value_name=initial_value_name,
        control_name=control_name,
        terminal_weight=terminal_weight,
        sampling_mode=sampling_mode,
        fixed_paths=fixed_paths,
        fixed_paths_key=jr.fold_in(root_key, 100),
        label="deep-bsde-shooting",
    )
    base_term_count = len(solver.terms)
    temporary = solver._append_training_terms(
        objective,
        key=jr.fold_in(root_key, 101),
    )
    trained = temporary.solve(
        num_iter=steps,
        gradient_accumulation=gradient_accumulation,
        optim=optim,
        seed=int(seed),
        jit=jit,
        keep_best=keep_best,
        log_every=log_every,
    )
    trained = trained._retain_training_prefix(base_term_count)
    paths = (
        problem.sample(jr.fold_in(root_key, 200))
        if validation_paths is None
        else validation_paths
    )
    trained_functions = trained.ansatz_functions()
    rollout = deep_bsde_rollout(
        problem,
        paths,
        trained_functions[initial_value_name],
        trained_functions[control_name],
        key=jr.fold_in(root_key, 201),
    )
    diagnostics = deep_bsde_shooting_diagnostics(rollout)
    return DeepBSDEResult(
        solver=trained,
        rollout=rollout,
        diagnostics=diagnostics,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        initial_value_name=initial_value_name,
        control_name=control_name,
    )


__all__ = ["DeepBSDEResult", "solve_deep_bsde"]
