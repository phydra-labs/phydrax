#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import optax
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from jaxtyping import Array, Key

from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from .._term import AbstractScalarTerm
from .._training import EvaluationParametersFn
from ..enforcement import EnforcementProgram
from ..optim._kfac._config import KFAC
from ..optim._riemannian import AbstractRiemannianOptimizer
from ._functional_objective import (
    _FunctionalObjective,
    evaluate_prepared_objective,
)


class FunctionalSolver(StrictModule):
    """Optimize one ordered collection of real scalar terms over named fields.

    Penalties and signed objectives share the same term contract. A precompiled
    `EnforcementProgram`, when supplied, is applied before every term evaluation.
    """

    functions: frozendict[str, DomainFunction]
    objective: _FunctionalObjective
    training_diagnostics: frozendict[str, Array]

    def __init__(
        self,
        *,
        functions: Mapping[str, DomainFunction],
        terms: AbstractScalarTerm | Sequence[AbstractScalarTerm],
        evaluation_terms: AbstractScalarTerm | Sequence[AbstractScalarTerm] = (),
        enforcement: EnforcementProgram | None = None,
        collocation_key: Key[Array, ""] = DOC_KEY0,
    ):
        """Create a solver from fields, scalar terms, and optional enforcement."""
        self.functions = frozendict(functions)
        self.objective = _FunctionalObjective(
            terms=terms,
            evaluation_terms=evaluation_terms,
            enforcement=enforcement,
            collocation_key=collocation_key,
        )
        self.training_diagnostics = frozendict()

    @property
    def terms(self) -> tuple[AbstractScalarTerm, ...]:
        """Return the ordered training terms."""
        return self.objective.terms

    @property
    def evaluation_terms(self) -> tuple[AbstractScalarTerm, ...]:
        """Return the ordered diagnostic-only terms."""
        return self.objective.evaluation_terms

    @property
    def enforcement(self) -> EnforcementProgram | None:
        """Return the optional exact-enforcement program."""
        return self.objective.enforcement

    @property
    def collocation(self) -> tuple[Any | None, ...]:
        """Return the population aligned with each training term."""
        return self.objective.populations

    def _with_collocation(
        self,
        collocation: Sequence[Any | None],
        /,
    ) -> "FunctionalSolver":
        objective = self.objective.with_populations(collocation)
        return eqx.tree_at(lambda solver: solver.objective, self, objective)

    def _append_training_terms(
        self,
        terms: AbstractScalarTerm | Sequence[AbstractScalarTerm],
        /,
        *,
        key: Key[Array, ""],
    ) -> "FunctionalSolver":
        objective = self.objective.append_training_terms(terms, key=key)
        return eqx.tree_at(lambda solver: solver.objective, self, objective)

    def _retain_training_prefix(self, count: int, /) -> "FunctionalSolver":
        objective = self.objective.retain_training_prefix(count)
        return eqx.tree_at(lambda solver: solver.objective, self, objective)

    def ansatz_functions(self) -> frozendict[str, DomainFunction]:
        r"""Return the current field mapping after applying enforcement (if configured)."""
        if self.enforcement is None:
            return self.functions
        return self.enforcement.apply(self.functions)

    def __getitem__(self, var: str) -> DomainFunction:
        """Convenience accessor: return the (ansatz) field named `var`."""
        return self.ansatz_functions()[var]

    def save_onnx(self, var: str, path: str | Path, /, **kwargs: Any) -> Any:
        """Export one named ansatz function to ONNX.

        This is a thin convenience wrapper around `phydrax.export.save_onnx`.
        It exports the inference function `self[var]`, not the solver, loss, or
        scalar terms.
        """
        from ..export import save_onnx

        return save_onnx(self[var], path, **kwargs)

    def partition_functions(self) -> tuple[Any, Any]:
        """Return `(trainable, non_trainable)` function PyTrees used by `solve()`."""
        from .._trainable import partition_trainable

        return partition_trainable(self.functions)

    def trainable_functions(self) -> Any:
        """Return the trainable function PyTree used as optimizer/evolution state."""
        trainable, _non_trainable = self.partition_functions()
        return trainable

    def loss(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        step: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        """Evaluate all scalar terms and model-attached losses."""
        prepared = self.objective.prepare_training(
            range(len(self.terms)),
            scale=1.0,
            evaluation_key=key,
            sampling_key=key,
            iteration=step,
            evaluation_kwargs=kwargs,
        )
        return evaluate_prepared_objective(prepared, self.functions).total

    def solve(
        self,
        *,
        num_iter: int,
        optim: optax.GradientTransformation
        | optax.GradientTransformationExtraArgs
        | DistributionBasedAlgorithm
        | KFAC
        | AbstractRiemannianOptimizer
        | None = None,
        evaluation_parameters: EvaluationParametersFn | None = None,
        seed: int = 0,
        jit: bool = True,
        keep_best: bool = True,
        log_every: int = 1,
        log_terms: bool = True,
        log_path: str | Path | None = None,
        tensorboard_log_dir: str | Path | None = None,
        tensorboard_every: int | None = None,
        tensorboard_flush_every: int = 10,
        profile_adaptive: bool = False,
        train_term_sample_size: int | None = None,
    ) -> "FunctionalSolver":
        """Run the training loop and return an updated solver.

        The optimization updates trainable inexact-array leaves of `self.functions`.
        Domains and fixed observed-data state are kept non-trainable.

        - Standard and extra-argument Optax transformations are accepted.
        - `phydrax.optim.kfac(...)` configurations are accepted and receive frozen
          sampled residual terms from this solver.
        - Phydrax Riemannian optimizers are accepted and update explicitly selected
          trainable leaves through their declared metrics and retractions.
        - Evosax distribution-based algorithms are accepted.
        - Evosax population-based algorithms require an explicit search-space contract
          and are therefore rejected; bounded geometry design uses
          `DesignConstraintSystem.search(...)`.
        - `evaluation_parameters`, when provided, maps Optax optimizer state and raw
          training parameters to the parameter view used for diagnostics, model
          selection, and the returned solver. Riemannian optimizers reject ambient
          evaluation transforms because they need not preserve manifold membership.

        During training, each term receives the one-based iteration index as the
        JAX scalar keyword `iter_`, enabling scheduled coefficients.
        If `SIGINT` or `SIGTERM` is received while the loop is active, the current
        loop exits gracefully and `solve(...)` returns the best/current solver state
        instead of terminating the calling program.

        Logging:

        - If `log_every > 0`, prints a progress line every `log_every` iterations.
        - If `log_terms=True`, also prints the per-term loss breakdown.
        - If `log_path` is provided, logs are written to that file instead of stdout.
        - If `tensorboard_log_dir` is provided, scalar training logs are written as
          TensorBoard event files. `tensorboard_every` controls the event cadence
          and defaults to `log_every` when positive, otherwise every iteration.
        - If `profile_adaptive=True`, device-synchronized refresh and optimizer wall
          times are returned in `training_diagnostics`.
        - `train_term_sample_size` optionally samples a fixed-size subset of
          stochastic training terms per optimizer step and rescales their values
          to preserve an unbiased estimate of the complete term sum.
        """
        num_iter = int(num_iter)
        if num_iter < 0:
            raise ValueError("num_iter must be non-negative.")
        if num_iter == 0:
            return self

        from ._functional_backend import solve as _solve
        from ._functional_run import FunctionalSolveConfig

        optimizer = optax.rprop(1e-3) if optim is None else optim
        config = FunctionalSolveConfig(
            num_iter=num_iter,
            evaluation_parameters=evaluation_parameters,
            seed=seed,
            jit=jit,
            keep_best=keep_best,
            log_every=log_every,
            log_terms=log_terms,
            log_path=log_path,
            tensorboard_log_dir=tensorboard_log_dir,
            tensorboard_every=tensorboard_every,
            tensorboard_flush_every=tensorboard_flush_every,
            profile_adaptive=profile_adaptive,
            train_term_sample_size=train_term_sample_size,
        )
        return _solve(self, optim=optimizer, config=config)
