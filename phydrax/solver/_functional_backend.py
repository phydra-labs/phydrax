#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import optax
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm

from ..optim._kfac._config import KFAC
from ..optim._riemannian import AbstractRiemannianOptimizer
from ._functional_evosax import _solve_evosax_distribution
from ._functional_gradient import solve_gradient
from ._functional_run import FunctionalSolveConfig


if TYPE_CHECKING:
    from ._functional_solver import FunctionalSolver


class _FunctionalBackend(Protocol):
    """One backend consuming the shared prepared-objective run contract."""

    def run(
        self,
        solver: "FunctionalSolver",
        config: FunctionalSolveConfig,
        /,
    ) -> "FunctionalSolver": ...


@dataclass(frozen=True, slots=True)
class _GradientBackend:
    optimizer: Any

    def run(
        self,
        solver: "FunctionalSolver",
        config: FunctionalSolveConfig,
        /,
    ) -> "FunctionalSolver":
        return solve_gradient(
            solver,
            num_iter=config.num_iter,
            optim=self.optimizer,
            evaluation_parameters=config.evaluation_parameters,
            seed=config.seed,
            jit=config.jit,
            keep_best=config.keep_best,
            log_every=config.log_every,
            log_terms=config.log_terms,
            log_path=config.log_path,
            tensorboard_log_dir=config.tensorboard_log_dir,
            tensorboard_every=config.tensorboard_every,
            tensorboard_flush_every=config.tensorboard_flush_every,
            profile_adaptive=config.profile_adaptive,
            train_term_sample_size=config.train_term_sample_size,
        )


@dataclass(frozen=True, slots=True)
class _EvosaxBackend:
    algorithm: DistributionBasedAlgorithm

    def run(
        self,
        solver: "FunctionalSolver",
        config: FunctionalSolveConfig,
        /,
    ) -> "FunctionalSolver":
        return _solve_evosax_distribution(
            solver,
            num_iter=config.num_iter,
            algo=self.algorithm,
            seed=config.seed,
            jit=config.jit,
            keep_best=config.keep_best,
            log_every=config.log_every,
            log_terms=config.log_terms,
            log_path=config.log_path,
            tensorboard_log_dir=config.tensorboard_log_dir,
            tensorboard_every=config.tensorboard_every,
            tensorboard_flush_every=config.tensorboard_flush_every,
            profile_adaptive=config.profile_adaptive,
            train_term_sample_size=config.train_term_sample_size,
        )


@dataclass(frozen=True, slots=True)
class _KFACBackend:
    optimizer: KFAC

    def run(
        self,
        solver: "FunctionalSolver",
        config: FunctionalSolveConfig,
        /,
    ) -> "FunctionalSolver":
        from ._kfac_solver import solve_kfac

        return solve_kfac(
            solver,
            num_iter=config.num_iter,
            optim=self.optimizer,
            evaluation_parameters=config.evaluation_parameters,
            seed=config.seed,
            jit=config.jit,
            keep_best=config.keep_best,
            log_every=config.log_every,
            log_terms=config.log_terms,
            log_path=config.log_path,
            tensorboard_log_dir=config.tensorboard_log_dir,
            tensorboard_every=config.tensorboard_every,
            tensorboard_flush_every=config.tensorboard_flush_every,
            profile_adaptive=config.profile_adaptive,
            train_term_sample_size=config.train_term_sample_size,
        )


def _resolve_backend(
    optimizer: Any,
    /,
    *,
    evaluation_parameters: Any,
) -> _FunctionalBackend:
    if isinstance(optimizer, str):
        raise TypeError(
            "optim must be an optimizer object (e.g. phydrax.optim.riemannian_sgd(...), "
            "optax.adam(...), optax.lbfgs(...), or an evosax distribution-based "
            "algorithm instance), not a string."
        )
    if isinstance(optimizer, PopulationBasedAlgorithm):
        if evaluation_parameters is not None:
            raise ValueError(
                "evaluation_parameters is supported only for Optax optimizers."
            )
        raise NotImplementedError(
            "FunctionalSolver does not accept Evosax population-based algorithms: "
            "they require an explicit initial population and finite search-space "
            "semantics. For bounded geometry design, use "
            "DesignConstraintSystem.search(...)."
        )
    if isinstance(optimizer, DistributionBasedAlgorithm):
        if evaluation_parameters is not None:
            raise ValueError(
                "evaluation_parameters is supported only for Optax optimizers."
            )
        return _EvosaxBackend(optimizer)
    if isinstance(optimizer, KFAC):
        return _KFACBackend(optimizer)
    if isinstance(
        optimizer,
        (
            AbstractRiemannianOptimizer,
            optax.GradientTransformationExtraArgs,
            optax.GradientTransformation,
        ),
    ):
        return _GradientBackend(optimizer)
    raise TypeError(
        "optim must be a Phydrax Riemannian optimizer, an Optax transformation, "
        "a KFAC configuration, or an Evosax distribution-based algorithm instance."
    )


def solve(
    solver: "FunctionalSolver",
    *,
    optim: Any,
    config: FunctionalSolveConfig,
) -> "FunctionalSolver":
    """Dispatch one functional run through the common backend interface."""
    backend = _resolve_backend(
        optim,
        evaluation_parameters=config.evaluation_parameters,
    )
    return backend.run(solver, config)


__all__ = ["FunctionalSolveConfig", "solve"]
