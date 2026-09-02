#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import optax
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm

from ..optim._iterative import (
    AbstractCompositeLeastSquaresMethod,
    AbstractLeastSquaresMethod,
    AbstractScalarIterativeMethod,
)
from ..optim._kfac._config import KFAC
from ..optim._mirror_descent import AbstractMirrorOptimizer
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
            parameter_paths=config.parameter_paths,
            parameter_shapes=config.parameter_shapes,
            parameter_dtypes=config.parameter_dtypes,
            parameter_alias_groups=config.parameter_alias_groups,
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
            precision=config.precision,
            training=config.training,
            resume=config.resume,
            accepted_update_hook=config.accepted_update_hook,
            target_policy=config.target_policy,
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
        if config.parameter_paths is not None:
            raise ValueError(
                "Explicit parameter subspaces are unsupported by Evosax backends."
            )
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
        if config.parameter_paths is not None:
            raise ValueError("Explicit parameter subspaces are unsupported by KFAC.")
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
            training=config.training,
            resume=config.resume,
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
            "optim must be an optimizer object (e.g. phydrax.optim.mirror_descent(...), "
            "phydrax.optim.riemannian_sgd(...), optax.adam(...), optax.lbfgs(...), "
            "or an evosax distribution-based algorithm instance), not a string."
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
            AbstractCompositeLeastSquaresMethod,
            AbstractLeastSquaresMethod,
            AbstractScalarIterativeMethod,
            AbstractRiemannianOptimizer,
            AbstractMirrorOptimizer,
            optax.GradientTransformationExtraArgs,
            optax.GradientTransformation,
        ),
    ):
        return _GradientBackend(optimizer)
    raise TypeError(
        "optim must be a Phydrax iterative, mirror, or Riemannian optimizer, an "
        "Optax transformation, a KFAC configuration, or an Evosax "
        "distribution-based algorithm instance."
    )


def solve(
    solver: "FunctionalSolver",
    *,
    optim: Any,
    config: FunctionalSolveConfig,
) -> "FunctionalSolver":
    """Dispatch one functional run through the common backend interface."""
    if config.precision is not None and not isinstance(
        optim,
        optax.GradientTransformation,
    ):
        raise ValueError(
            "Functional precision currently supports standard Optax transforms only."
        )
    backend = _resolve_backend(
        optim,
        evaluation_parameters=config.evaluation_parameters,
    )
    if config.accepted_update_hook is not None and not isinstance(
        backend, _GradientBackend
    ):
        raise ValueError(
            "Accepted-update hooks are supported only by functional gradient backends."
        )
    if config.target_policy is not None and not isinstance(backend, _GradientBackend):
        raise ValueError(
            "Target policies are supported only by functional gradient backends."
        )
    if (config.training is not None or config.resume) and isinstance(
        backend, _EvosaxBackend
    ):
        raise ValueError(
            "Functional training plans and resume are unsupported by Evosax backends."
        )
    return backend.run(solver, config)


__all__ = ["FunctionalSolveConfig", "solve"]
