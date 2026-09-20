#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import optax

from ..optim._evolution_strategy import AbstractDistributionEvolutionMethod
from ..optim._iterative import (
    AbstractCompositeLeastSquaresMethod,
    AbstractLeastSquaresMethod,
    AbstractScalarIterativeMethod,
)
from ..optim._kfac._config import KFAC
from ..optim._mirror_descent import AbstractMirrorOptimizer
from ..optim._riemannian import AbstractRiemannianOptimizer
from ._functional_evolution import _solve_distribution_evolution
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
            session=config.session,
            session_every=config.session_every,
            tensorboard_log_dir=config.tensorboard_log_dir,
            tensorboard_every=config.tensorboard_every,
            tensorboard_flush_every=config.tensorboard_flush_every,
            profile_adaptive=config.profile_adaptive,
            train_term_sample_size=config.train_term_sample_size,
            gradient_accumulation=config.gradient_accumulation,
            precision=config.precision,
            training=config.training,
            resume=config.resume,
            accepted_update_hook=config.accepted_update_hook,
            target_policy=config.target_policy,
        )


@dataclass(frozen=True, slots=True)
class _EvolutionBackend:
    algorithm: AbstractDistributionEvolutionMethod

    def run(
        self,
        solver: "FunctionalSolver",
        config: FunctionalSolveConfig,
        /,
    ) -> "FunctionalSolver":
        if config.training is not None and config.training.update_alignment is not None:
            raise ValueError(
                "Functional update alignment is unsupported by evolution backends."
            )
        if config.parameter_paths is not None:
            raise ValueError(
                "Explicit parameter subspaces are unsupported by evolution backends."
            )
        return _solve_distribution_evolution(
            solver,
            num_iter=config.num_iter,
            algo=self.algorithm,
            seed=config.seed,
            jit=config.jit,
            keep_best=config.keep_best,
            log_every=config.log_every,
            log_terms=config.log_terms,
            session=config.session,
            session_every=config.session_every,
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
        if config.training is not None and config.training.update_alignment is not None:
            raise ValueError(
                "Functional update alignment is unsupported by KFAC backends."
            )
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
            session=config.session,
            session_every=config.session_every,
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
            "phydrax.optim.riemannian_sgd(...), phydrax.optim.OpenEvolutionStrategy(...), "
            "optax.adam(...), or optax.lbfgs(...)), not a string."
        )
    if isinstance(optimizer, AbstractDistributionEvolutionMethod):
        if evaluation_parameters is not None:
            raise ValueError(
                "evaluation_parameters is supported only for Optax optimizers."
            )
        return _EvolutionBackend(optimizer)
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
        "optim must be a Phydrax iterative, mirror, Riemannian, or distribution "
        "evolution method, an Optax transformation, or a KFAC configuration."
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
        backend, _EvolutionBackend
    ):
        raise ValueError(
            "Functional training plans and resume are unsupported by evolution backends."
        )
    if config.gradient_accumulation > 1:
        if not isinstance(backend, _GradientBackend):
            raise ValueError(
                "gradient_accumulation > 1 is supported only by standard Optax."
            )
        training = config.training
        if training is not None and (
            training.stateful or training.causal or training.diagnostics is not None
        ):
            raise ValueError(
                "gradient_accumulation > 1 does not support stateful, causal, "
                "term-balancing, or diagnostic functional training policies."
            )
        if (
            config.keep_best
            and config.evaluation_parameters is not None
            and (training is None or training.selection is None)
        ):
            raise ValueError(
                "Accumulated evaluation-parameter selection requires a fixed "
                "FunctionalSelectionPolicy or keep_best=False."
            )
    return backend.run(solver, config)


__all__ = ["FunctionalSolveConfig", "solve"]
