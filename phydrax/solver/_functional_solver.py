#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import optax
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from jaxtyping import Array, Key

from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from .._term import AbstractScalarTerm, evaluate
from .._training import EvaluationParametersFn
from ..enforcement import EnforcementProgram
from ..integration import AdaptiveIntegration
from ..operators.differential._runtime import derivative_runtime_context
from ..optim._kfac._config import KFAC
from ..optim._riemannian import AbstractRiemannianOptimizer
from ..terms._residual import ResidualPenalty
from ._model_losses import function_model_loss_values


def _terms_tuple(
    value: AbstractScalarTerm | Sequence[AbstractScalarTerm],
    /,
    *,
    name: str,
) -> tuple[AbstractScalarTerm, ...]:
    out = (value,) if isinstance(value, AbstractScalarTerm) else tuple(value)
    bad = tuple(term for term in out if not isinstance(term, AbstractScalarTerm))
    if bad:
        raise TypeError(
            f"All {name} must be scalar terms; got "
            f"{tuple(type(term).__name__ for term in bad)!r}."
        )
    return out


def _adaptive_term(term: Any, /) -> ResidualPenalty:
    if not isinstance(term, ResidualPenalty) or not isinstance(
        term.source, AdaptiveIntegration
    ):
        raise TypeError(
            "Adaptive collocation requires ResidualPenalty with an "
            "AdaptiveIntegration source."
        )
    return term


class FunctionalSolver(StrictModule):
    """Optimize one ordered collection of real scalar terms over named fields.

    Penalties and signed objectives share the same term contract. A precompiled
    `EnforcementProgram`, when supplied, is applied before every term evaluation.
    """

    functions: frozendict[str, DomainFunction]
    terms: tuple[AbstractScalarTerm, ...]
    evaluation_terms: tuple[AbstractScalarTerm, ...]
    enforcement: EnforcementProgram | None
    collocation: tuple[Any | None, ...]
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
        self.terms = _terms_tuple(terms, name="terms")
        self.evaluation_terms = _terms_tuple(
            evaluation_terms,
            name="evaluation_terms",
        )
        adaptive_evaluation_terms = tuple(
            term
            for term in self.evaluation_terms
            if isinstance(term, ResidualPenalty)
            and isinstance(term.source, AdaptiveIntegration)
        )
        if adaptive_evaluation_terms:
            raise ValueError(
                "AdaptiveIntegration sources are only supported in training terms, "
                "which own solver-managed collocation populations."
            )

        if enforcement is not None and not isinstance(enforcement, EnforcementProgram):
            raise TypeError("enforcement must be an EnforcementProgram or None.")
        self.enforcement = enforcement
        collocation_keys = jr.split(collocation_key, len(self.terms))
        self.collocation = tuple(
            (
                term.policy.initialize(term, key=term_key)
                if isinstance(term, ResidualPenalty)
                and isinstance(term.source, AdaptiveIntegration)
                else None
            )
            for term, term_key in zip(self.terms, collocation_keys, strict=True)
        )
        self.training_diagnostics = frozendict()

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
        functions = self.ansatz_functions()
        keys = jr.split(key, len(self.terms))
        total = jnp.array(0.0, dtype=float)
        with derivative_runtime_context():
            for term, population, term_key in zip(
                self.terms,
                self.collocation,
                keys,
                strict=True,
            ):
                term_kwargs = dict(kwargs)
                if population is not None:
                    adaptive_term = _adaptive_term(term)
                    batch, local_weight = adaptive_term.policy.loss_batch_and_weight(
                        population
                    )
                    term_kwargs["realization"] = adaptive_term._adaptive_realization(
                        batch,
                        local_weight,
                        key=term_key,
                    )
                total = (
                    total
                    + evaluate(
                        term,
                        functions,
                        key=term_key,
                        step=step,
                        **term_kwargs,
                    ).value
                )
            iter_ = step
            for model_loss in function_model_loss_values(
                self.functions,
                key=jr.fold_in(key, len(self.terms)),
                iter_=iter_,
            ):
                total = total + model_loss
        return total

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
        from ._functional_train import solve as _solve

        if optim is None:
            optim = optax.rprop(1e-3)

        if isinstance(optim, KFAC):
            from ._kfac_solver import solve_kfac

            return solve_kfac(
                self,
                num_iter=num_iter,
                optim=optim,
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

        return _solve(
            self,
            num_iter=num_iter,
            optim=optim,
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
