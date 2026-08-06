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
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._objective import AbstractObjectiveTerm
from .._strict import StrictModule
from .._training import EvaluationParametersFn
from ..constraints._base import AbstractConstraint
from ..constraints._functional import FunctionalConstraint
from ..domain._base import EnforcementGateMethod
from ..domain._function import DomainFunction
from ..operators.differential._runtime import derivative_runtime_context
from ._enforced_constraint_pipeline import (
    EnforcedConstraintPipelines,
    EnforcedInteriorData,
    MultiFieldEnforcedConstraint,
    SingleFieldEnforcedConstraint,
)
from ._model_losses import function_model_loss_values


def _constraints_tuple(
    value: AbstractConstraint | Sequence[AbstractConstraint],
    /,
    *,
    name: str,
) -> tuple[AbstractConstraint, ...]:
    if isinstance(value, AbstractConstraint):
        out = (value,)
    else:
        out = tuple(value)
    bad = tuple(c for c in out if not isinstance(c, AbstractConstraint))
    if bad:
        raise TypeError(
            f"All {name} must be instances of AbstractConstraint; got "
            f"{tuple(type(c).__name__ for c in bad)!r}."
        )
    return out


def _objectives_tuple(
    value: AbstractObjectiveTerm | Sequence[AbstractObjectiveTerm],
    /,
) -> tuple[AbstractObjectiveTerm, ...]:
    if isinstance(value, AbstractObjectiveTerm):
        out = (value,)
    else:
        out = tuple(value)
    bad = tuple(term for term in out if not isinstance(term, AbstractObjectiveTerm))
    if bad:
        raise TypeError(
            "All objectives must be instances of AbstractObjectiveTerm; got "
            f"{tuple(type(term).__name__ for term in bad)!r}."
        )
    return out


class FunctionalSolver(StrictModule):
    r"""Assemble constraints, raw objectives, and model losses into a scalar functional.

    A `FunctionalSolver` holds:

    - a mapping of named fields (as `DomainFunction`s), e.g. $u_\theta$;
    - a collection of constraints $\ell_i$ producing scalar penalties;
    - optional raw signed objective terms $\mathcal F_j$;
    - optional model-level losses attached to the trainable models;
    - optional eval-only constraints for validation diagnostics.

    The solver functional is

    $$
    \mathcal J = \sum_i \ell_i + \sum_j \mathcal F_j + \sum_k r_k.
    $$

    Optionally, *enforced constraint pipelines* can be applied to replace the raw fields
    with ansatz functions that satisfy selected boundary/initial conditions exactly.

    **Evaluation**

    - `ansatz_functions()` applies any enforced pipelines and returns the effective field
      mapping used by constraints.
    - `loss(key=...)` splits the provided PRNG key into one subkey per constraint,
      evaluates any attached model losses, and sums the resulting scalar losses.

    **Training**

    `solve(...)` optimizes trainable inexact-array leaves inside `functions` (via a
    Phydrax-aware Equinox partition). Domains and fixed observed-data state remain
    numeric/JAX-traceable but are excluded from gradients and optimizer updates.
    The solver also passes an `iter_` counter through to constraint losses so that
    constraints can implement schedules.
    """

    functions: frozendict[str, DomainFunction]
    constraints: tuple[AbstractConstraint, ...]
    objectives: tuple[AbstractObjectiveTerm, ...]
    eval_constraints: tuple[AbstractConstraint, ...]
    constraint_pipelines: EnforcedConstraintPipelines | None
    collocation: tuple[Any | None, ...]
    training_diagnostics: frozendict[str, Array]

    def __init__(
        self,
        *,
        functions: Mapping[str, DomainFunction],
        constraints: AbstractConstraint | Sequence[AbstractConstraint],
        eval_constraints: AbstractConstraint | Sequence[AbstractConstraint] = (),
        objectives: AbstractObjectiveTerm | Sequence[AbstractObjectiveTerm] = (),
        constraint_pipelines: EnforcedConstraintPipelines | None = None,
        constraint_terms: Sequence[
            SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint
        ] = (),
        interior_data_terms: Sequence[EnforcedInteriorData] = (),
        evolution_var: str = "t",
        include_identity_remainder: bool = True,
        gate_method: EnforcementGateMethod = "auto",
        gate_saturation_fraction: float = 0.5,
        gate_linear_fraction: float = 0.5,
        boundary_weight_num_reference: int = 500_000,
        boundary_weight_sampler: str = "latin_hypercube",
        boundary_weight_key: Key[Array, ""] = DOC_KEY0,
        collocation_key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""Create a functional solver.

        **Arguments:**

        - `functions`: Mapping `{name: DomainFunction}` defining the fields.
        - `constraints`: One or more `AbstractConstraint` instances.
        - `objectives`: Optional raw scalar objective terms, including signed integral functionals.
        - `eval_constraints`: Optional constraints evaluated only for logging/diagnostics.
        - `constraint_pipelines`: Optional pre-built enforced constraint pipelines. If provided,
          do not also pass `constraint_terms`/`interior_data_terms`.
        - `constraint_terms`: Enforced constraint terms used to build `EnforcedConstraintPipelines`
          (boundary/initial ansätze).
        - `interior_data_terms`: Enforced interior data sources used to build `EnforcedConstraintPipelines`.
        - `evolution_var`: Name of the time-like label used for initial staging (default `"t"`).
        - `include_identity_remainder`: Boundary blending option for enforced pipelines.
        - `gate_method`: CAD enforcement-gate implementation. ``"auto"`` selects the
          global R-equivalence gate; ``"compact"`` selects the compact fallback.
        - `gate_saturation_fraction`: Relative extent of compact CAD gates.
        - `gate_linear_fraction`: Fraction of the compact gate extent retaining a
          linear boundary profile.
        - `boundary_weight_num_reference`: Number of reference samples used for boundary blending weights.
        - `boundary_weight_sampler`: Sampler used to draw boundary blending references.
        - `boundary_weight_key`: PRNG key used to draw boundary blending references.
        """
        self.functions = frozendict(functions)
        self.constraints = _constraints_tuple(constraints, name="constraints")
        self.objectives = _objectives_tuple(objectives)
        self.eval_constraints = _constraints_tuple(
            eval_constraints,
            name="eval_constraints",
        )

        if constraint_pipelines is not None and (constraint_terms or interior_data_terms):
            raise ValueError(
                "Provide either constraint_pipelines=... or constraint_terms/interior_data_terms, not both."
            )

        if constraint_pipelines is None and (constraint_terms or interior_data_terms):
            constraint_pipelines = EnforcedConstraintPipelines.build(
                functions=self.functions,
                constraints=constraint_terms,
                interior_data=interior_data_terms,
                evolution_var=str(evolution_var),
                gate_method=gate_method,
                include_identity_remainder=bool(include_identity_remainder),
                gate_saturation_fraction=gate_saturation_fraction,
                gate_linear_fraction=gate_linear_fraction,
                num_reference=int(boundary_weight_num_reference),
                sampler=str(boundary_weight_sampler),
                key=boundary_weight_key,
            )

        self.constraint_pipelines = constraint_pipelines
        collocation_keys = jr.split(collocation_key, len(self.constraints))
        self.collocation = tuple(
            (
                constraint.collocation_policy.initialize(constraint, key=constraint_key)
                if isinstance(constraint, FunctionalConstraint)
                and constraint.collocation_policy is not None
                else None
            )
            for constraint, constraint_key in zip(
                self.constraints, collocation_keys, strict=True
            )
        )
        self.training_diagnostics = frozendict()

    def ansatz_functions(self) -> frozendict[str, DomainFunction]:
        r"""Return the current field mapping after applying enforced pipelines (if configured)."""
        if self.constraint_pipelines is None:
            return self.functions
        return self.constraint_pipelines.apply(self.functions)

    def enforced_functions(self) -> frozendict[str, DomainFunction]:
        """Alias for `ansatz_functions()`."""
        return self.ansatz_functions()

    def __getitem__(self, var: str) -> DomainFunction:
        """Convenience accessor: return the (ansatz) field named `var`."""
        return self.ansatz_functions()[var]

    def save_onnx(self, var: str, path: str | Path, /, **kwargs: Any) -> Any:
        """Export one named ansatz function to ONNX.

        This is a thin convenience wrapper around `phydrax.export.save_onnx`.
        It exports the inference function `self[var]`, not the solver, loss, or
        constraints.
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
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate constraints, raw objectives, and attached model losses.

        This:

        1) applies enforced pipelines (if configured),
        2) splits `key` across constraints and raw objectives,
        3) sums every `loss(...)` value,
        4) adds scalar model losses attached via `model.add_model_loss(...)` or
           model `__loss__` hooks.

        Additional keyword arguments are forwarded to constraints and objectives.
        """
        functions = self.ansatz_functions()
        num_terms = len(self.constraints) + len(self.objectives)
        keys = jr.split(key, num_terms)
        constraint_keys = keys[: len(self.constraints)]
        objective_keys = keys[len(self.constraints) :]
        total = jnp.array(0.0, dtype=float)
        with derivative_runtime_context():
            for c, population, k in zip(
                self.constraints, self.collocation, constraint_keys, strict=True
            ):
                if population is not None:
                    if not isinstance(c, FunctionalConstraint):
                        raise TypeError(
                            "Adaptive collocation is only valid for FunctionalConstraint."
                        )
                    policy = c.collocation_policy
                    if policy is None:
                        raise ValueError(
                            "Adaptive population requires a collocation policy."
                        )
                    batch, batch_weight = policy.loss_batch_and_weight(population)
                    term = c.loss(
                        functions,
                        key=k,
                        batch=batch,
                        batch_weight=batch_weight,
                        **kwargs,
                    )
                else:
                    term = c.loss(functions, key=k, **kwargs)
                total = total + term
            for objective, objective_key in zip(
                self.objectives, objective_keys, strict=True
            ):
                total = total + objective.loss(functions, key=objective_key, **kwargs)
            iter_ = kwargs.get("iter_", None)
            for term in function_model_loss_values(
                self.functions,
                key=jr.fold_in(key, num_terms),
                iter_=iter_,
            ):
                total = total + term
        return total

    def solve(
        self,
        *,
        num_iter: int,
        optim: optax.GradientTransformation
        | optax.GradientTransformationExtraArgs
        | Any
        | None = None,
        evaluation_parameters: EvaluationParametersFn | None = None,
        seed: int = 0,
        jit: bool = True,
        keep_best: bool = True,
        log_every: int = 1,
        log_constraints: bool = True,
        log_path: str | Path | None = None,
        tensorboard_log_dir: str | Path | None = None,
        tensorboard_every: int | None = None,
        tensorboard_flush_every: int = 10,
        profile_adaptive: bool = False,
        train_constraint_sample_size: int | None = None,
    ) -> "FunctionalSolver":
        """Run the training loop and return an updated solver.

        The optimization updates trainable inexact-array leaves of `self.functions`.
        Domains and fixed observed-data state are kept non-trainable.

        - If `optim` is an Optax `GradientTransformation`, a standard gradient step is used.
        - If `optim` is an Optax `GradientTransformationExtraArgs`, a line-search style update is used.
        - Otherwise, `optim` is treated as an evosax algorithm instance.
        - `evaluation_parameters`, when provided, maps optimizer state and raw training
          parameters to the parameter view used for diagnostics, model selection, and
          the returned solver. This supports optimizers such as Optax schedule-free
          transformations without changing the gradient/update parameter lifecycle.

        During training, each constraint loss receives an `iter_` keyword argument (the
        1-based iteration index as a JAX scalar) to enable schedules.
        If `SIGINT` or `SIGTERM` is received while the loop is active, the current
        loop exits gracefully and `solve(...)` returns the best/current solver state
        instead of terminating the calling program.

        Logging:

        - If `log_every > 0`, prints a progress line every `log_every` iterations.
        - If `log_constraints=True`, also prints the per-constraint loss breakdown.
        - If `log_path` is provided, logs are written to that file instead of stdout.
        - If `tensorboard_log_dir` is provided, scalar training logs are written as
          TensorBoard event files. `tensorboard_every` controls the event cadence
          and defaults to `log_every` when positive, otherwise every iteration.
        - If `profile_adaptive=True`, device-synchronized refresh and optimizer wall
          times are returned in `training_diagnostics`.
        - `train_constraint_sample_size` optionally samples a fixed-size subset of
          training constraints per optimizer step and rescales their losses to
          keep an unbiased estimate of the full constraint sum. This can reduce
          JIT compile time when many constraints have different static shapes.
        """
        from ._functional_train import solve as _solve

        if optim is None:
            optim = optax.rprop(1e-3)

        return _solve(
            self,
            num_iter=num_iter,
            optim=optim,
            evaluation_parameters=evaluation_parameters,
            seed=seed,
            jit=jit,
            keep_best=keep_best,
            log_every=log_every,
            log_constraints=log_constraints,
            log_path=log_path,
            tensorboard_log_dir=tensorboard_log_dir,
            tensorboard_every=tensorboard_every,
            tensorboard_flush_every=tensorboard_flush_every,
            profile_adaptive=profile_adaptive,
            train_constraint_sample_size=train_constraint_sample_size,
        )
