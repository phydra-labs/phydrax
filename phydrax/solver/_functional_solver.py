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
from .._strict import StrictModule
from ..constraints._base import AbstractConstraint
from ..domain._function import DomainFunction
from ._enforced_constraint_pipeline import (
    EnforcedConstraintPipelines,
    EnforcedInteriorData,
    MultiFieldEnforcedConstraint,
    SingleFieldEnforcedConstraint,
)


class FunctionalSolver(StrictModule):
    r"""Assemble constraints into a differentiable scalar loss.

    A `FunctionalSolver` holds:

    - a mapping of named fields (as `DomainFunction`s), e.g. $u_\theta$;
    - a collection of constraints $\ell_i$ producing scalar penalties.

    The solver loss is the (weighted) sum

    $$
    L = \sum_i \ell_i.
    $$

    Optionally, *enforced constraint pipelines* can be applied to replace the raw fields
    with ansatz functions that satisfy selected boundary/initial conditions exactly.

    **Evaluation**

    - `ansatz_functions()` applies any enforced pipelines and returns the effective field
      mapping used by constraints.
    - `loss(key=...)` splits the provided PRNG key into one subkey per constraint and
      sums the resulting scalar losses.

    **Training**

    `solve(...)` optimizes the inexact-array leaves inside `functions` (via Equinox
    partitioning), and passes an `iter_` counter through to constraint losses so that
    constraints can implement schedules.
    """

    functions: frozendict[str, DomainFunction]
    constraints: tuple[AbstractConstraint, ...]
    constraint_pipelines: EnforcedConstraintPipelines | None

    def __init__(
        self,
        *,
        functions: Mapping[str, DomainFunction],
        constraints: AbstractConstraint | Sequence[AbstractConstraint],
        constraint_pipelines: EnforcedConstraintPipelines | None = None,
        constraint_terms: Sequence[
            SingleFieldEnforcedConstraint | MultiFieldEnforcedConstraint
        ] = (),
        interior_data_terms: Sequence[EnforcedInteriorData] = (),
        evolution_var: str = "t",
        include_identity_remainder: bool = True,
        boundary_weight_num_reference: int = 500_000,
        boundary_weight_sampler: str = "latin_hypercube",
        boundary_weight_key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""Create a functional solver.

        **Arguments:**

        - `functions`: Mapping `{name: DomainFunction}` defining the fields.
        - `constraints`: One or more `AbstractConstraint` instances.
        - `constraint_pipelines`: Optional pre-built enforced constraint pipelines. If provided,
          do not also pass `constraint_terms`/`interior_data_terms`.
        - `constraint_terms`: Enforced constraint terms used to build `EnforcedConstraintPipelines`
          (boundary/initial ansätze).
        - `interior_data_terms`: Enforced interior data sources used to build `EnforcedConstraintPipelines`.
        - `evolution_var`: Name of the time-like label used for initial staging (default `"t"`).
        - `include_identity_remainder`: Boundary blending option for enforced pipelines.
        - `boundary_weight_num_reference`: Number of reference samples used for boundary blending weights.
        - `boundary_weight_sampler`: Sampler used to draw boundary blending references.
        - `boundary_weight_key`: PRNG key used to draw boundary blending references.
        """
        self.functions = frozendict(functions)

        if isinstance(constraints, AbstractConstraint):
            self.constraints = (constraints,)
        else:
            self.constraints = tuple(constraints)
            bad = tuple(
                c for c in self.constraints if not isinstance(c, AbstractConstraint)
            )
            if bad:
                raise TypeError(
                    "All constraints must be instances of AbstractConstraint; got "
                    f"{tuple(type(c).__name__ for c in bad)!r}."
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
                include_identity_remainder=bool(include_identity_remainder),
                num_reference=int(boundary_weight_num_reference),
                sampler=str(boundary_weight_sampler),
                key=boundary_weight_key,
            )

        self.constraint_pipelines = constraint_pipelines

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

    def loss(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate the total loss $L=\sum_i \ell_i$ over all configured constraints.

        This:

        1) applies enforced pipelines (if configured),
        2) splits `key` into one subkey per constraint,
        3) sums `constraint.loss(...)` over all constraints.

        Any additional keyword arguments are forwarded to each constraint.
        """
        if not self.constraints:
            return jnp.array(0.0, dtype=float)

        functions = self.ansatz_functions()
        keys = jr.split(key, len(self.constraints))
        total = jnp.array(0.0, dtype=float)
        for c, k in zip(self.constraints, keys, strict=True):
            total = total + c.loss(functions, key=k, **kwargs)
        return total

    def solve(
        self,
        *,
        num_iter: int,
        optim: optax.GradientTransformation
        | optax.GradientTransformationExtraArgs
        | Any
        | None = None,
        seed: int = 0,
        jit: bool = True,
        keep_best: bool = True,
        log_every: int = 1,
        log_constraints: bool = True,
        log_path: str | Path | None = None,
    ) -> "FunctionalSolver":
        """Run the training loop and return an updated solver.

        The optimization updates the inexact-array leaves of `self.functions`.

        - If `optim` is an Optax `GradientTransformation`, a standard gradient step is used.
        - If `optim` is an Optax `GradientTransformationExtraArgs`, a line-search style update is used.
        - Otherwise, `optim` is treated as an evosax algorithm instance.

        During training, each constraint loss receives an `iter_` keyword argument (the
        1-based iteration index as a JAX scalar) to enable schedules.

        Logging:

        - If `log_every > 0`, prints a progress line every `log_every` iterations.
        - If `log_constraints=True`, also prints the per-constraint loss breakdown.
        - If `log_path` is provided, logs are written to that file instead of stdout.
        """
        from ._functional_train import solve as _solve

        if optim is None:
            optim = optax.rprop(1e-3)

        return _solve(
            self,
            num_iter=num_iter,
            optim=optim,
            seed=seed,
            jit=jit,
            keep_best=keep_best,
            log_every=log_every,
            log_constraints=log_constraints,
            log_path=log_path,
        )
