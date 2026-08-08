#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._term import AbstractSamplingTerm
from ..stochastic._bsde import (
    bsde_objective_loss,
    BSDEControlMode,
    BSDEObjectiveMode,
    BSDEPathBatch,
    BSDEProblem,
    BSDEQuadrature,
    evaluate_bsde,
)


class BSDETerm(AbstractSamplingTerm):
    """Sampled BSDE residual term compatible with ``FunctionalSolver``."""

    problem: BSDEProblem
    fixed_paths: BSDEPathBatch | None
    terminal_weight: Array
    local_weight: Array
    global_weight: Array
    value_name: str = eqx.field(static=True)
    control_name: str | None = eqx.field(static=True)
    mode: BSDEObjectiveMode = eqx.field(static=True)
    control_mode: BSDEControlMode = eqx.field(static=True)
    quadrature: BSDEQuadrature = eqx.field(static=True)
    sampling_mode: Literal["resample", "fixed"] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        problem: BSDEProblem,
        /,
        *,
        value_name: str,
        control_name: str | None = None,
        mode: BSDEObjectiveMode = "joint",
        control_mode: BSDEControlMode = "explicit",
        quadrature: BSDEQuadrature = "left",
        terminal_weight: ArrayLike = 1.0,
        local_weight: ArrayLike = 1.0,
        global_weight: ArrayLike = 1.0,
        sampling_mode: Literal["resample", "fixed"] = "resample",
        fixed_paths: BSDEPathBatch | None = None,
        fixed_paths_key: Key[Array, ""] = jr.key(0),
        label: str | None = None,
    ):
        if not isinstance(problem, BSDEProblem):
            raise TypeError("problem must be a BSDEProblem.")
        if not isinstance(value_name, str) or not value_name:
            raise ValueError("value_name must be a non-empty string.")
        if control_mode not in ("explicit", "autodiff"):
            raise ValueError("control_mode must be 'explicit' or 'autodiff'.")
        if control_mode == "explicit" and (
            not isinstance(control_name, str) or not control_name
        ):
            raise ValueError("Explicit control requires a non-empty control_name.")
        if control_mode == "autodiff" and control_name is not None:
            raise ValueError("Autodiff control does not accept control_name.")
        if mode not in ("terminal", "local", "global", "joint"):
            raise ValueError("Unknown BSDE objective mode.")
        if quadrature not in ("left", "trapezoid"):
            raise ValueError("Unknown BSDE quadrature.")
        if sampling_mode not in ("resample", "fixed"):
            raise ValueError("sampling_mode must be 'resample' or 'fixed'.")
        if fixed_paths is not None and not isinstance(fixed_paths, BSDEPathBatch):
            raise TypeError("fixed_paths must be a BSDEPathBatch or None.")
        if sampling_mode == "resample" and fixed_paths is not None:
            raise ValueError("fixed_paths is valid only for fixed sampling.")
        weights = tuple(
            jnp.asarray(value, dtype=float).reshape(())
            for value in (terminal_weight, local_weight, global_weight)
        )
        if any(bool(~jnp.isfinite(value)) or float(value) < 0.0 for value in weights):
            raise ValueError("BSDE objective weights must be finite and nonnegative.")
        self.problem = problem
        self.fixed_paths = (
            problem.sample(fixed_paths_key)
            if sampling_mode == "fixed" and fixed_paths is None
            else fixed_paths
        )
        self.terminal_weight, self.local_weight, self.global_weight = weights
        self.value_name = value_name
        self.control_name = control_name
        self.mode = mode
        self.control_mode = control_mode
        self.quadrature = quadrature
        self.sampling_mode = sampling_mode
        self.label = None if label is None else str(label)

    def sample(self, *, key: Key[Array, ""] = jr.key(0)) -> BSDEPathBatch:
        if self.sampling_mode == "fixed":
            if self.fixed_paths is None:
                raise ValueError("Fixed BSDE objective has no fixed_paths.")
            return self.fixed_paths
        return self.problem.sample(key)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        batch: BSDEPathBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del kwargs
        if self.value_name not in functions:
            raise KeyError(f"Missing BSDE value function {self.value_name!r}.")
        value = functions[self.value_name]
        control = (
            None
            if self.control_name is None
            else functions[self.control_name]
            if self.control_name in functions
            else None
        )
        if self.control_name is not None and control is None:
            raise KeyError(f"Missing BSDE control function {self.control_name!r}.")
        sampling_key, evaluation_key = jr.split(key)
        paths = self.sample(key=sampling_key) if batch is None else batch
        evaluation = evaluate_bsde(
            self.problem,
            paths,
            value,
            control_predictor=control,
            control_mode=self.control_mode,
            quadrature=self.quadrature,
            key=evaluation_key,
        )
        return bsde_objective_loss(
            evaluation,
            mode=self.mode,
            terminal_weight=self.terminal_weight,
            local_weight=self.local_weight,
            global_weight=self.global_weight,
        )


__all__ = ["BSDETerm"]
