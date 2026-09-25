#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Training-kernel update rules for Phydrax native optimizers.

Native least-squares, scalar iterative, and Riemannian line-search methods run
their own damping or globalization transaction: one `step` either moves to an
accepted point or keeps the parameters and records the rejection (damping,
counters, line-search memory) in the method state. Their kernel rules forward
that decision and commit the whole method state on a finite rejection, because
every field of a rejected method state is defined at the unchanged parameters.
Mirror and non-searching Riemannian optimizers accept every finite update.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._training_kernel import AbstractKernelUpdateRule, KernelUpdateContext
from ._iterative import (
    AbstractCompositeLeastSquaresMethod,
    AbstractLeastSquaresMethod,
    AbstractScalarIterativeMethod,
    OptimizationTermination,
)
from ._mirror_descent import AbstractMirrorOptimizer
from ._riemannian import (
    AbstractRiemannianLineSearchOptimizer,
    AbstractRiemannianOptimizer,
)


# Payload-bound problem builders: `(model_state, fixed, payload)` of one attempt.
ResidualBuilder = Callable[[Any, Any, Any], Callable[[PyTree[Any]], PyTree[Any]]]
CompositeProblemBuilder = Callable[[Any, Any, Any], Any]

# Diagnostic fields of native method states that hold not-a-number sentinels
# until a method has produced them; they never gate a commit.
_SENTINEL_FIELDS = frozenset({"metrics", "initial_optimality_norm"})


@final
class NativeMethodState(StrictModule):
    """Kernel rule state wrapping one native optimizer state."""

    method_state: Any


def _finite_numeric_state(state: Any, /) -> Array:
    """Finite check of a native method state, skipping sentinel diagnostics.

    Sentinel fields (`metrics`, `initial_optimality_norm`) are reported values,
    never inputs to the next step's arithmetic, so they do not gate a commit.
    """
    values = (
        [
            getattr(state, field.name)
            for field in dataclasses.fields(state)
            if field.name not in _SENTINEL_FIELDS
        ]
        if dataclasses.is_dataclass(state)
        else [state]
    )
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(values):
        if eqx.is_inexact_array(leaf):
            finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


class _AbstractNativeMethodRule(AbstractKernelUpdateRule):
    """Native method running its own accept/reject transaction per attempt.

    A native method state may depend on the problem's structure (for example a
    prepared linear-refresh state), which its first step would otherwise add.
    The kernel requires a structure-stable rule state, so the frontend passes
    `prepared_state` (the method's `prepare_state` on a representative attempt)
    and `init` starts from it.
    """

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ("method_state",)
    method: eqx.AbstractVar[Any]
    prepared_state: eqx.AbstractVar[Any]

    def init(self, parameters: PyTree[Any], /) -> NativeMethodState:
        if self.prepared_state is None:
            return NativeMethodState(self.method.init(parameters))
        return NativeMethodState(self.prepared_state)

    @property
    def forms_own_derivatives(self) -> bool:
        return True

    def rule_state_finite(self, rule_state: NativeMethodState, /) -> Array:
        return _finite_numeric_state(rule_state.method_state)


@final
class LeastSquaresUpdateRule(_AbstractNativeMethodRule):
    """Native nonlinear least-squares method as a training-kernel rule.

    `residual(model_state, fixed, payload)` returns the residual function of the
    attempt; the method forms its own Jacobian products. The method's accepted
    flag decides acceptance; a finite rejection commits the method state (damping,
    counters, linear refresh state), which it defines at the unchanged parameters.
    """

    method: AbstractLeastSquaresMethod
    prepared_state: Any
    residual: ResidualBuilder = eqx.field(static=True)
    termination: OptimizationTermination | None = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractLeastSquaresMethod,
        residual: ResidualBuilder,
        /,
        *,
        termination: OptimizationTermination | None,
        rule_id: str,
        prepared_state: Any = None,
    ):
        if not isinstance(method, AbstractLeastSquaresMethod):
            raise TypeError("method must be an AbstractLeastSquaresMethod.")
        if not callable(residual):
            raise TypeError("residual must be callable.")
        self.method = method
        self.prepared_state = prepared_state
        self.residual = residual
        self.termination = termination
        self.rule_id = canonical_fingerprint(
            {
                "kind": "least-squares-rule",
                "method": method.method_id,
                "frontend": rule_id,
            }
        )

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: NativeMethodState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], NativeMethodState, NativeMethodState, Array]:
        del gradients, value
        residual = self.residual(context.model_state, context.fixed, context.payload)
        candidate, method_state, _ = self.method.step(
            residual,
            parameters,
            rule_state.method_state,
            termination=self.termination,
        )
        state = NativeMethodState(method_state)
        return candidate, state, state, self.method.step_metrics(method_state).accepted


@final
class CompositeLeastSquaresUpdateRule(_AbstractNativeMethodRule):
    """Native composite (residual plus scalar) least-squares method as a rule.

    `problem(model_state, fixed, payload)` returns the composite problem of the
    attempt; acceptance and rejection state follow `LeastSquaresUpdateRule`.
    """

    method: AbstractCompositeLeastSquaresMethod
    prepared_state: Any
    problem: CompositeProblemBuilder = eqx.field(static=True)
    termination: OptimizationTermination | None = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractCompositeLeastSquaresMethod,
        problem: CompositeProblemBuilder,
        /,
        *,
        termination: OptimizationTermination | None,
        rule_id: str,
        prepared_state: Any = None,
    ):
        if not isinstance(method, AbstractCompositeLeastSquaresMethod):
            raise TypeError("method must be an AbstractCompositeLeastSquaresMethod.")
        if not callable(problem):
            raise TypeError("problem must be callable.")
        self.method = method
        self.prepared_state = prepared_state
        self.problem = problem
        self.termination = termination
        self.rule_id = canonical_fingerprint(
            {
                "kind": "composite-least-squares-rule",
                "method": method.method_id,
                "frontend": rule_id,
            }
        )

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: NativeMethodState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], NativeMethodState, NativeMethodState, Array]:
        del gradients, value
        problem = self.problem(context.model_state, context.fixed, context.payload)
        candidate, method_state, _ = self.method.step(
            problem,
            parameters,
            rule_state.method_state,
            termination=self.termination,
            args=None,
        )
        state = NativeMethodState(method_state)
        return candidate, state, state, self.method.step_metrics(method_state).accepted


@final
class ScalarIterativeUpdateRule(_AbstractNativeMethodRule):
    """Native scalar iterative method (Newton-Krylov, quasi-Newton) as a rule.

    The method minimizes the kernel's admission-masked `objective_value` of the
    attempt and forms its own derivatives of it; acceptance and rejection state
    follow the method's own globalization.
    """

    method: AbstractScalarIterativeMethod
    prepared_state: Any
    termination: OptimizationTermination | None = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractScalarIterativeMethod,
        /,
        *,
        termination: OptimizationTermination | None,
        rule_id: str,
        prepared_state: Any = None,
    ):
        if not isinstance(method, AbstractScalarIterativeMethod):
            raise TypeError("method must be an AbstractScalarIterativeMethod.")
        self.method = method
        self.prepared_state = prepared_state
        self.termination = termination
        self.rule_id = canonical_fingerprint(
            {
                "kind": "scalar-iterative-rule",
                "method": method.method_id,
                "frontend": rule_id,
            }
        )

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: NativeMethodState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], NativeMethodState, NativeMethodState, Array]:
        del gradients, value
        candidate, method_state, _ = self.method.step(
            context.objective_value,
            parameters,
            rule_state.method_state,
            termination=self.termination,
        )
        state = NativeMethodState(method_state)
        return candidate, state, state, self.method.step_metrics(method_state).accepted


@final
class MirrorUpdateRule(AbstractKernelUpdateRule):
    """Mirror-descent optimizer that accepts every finite update."""

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ()
    optimizer: AbstractMirrorOptimizer
    rule_id: str = eqx.field(static=True)

    def __init__(self, optimizer: AbstractMirrorOptimizer, /, *, rule_id: str):
        if not isinstance(optimizer, AbstractMirrorOptimizer):
            raise TypeError("optimizer must be an AbstractMirrorOptimizer.")
        self.optimizer = optimizer
        self.rule_id = canonical_fingerprint(
            {
                "kind": "mirror-rule",
                "optimizer": optimizer.optimizer_id,
                "frontend": rule_id,
            }
        )

    def init(self, parameters: PyTree[Any], /) -> Any:
        return self.optimizer.init(parameters)

    def rule_state_finite(self, rule_state: Any, /) -> Array:
        return _finite_numeric_state(rule_state)

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: Any,
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], Any, Any, Array]:
        del value, context
        candidate, next_state = self.optimizer.update(gradients, rule_state, parameters)
        return candidate, next_state, rule_state, jnp.asarray(True)


@final
class RiemannianUpdateRule(AbstractKernelUpdateRule):
    """Riemannian optimizer without a line search; accepts every finite update."""

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ()
    optimizer: AbstractRiemannianOptimizer
    rule_id: str = eqx.field(static=True)

    def __init__(self, optimizer: AbstractRiemannianOptimizer, /, *, rule_id: str):
        if not isinstance(optimizer, AbstractRiemannianOptimizer) or isinstance(
            optimizer, AbstractRiemannianLineSearchOptimizer
        ):
            raise TypeError(
                "optimizer must be a Riemannian optimizer without a line search."
            )
        self.optimizer = optimizer
        self.rule_id = canonical_fingerprint(
            {
                "kind": "riemannian-rule",
                "optimizer": optimizer.optimizer_id,
                "frontend": rule_id,
            }
        )

    def init(self, parameters: PyTree[Any], /) -> Any:
        return self.optimizer.init(parameters)

    def rule_state_finite(self, rule_state: Any, /) -> Array:
        return _finite_numeric_state(rule_state)

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: Any,
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], Any, Any, Array]:
        del value, context
        candidate, next_state = self.optimizer.update(gradients, rule_state, parameters)
        return candidate, next_state, rule_state, jnp.asarray(True)


@final
class RiemannianLineSearchUpdateRule(AbstractKernelUpdateRule):
    """Riemannian line-search optimizer (conjugate gradient, L-BFGS) as a rule.

    The line search consumes the kernel's admission-masked `objective_value`.
    Its accepted flag decides acceptance; a finite rejection commits the method
    state, which records the failed search at the unchanged parameters (so the
    next attempt restarts along steepest descent).
    """

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ("method_state",)
    optimizer: AbstractRiemannianLineSearchOptimizer
    rule_id: str = eqx.field(static=True)

    def __init__(
        self, optimizer: AbstractRiemannianLineSearchOptimizer, /, *, rule_id: str
    ):
        if not isinstance(optimizer, AbstractRiemannianLineSearchOptimizer):
            raise TypeError("optimizer must be a Riemannian line-search optimizer.")
        self.optimizer = optimizer
        self.rule_id = canonical_fingerprint(
            {
                "kind": "riemannian-line-search-rule",
                "optimizer": optimizer.optimizer_id,
                "frontend": rule_id,
            }
        )

    def init(self, parameters: PyTree[Any], /) -> NativeMethodState:
        return NativeMethodState(self.optimizer.init(parameters))

    def rule_state_finite(self, rule_state: NativeMethodState, /) -> Array:
        return _finite_numeric_state(rule_state.method_state)

    def propose(
        self,
        parameters: PyTree[Any],
        gradients: PyTree[Any],
        value: Array,
        rule_state: NativeMethodState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[PyTree[Any], NativeMethodState, NativeMethodState, Array]:
        candidate, method_state = self.optimizer.update(
            gradients,
            rule_state.method_state,
            parameters,
            value=value,
            value_fn=context.objective_value,
        )
        state = NativeMethodState(method_state)
        accepted = self.optimizer.step_metrics(method_state).line_search_accepted
        return candidate, state, state, accepted


__all__ = [
    "CompositeLeastSquaresUpdateRule",
    "LeastSquaresUpdateRule",
    "MirrorUpdateRule",
    "NativeMethodState",
    "RiemannianLineSearchUpdateRule",
    "RiemannianUpdateRule",
    "ScalarIterativeUpdateRule",
]
