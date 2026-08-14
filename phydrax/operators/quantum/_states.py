#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from phydrax.domain import DomainFunction

from ..._strict import StrictModule
from ._validation import (
    join_function_arguments,
    validate_matrix_value,
    validate_vector_value,
)


class _StateNormCallable(StrictModule):
    state: DomainFunction

    def __init__(self, state: DomainFunction):
        self.state = state

    def __call__(self, *args, key=None, **kwargs):
        state = validate_vector_value(
            self.state.func(*args, key=key, **kwargs),
            role="quantum state",
        )
        return jnp.real(jnp.vdot(state, state)) - 1.0


class _StateObservableCallable(StrictModule):
    state: DomainFunction
    observable: DomainFunction
    state_positions: tuple[int, ...]
    observable_positions: tuple[int, ...]
    operation: str

    def __init__(
        self,
        state: DomainFunction,
        observable: DomainFunction,
        state_positions: tuple[int, ...],
        observable_positions: tuple[int, ...],
        operation: Literal["expectation", "variance"],
    ):
        self.state = state
        self.observable = observable
        self.state_positions = state_positions
        self.observable_positions = observable_positions
        self.operation = operation

    def __call__(self, *args, key=None, **kwargs):
        state_args = tuple(args[index] for index in self.state_positions)
        observable_args = tuple(args[index] for index in self.observable_positions)
        state = validate_vector_value(
            self.state.func(*state_args, key=key, **kwargs),
            role="quantum state",
        )
        observable = validate_matrix_value(
            self.observable.func(*observable_args, key=key, **kwargs),
            role="observable",
        )
        if int(observable.shape[1]) != int(state.shape[0]):
            raise ValueError(
                "Observable and quantum-state dimensions must match; "
                f"got {observable.shape} and {state.shape}."
            )
        action = observable @ state
        expectation = jnp.vdot(state, action)
        if self.operation == "expectation":
            return expectation
        if self.operation == "variance":
            second_moment = jnp.vdot(state, observable @ action)
            return second_moment - expectation * expectation
        raise RuntimeError(f"Unknown state-observable operation {self.operation!r}.")


class _DensityExpectationCallable(StrictModule):
    density: DomainFunction
    observable: DomainFunction
    density_positions: tuple[int, ...]
    observable_positions: tuple[int, ...]

    def __init__(
        self,
        density: DomainFunction,
        observable: DomainFunction,
        density_positions: tuple[int, ...],
        observable_positions: tuple[int, ...],
    ):
        self.density = density
        self.observable = observable
        self.density_positions = density_positions
        self.observable_positions = observable_positions

    def __call__(self, *args, key=None, **kwargs):
        density_args = tuple(args[index] for index in self.density_positions)
        observable_args = tuple(args[index] for index in self.observable_positions)
        density = validate_matrix_value(
            self.density.func(*density_args, key=key, **kwargs),
            role="density operator",
        )
        observable = validate_matrix_value(
            self.observable.func(*observable_args, key=key, **kwargs),
            role="observable",
        )
        if density.shape != observable.shape:
            raise ValueError(
                "Density-operator and observable dimensions must match; "
                f"got {density.shape} and {observable.shape}."
            )
        return jnp.trace(density @ observable)


class _DensityFromFactorCallable(StrictModule):
    factor: DomainFunction

    def __init__(self, factor: DomainFunction):
        self.factor = factor

    def __call__(self, *args, key=None, **kwargs):
        factor = jnp.asarray(self.factor.func(*args, key=key, **kwargs))
        if factor.ndim != 2 or int(factor.shape[0]) == 0 or int(factor.shape[1]) == 0:
            raise ValueError(
                "density_from_factor factor must have shape (n, r) with n, r > 0; "
                f"got {factor.shape}."
            )
        unnormalized = factor @ jnp.conj(factor.T)
        trace = jnp.real(jnp.trace(unnormalized))
        checked_trace = eqx.error_if(
            trace,
            trace == 0,
            "density_from_factor factor must have nonzero Frobenius norm.",
        )
        return unnormalized / checked_trace


def state_norm_residual(state: DomainFunction, /) -> DomainFunction:
    r"""Return the normalization residual $\langle\psi|\psi\rangle-1$.

    The state must be vector-valued. The returned residual is explicitly real.
    """
    if not isinstance(state, DomainFunction):
        raise TypeError("state_norm_residual expects a DomainFunction.")
    return DomainFunction(
        domain=state.domain,
        deps=state.deps,
        func=_StateNormCallable(state),
        metadata={},
    )


def _state_observable_operation(
    state: DomainFunction,
    observable: DomainFunction,
    /,
    *,
    operation: Literal["expectation", "variance"],
) -> DomainFunction:
    if not isinstance(state, DomainFunction) or not isinstance(
        observable, DomainFunction
    ):
        raise TypeError(f"{operation} expects state and observable DomainFunctions.")
    domain, deps, promoted, positions = join_function_arguments(state, observable)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_StateObservableCallable(
            promoted[0],
            promoted[1],
            positions[0],
            positions[1],
            operation,
        ),
        metadata={},
    )


def state_expectation(
    state: DomainFunction,
    observable: DomainFunction,
    /,
) -> DomainFunction:
    r"""Return $\langle\psi|A|\psi\rangle$ for a normalized state vector.

    No normalization or Hermiticity assumption is imposed implicitly. Consequently,
    a non-Hermitian observable may produce a complex value.
    """
    return _state_observable_operation(state, observable, operation="expectation")


def density_expectation(
    density: DomainFunction,
    observable: DomainFunction,
    /,
) -> DomainFunction:
    r"""Return the density-operator expectation $\operatorname{tr}(\rho A)$.

    Both operands must be equally sized square matrices. Physicality of ``density``
    and Hermiticity of ``observable`` are not imposed implicitly.
    """
    if not isinstance(density, DomainFunction) or not isinstance(
        observable, DomainFunction
    ):
        raise TypeError(
            "density_expectation expects density and observable DomainFunctions."
        )
    domain, deps, promoted, positions = join_function_arguments(density, observable)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_DensityExpectationCallable(
            promoted[0],
            promoted[1],
            positions[0],
            positions[1],
        ),
        metadata={},
    )


def observable_variance(
    state: DomainFunction,
    observable: DomainFunction,
    /,
) -> DomainFunction:
    r"""Return $\langle A^2\rangle-\langle A\rangle^2$ for a normalized state.

    The expression is returned without discarding its imaginary part. It is real and
    nonnegative when the state is normalized and the observable is Hermitian, up to
    floating-point error.
    """
    return _state_observable_operation(state, observable, operation="variance")


def density_from_factor(factor: DomainFunction, /) -> DomainFunction:
    r"""Construct $\rho=TT^\dagger/\operatorname{tr}(TT^\dagger)$ pointwise.

    ``factor`` may have rectangular value shape ``(n, r)``, enabling rank-limited
    density operators. It must have nonzero Frobenius norm at every evaluated point.
    The result is Hermitian, positive semidefinite, and unit trace by construction.
    """
    if not isinstance(factor, DomainFunction):
        raise TypeError("density_from_factor expects a DomainFunction.")
    return DomainFunction(
        domain=factor.domain,
        deps=factor.deps,
        func=_DensityFromFactorCallable(factor),
        metadata={},
    )


__all__ = [
    "density_expectation",
    "density_from_factor",
    "observable_variance",
    "state_expectation",
    "state_norm_residual",
]
