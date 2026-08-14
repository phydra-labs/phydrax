#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

from .._plans import LinearBackend, LinearSolvePlan
from ._jax_dense import prepare_dense, solve_dense, solve_dense_transformed
from ._jax_sparse import (
    HostSparseState,
    prepare_sparse,
    solve_host_sparse_transformed,
    solve_sparse,
)
from ._jax_structured import prepare_structured, solve_structured
from ._lineax import prepare_lineax, solve_lineax
from ._matfree import prepare_matfree, solve_matfree
from ._native_krylov import prepare_native_krylov, solve_native_krylov


class AbstractLinearProvider(abc.ABC):
    """Internal provider boundary used by the public prepare/solve lifecycle."""

    backends: tuple[LinearBackend, ...]
    accepts_initial_guess: bool = False
    supports_implicit_differentiation: bool = False

    def accepts(self, backend: LinearBackend, /) -> bool:
        return backend in self.backends

    @abc.abstractmethod
    def prepare(self, problem: Any, plan: LinearSolvePlan, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
    ) -> Any:
        raise NotImplementedError

    def supports_transformed(self, state: Any, /) -> bool:
        return False

    def solve_transformed(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        adjoint: bool,
    ) -> Any:
        raise ValueError("This provider cannot reuse state for transformed solves.")


class _StructuredProvider(AbstractLinearProvider):
    backends = ("jax-structured",)
    supports_implicit_differentiation = True

    def prepare(self, problem, plan, /):
        return prepare_structured(problem, plan)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_structured(state, rhs, plan)


class _DenseProvider(AbstractLinearProvider):
    backends = ("jax-dense",)
    supports_implicit_differentiation = True

    def prepare(self, problem, plan, /):
        return prepare_dense(problem, plan)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_dense(state, rhs, plan)

    def supports_transformed(self, state, /) -> bool:
        from ._jax_dense import DenseCholeskyState, DenseLUState

        return isinstance(state, (DenseLUState, DenseCholeskyState))

    def solve_transformed(self, state, rhs, plan, /, *, adjoint):
        return solve_dense_transformed(state, rhs, plan, adjoint=adjoint)


class _SparseProvider(AbstractLinearProvider):
    backends = ("jax-sparse", "host-sparse")
    supports_implicit_differentiation = True

    def prepare(self, problem, plan, /):
        return prepare_sparse(problem, plan)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_sparse(state, rhs, plan)

    def supports_transformed(self, state, /) -> bool:
        return isinstance(state, HostSparseState)

    def solve_transformed(self, state, rhs, plan, /, *, adjoint):
        if not isinstance(state, HostSparseState):
            return super().solve_transformed(state, rhs, plan, adjoint=adjoint)
        return solve_host_sparse_transformed(state, rhs, adjoint=adjoint)


class _NativeKrylovProvider(AbstractLinearProvider):
    backends = ("native-krylov",)
    accepts_initial_guess = True
    supports_implicit_differentiation = True

    def prepare(self, problem, plan, /):
        return prepare_native_krylov(problem, plan)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_native_krylov(state, rhs, plan, initial_guess=initial_guess)


class _MatfreeProvider(AbstractLinearProvider):
    backends = ("matfree",)
    accepts_initial_guess = True
    supports_implicit_differentiation = True

    def prepare(self, problem, plan, /):
        return prepare_matfree(problem, plan)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_matfree(state, rhs, plan, initial_guess=initial_guess)


class _LineaxProvider(AbstractLinearProvider):
    backends = ("lineax",)
    supports_implicit_differentiation = True
    accepts_initial_guess = True

    def prepare(self, problem, plan, /):
        return prepare_lineax(problem, plan)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_lineax(state, rhs, plan, initial_guess=initial_guess)


_PROVIDERS: tuple[AbstractLinearProvider, ...] = (
    _StructuredProvider(),
    _DenseProvider(),
    _SparseProvider(),
    _NativeKrylovProvider(),
    _MatfreeProvider(),
    _LineaxProvider(),
)


def provider_for(backend: LinearBackend, /) -> AbstractLinearProvider:
    for provider in _PROVIDERS:
        if provider.accepts(backend):
            return provider
    raise ValueError(f"Unknown backend {backend!r}.")


__all__ = ["AbstractLinearProvider", "provider_for"]
