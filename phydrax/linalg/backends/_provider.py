#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

from .._plans import LinearBackend, LinearSolvePlan
from .._preconditioners import AbstractPreconditioner
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
from ._native_block_krylov import (
    prepare_native_block_krylov,
    solve_native_block_krylov,
)
from ._native_krylov import prepare_native_krylov, solve_native_krylov


class AbstractLinearProvider(abc.ABC):
    """Internal provider boundary used by the public template/bind/solve lifecycle."""

    backends: tuple[LinearBackend, ...]
    accepts_initial_guess: bool = False
    supports_implicit_differentiation: bool = False

    def accepts(self, backend: LinearBackend, /) -> bool:
        return backend in self.backends

    def analyze(self, problem: Any, plan: LinearSolvePlan, /) -> Any:
        """Return coefficient-independent symbolic state for one solve plan."""
        del problem, plan
        return None

    @abc.abstractmethod
    def bind(
        self,
        symbolic_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> Any:
        """Bind current coefficients without rediscovering symbolic structure."""
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

    def bind(self, symbolic_state, problem, plan, /, *, preconditioner=None):
        del symbolic_state
        if preconditioner is not None:
            raise ValueError("Structured direct binding rejects preconditioning.")
        return prepare_structured(problem, plan)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_structured(state, rhs, plan)


class _DenseProvider(AbstractLinearProvider):
    backends = ("jax-dense",)
    supports_implicit_differentiation = True

    def bind(self, symbolic_state, problem, plan, /, *, preconditioner=None):
        del symbolic_state
        if preconditioner is not None:
            raise ValueError("Dense binding rejects preconditioning.")
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

    def bind(self, symbolic_state, problem, plan, /, *, preconditioner=None):
        del symbolic_state
        if preconditioner is not None:
            raise ValueError("Sparse direct binding rejects preconditioning.")
        return prepare_sparse(problem, plan)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_sparse(state, rhs, plan)

    def supports_transformed(self, state, /) -> bool:
        return isinstance(state, HostSparseState)

    def solve_transformed(self, state, rhs, plan, /, *, adjoint):
        if not isinstance(state, HostSparseState):
            return super().solve_transformed(state, rhs, plan, adjoint=adjoint)
        return solve_host_sparse_transformed(state, rhs, adjoint=adjoint)


class _NativeBlockKrylovProvider(AbstractLinearProvider):
    backends = ("native-block-krylov",)
    accepts_initial_guess = True
    supports_implicit_differentiation = True

    def bind(self, symbolic_state, problem, plan, /, *, preconditioner=None):
        del symbolic_state
        return prepare_native_block_krylov(
            problem,
            plan,
            preconditioner=preconditioner,
        )

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_native_block_krylov(
            state,
            rhs,
            plan,
            initial_guess=initial_guess,
        )


class _NativeKrylovProvider(AbstractLinearProvider):
    backends = ("native-krylov",)
    accepts_initial_guess = True
    supports_implicit_differentiation = True

    def bind(self, symbolic_state, problem, plan, /, *, preconditioner=None):
        del symbolic_state
        return prepare_native_krylov(
            problem,
            plan,
            preconditioner=preconditioner,
        )

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_native_krylov(state, rhs, plan, initial_guess=initial_guess)


class _MatfreeProvider(AbstractLinearProvider):
    backends = ("matfree",)
    accepts_initial_guess = True
    supports_implicit_differentiation = True

    def bind(self, symbolic_state, problem, plan, /, *, preconditioner=None):
        del symbolic_state
        if preconditioner is not None:
            raise ValueError("Matfree binding rejects preconditioning.")
        return prepare_matfree(problem, plan)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_matfree(state, rhs, plan, initial_guess=initial_guess)


class _LineaxProvider(AbstractLinearProvider):
    backends = ("lineax",)
    supports_implicit_differentiation = True
    accepts_initial_guess = True

    def bind(self, symbolic_state, problem, plan, /, *, preconditioner=None):
        del symbolic_state
        return prepare_lineax(problem, plan, preconditioner=preconditioner)

    def solve(self, state, rhs, plan, /, *, initial_guess=None):
        return solve_lineax(state, rhs, plan, initial_guess=initial_guess)


_PROVIDERS: tuple[AbstractLinearProvider, ...] = (
    _StructuredProvider(),
    _DenseProvider(),
    _SparseProvider(),
    _NativeBlockKrylovProvider(),
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
