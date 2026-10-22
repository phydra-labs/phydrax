#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING

from jax import Array

from ..._iteration import (
    IterationCapabilities,
    IterationPlan,
    IterationRuntimeState,
)
from .._bordered import _bordered_base, _bordered_structure
from .._plans import LinearBackend, LinearSolvePlan
from .._policies import LinearSolveControl
from .._preconditioners import AbstractPreconditioner
from .._structured_operators import BandedLinearOperator
from ._cudss import (
    analyze_cudss,
    bind_cudss,
    refresh_cudss,
    release_cudss,
    solve_cudss,
)
from ._jax_dense import prepare_dense, solve_dense, solve_dense_transformed
from ._jax_sparse import (
    HostSparseState,
    prepare_sparse,
    solve_host_sparse_transformed,
    solve_sparse,
)
from ._jax_structured import (
    _StructuredSymbolicState,
    prepare_structured,
    solve_structured,
    solve_structured_transformed,
    StructuredState,
)
from ._lineax import prepare_lineax, solve_lineax
from ._native_block_krylov import (
    prepare_native_block_krylov,
    solve_native_block_krylov,
)
from ._native_krylov import prepare_native_krylov, solve_native_krylov


if TYPE_CHECKING:
    from ._cudss import (
        CudssBackendOutput,
        CudssFactorState,
        CudssSymbolicState,
    )
    from ._jax_dense import DenseBackendOutput
    from ._jax_sparse import SparseBackendOutput
    from ._jax_structured import StructuredBackendOutput
    from ._lineax import LineaxBackendOutput, LineaxState
    from ._native_block_krylov import (
        NativeBlockKrylovBackendOutput,
        NativeBlockKrylovState,
    )
    from ._native_krylov import NativeKrylovBackendOutput, NativeKrylovState


class AbstractLinearProvider(abc.ABC):
    """Internal provider boundary used by the public template/bind/solve lifecycle."""

    backends: tuple[LinearBackend, ...]
    accepts_initial_guess: bool = False
    supports_implicit_differentiation: bool = False
    # Prepared transpose/adjoint solves of batched operators reuse the factors.
    supports_batched_transformed: bool = False
    # Solve is a pure device program over prepared state: the runtime executes
    # it through its stable compiled entry so loops are traced once per structure.
    compiled_execution: bool = False
    iteration_capabilities = IterationCapabilities.terminal_only()

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

    def refresh(
        self,
        symbolic_state: Any,
        previous_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> Any:
        """Refresh numeric coefficients while retaining provider-owned structure."""
        del previous_state
        return self.bind(
            symbolic_state,
            problem,
            plan,
            preconditioner=preconditioner,
        )

    def release(self, state: Any, /) -> bool | Array:
        """Release provider resources; array-only providers are no-ops."""
        del state
        return False

    @abc.abstractmethod
    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
        control: LinearSolveControl | None = None,
        iteration: IterationPlan | None = None,
        iteration_state: IterationRuntimeState | None = None,
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

    def analyze(self, problem: Any, plan: LinearSolvePlan, /) -> _StructuredSymbolicState:
        del plan
        return _StructuredSymbolicState(_bordered_structure(problem.operator))

    def bind(
        self,
        symbolic_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> StructuredState:
        if not isinstance(symbolic_state, _StructuredSymbolicState):
            raise TypeError("Structured binding requires native symbolic state.")
        if symbolic_state.bordered_structure != _bordered_structure(problem.operator):
            raise ValueError(
                "Structured binding cannot change the bordered coordinate layout."
            )
        if preconditioner is not None:
            raise ValueError("Structured direct binding rejects preconditioning.")
        return prepare_structured(problem, plan)

    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
        control: LinearSolveControl | None = None,
        iteration: IterationPlan | None = None,
        iteration_state: IterationRuntimeState | None = None,
    ) -> StructuredBackendOutput:
        del iteration, iteration_state
        return solve_structured(state, rhs, plan)

    def supports_transformed(self, state: Any, /) -> bool:
        return isinstance(state, StructuredState) and (
            _bordered_base(state.operator) is not None
            or isinstance(state.operator, BandedLinearOperator)
        )

    def solve_transformed(
        self, state: StructuredState, rhs: Any, plan: LinearSolvePlan, /, *, adjoint: bool
    ) -> StructuredBackendOutput:
        return solve_structured_transformed(state, rhs, plan, adjoint=adjoint)


class _DenseProvider(AbstractLinearProvider):
    backends = ("jax-dense",)
    supports_implicit_differentiation = True

    def bind(
        self,
        symbolic_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> Any:
        del symbolic_state
        if preconditioner is not None:
            raise ValueError("Dense binding rejects preconditioning.")
        return prepare_dense(problem, plan)

    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
        control: LinearSolveControl | None = None,
        iteration: IterationPlan | None = None,
        iteration_state: IterationRuntimeState | None = None,
    ) -> DenseBackendOutput:
        del iteration, iteration_state
        return solve_dense(state, rhs, plan)

    def supports_transformed(self, state: Any, /) -> bool:
        from ._jax_dense import (
            DenseCholeskyState,
            DenseLUState,
            DenseMixedPrecisionLUState,
        )

        return isinstance(
            state,
            (DenseLUState, DenseMixedPrecisionLUState, DenseCholeskyState),
        )

    def solve_transformed(
        self, state: Any, rhs: Any, plan: LinearSolvePlan, /, *, adjoint: bool
    ) -> DenseBackendOutput:
        return solve_dense_transformed(state, rhs, plan, adjoint=adjoint)


class _SparseProvider(AbstractLinearProvider):
    backends = ("jax-sparse", "host-sparse")
    supports_implicit_differentiation = True

    def bind(
        self,
        symbolic_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> Any:
        del symbolic_state
        if preconditioner is not None:
            raise ValueError("Sparse direct binding rejects preconditioning.")
        return prepare_sparse(problem, plan)

    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
        control: LinearSolveControl | None = None,
        iteration: IterationPlan | None = None,
        iteration_state: IterationRuntimeState | None = None,
    ) -> SparseBackendOutput:
        del iteration, iteration_state
        return solve_sparse(state, rhs, plan)

    def supports_transformed(self, state: Any, /) -> bool:
        return isinstance(state, HostSparseState)

    def solve_transformed(
        self, state: Any, rhs: Any, plan: LinearSolvePlan, /, *, adjoint: bool
    ) -> SparseBackendOutput:
        if not isinstance(state, HostSparseState):
            return super().solve_transformed(state, rhs, plan, adjoint=adjoint)
        return solve_host_sparse_transformed(state, rhs, adjoint=adjoint)


class _CudssProvider(AbstractLinearProvider):
    backends = ("cudss",)
    supports_implicit_differentiation = True
    supports_batched_transformed = True

    def analyze(self, problem: Any, plan: LinearSolvePlan, /) -> CudssSymbolicState:
        return analyze_cudss(problem, plan)

    def bind(
        self,
        symbolic_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> CudssFactorState:
        if preconditioner is not None:
            raise ValueError("cuDSS direct binding rejects preconditioning.")
        return bind_cudss(symbolic_state, problem, plan)

    def refresh(
        self,
        symbolic_state: Any,
        previous_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> CudssFactorState:
        if preconditioner is not None:
            raise ValueError("cuDSS direct refresh rejects preconditioning.")
        return refresh_cudss(symbolic_state, previous_state, problem, plan)

    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
        control: LinearSolveControl | None = None,
        iteration: IterationPlan | None = None,
        iteration_state: IterationRuntimeState | None = None,
    ) -> CudssBackendOutput:
        del iteration, iteration_state
        return solve_cudss(state, rhs, plan)

    def supports_transformed(self, state: Any, /) -> bool:
        return True

    def solve_transformed(
        self, state: Any, rhs: Any, plan: LinearSolvePlan, /, *, adjoint: bool
    ) -> CudssBackendOutput:
        del adjoint
        return solve_cudss(state, rhs, plan)

    def release(self, state: Any, /) -> bool | Array:
        return release_cudss(state)


class _CudssMGMNProvider(AbstractLinearProvider):
    backends = ("cudss-mgmn",)
    # The mathematical primitive owns collective derivatives. Wrapping it in
    # the generic local root derivative would introduce a second solve route.
    supports_implicit_differentiation = False

    def analyze(self, problem: Any, plan: LinearSolvePlan, /) -> Any:
        from ._cudss_mgmn import analyze_cudss_mgmn

        return analyze_cudss_mgmn(problem, plan)

    def bind(
        self,
        symbolic_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> Any:
        from ._cudss_mgmn import bind_cudss_mgmn

        if preconditioner is not None:
            raise ValueError("MGMN direct binding rejects preconditioning.")
        return bind_cudss_mgmn(symbolic_state, problem, plan)

    def refresh(
        self,
        symbolic_state: Any,
        previous_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> Any:
        from ._cudss_mgmn import refresh_cudss_mgmn

        if preconditioner is not None:
            raise ValueError("MGMN direct refresh rejects preconditioning.")
        return refresh_cudss_mgmn(symbolic_state, previous_state, problem, plan)

    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
        control: LinearSolveControl | None = None,
        iteration: IterationPlan | None = None,
        iteration_state: IterationRuntimeState | None = None,
    ) -> CudssBackendOutput:
        from ._cudss_mgmn import solve_cudss_mgmn

        del initial_guess, control, iteration, iteration_state
        return solve_cudss_mgmn(state, rhs, plan)

    def supports_transformed(self, state: Any, /) -> bool:
        return True

    def solve_transformed(
        self, state: Any, rhs: Any, plan: LinearSolvePlan, /, *, adjoint: bool
    ) -> CudssBackendOutput:
        from ._cudss_mgmn import solve_cudss_mgmn

        # Real symmetry reuses the factors, not the independent stored-value
        # derivative orientation of the transpose/adjoint solve.
        del adjoint
        return solve_cudss_mgmn(state, rhs, plan, transpose=True)

    def release(self, state: Any, /) -> bool | Array:
        from ._cudss_mgmn import release_cudss_mgmn

        return release_cudss_mgmn(state)


class _NativeBlockKrylovProvider(AbstractLinearProvider):
    backends = ("native-block-krylov",)
    accepts_initial_guess = True
    supports_implicit_differentiation = True
    compiled_execution = True
    iteration_capabilities = IterationCapabilities(
        ("terminal", "inner-iteration"),
        device_stop=True,
        mapped_records=True,
    )

    def bind(
        self,
        symbolic_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> NativeBlockKrylovState:
        del symbolic_state
        return prepare_native_block_krylov(
            problem,
            plan,
            preconditioner=preconditioner,
        )

    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
        control: LinearSolveControl | None = None,
        iteration: IterationPlan | None = None,
        iteration_state: IterationRuntimeState | None = None,
    ) -> NativeBlockKrylovBackendOutput:
        return solve_native_block_krylov(
            state,
            rhs,
            plan,
            initial_guess=initial_guess,
            iteration=iteration,
            iteration_state=iteration_state,
        )


class _NativeKrylovProvider(AbstractLinearProvider):
    backends = ("native-krylov",)
    accepts_initial_guess = True
    supports_implicit_differentiation = True
    compiled_execution = True
    iteration_capabilities = IterationCapabilities(
        ("terminal", "inner-iteration"),
        device_stop=True,
        mapped_records=True,
    )

    def bind(
        self,
        symbolic_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> NativeKrylovState:
        del symbolic_state
        return prepare_native_krylov(
            problem,
            plan,
            preconditioner=preconditioner,
        )

    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
        control: LinearSolveControl | None = None,
        iteration: IterationPlan | None = None,
        iteration_state: IterationRuntimeState | None = None,
    ) -> NativeKrylovBackendOutput:
        return solve_native_krylov(
            state,
            rhs,
            plan,
            initial_guess=initial_guess,
            control=control,
            iteration=iteration,
            iteration_state=iteration_state,
        )


class _LineaxProvider(AbstractLinearProvider):
    backends = ("lineax",)
    supports_implicit_differentiation = True
    accepts_initial_guess = True

    def bind(
        self,
        symbolic_state: Any,
        problem: Any,
        plan: LinearSolvePlan,
        /,
        *,
        preconditioner: AbstractPreconditioner | None = None,
    ) -> LineaxState:
        del symbolic_state
        return prepare_lineax(problem, plan, preconditioner=preconditioner)

    def solve(
        self,
        state: Any,
        rhs: Any,
        plan: LinearSolvePlan,
        /,
        *,
        initial_guess: Any = None,
        control: LinearSolveControl | None = None,
        iteration: IterationPlan | None = None,
        iteration_state: IterationRuntimeState | None = None,
    ) -> LineaxBackendOutput:
        del iteration, iteration_state
        return solve_lineax(state, rhs, plan, initial_guess=initial_guess)


_PROVIDERS: tuple[AbstractLinearProvider, ...] = (
    _StructuredProvider(),
    _DenseProvider(),
    _SparseProvider(),
    _CudssProvider(),
    _CudssMGMNProvider(),
    _NativeBlockKrylovProvider(),
    _NativeKrylovProvider(),
    _LineaxProvider(),
)


def provider_for(backend: LinearBackend, /) -> AbstractLinearProvider:
    for provider in _PROVIDERS:
        if provider.accepts(backend):
            return provider
    raise ValueError(f"Unknown backend {backend!r}.")


__all__ = ["AbstractLinearProvider", "provider_for"]
