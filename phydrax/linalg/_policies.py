#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from typing import Literal, TypeAlias

import equinox as eqx

from .._strict import StrictModule
from ._materialization import MaterializationPolicy
from ._preconditioners import AbstractPreconditioner


FailureMode: TypeAlias = Literal["status", "error"]
DifferentiationMode: TypeAlias = Literal[
    "mathematical",
    "rhs-only",
    "algorithmic",
    "none",
]


class AbstractLinearMethod(StrictModule):
    """Explicit numerical method selected by a linear-solve policy."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class AutoLinearMethod(AbstractLinearMethod):
    """Deterministic capability-based method selection."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "auto"


class DenseLU(AbstractLinearMethod):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "dense-lu"


class DenseCholesky(AbstractLinearMethod):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "dense-cholesky"


class DenseQR(AbstractLinearMethod):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "dense-qr"


class DenseSVD(AbstractLinearMethod):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "dense-svd"


class StructuredDirect(AbstractLinearMethod):
    """Exact native execution for an operator with recognized structure."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "structured-direct"


class SparseDirect(AbstractLinearMethod):
    """Native device sparse direct solve; currently CUDA CSR QR."""

    reorder: int = eqx.field(static=True)

    def __init__(self, *, reorder: int = 1):
        reorder_ = int(reorder)
        if reorder_ not in (0, 1, 2, 3):
            raise ValueError("SparseDirect reorder must be one of 0, 1, 2, or 3.")
        self.reorder = reorder_

    @property
    def name(self) -> str:
        return "sparse-direct"


class HostSparseLU(AbstractLinearMethod):
    """Explicit non-JIT SciPy SuperLU factorization on the host."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "host-sparse-lu"


class ConjugateGradient(AbstractLinearMethod):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "cg"


class PCG(AbstractLinearMethod):
    """Pairing-aware preconditioned conjugate gradients."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "pcg"


class MINRES(AbstractLinearMethod):
    """Minimum residual iteration for self-adjoint indefinite systems."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "minres"


class GMRES(AbstractLinearMethod):
    restart: int = eqx.field(static=True)
    stagnation_iterations: int = eqx.field(static=True)

    def __init__(self, *, restart: int = 20, stagnation_iterations: int = 20):
        restart_ = int(restart)
        stagnation = int(stagnation_iterations)
        if restart_ < 1 or stagnation < 1:
            raise ValueError("GMRES restart and stagnation_iterations must be positive.")
        self.restart = restart_
        self.stagnation_iterations = stagnation

    @property
    def name(self) -> str:
        return "gmres"


class FGMRES(AbstractLinearMethod):
    """Restarted flexible GMRES with variable-preconditioner semantics."""

    restart: int = eqx.field(static=True)
    stagnation_iterations: int = eqx.field(static=True)

    def __init__(self, *, restart: int = 30, stagnation_iterations: int = 30):
        restart_ = int(restart)
        stagnation = int(stagnation_iterations)
        if restart_ < 1 or stagnation < 1:
            raise ValueError("FGMRES restart and stagnation_iterations must be positive.")
        self.restart = restart_
        self.stagnation_iterations = stagnation

    @property
    def name(self) -> str:
        return "fgmres"


class BiCGStab(AbstractLinearMethod):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "bicgstab"


class LSMR(AbstractLinearMethod):
    condition_limit: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)

    def __init__(self, *, condition_limit: float = 1e8, damping: float = 0.0):
        condition = float(condition_limit)
        damping_ = float(damping)
        if not math.isfinite(condition) or condition <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        if not math.isfinite(damping_) or damping_ < 0.0:
            raise ValueError("damping must be finite and non-negative.")
        self.condition_limit = condition
        self.damping = damping_

    @property
    def name(self) -> str:
        return "lsmr"


class GeneralizedLSMR(AbstractLinearMethod):
    """Pairing-aware generalized LSMR with explicit adjoint action."""

    condition_limit: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)

    def __init__(self, *, condition_limit: float = 1e8, damping: float = 0.0):
        condition = float(condition_limit)
        damping_ = float(damping)
        if not math.isfinite(condition) or condition <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        if not math.isfinite(damping_) or damping_ < 0.0:
            raise ValueError("damping must be finite and non-negative.")
        self.condition_limit = condition
        self.damping = damping_

    @property
    def name(self) -> str:
        return "generalized-lsmr"


class TolerancePolicy(StrictModule):
    relative: float = eqx.field(static=True)
    absolute: float = eqx.field(static=True)
    max_steps: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative: float = 1e-8,
        absolute: float = 1e-10,
        max_steps: int | None = None,
    ):
        relative_ = float(relative)
        absolute_ = float(absolute)
        steps = None if max_steps is None else int(max_steps)
        if (
            not math.isfinite(relative_)
            or not math.isfinite(absolute_)
            or relative_ < 0.0
            or absolute_ < 0.0
        ):
            raise ValueError("Tolerances must be finite and non-negative.")
        if steps is not None and steps < 1:
            raise ValueError("max_steps must be positive or None.")
        self.relative = relative_
        self.absolute = absolute_
        self.max_steps = steps


class RankPolicy(StrictModule):
    relative_cutoff: float | None = eqx.field(static=True)
    require_full_rank: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_cutoff: float | None = None,
        require_full_rank: bool = False,
    ):
        cutoff = None if relative_cutoff is None else float(relative_cutoff)
        if cutoff is not None and (not math.isfinite(cutoff) or cutoff < 0.0):
            raise ValueError("relative_cutoff must be non-negative and finite or None.")
        self.relative_cutoff = cutoff
        self.require_full_rank = bool(require_full_rank)


class FailurePolicy(StrictModule):
    mode: FailureMode = eqx.field(static=True)

    def __init__(self, mode: FailureMode = "status", /):
        if mode not in ("status", "error"):
            raise ValueError("Failure mode must be 'status' or 'error'.")
        self.mode = mode


class DifferentiationPolicy(StrictModule):
    """Requested derivative of the mathematical or executed solve map."""

    mode: DifferentiationMode = eqx.field(static=True)

    def __init__(self, mode: DifferentiationMode = "mathematical", /):
        if mode not in ("mathematical", "rhs-only", "algorithmic", "none"):
            raise ValueError("Unknown differentiation mode.")
        self.mode = mode


class SolveResourcePolicy(StrictModule):
    """Independent byte budgets for preparation and batched solve execution."""

    factorization_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    krylov_basis_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        factorization_bytes: int = 512 * 1024 * 1024,
        workspace_bytes: int = 512 * 1024 * 1024,
        krylov_basis_bytes: int = 256 * 1024 * 1024,
    ):
        values = tuple(
            int(value)
            for value in (
                factorization_bytes,
                workspace_bytes,
                krylov_basis_bytes,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("Resource budgets must be non-negative.")
        (
            self.factorization_bytes,
            self.workspace_bytes,
            self.krylov_basis_bytes,
        ) = values


class LinearSolvePolicy(StrictModule):
    """Composable mathematical, differentiation, and resource requirements."""

    method: AbstractLinearMethod
    tolerance: TolerancePolicy
    rank: RankPolicy
    materialization: MaterializationPolicy
    preconditioner: AbstractPreconditioner | None
    differentiation: DifferentiationPolicy
    failure: FailurePolicy
    resources: SolveResourcePolicy

    def __init__(
        self,
        method: AbstractLinearMethod | None = None,
        /,
        *,
        tolerance: TolerancePolicy | None = None,
        rank: RankPolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        preconditioner: AbstractPreconditioner | None = None,
        differentiation: DifferentiationPolicy | None = None,
        failure: FailurePolicy | None = None,
        resources: SolveResourcePolicy | None = None,
    ):
        method_ = AutoLinearMethod() if method is None else method
        tolerance_ = TolerancePolicy() if tolerance is None else tolerance
        rank_ = RankPolicy() if rank is None else rank
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        if preconditioner is not None and not isinstance(
            preconditioner, AbstractPreconditioner
        ):
            raise TypeError("preconditioner must be an AbstractPreconditioner or None.")
        differentiation_ = (
            DifferentiationPolicy() if differentiation is None else differentiation
        )
        failure_ = FailurePolicy() if failure is None else failure
        resources_ = SolveResourcePolicy() if resources is None else resources
        if not isinstance(method_, AbstractLinearMethod):
            raise TypeError("method must be an AbstractLinearMethod.")
        if not isinstance(tolerance_, TolerancePolicy):
            raise TypeError("tolerance must be a TolerancePolicy.")
        if not isinstance(rank_, RankPolicy):
            raise TypeError("rank must be a RankPolicy.")
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        if not isinstance(differentiation_, DifferentiationPolicy):
            raise TypeError("differentiation must be a DifferentiationPolicy.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy.")
        if not isinstance(resources_, SolveResourcePolicy):
            raise TypeError("resources must be a SolveResourcePolicy.")
        self.method = method_
        self.tolerance = tolerance_
        self.rank = rank_
        self.materialization = materialization_
        self.preconditioner = preconditioner
        self.differentiation = differentiation_
        self.failure = failure_
        self.resources = resources_


__all__ = [
    "AbstractLinearMethod",
    "AutoLinearMethod",
    "BiCGStab",
    "ConjugateGradient",
    "DenseCholesky",
    "DenseLU",
    "DenseQR",
    "DenseSVD",
    "DifferentiationMode",
    "DifferentiationPolicy",
    "FailureMode",
    "FailurePolicy",
    "FGMRES",
    "GeneralizedLSMR",
    "GMRES",
    "HostSparseLU",
    "LinearSolvePolicy",
    "LSMR",
    "MINRES",
    "PCG",
    "RankPolicy",
    "SolveResourcePolicy",
    "SparseDirect",
    "StructuredDirect",
    "TolerancePolicy",
]
