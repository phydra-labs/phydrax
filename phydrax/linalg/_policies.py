#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import core as jax_core
from jax.typing import DTypeLike
from jaxtyping import Array

from .._precision import precision_dtype_name
from .._strict import StrictModule
from ._materialization import MaterializationPolicy
from ._preconditioning import PreconditioningPolicy
from ._recycling_policy import (
    RecyclingExtraction,
    RecyclingPolicy,
    RecyclingRefresh,
)


FailureMode: TypeAlias = Literal["status", "error"]
DifferentiationMode: TypeAlias = Literal[
    "mathematical",
    "rhs-only",
    "algorithmic",
    "none",
]
PrecisionDType: TypeAlias = Literal[
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
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
    """Economy SVD with optional scalar Tikhonov damping."""

    damping: float = eqx.field(static=True)

    def __init__(self, *, damping: float = 0.0):
        damping_ = float(damping)
        if not math.isfinite(damping_) or damping_ < 0.0:
            raise ValueError("damping must be finite and non-negative.")
        self.damping = damping_

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


class SparseLU(AbstractLinearMethod):
    """Sparse LU with an explicit immutable host provider selection."""

    provider: Literal["auto", "scipy-superlu", "umfpack"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        provider: Literal["auto", "scipy-superlu", "umfpack"] = "auto",
    ):
        if provider not in ("auto", "scipy-superlu", "umfpack"):
            raise ValueError(f"Unknown sparse LU provider {provider!r}.")
        self.provider = provider

    @property
    def name(self) -> str:
        return "sparse-lu"


class SparseCholesky(AbstractLinearMethod):
    """Sparse Cholesky with an explicit CHOLMOD provider."""

    provider: Literal["cholmod"] = eqx.field(static=True)

    def __init__(self, *, provider: Literal["cholmod"] = "cholmod"):
        if provider != "cholmod":
            raise ValueError(f"Unknown sparse Cholesky provider {provider!r}.")
        self.provider = provider

    @property
    def name(self) -> str:
        return "sparse-cholesky"


class SparseQR(AbstractLinearMethod):
    """Sparse QR with an explicit device JAX or host SPQR provider."""

    provider: Literal["jax-cuda", "spqr"] = eqx.field(static=True)
    reorder: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        provider: Literal["jax-cuda", "spqr"] = "jax-cuda",
        reorder: int = 1,
    ):
        if provider not in ("jax-cuda", "spqr"):
            raise ValueError(f"Unknown sparse QR provider {provider!r}.")
        reorder_ = int(reorder)
        if reorder_ not in (0, 1, 2, 3):
            raise ValueError("SparseQR reorder must be one of 0, 1, 2, or 3.")
        self.provider = provider
        self.reorder = reorder_

    @property
    def name(self) -> str:
        return "sparse-qr"


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


class ProjectedPCG(AbstractLinearMethod):
    """PCG on the certified orthogonal complement of a complete kernel."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "projected-pcg"


class BlockCG(AbstractLinearMethod):
    """True shared-space block conjugate gradients."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "block-cg"


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


class BlockGMRES(AbstractLinearMethod):
    """Restarted true shared-space block GMRES."""

    restart: int = eqx.field(static=True)

    def __init__(self, *, restart: int = 20):
        restart_ = int(restart)
        if restart_ < 1:
            raise ValueError("BlockGMRES restart must be positive.")
        self.restart = restart_

    @property
    def name(self) -> str:
        return "block-gmres"


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


class LinearSolveControl(StrictModule):
    """Dynamic per-invocation controls for a prepared native Krylov solve."""

    relative_tolerance: Array | None
    absolute_tolerance: Array | None
    maximum_steps: Array | None

    def __init__(
        self,
        *,
        relative_tolerance: Any | None = None,
        absolute_tolerance: Any | None = None,
        maximum_steps: Any | None = None,
    ):
        self.relative_tolerance = _runtime_tolerance(
            relative_tolerance,
            "relative_tolerance",
        )
        self.absolute_tolerance = _runtime_tolerance(
            absolute_tolerance,
            "absolute_tolerance",
        )
        self.maximum_steps = _runtime_maximum_steps(maximum_steps)


def _runtime_tolerance(value: Any | None, name: str, /) -> Array | None:
    if value is None:
        return None
    scalar = jnp.asarray(value)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be scalar or None.")
    if not (
        jnp.issubdtype(scalar.dtype, jnp.integer)
        or jnp.issubdtype(scalar.dtype, jnp.floating)
    ):
        raise TypeError(f"{name} must have a real numeric dtype.")
    scalar = scalar.astype(jnp.result_type(scalar, 0.0))
    invalid = ~jnp.isfinite(scalar) | (scalar < 0.0)
    if isinstance(invalid, jax_core.Tracer):
        return eqx.error_if(
            scalar,
            invalid,
            f"{name} must be finite and non-negative.",
        )
    if bool(invalid):
        raise ValueError(f"{name} must be finite and non-negative.")
    return scalar


def _runtime_maximum_steps(value: Any | None, /) -> Array | None:
    if value is None:
        return None
    scalar = jnp.asarray(value)
    if scalar.ndim != 0:
        raise ValueError("maximum_steps must be scalar or None.")
    if not jnp.issubdtype(scalar.dtype, jnp.integer):
        raise TypeError("maximum_steps must have an integer dtype.")
    invalid = scalar < 1
    if isinstance(invalid, jax_core.Tracer):
        return eqx.error_if(scalar, invalid, "maximum_steps must be positive.")
    if bool(invalid):
        raise ValueError("maximum_steps must be positive.")
    return scalar


class MixedPrecisionPolicy(StrictModule):
    """Explicit arithmetic precision for each linear-solve execution stage."""

    operator_dtype: PrecisionDType | None = eqx.field(static=True)
    factorization_dtype: PrecisionDType | None = eqx.field(static=True)
    preconditioner_dtype: PrecisionDType | None = eqx.field(static=True)
    krylov_dtype: PrecisionDType | None = eqx.field(static=True)
    residual_dtype: PrecisionDType | None = eqx.field(static=True)
    accumulation_dtype: PrecisionDType | None = eqx.field(static=True)
    maximum_refinement_steps: int = eqx.field(static=True)
    condition_limit: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        operator_dtype: DTypeLike | None = None,
        factorization_dtype: DTypeLike | None = None,
        preconditioner_dtype: DTypeLike | None = None,
        krylov_dtype: DTypeLike | None = None,
        residual_dtype: DTypeLike | None = None,
        accumulation_dtype: DTypeLike | None = None,
        maximum_refinement_steps: int = 0,
        condition_limit: float | None = None,
    ):
        self.operator_dtype = _precision_dtype(operator_dtype, "operator_dtype")
        self.factorization_dtype = _precision_dtype(
            factorization_dtype,
            "factorization_dtype",
        )
        self.preconditioner_dtype = _precision_dtype(
            preconditioner_dtype,
            "preconditioner_dtype",
        )
        self.krylov_dtype = _precision_dtype(krylov_dtype, "krylov_dtype")
        self.residual_dtype = _precision_dtype(residual_dtype, "residual_dtype")
        self.accumulation_dtype = _precision_dtype(
            accumulation_dtype,
            "accumulation_dtype",
        )
        steps = int(maximum_refinement_steps)
        if steps < 0:
            raise ValueError("maximum_refinement_steps must be non-negative.")
        limit = None if condition_limit is None else float(condition_limit)
        if limit is not None and (not math.isfinite(limit) or limit <= 1.0):
            raise ValueError("condition_limit must be finite and greater than one.")
        self.maximum_refinement_steps = steps
        self.condition_limit = limit


def _precision_dtype(
    value: DTypeLike | None,
    name: str,
    /,
) -> PrecisionDType | None:
    if value is None:
        return None
    precision = precision_dtype_name(value)
    supported = (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
    if precision not in supported:
        raise ValueError(
            f"{name} must name a supported real or complex floating dtype; "
            f"got {precision!r}."
        )
    return precision


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
    preconditioner_bytes: int = eqx.field(static=True)
    recycling_state_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        factorization_bytes: int = 512 * 1024 * 1024,
        workspace_bytes: int = 512 * 1024 * 1024,
        krylov_basis_bytes: int = 256 * 1024 * 1024,
        preconditioner_bytes: int = 256 * 1024 * 1024,
        recycling_state_bytes: int = 256 * 1024 * 1024,
    ):
        values = tuple(
            int(value)
            for value in (
                factorization_bytes,
                workspace_bytes,
                krylov_basis_bytes,
                preconditioner_bytes,
                recycling_state_bytes,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("Resource budgets must be non-negative.")
        (
            self.factorization_bytes,
            self.workspace_bytes,
            self.krylov_basis_bytes,
            self.preconditioner_bytes,
            self.recycling_state_bytes,
        ) = values


class LinearSolvePolicy(StrictModule):
    """Composable mathematical, differentiation, and resource requirements."""

    method: AbstractLinearMethod
    tolerance: TolerancePolicy
    rank: RankPolicy
    materialization: MaterializationPolicy
    preconditioning: PreconditioningPolicy | None
    recycling: RecyclingPolicy | None
    differentiation: DifferentiationPolicy
    failure: FailurePolicy
    resources: SolveResourcePolicy
    precision: MixedPrecisionPolicy | None
    require_device_binding: bool = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractLinearMethod | None = None,
        /,
        *,
        tolerance: TolerancePolicy | None = None,
        rank: RankPolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        preconditioning: PreconditioningPolicy | None = None,
        recycling: RecyclingPolicy | None = None,
        differentiation: DifferentiationPolicy | None = None,
        failure: FailurePolicy | None = None,
        resources: SolveResourcePolicy | None = None,
        precision: MixedPrecisionPolicy | None = None,
        require_device_binding: bool = False,
    ):
        method_ = AutoLinearMethod() if method is None else method
        tolerance_ = TolerancePolicy() if tolerance is None else tolerance
        rank_ = RankPolicy() if rank is None else rank
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        if preconditioning is not None and not isinstance(
            preconditioning, PreconditioningPolicy
        ):
            raise TypeError("preconditioning must be a PreconditioningPolicy or None.")
        if recycling is not None and not isinstance(recycling, RecyclingPolicy):
            raise TypeError("recycling must be a RecyclingPolicy or None.")
        differentiation_ = (
            DifferentiationPolicy() if differentiation is None else differentiation
        )
        failure_ = FailurePolicy() if failure is None else failure
        resources_ = SolveResourcePolicy() if resources is None else resources
        if precision is not None and not isinstance(precision, MixedPrecisionPolicy):
            raise TypeError("precision must be a MixedPrecisionPolicy or None.")
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
        self.preconditioning = preconditioning
        self.recycling = recycling
        self.differentiation = differentiation_
        self.failure = failure_
        self.precision = precision
        self.resources = resources_
        self.require_device_binding = bool(require_device_binding)


__all__ = [
    "AbstractLinearMethod",
    "AutoLinearMethod",
    "BiCGStab",
    "BlockCG",
    "BlockGMRES",
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
    "SparseCholesky",
    "LinearSolveControl",
    "LinearSolvePolicy",
    "LSMR",
    "MINRES",
    "PCG",
    "MixedPrecisionPolicy",
    "PrecisionDType",
    "ProjectedPCG",
    "RankPolicy",
    "SolveResourcePolicy",
    "SparseLU",
    "SparseQR",
    "StructuredDirect",
    "TolerancePolicy",
    "RecyclingExtraction",
    "RecyclingPolicy",
    "RecyclingRefresh",
]
