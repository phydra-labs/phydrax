#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._materialization import MaterializationPolicy
from .._policies import FailurePolicy
from .._preconditioning import PreconditioningPolicy


EigenTarget: TypeAlias = Literal[
    "smallest-algebraic",
    "largest-algebraic",
    "smallest-magnitude",
    "largest-magnitude",
]
EigenDifferentiationMode: TypeAlias = Literal["none", "eigenvalues"]


class AutoEigenMethod(StrictModule):
    """Deterministically try LOBPCG before restarted Lanczos."""

    def __init__(self):
        return

    @property
    def name(self) -> str:
        return "auto"


class DenseEigh(StrictModule):
    """Full dense Hermitian eigendecomposition with an explicit backend."""

    backend: Literal["jax", "eigh-ffi"] = eqx.field(static=True)

    def __init__(self, *, backend: Literal["jax", "eigh-ffi"] = "jax"):
        if backend not in ("jax", "eigh-ffi"):
            raise ValueError("DenseEigh backend must be 'jax' or 'eigh-ffi'.")
        self.backend = backend

    @property
    def name(self) -> str:
        return "dense-eigh" if self.backend == "jax" else "dense-eigh-ffi"


class LOBPCG(StrictModule):
    """Locally optimal block preconditioned conjugate gradients."""

    block_dimension: int | None = eqx.field(static=True)

    def __init__(self, *, block_dimension: int | None = None):
        dimension = None if block_dimension is None else int(block_dimension)
        if dimension is not None and dimension < 1:
            raise ValueError("LOBPCG block_dimension must be positive or None.")
        self.block_dimension = dimension

    @property
    def name(self) -> str:
        return "lobpcg"


class RestartedLanczos(StrictModule):
    """Fixed-capacity thick-restarted self-adjoint Lanczos iteration."""

    subspace_dimension: int | None = eqx.field(static=True)
    restart_dimension: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        subspace_dimension: int | None = None,
        restart_dimension: int | None = None,
    ):
        subspace = None if subspace_dimension is None else int(subspace_dimension)
        restart = None if restart_dimension is None else int(restart_dimension)
        if subspace is not None and subspace < 2:
            raise ValueError("subspace_dimension must be at least two or None.")
        if restart is not None and restart < 1:
            raise ValueError("restart_dimension must be positive or None.")
        if subspace is not None and restart is not None and restart >= subspace:
            raise ValueError("restart_dimension must be smaller than subspace_dimension.")
        self.subspace_dimension = subspace
        self.restart_dimension = restart

    @property
    def name(self) -> str:
        return "restarted-lanczos"


EigenMethod: TypeAlias = AutoEigenMethod | DenseEigh | LOBPCG | RestartedLanczos


class EigenTolerancePolicy(StrictModule):
    """Residual and orthogonality requirements for certified modes."""

    relative: float = eqx.field(static=True)
    absolute: float = eqx.field(static=True)
    orthogonality: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative: float = 1e-6,
        absolute: float = 1e-8,
        orthogonality: float = 1e-6,
    ):
        scalars = tuple(float(value) for value in (relative, absolute, orthogonality))
        if any(not math.isfinite(value) or value < 0.0 for value in scalars):
            raise ValueError("Eigen tolerances must be finite and non-negative.")
        self.relative, self.absolute, self.orthogonality = scalars


class EigenResourcePolicy(StrictModule):
    """Hard budgets for fixed-shape eigen preparation and iteration."""

    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    krylov_basis_bytes: int = eqx.field(static=True)
    preconditioner_bytes: int = eqx.field(static=True)
    operator_matvecs: int = eqx.field(static=True)
    metric_matvecs: int = eqx.field(static=True)
    preconditioner_applies: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        preparation_bytes: int = 256 * 1024 * 1024,
        workspace_bytes: int = 512 * 1024 * 1024,
        krylov_basis_bytes: int = 512 * 1024 * 1024,
        preconditioner_bytes: int = 256 * 1024 * 1024,
        operator_matvecs: int = 1_000_000,
        metric_matvecs: int = 1_000_000,
        preconditioner_applies: int = 1_000_000,
    ):
        values = tuple(
            int(value)
            for value in (
                preparation_bytes,
                workspace_bytes,
                krylov_basis_bytes,
                preconditioner_bytes,
                operator_matvecs,
                metric_matvecs,
                preconditioner_applies,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("Eigen resource budgets must be non-negative.")
        (
            self.preparation_bytes,
            self.workspace_bytes,
            self.krylov_basis_bytes,
            self.preconditioner_bytes,
            self.operator_matvecs,
            self.metric_matvecs,
            self.preconditioner_applies,
        ) = values


class EigenSolvePolicy(StrictModule):
    """Numerical contract; ``eigenvalues`` requests first-order derivatives only."""

    method: EigenMethod
    count: int = eqx.field(static=True)
    which: EigenTarget = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    tolerance: EigenTolerancePolicy
    resources: EigenResourcePolicy
    materialization: MaterializationPolicy
    initial_basis: Array | None
    key: Array | None
    preconditioning: PreconditioningPolicy | None
    differentiation: EigenDifferentiationMode = eqx.field(static=True)
    failure: FailurePolicy

    def __init__(
        self,
        method: EigenMethod | None = None,
        /,
        *,
        count: int = 1,
        which: EigenTarget = "smallest-algebraic",
        max_steps: int = 100,
        tolerance: EigenTolerancePolicy | None = None,
        resources: EigenResourcePolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        initial_basis: ArrayLike | None = None,
        key: ArrayLike | None = None,
        preconditioning: PreconditioningPolicy | None = None,
        differentiation: EigenDifferentiationMode = "none",
        failure: FailurePolicy | None = None,
    ):
        method_ = AutoEigenMethod() if method is None else method
        if not isinstance(
            method_,
            (AutoEigenMethod, DenseEigh, LOBPCG, RestartedLanczos),
        ):
            raise TypeError(
                "method must be AutoEigenMethod, DenseEigh, LOBPCG, or RestartedLanczos."
            )
        count_ = int(count)
        steps = int(max_steps)
        if count_ < 1:
            raise ValueError("Eigenvalue count must be positive.")
        if steps < 1:
            raise ValueError("max_steps must be positive.")
        if which not in (
            "smallest-algebraic",
            "largest-algebraic",
            "smallest-magnitude",
            "largest-magnitude",
        ):
            raise ValueError("Unknown eigenvalue target.")
        tolerance_ = EigenTolerancePolicy() if tolerance is None else tolerance
        resources_ = EigenResourcePolicy() if resources is None else resources
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(tolerance_, EigenTolerancePolicy):
            raise TypeError("tolerance must be an EigenTolerancePolicy.")
        if not isinstance(resources_, EigenResourcePolicy):
            raise TypeError("resources must be an EigenResourcePolicy.")
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        if preconditioning is not None and not isinstance(
            preconditioning, PreconditioningPolicy
        ):
            raise TypeError("preconditioning must be a PreconditioningPolicy or None.")
        if differentiation not in ("none", "eigenvalues"):
            raise ValueError("differentiation must be 'none' or 'eigenvalues'.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy.")
        basis_ = _initial_basis(initial_basis)
        key_ = _random_key(key)
        self.method = method_
        self.count = count_
        self.which = which
        self.max_steps = steps
        self.tolerance = tolerance_
        self.resources = resources_
        self.materialization = materialization_
        self.initial_basis = basis_
        self.key = key_
        self.preconditioning = preconditioning
        self.differentiation = differentiation
        self.failure = failure_


def _initial_basis(value: ArrayLike | None, /) -> Array | None:
    if value is None:
        return None
    basis = jnp.asarray(value)
    if basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] < 1:
        raise ValueError("initial_basis must be a non-empty rank-two coordinate array.")
    if not jnp.issubdtype(basis.dtype, jnp.inexact):
        raise TypeError("initial_basis must use an inexact dtype.")
    return eqx.error_if(
        basis,
        jnp.any(~jnp.isfinite(basis)),
        "initial_basis entries must be finite.",
    )


def _random_key(value: ArrayLike | None, /) -> Array | None:
    if value is None:
        return None
    key = jnp.asarray(value)
    typed_key = jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
    legacy_key = key.dtype == jnp.dtype(jnp.uint32) and key.shape == (2,)
    if typed_key and key.shape != ():
        raise ValueError("A typed PRNG key must be scalar.")
    if not typed_key and not legacy_key:
        raise TypeError(
            "key must be a scalar typed PRNG key or a uint32 key of shape (2,)."
        )
    return key


__all__ = [
    "AutoEigenMethod",
    "EigenDifferentiationMode",
    "DenseEigh",
    "EigenResourcePolicy",
    "EigenSolvePolicy",
    "EigenTarget",
    "EigenTolerancePolicy",
    "LOBPCG",
    "RestartedLanczos",
]
