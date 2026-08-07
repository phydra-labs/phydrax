#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Real
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from phydrax.domain import DomainFunction

from ..._strict import StrictModule
from ._validation import (
    join_function_arguments,
    validate_matrix_value,
    validate_vector_value,
)


def _coerce_entropy_base(base: ArrayLike, /) -> Array:
    value = jnp.asarray(base)
    if value.shape != ():
        raise ValueError(f"entropy base must be a scalar, got shape {value.shape}.")
    if jnp.iscomplexobj(value):
        raise TypeError("entropy base must be real.")
    if isinstance(base, Real) and (isinstance(base, bool) or base <= 0 or base == 1):
        raise ValueError("entropy base must be positive and unequal to one.")
    return value


def _density_eigh(value: Any, /, *, role: str) -> tuple[Array, Array]:
    density = validate_matrix_value(value, role=role)
    if int(density.shape[0]) == 0:
        raise ValueError(f"{role} must be nonempty.")
    density = density * jnp.asarray(1.0)
    adjoint = jnp.conj(density.T)
    real_dtype = jnp.real(density).dtype
    tolerance = (
        100.0
        * int(density.shape[0])
        * jnp.finfo(real_dtype).eps
        * jnp.maximum(1.0, jnp.max(jnp.abs(density)))
    )
    checked_density = eqx.error_if(
        density,
        jnp.max(jnp.abs(density - adjoint)) > tolerance,
        f"{role} must be Hermitian.",
    )
    hermitian = 0.5 * (checked_density + jnp.conj(checked_density.T))
    eigenvalues, eigenvectors = jnp.linalg.eigh(hermitian)
    checked_eigenvalues = eqx.error_if(
        eigenvalues,
        jnp.min(eigenvalues) < -tolerance,
        f"{role} must be positive semidefinite.",
    )
    return jnp.maximum(checked_eigenvalues, 0.0), eigenvectors


class _PurityCallable(StrictModule):
    density: DomainFunction

    def __init__(self, density: DomainFunction):
        self.density = density

    def __call__(self, *args, key=None, **kwargs):
        density = validate_matrix_value(
            self.density.func(*args, key=key, **kwargs),
            role="density operator",
        )
        return jnp.real(oe.contract("ij,ji->", density, density))


class _EntropyCallable(StrictModule):
    density: DomainFunction
    base: Array

    def __init__(self, density: DomainFunction, base: Array):
        self.density = density
        self.base = base

    def __call__(self, *args, key=None, **kwargs):
        eigenvalues, _ = _density_eigh(
            self.density.func(*args, key=key, **kwargs),
            role="density operator",
        )
        checked_base = eqx.error_if(
            self.base,
            (self.base <= 0) | (self.base == 1),
            "entropy base must be positive and unequal to one.",
        )
        return -jnp.sum(jsp.special.xlogy(eigenvalues, eigenvalues)) / jnp.log(
            checked_base
        )


class _StateFidelityCallable(StrictModule):
    left: DomainFunction
    right: DomainFunction
    left_positions: tuple[int, ...]
    right_positions: tuple[int, ...]

    def __init__(
        self,
        left: DomainFunction,
        right: DomainFunction,
        left_positions: tuple[int, ...],
        right_positions: tuple[int, ...],
    ):
        self.left = left
        self.right = right
        self.left_positions = left_positions
        self.right_positions = right_positions

    def __call__(self, *args, key=None, **kwargs):
        left_args = tuple(args[index] for index in self.left_positions)
        right_args = tuple(args[index] for index in self.right_positions)
        left = validate_vector_value(
            self.left.func(*left_args, key=key, **kwargs),
            role="left quantum state",
        )
        right = validate_vector_value(
            self.right.func(*right_args, key=key, **kwargs),
            role="right quantum state",
        )
        if left.shape != right.shape:
            raise ValueError(
                "Quantum-state dimensions must match for state_fidelity; "
                f"got {left.shape} and {right.shape}."
            )
        return jnp.abs(jnp.vdot(left, right)) ** 2


class _DensityFidelityCallable(StrictModule):
    left: DomainFunction
    right: DomainFunction
    left_positions: tuple[int, ...]
    right_positions: tuple[int, ...]

    def __init__(
        self,
        left: DomainFunction,
        right: DomainFunction,
        left_positions: tuple[int, ...],
        right_positions: tuple[int, ...],
    ):
        self.left = left
        self.right = right
        self.left_positions = left_positions
        self.right_positions = right_positions

    def __call__(self, *args, key=None, **kwargs):
        left_args = tuple(args[index] for index in self.left_positions)
        right_args = tuple(args[index] for index in self.right_positions)
        left_eigenvalues, left_eigenvectors = _density_eigh(
            self.left.func(*left_args, key=key, **kwargs),
            role="left density operator",
        )
        right_eigenvalues, right_eigenvectors = _density_eigh(
            self.right.func(*right_args, key=key, **kwargs),
            role="right density operator",
        )
        if left_eigenvectors.shape != right_eigenvectors.shape:
            raise ValueError(
                "Density-operator dimensions must match for density_fidelity; "
                f"got {left_eigenvectors.shape} and {right_eigenvectors.shape}."
            )
        left_root = (left_eigenvectors * jnp.sqrt(left_eigenvalues)[None, :]) @ jnp.conj(
            left_eigenvectors.T
        )
        right_root = (
            right_eigenvectors * jnp.sqrt(right_eigenvalues)[None, :]
        ) @ jnp.conj(right_eigenvectors.T)
        singular_values = jnp.linalg.svd(
            left_root @ right_root,
            compute_uv=False,
        )
        return jnp.sum(singular_values) ** 2


class _TraceDistanceCallable(StrictModule):
    left: DomainFunction
    right: DomainFunction
    left_positions: tuple[int, ...]
    right_positions: tuple[int, ...]

    def __init__(
        self,
        left: DomainFunction,
        right: DomainFunction,
        left_positions: tuple[int, ...],
        right_positions: tuple[int, ...],
    ):
        self.left = left
        self.right = right
        self.left_positions = left_positions
        self.right_positions = right_positions

    def __call__(self, *args, key=None, **kwargs):
        left_args = tuple(args[index] for index in self.left_positions)
        right_args = tuple(args[index] for index in self.right_positions)
        left = validate_matrix_value(
            self.left.func(*left_args, key=key, **kwargs),
            role="left density operator",
        )
        right = validate_matrix_value(
            self.right.func(*right_args, key=key, **kwargs),
            role="right density operator",
        )
        if left.shape != right.shape:
            raise ValueError(
                "Density-operator dimensions must match for trace_distance; "
                f"got {left.shape} and {right.shape}."
            )
        return 0.5 * jnp.sum(jnp.linalg.svd(left - right, compute_uv=False))


def purity(density: DomainFunction, /) -> DomainFunction:
    r"""Return the pointwise purity $\operatorname{tr}(\rho^2)$.

    ``density`` must be square-matrix-valued. The result is explicitly real. A physical
    unit-trace density has purity in $[1/n,1]$, but physicality is not imposed
    implicitly.
    """
    if not isinstance(density, DomainFunction):
        raise TypeError("purity expects a DomainFunction.")
    return DomainFunction(
        domain=density.domain,
        deps=density.deps,
        func=_PurityCallable(density),
        metadata={},
    )


def von_neumann_entropy(
    density: DomainFunction,
    /,
    *,
    base: ArrayLike = 2.0,
) -> DomainFunction:
    r"""Return $-\operatorname{tr}(\rho\log_{\mathtt{base}}\rho)$ pointwise.

    ``density`` must be a nonempty Hermitian positive-semidefinite matrix. Zero
    eigenvalues contribute zero. Unit trace is assumed when interpreting the result as
    an entropy but is not imposed or restored implicitly. ``base`` must be a positive
    real scalar unequal to one and defaults to two, so the result is measured in bits.
    """
    if not isinstance(density, DomainFunction):
        raise TypeError("von_neumann_entropy expects a DomainFunction.")
    return DomainFunction(
        domain=density.domain,
        deps=density.deps,
        func=_EntropyCallable(density, _coerce_entropy_base(base)),
        metadata={},
    )


def state_fidelity(
    left: DomainFunction,
    right: DomainFunction,
    /,
) -> DomainFunction:
    r"""Return the pure-state fidelity $|\langle\psi|\phi\rangle|^2$ pointwise.

    Both operands must be equally sized vectors. Normalization is assumed and is not
    imposed implicitly.
    """
    if not isinstance(left, DomainFunction) or not isinstance(right, DomainFunction):
        raise TypeError("state_fidelity expects two DomainFunctions.")
    domain, deps, promoted, positions = join_function_arguments(left, right)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_StateFidelityCallable(
            promoted[0],
            promoted[1],
            positions[0],
            positions[1],
        ),
        metadata={},
    )


def density_fidelity(
    left: DomainFunction,
    right: DomainFunction,
    /,
) -> DomainFunction:
    r"""Return the squared Uhlmann fidelity between two density operators.

    The convention is

    $$
    F(\rho,\sigma)=\left\|\sqrt{\rho}\sqrt{\sigma}\right\|_1^2.
    $$

    Both operands must be equally sized, nonempty Hermitian positive-semidefinite
    matrices. Unit trace is assumed when interpreting the result in $[0,1]$ but is not
    imposed or restored implicitly.
    """
    if not isinstance(left, DomainFunction) or not isinstance(right, DomainFunction):
        raise TypeError("density_fidelity expects two DomainFunctions.")
    domain, deps, promoted, positions = join_function_arguments(left, right)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_DensityFidelityCallable(
            promoted[0],
            promoted[1],
            positions[0],
            positions[1],
        ),
        metadata={},
    )


def trace_distance(
    left: DomainFunction,
    right: DomainFunction,
    /,
) -> DomainFunction:
    r"""Return $\tfrac12\|\rho-\sigma\|_1$ pointwise.

    Both operands must be equally sized square matrices. For physical unit-trace
    densities the result lies in $[0,1]$. Physicality is not imposed implicitly.
    """
    if not isinstance(left, DomainFunction) or not isinstance(right, DomainFunction):
        raise TypeError("trace_distance expects two DomainFunctions.")
    domain, deps, promoted, positions = join_function_arguments(left, right)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_TraceDistanceCallable(
            promoted[0],
            promoted[1],
            positions[0],
            positions[1],
        ),
        metadata={},
    )


__all__ = [
    "density_fidelity",
    "purity",
    "state_fidelity",
    "trace_distance",
    "von_neumann_entropy",
]
