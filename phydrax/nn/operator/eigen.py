#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...linalg import AbstractVectorSpace
from ...linalg.eigen import (
    Eigenproblem,
    EigenSolvePolicy,
    GeneralizedEigenproblem,
    rayleigh_ritz,
    TrialSubspaceRitzResult,
    warm_started_eigensolve,
    WarmStartedEigenResult,
)
from .data import FunctionSamples


class OperatorTrialSubspace(StrictModule):
    """One neural-operator prediction lowered to eigensolver coordinates."""

    basis: Array
    support_id: str = eqx.field(static=True)
    measure_id: str | None = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    masked: bool = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.basis.shape[1])


def operator_trial_subspace(
    samples: FunctionSamples,
    space: AbstractVectorSpace,
    /,
) -> OperatorTrialSubspace:
    """Lower trailing output modes on one support to coordinate basis columns."""
    if not isinstance(samples, FunctionSamples):
        raise TypeError("samples must be FunctionSamples.")
    if not isinstance(space, AbstractVectorSpace):
        raise TypeError("space must be an AbstractVectorSpace.")
    if samples.values is None:
        raise ValueError("Trial-subspace FunctionSamples require values.")
    values = jnp.asarray(samples.values)
    sample_shape = samples.sample_shape
    sample_rank = len(sample_shape)
    if values.ndim < sample_rank + 1:
        raise ValueError(
            "Trial-subspace values must append one mode axis to the sample/value axes."
        )
    if tuple(int(size) for size in values.shape[:sample_rank]) != sample_shape:
        raise ValueError(
            "Trial-subspace values must describe exactly one case on the declared support."
        )
    capacity = int(values.shape[-1])
    if capacity < 1:
        raise ValueError("Trial-subspace mode capacity must be positive.")
    value_shape = tuple(int(size) for size in values.shape[sample_rank:-1])
    coordinate_size = math.prod(sample_shape + value_shape)
    if coordinate_size != space.size:
        raise ValueError(
            "Flattened trial-subspace values must match the eigensolver space size; "
            f"got {coordinate_size} and {space.size}."
        )
    if samples.mask is not None:
        mask = jnp.asarray(samples.mask, dtype=bool)
        if tuple(int(size) for size in mask.shape) != sample_shape:
            raise ValueError(
                "Trial-subspace masks must describe one unbatched sample support."
            )
        mask_shape = sample_shape + (1,) * (values.ndim - sample_rank)
        values = jnp.where(mask.reshape(mask_shape), values, 0)
    basis = values.reshape((space.size, capacity))
    space.unflatten(basis[:, 0])
    return OperatorTrialSubspace(
        basis=basis,
        support_id=samples.support_id,
        measure_id=samples.measure_id,
        sample_shape=sample_shape,
        value_shape=value_shape,
        masked=samples.mask is not None,
    )


def rayleigh_ritz_from_samples(
    problem: Eigenproblem | GeneralizedEigenproblem,
    samples: FunctionSamples,
    /,
    *,
    count: int | None = None,
    which: str = "smallest-algebraic",
    tolerance: float = 1e-10,
) -> TrialSubspaceRitzResult:
    """Certify a neural-operator trial-space prediction by physical projection."""
    trial = operator_trial_subspace(samples, problem.operator.source)
    return rayleigh_ritz(
        problem,
        trial.basis,
        count=count,
        which=which,
        tolerance=tolerance,
    )


def warm_started_eigensolve_from_samples(
    problem: Eigenproblem | GeneralizedEigenproblem,
    samples: FunctionSamples,
    /,
    *,
    policy: EigenSolvePolicy | None = None,
    tolerance: float = 1e-10,
) -> WarmStartedEigenResult:
    """Refine a neural-operator trial-space prediction with the native solver."""
    trial = operator_trial_subspace(samples, problem.operator.source)
    return warm_started_eigensolve(
        problem,
        trial.basis,
        policy=policy,
        tolerance=tolerance,
    )


__all__ = [
    "OperatorTrialSubspace",
    "operator_trial_subspace",
    "rayleigh_ritz_from_samples",
    "warm_started_eigensolve_from_samples",
]
