#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

import phydrax.ein as ein

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractScalarTerm
from .._trainable import NonTrainableState
from ..conditions import AbstractResidualCondition
from ..domain import DomainComponent, DomainFunction


class LocalTestSpaceEvidence(StrictModule, NonTrainableState):
    minimum_gram_eigenvalue: float = eqx.field(static=True)
    gram_condition: float = eqx.field(static=True)
    orthonormal: bool = eqx.field(static=True)
    verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_gram_eigenvalue: float,
        gram_condition: float,
        orthonormal: bool,
        verified: bool,
    ):
        self.minimum_gram_eigenvalue = float(minimum_gram_eigenvalue)
        self.gram_condition = float(gram_condition)
        self.orthonormal = bool(orthonormal)
        self.verified = bool(verified)


class LocalTestSpace(StrictModule, NonTrainableState):
    """Fixed local test basis, quadrature, and discrete Riesz map."""

    component: DomainComponent
    points: Any
    basis_values: Array
    weights: Array
    gram_inverse: Array
    evidence: LocalTestSpaceEvidence
    test_space_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: DomainComponent,
        points: Any,
        basis_values: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        test_space_id: str,
        gram_tolerance: float = 1.0e-12,
    ):
        if not isinstance(component, DomainComponent):
            raise TypeError("component must be a DomainComponent.")
        basis = np.asarray(basis_values, dtype=float)
        if basis.ndim != 2 or basis.shape[0] <= 0 or basis.shape[1] <= 0:
            raise ValueError("basis_values must have shape (points, modes).")
        weights_ = (
            np.full((basis.shape[0],), 1.0 / basis.shape[0], dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if weights_.shape != (basis.shape[0],):
            raise ValueError("weights must contain one quadrature weight per point.")
        if np.any(weights_ <= 0.0) or not np.all(np.isfinite(weights_)):
            raise ValueError("Local test-space weights must be finite and positive.")
        gram = basis.T @ (weights_[:, None] * basis)
        eigenvalues = np.linalg.eigvalsh(gram)
        minimum = float(eigenvalues[0])
        condition = float(eigenvalues[-1] / minimum) if minimum > 0.0 else np.inf
        verified = bool(minimum > float(gram_tolerance) and np.isfinite(condition))
        if not verified:
            raise ValueError("Local test basis has a singular discrete Gram matrix.")
        identity = np.eye(gram.shape[0])
        orthonormal = bool(np.allclose(gram, identity, atol=1.0e-10, rtol=1.0e-10))
        identifier = str(test_space_id)
        if not identifier:
            raise ValueError("test_space_id must be non-empty.")
        self.component = component
        self.points = points
        self.basis_values = jnp.asarray(basis)
        self.weights = jnp.asarray(weights_)
        self.gram_inverse = jnp.asarray(np.linalg.solve(gram, identity))
        self.evidence = LocalTestSpaceEvidence(
            minimum_gram_eigenvalue=minimum,
            gram_condition=condition,
            orthonormal=orthonormal,
            verified=verified,
        )
        self.test_space_id = identifier


class LocalizedResidualNorm(AbstractScalarTerm):
    """Riesz-weighted local weak residual norm over a fixed test space."""

    condition: AbstractResidualCondition
    test_space: LocalTestSpace
    scale: Array
    fields: tuple[str, ...] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        condition: AbstractResidualCondition,
        test_space: LocalTestSpace,
        /,
        *,
        scale: float = 1.0,
        label: str | None = None,
    ):
        if not isinstance(condition, AbstractResidualCondition):
            raise TypeError("condition must be an AbstractResidualCondition.")
        if not isinstance(test_space, LocalTestSpace):
            raise TypeError("test_space must be a LocalTestSpace.")
        if not condition.on.domain.same_support(test_space.component.domain):
            raise ValueError("Condition and local test space must share one domain.")
        scale_ = jnp.asarray(scale, dtype=float).reshape(())
        if not bool(jnp.isfinite(scale_)) or float(scale_) < 0.0:
            raise ValueError("scale must be finite and non-negative.")
        self.condition = condition
        self.test_space = test_space
        self.scale = scale_
        self.fields = condition.fields
        self.label = condition.label if label is None else str(label)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        residual = self.condition.residual(functions)(
            self.test_space.points,
            key=key,
        ).data
        moments = ein.contract(
            "nm,n...,n->m...",
            self.test_space.basis_values,
            residual,
            self.test_space.weights,
        )
        flat = moments.reshape((moments.shape[0], -1))
        riesz = ein.contract("mn,nk->mk", self.test_space.gram_inverse, flat)
        return self.scale * jnp.real(jnp.sum(jnp.conj(flat) * riesz))


def polynomial_test_space(
    component: DomainComponent,
    points: Any,
    coordinate_values: ArrayLike,
    order: int,
    /,
    *,
    weights: ArrayLike | None = None,
    test_space_id: str,
) -> LocalTestSpace:
    """Build Legendre test modes on one normalized scalar coordinate design."""
    order_ = int(order)
    if order_ < 0:
        raise ValueError("order must be non-negative.")
    coordinate = np.asarray(coordinate_values, dtype=float).reshape((-1,))
    if np.any(coordinate < -1.0) or np.any(coordinate > 1.0):
        raise ValueError("Polynomial test coordinates must lie in [-1, 1].")
    basis = np.polynomial.legendre.legvander(coordinate, order_)
    return LocalTestSpace(
        component,
        points,
        basis,
        weights=weights,
        test_space_id=test_space_id,
    )


__all__ = [
    "LocalTestSpace",
    "LocalTestSpaceEvidence",
    "LocalizedResidualNorm",
    "polynomial_test_space",
]
