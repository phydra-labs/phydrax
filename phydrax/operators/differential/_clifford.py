#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp

from ..._strict import StrictModule
from ...domain import DomainFunction
from ...metrix.clifford import (
    basis_blade,
    CliffordAlgebraSpec,
    CliffordBladeLayout,
    CliffordProductPlan,
    prepare_product,
)
from ._domain_ops import _factor_and_dim, partial


class _CliffordDiracCallable(StrictModule):
    derivatives: tuple[DomainFunction, ...]
    products: tuple[CliffordProductPlan, ...]
    reciprocal_vectors: tuple[jnp.ndarray, ...]

    def __init__(
        self,
        derivatives: tuple[DomainFunction, ...],
        products: tuple[CliffordProductPlan, ...],
        reciprocal_vectors: tuple[jnp.ndarray, ...],
        /,
    ):
        self.derivatives = derivatives
        self.products = products
        self.reciprocal_vectors = reciprocal_vectors

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        total = None
        for derivative, product, reciprocal in zip(
            self.derivatives,
            self.products,
            self.reciprocal_vectors,
        ):
            values = jnp.asarray(derivative.func(*args, key=key, **kwargs))
            contribution = product(reciprocal, values)
            total = contribution if total is None else total + contribution
        if total is None:
            raise RuntimeError("Clifford Dirac operator has no coordinate derivatives.")
        return total


def clifford_dirac(
    field: DomainFunction,
    algebra: CliffordAlgebraSpec,
    layout: CliffordBladeLayout,
    /,
    *,
    var: str = "x",
    side: Literal["left"] = "left",
    mode: Literal["reverse", "forward"] = "forward",
) -> DomainFunction:
    """Return the flat constant-metric left Dirac derivative of a Clifford field."""
    if not isinstance(field, DomainFunction):
        raise TypeError("field must be a DomainFunction.")
    if not isinstance(algebra, CliffordAlgebraSpec):
        raise TypeError("algebra must be a CliffordAlgebraSpec.")
    if not isinstance(layout, CliffordBladeLayout):
        raise TypeError("layout must be a CliffordBladeLayout.")
    algebra.require_compatible(layout.algebra)
    if side != "left":
        raise ValueError("Initial Clifford Dirac support is left-sided only.")
    if not algebra.nondegenerate:
        raise ValueError("Clifford Dirac operator requires a nondegenerate metric.")
    _, dimension = _factor_and_dim(field, var)
    if dimension != algebra.dimension:
        raise ValueError("Dirac variable and Clifford algebra dimensions do not match.")
    if not layout.complete_grades:
        raise ValueError("Dirac operator requires a union of complete Clifford grades.")

    derivatives = tuple(
        partial(field, var=var, axis=axis, mode=mode) for axis in range(algebra.dimension)
    )
    products = tuple(
        prepare_product(
            algebra,
            layout,
            layout,
            output_layout=layout,
            backend="sparse",
        )
        for _ in range(algebra.dimension)
    )
    reciprocal_vectors = tuple(
        algebra.diagonal[axis] * basis_blade(layout, 1 << axis)
        for axis in range(algebra.dimension)
    )
    metadata = {
        key: value
        for key, value in field.metadata.items()
        if key != "trial_space_certificate"
    }
    return DomainFunction(
        domain=field.domain,
        deps=field.deps,
        func=_CliffordDiracCallable(derivatives, products, reciprocal_vectors),
        metadata=metadata,
    )


__all__ = ["clifford_dirac"]
