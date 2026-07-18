#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from ..._strict import StrictModule
from ...domain._function import _drop_derivative_hook_metadata, DomainFunction
from ._domain_ops import directional_derivative


_ADEngine = Literal["auto", "reverse", "forward", "jvp"]


class _VectorFieldCallable(StrictModule):
    source: DomainFunction
    dimension: int
    role: str

    def __init__(self, source: DomainFunction, dimension: int, role: str):
        self.source = source
        self.dimension = int(dimension)
        self.role = role

    def __call__(self, *args, key=None, **kwargs):
        value = jnp.asarray(self.source.func(*args, key=key, **kwargs))
        if value.ndim != 1 or int(value.shape[0]) != self.dimension:
            raise ValueError(
                f"lie_bracket {self.role} must be a vector of shape "
                f"({self.dimension},), got {value.shape}."
            )
        return value


def _validated_vector_field(
    field: DomainFunction,
    /,
    *,
    dimension: int,
    role: str,
) -> DomainFunction:
    return DomainFunction(
        domain=field.domain,
        deps=field.deps,
        func=_VectorFieldCallable(field, dimension, role),
        metadata=_drop_derivative_hook_metadata(field.metadata),
    )


def lie_bracket(
    x: DomainFunction,
    y: DomainFunction,
    /,
    *,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
    ad_engine: _ADEngine = "auto",
) -> DomainFunction:
    r"""Lie bracket $[X,Y]=D_XY-D_YX$ of two vector fields.

    The fields must take values in the tangent space of the geometry coordinate
    selected by ``var``. This operator is the Lie bracket of vector fields; matrix
    operator commutators are exposed separately as ``commutator``.
    """
    if not isinstance(x, DomainFunction) or not isinstance(y, DomainFunction):
        raise TypeError("lie_bracket expects two DomainFunction vector fields.")
    if var not in x.domain.labels or var not in y.domain.labels:
        raise ValueError(f"Both vector fields must be defined over variable {var!r}.")

    x_dimension = int(x.domain.factor(var).var_dim)
    y_dimension = int(y.domain.factor(var).var_dim)
    if x_dimension != y_dimension:
        raise ValueError(
            "lie_bracket vector-field dimensions must match; "
            f"got {x_dimension} and {y_dimension}."
        )

    x_ = _validated_vector_field(x, dimension=x_dimension, role="left operand")
    y_ = _validated_vector_field(y, dimension=x_dimension, role="right operand")
    derivative_kwargs = {
        "var": var,
        "mode": mode,
        "backend": backend,
        "basis": basis,
        "periodic": periodic,
        "ad_engine": ad_engine,
    }
    return directional_derivative(y_, x_, **derivative_kwargs) - directional_derivative(
        x_, y_, **derivative_kwargs
    )


__all__ = ["lie_bracket"]
