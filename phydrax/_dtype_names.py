#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical scalar dtype names and dtype rules.

This leaf owns the supported precision dtype table and dtype classification. It
imports no other Phydrax module so that strict-module construction and the typing
plan can depend on it without cycles. It never allocates arrays or inspects array
values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike


RealPrecisionDType: TypeAlias = Literal[
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float16",
    "bfloat16",
    "float32",
    "float64",
]
ComplexPrecisionDType: TypeAlias = Literal["complex64", "complex128"]
ScalarPrecisionDType: TypeAlias = RealPrecisionDType | ComplexPrecisionDType
DTypeCategory: TypeAlias = Literal[
    "boolean", "integer", "floating", "complex", "inexact", "numeric", "prng_key"
]

# The Literal aliases are the single declaration of the supported names; the
# typed tuples let membership checks narrow a canonical name to its alias.
_REAL_PRECISION_DTYPES: tuple[RealPrecisionDType, ...] = get_args(RealPrecisionDType)
_PRECISION_DTYPES: tuple[ScalarPrecisionDType, ...] = (
    *_REAL_PRECISION_DTYPES,
    *get_args(ComplexPrecisionDType),
)


def canonical_dtype(value: DTypeLike, /) -> np.dtype:
    """Return the JAX-canonical dtype object named by `value`."""
    return jax.dtypes.canonicalize_dtype(jnp.dtype(value))


def precision_dtype_name(value: DTypeLike, /) -> ScalarPrecisionDType:
    """Return one canonical supported JAX scalar dtype name."""
    name = canonical_dtype(value).name
    if name not in _PRECISION_DTYPES:
        raise ValueError(f"Unsupported precision dtype {name!r}.")
    return name


def real_precision_dtype_name(value: DTypeLike, /) -> RealPrecisionDType:
    """Return one canonical supported real floating dtype name."""
    name = precision_dtype_name(value)
    if name not in _REAL_PRECISION_DTYPES:
        raise ValueError(f"Precision dtype {name!r} is not real floating-point.")
    return name


def complex_precision_dtype(value: DTypeLike, /) -> ComplexPrecisionDType:
    """Return the complex companion used for one real precision dtype."""
    name = real_precision_dtype_name(value)
    return "complex128" if name == "float64" else "complex64"


def inexact_result_type(*values: object) -> np.dtype:
    """Return the JAX result dtype of ``values``, promoted to an inexact dtype.

    Floating and complex inputs keep their precision (``float32`` stays
    ``float32``); integer, boolean, and empty inputs use JAX's canonical default
    floating dtype. Equivalent to ``jnp.result_type(*values, float)``, where the
    weakly typed Python ``float`` never widens an inexact input.
    """
    if values:
        dtype = jnp.result_type(*values)
        if jnp.issubdtype(dtype, jnp.inexact):
            return dtype
    return jax.dtypes.canonicalize_dtype(jnp.float64)


@dataclass(frozen=True, slots=True)
class DTypeRule:
    """One accepted dtype: an exact canonical dtype or a dtype category."""

    exact: np.dtype | None
    category: DTypeCategory | None

    def __post_init__(self) -> None:
        if (self.exact is None) == (self.category is None):
            raise ValueError("A dtype rule names exactly one exact dtype or category.")

    @property
    def label(self) -> str:
        return self.exact.name if self.exact is not None else str(self.category)


def exact_dtype_rule(value: DTypeLike, /) -> DTypeRule:
    """Return the rule accepting only the canonical dtype named by `value`."""
    return DTypeRule(canonical_dtype(value), None)


def category_dtype_rule(category: DTypeCategory, /) -> DTypeRule:
    """Return the rule accepting every dtype of one category."""
    return DTypeRule(None, category)


def dtype_matches(rule: DTypeRule, dtype: np.dtype, /) -> bool:
    """Return whether `dtype` satisfies `rule`, without inspecting array values."""
    if rule.exact is not None:
        return dtype == rule.exact
    match rule.category:
        case "boolean":
            return dtype == np.dtype(np.bool_)
        case "integer":
            return bool(jnp.issubdtype(dtype, jnp.integer))
        case "floating":
            return bool(jnp.issubdtype(dtype, jnp.floating))
        case "complex":
            return bool(jnp.issubdtype(dtype, jnp.complexfloating))
        case "inexact":
            return bool(jnp.issubdtype(dtype, jnp.inexact))
        case "numeric":
            return bool(jnp.issubdtype(dtype, jnp.number))
        case "prng_key":
            return bool(jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key))
        case None:
            raise ValueError("A dtype rule names exactly one exact dtype or category.")


__all__ = [
    "ComplexPrecisionDType",
    "RealPrecisionDType",
    "ScalarPrecisionDType",
    "complex_precision_dtype",
    "precision_dtype_name",
    "real_precision_dtype_name",
]
