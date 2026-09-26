#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phydrax annotation vocabulary objects.

Static checkers see every form as its base type (`jax.Array`, `numpy.ndarray`,
`int`, `str`, `tuple[str, ...]`); the structural contract of each tensor form is
recorded in an immutable registration table read by the private plan compiler.
Nothing in this module validates values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Literal, NoReturn, Protocol, TypeAlias, TypeVar

import jax
import numpy as np
import numpy.typing as npt
from jax.typing import ArrayLike
from typing_extensions import TypeAliasType, TypeVarTuple

from ._dtype_names import category_dtype_rule, DTypeRule, exact_dtype_rule


_Shape = TypeVarTuple("_Shape")
_SizeVar = TypeVar("_SizeVar")
_FormVar = TypeVar("_FormVar")


class Dim:
    """Nominal size variable: a named extent that must agree wherever it appears.

    Declare one by subclassing, optionally with a minimum extent::

        class ComponentDim(Dim, minimum=1):
            \"\"\"Number of chemical components.\"\"\"

    Identity is the class object. A dimension is a size variable only; it is not a
    scientific axis identity (see `phydrax.axes.AxisKey`). By convention ordinary
    dimensions end in ``Dim`` and variadic groups in ``Dims``.
    """

    minimum: ClassVar[int] = 0

    def __init_subclass__(cls, *, minimum: int = 0, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if type(minimum) is not int:
            raise TypeError("A dimension minimum must be an int.")
        if minimum < 0:
            raise ValueError("A dimension minimum must be nonnegative.")
        cls.minimum = minimum

    def __new__(cls, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError(
            f"{cls.__name__} is a dimension token and cannot be instantiated."
        )


class VariadicDim(Dim):
    """Nominal group of zero or more extents that must agree wherever it appears."""


class _Token:
    """Fixed shape tokens: neither instantiable nor extensible."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if cls.__module__ != __name__:
            raise TypeError(f"{cls.__mro__[1].__name__} cannot be subclassed.")
        super().__init_subclass__(**kwargs)

    def __new__(cls, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError(f"{cls.__name__} is a shape token and cannot be instantiated.")


class AnyDim(_Token):
    """One extent of any size that binds nothing."""


class AnyShape(_Token):
    """Any rank and any extents; it must be the only shape argument."""


class Scalar(_Token):
    """Rank zero; it must be the only shape argument."""


class Broadcast(Generic[_SizeVar]):
    """An extent that is either one or the extent bound to the wrapped dimension."""

    def __new__(cls, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("Broadcast is a shape marker and cannot be instantiated.")


Backend: TypeAlias = Literal["jax", "host"]


@dataclass(frozen=True, slots=True)
class TensorForm:
    """Registered contract of one tensor alias; `dtype=None` accepts any dtype."""

    name: str
    backend: Backend
    dtype: DTypeRule | None


Bool = TypeAliasType("Bool", jax.Array, type_params=(_Shape,))
Int32 = TypeAliasType("Int32", jax.Array, type_params=(_Shape,))
Int64 = TypeAliasType("Int64", jax.Array, type_params=(_Shape,))
UInt32 = TypeAliasType("UInt32", jax.Array, type_params=(_Shape,))
Float32 = TypeAliasType("Float32", jax.Array, type_params=(_Shape,))
Float64 = TypeAliasType("Float64", jax.Array, type_params=(_Shape,))
Complex64 = TypeAliasType("Complex64", jax.Array, type_params=(_Shape,))
Complex128 = TypeAliasType("Complex128", jax.Array, type_params=(_Shape,))
Integer = TypeAliasType("Integer", jax.Array, type_params=(_Shape,))
Float = TypeAliasType("Float", jax.Array, type_params=(_Shape,))
Complex = TypeAliasType("Complex", jax.Array, type_params=(_Shape,))
Inexact = TypeAliasType("Inexact", jax.Array, type_params=(_Shape,))
Shaped = TypeAliasType("Shaped", jax.Array, type_params=(_Shape,))

HostBool = TypeAliasType("HostBool", npt.NDArray[np.bool_], type_params=(_Shape,))
HostInt32 = TypeAliasType("HostInt32", npt.NDArray[np.int32], type_params=(_Shape,))
HostInt64 = TypeAliasType("HostInt64", npt.NDArray[np.int64], type_params=(_Shape,))
HostFloat32 = TypeAliasType("HostFloat32", npt.NDArray[np.float32], type_params=(_Shape,))
HostFloat64 = TypeAliasType("HostFloat64", npt.NDArray[np.float64], type_params=(_Shape,))
HostComplex128 = TypeAliasType(
    "HostComplex128", npt.NDArray[np.complex128], type_params=(_Shape,)
)
# Category host forms are statically broader than their runtime contract: NumPy's
# generic scalar hierarchy is the most precise static description available.
HostInteger = TypeAliasType(
    "HostInteger", npt.NDArray[np.integer[Any]], type_params=(_Shape,)
)
HostFloat = TypeAliasType(
    "HostFloat", npt.NDArray[np.floating[Any]], type_params=(_Shape,)
)
HostInexact = TypeAliasType(
    "HostInexact", npt.NDArray[np.inexact[Any]], type_params=(_Shape,)
)
HostShaped = TypeAliasType("HostShaped", npt.NDArray[Any], type_params=(_Shape,))

PRNGKey = TypeAliasType("PRNGKey", jax.Array)
"""A typed JAX scalar PRNG key (`jax.random.key`); legacy `uint32[2]` keys are not keys."""
Size = TypeAliasType("Size", int, type_params=(_SizeVar,))
Identifier = TypeAliasType("Identifier", str)
Identifiers = TypeAliasType("Identifiers", tuple[str, ...], type_params=(_SizeVar,))


class SupportsArray(Protocol):
    """Objects that export themselves through the NumPy array protocol."""

    def __array__(self) -> np.ndarray: ...


_NumericScalar: TypeAlias = bool | int | float | complex | np.generic
# Nested numeric sequences are accepted to a checker-enforceable depth of four;
# higher-rank host data must already be a NumPy or JAX array.
ConvertibleToArray: TypeAlias = (
    ArrayLike
    | SupportsArray
    | Sequence[_NumericScalar]
    | Sequence[Sequence[_NumericScalar]]
    | Sequence[Sequence[Sequence[_NumericScalar]]]
    | Sequence[Sequence[Sequence[Sequence[_NumericScalar]]]]
)
Like = TypeAliasType("Like", ConvertibleToArray, type_params=(_FormVar,))


def _tensor_forms() -> Mapping[TypeAliasType, TensorForm]:
    jax_forms: tuple[tuple[TypeAliasType, DTypeRule | None], ...] = (
        (Bool, category_dtype_rule("boolean")),
        (Int32, exact_dtype_rule(np.int32)),
        (Int64, exact_dtype_rule(np.int64)),
        (UInt32, exact_dtype_rule(np.uint32)),
        (Float32, exact_dtype_rule(np.float32)),
        (Float64, exact_dtype_rule(np.float64)),
        (Complex64, exact_dtype_rule(np.complex64)),
        (Complex128, exact_dtype_rule(np.complex128)),
        (Integer, category_dtype_rule("integer")),
        (Float, category_dtype_rule("floating")),
        (Complex, category_dtype_rule("complex")),
        (Inexact, category_dtype_rule("inexact")),
        (Shaped, None),
    )
    host_forms: tuple[tuple[TypeAliasType, DTypeRule | None], ...] = (
        (HostBool, category_dtype_rule("boolean")),
        (HostInt32, exact_dtype_rule(np.int32)),
        (HostInt64, exact_dtype_rule(np.int64)),
        (HostFloat32, exact_dtype_rule(np.float32)),
        (HostFloat64, exact_dtype_rule(np.float64)),
        (HostComplex128, exact_dtype_rule(np.complex128)),
        (HostInteger, category_dtype_rule("integer")),
        (HostFloat, category_dtype_rule("floating")),
        (HostInexact, category_dtype_rule("inexact")),
        (HostShaped, None),
    )
    forms: dict[TypeAliasType, TensorForm] = {}
    for backend, entries in (("jax", jax_forms), ("host", host_forms)):
        for alias, rule in entries:
            forms[alias] = TensorForm(alias.__name__, backend, rule)
    return MappingProxyType(forms)


TENSOR_FORMS: Mapping[TypeAliasType, TensorForm] = _tensor_forms()
"""Tensor alias registrations, keyed by alias identity."""
