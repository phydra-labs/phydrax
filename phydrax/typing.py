#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phydrax annotation vocabulary and explicit structural boundaries.

Static checkers see every form as its base type: JAX forms such as
``Float64[ComponentDim]`` are `jax.Array`, host forms such as
``HostFloat64[ComponentDim]`` are ``numpy.typing.NDArray[numpy.float64]``,
``Size[D]`` is `int`, ``Identifier`` is `str`, ``Identifiers[D]`` is
``tuple[str, ...]``, and ``PRNGKey`` is `jax.Array`. At runtime the same forms are a
closed structural contract language:

- tensor shapes are sequences of fixed extents (``Literal[3]``), nominal
  dimensions (`Dim` subclasses), anonymous extents (`AnyDim`), broadcastable
  extents (``Broadcast[D]``), and at most one variadic group (`VariadicDim`
  subclass); `Scalar` (rank zero) and `AnyShape` stand alone;
- a dimension binds once per `Scope` and must agree everywhere it appears;
- ``Size[D]`` binds `D` from an exact `int`; ``Identifiers[D]`` binds `D` from a
  tuple of unique canonical identifiers;
- ``Literal[...]`` and `Enum` subclasses are closed selectors;
- ``X | None``, unions of contract forms, and fixed tuples of contract forms
  compose.

Any other annotation is ordinary and static-only; Phydrax vocabulary in any other
placement is refused. Checks read only host metadata (kind, rank, extents, dtype,
static values): they never convert, synchronize, or add JAX operations. Wrong
kinds and dtypes raise `TypeError`; wrong ranks, extents, bindings, minimums, and
selector values raise `ValueError`. Numerical admissibility (finite, positive,
PSD, ...) is not part of this language and stays with its scientific owner.

`parse` validates a value against a form, `as_array`/`as_host_array` perform one
explicit conversion and then validate, and `validate` checks every contract field
of an opted-in module (``__strict_contract__ = True``). Opted-in modules are also
checked, read-only, when constructed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast, Literal, TypeAlias, TypeVar

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from typing_extensions import TypeForm

from ._dtype_names import dtype_matches
from ._strict import StrictModule
from ._typing_forms import (
    AnyDim,
    AnyShape,
    Bool,
    Broadcast,
    Complex,
    Complex64,
    Complex128,
    ConvertibleToArray,
    Dim,
    Float,
    Float32,
    Float64,
    HostBool,
    HostComplex128,
    HostFloat,
    HostFloat32,
    HostFloat64,
    HostInexact,
    HostInt32,
    HostInt64,
    HostInteger,
    HostShaped,
    Identifier,
    Identifiers,
    Inexact,
    Int32,
    Int64,
    Integer,
    Like,
    PRNGKey,
    Scalar,
    Shaped,
    Size,
    SupportsArray,
    UInt32,
    VariadicDim,
)
from ._typing_plan import (
    ArrayContract,
    check,
    parse_contract,
    raise_violation,
    Scope,
    validate_tree,
)


_T = TypeVar("_T")
Casting: TypeAlias = Literal["no", "equiv", "safe", "same_kind"]


def parse(
    value: object, form: TypeForm[_T], name: str, /, *, scope: Scope | None = None
) -> _T:
    """Validate `value` against one contract form and return it.

    The original object is returned unchanged, except that a Literal selector
    returns its declared literal when `value` merely compares equal to it (for
    example ``numpy.str_("dense")`` returns ``"dense"``).
    """
    contract = parse_contract(form)
    active_scope = Scope() if scope is None else scope
    mark = active_scope._mark()
    violation, result = check(contract, value, active_scope, name, canonicalize=True)
    if violation is not None:
        active_scope._rollback(mark)
        raise_violation(violation)
    # The structural check above establishes that `result` inhabits `form`.
    return cast(_T, result)


def _tensor_contract(form: object, backend: str, function: str, /) -> ArrayContract:
    contract = parse_contract(form)
    if not isinstance(contract, ArrayContract) or contract.backend != backend:
        raise TypeError(f"{function} requires a {backend} tensor form; got {form!r}.")
    return contract


def _host_values(value: object, name: str, /) -> np.ndarray:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{name}: {type(value).__name__} values are not numeric arrays.")
    values = np.asarray(value)
    if values.dtype.kind in "OUSV":
        raise TypeError(f"{name}: dtype {values.dtype} is not numeric.")
    return values


def _target_dtype(
    contract: ArrayContract, source: np.dtype, casting: Casting, name: str, /
) -> np.dtype:
    rule = contract.tensor.dtype
    if rule is None:
        return source
    if rule.exact is None:
        if not dtype_matches(rule, source):
            raise TypeError(f"{name}: dtype {source} is not {rule.label}.")
        return source
    if not np.can_cast(source, rule.exact, casting=casting):
        raise TypeError(
            f"{name}: cannot cast {source} to {rule.exact} under {casting!r} casting."
        )
    return rule.exact


def as_array(
    value: ConvertibleToArray,
    form: TypeForm[_T],
    name: str,
    /,
    *,
    scope: Scope | None = None,
    casting: Casting = "same_kind",
) -> _T:
    """Convert `value` once to a JAX array of `form` and validate it.

    JAX arrays stay on device (an exact form may cast them there); host values are
    normalized with `numpy.asarray` and transferred once. Category forms keep the
    input dtype when it belongs to the category and never promote across categories.
    """
    contract = _tensor_contract(form, "jax", "as_array")
    if isinstance(value, jax.Array):
        target = _target_dtype(contract, value.dtype, casting, name)
        array = value if value.dtype == target else value.astype(target)
    else:
        host = _host_values(value, name)
        array = jnp.asarray(
            host, dtype=_target_dtype(contract, host.dtype, casting, name)
        )
    return parse(array, form, name, scope=scope)


def as_host_array(
    value: ConvertibleToArray,
    form: TypeForm[_T],
    name: str,
    /,
    *,
    scope: Scope | None = None,
    casting: Casting = "same_kind",
) -> _T:
    """Convert `value` once to a NumPy array of `form` and validate it.

    Concrete JAX arrays are transferred to the host explicitly; tracers are refused.
    """
    contract = _tensor_contract(form, "host", "as_host_array")
    if isinstance(value, jax.core.Tracer):
        raise TypeError(f"{name}: traced values cannot be transferred to the host.")
    host = (
        np.asarray(jax.device_get(value))
        if isinstance(value, jax.Array)
        else _host_values(value, name)
    )
    target = _target_dtype(contract, host.dtype, casting, name)
    return parse(
        host if host.dtype == target else host.astype(target), form, name, scope=scope
    )


def validate(instance: object, /) -> None:
    """Check the structural contracts of one opted-in module and of its contents.

    `instance` must be a `StrictModule` that declares (or inherits)
    ``__strict_contract__ = True``. Its contract fields are checked in declaration
    order, then every opted-in module reachable through its fields, tuples, lists,
    and mappings. Construction does not make a module permanently valid:
    transformations such as `jax.tree_util.tree_map` or `equinox.tree_at` rebuild
    modules without running constructors, so consumers validate transformed values
    explicitly.
    """
    if not isinstance(instance, StrictModule) or not type(instance)._strict_contract_:
        raise TypeError(
            f"{type(instance).__qualname__} does not declare __strict_contract__ = True."
        )
    validate_tree(instance)


__all__ = [
    "AnyDim",
    "AnyShape",
    "Bool",
    "Broadcast",
    "Complex",
    "Complex64",
    "Complex128",
    "ConvertibleToArray",
    "Dim",
    "Float",
    "Float32",
    "Float64",
    "HostBool",
    "HostComplex128",
    "HostFloat",
    "HostFloat32",
    "HostFloat64",
    "HostInexact",
    "HostInt32",
    "HostInt64",
    "HostInteger",
    "HostShaped",
    "Identifier",
    "Identifiers",
    "Inexact",
    "Int32",
    "Int64",
    "Integer",
    "Like",
    "PRNGKey",
    "Scalar",
    "Scope",
    "Shaped",
    "Size",
    "SupportsArray",
    "UInt32",
    "VariadicDim",
    "as_array",
    "as_host_array",
    "parse",
    "validate",
]
