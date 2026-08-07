#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from ..._callable import _ensure_special_kwonly_args
from ..._doc import DOC_KEY0


PotentialCallable: TypeAlias = Callable[[Array, Array], ArrayLike]
PotentialLike: TypeAlias = PotentialCallable | DomainFunction


def _as_point_time_callable(
    function: PotentialLike,
    /,
    *,
    position_var: str,
    time_var: str,
    key: Key[Array, ""] = DOC_KEY0,
    role: str = "Function",
) -> PotentialCallable:
    if isinstance(function, DomainFunction):
        allowed = {position_var, time_var}
        unknown = tuple(dep for dep in function.deps if dep not in allowed)
        if unknown:
            raise ValueError(
                f"{role} dependencies must be position/time labels only; "
                f"got unsupported dependencies {unknown!r}."
            )

        positions = tuple(
            position_var if dep == position_var else time_var for dep in function.deps
        )

        def call_domain_function(q: Array, t: Array) -> ArrayLike:
            values = {position_var: q, time_var: t}
            args = tuple(values[label] for label in positions)
            return function.func(*args, key=key)

        return call_domain_function

    if not callable(function):
        raise TypeError(f"{role.lower()} must be callable or a DomainFunction.")
    adapted = _ensure_special_kwonly_args(function)

    def call_function(q: Array, t: Array) -> ArrayLike:
        return adapted(q, t, key=key)

    return call_function


def _as_potential_callable(
    potential: PotentialLike,
    /,
    *,
    position_var: str,
    time_var: str,
    key: Key[Array, ""] = DOC_KEY0,
) -> PotentialCallable:
    return _as_point_time_callable(
        potential,
        position_var=position_var,
        time_var=time_var,
        key=key,
        role="Potential",
    )


__all__ = ["PotentialCallable", "PotentialLike"]
