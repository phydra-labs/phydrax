#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compiled programs owned by one live object and released with it."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from typing import Any, TypeVar

import equinox as eqx
import jax

from ._strict import StrictModule


_T = TypeVar("_T")

_PROGRAMS: dict[tuple[int, str], Callable[..., Any]] = {}
_PROGRAMS_LOCK = threading.Lock()


class _StaticOutput(StrictModule):
    value: Any = eqx.field(static=True)


def run_owned_program(
    owner: Any, phase: str, method: Callable[..., _T], /, *operands: Any
) -> _T:
    """Run ``method(owner, *operands)`` as one compiled program per owner and phase.

    Executed op by op, every control-flow primitive (conditionals, loops,
    implicit root solves) traces fresh closures and so compiles and caches a
    new executable on each call. The owned program is traced once per owner
    and reused by later calls; inside an enclosing trace ``method`` runs
    inline. The owner alone holds the jitted function (no global wrapper
    cache keeps its static arguments alive), so its executables are released
    together with the owner.
    """
    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves((owner, operands))
    ):
        return method(owner, *operands)
    key = (id(owner), phase)
    with _PROGRAMS_LOCK:
        program = _PROGRAMS.get(key)
        if program is None:

            def run(dynamic: Any, static: Any, /) -> Any:
                bound, values = eqx.combine(
                    dynamic, jax.tree.unflatten(static[1], static[0])
                )
                output = method(bound, *values)
                output_dynamic, output_static = eqx.partition(output, eqx.is_array)
                return output_dynamic, _StaticOutput(output_static)

            program = jax.jit(run, static_argnums=1)
            _PROGRAMS[key] = program
            weakref.finalize(owner, _PROGRAMS.pop, key, None)
    dynamic, static = eqx.partition((owner, operands), eqx.is_array)
    static_leaves, static_structure = jax.tree.flatten(static)
    try:
        output_dynamic, output_static = program(
            dynamic, (tuple(static_leaves), static_structure)
        )
        jax.block_until_ready(output_dynamic)
    except jax.errors.JaxRuntimeError as error:
        # A failed ``eqx.error_if`` check surfaces as a runtime callback
        # failure; report it as the checked error, as ``eqx.filter_jit`` does.
        text = str(error)
        marker = "_EquinoxRuntimeError: "
        if marker not in text:
            raise
        message = text.rsplit(marker, 1)[1].strip().splitlines()[0]
        raise eqx.EquinoxRuntimeError(message) from None
    return eqx.combine(output_dynamic, output_static.value)


__all__ = ["run_owned_program"]
