#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import comb
from typing import Any, Literal, TypeAlias

import jax.numpy as jnp

from ..._strict import StrictModule
from ...domain._function import DomainFunction


DERIVATIVE_HOOK_KEY = "_optimized_derivative_hook"


DerivativeHook: TypeAlias = Callable[
    ...,
    DomainFunction | None,
]

DerivativeMode: TypeAlias = Literal["reverse", "forward"]
DerivativeBackend: TypeAlias = Literal["ad", "jet", "fd", "basis"]
DerivativeBasis: TypeAlias = Literal["poly", "fourier", "sine", "cosine"]


def get_derivative_hook(u: DomainFunction, /) -> DerivativeHook | None:
    hook = u.metadata.get(DERIVATIVE_HOOK_KEY)
    if hook is None:
        return None
    if not callable(hook):
        return None
    return hook


def with_derivative_hook(
    u: DomainFunction,
    hook: DerivativeHook,
    /,
) -> DomainFunction:
    return u.with_metadata(**{DERIVATIVE_HOOK_KEY: hook})


class _BlendWithGateCallable(StrictModule):
    base: DomainFunction
    overlay: DomainFunction
    gate: DomainFunction
    base_pos: tuple[int, ...]
    overlay_pos: tuple[int, ...]
    gate_pos: tuple[int, ...]

    def __init__(
        self,
        *,
        base: DomainFunction,
        overlay: DomainFunction,
        gate: DomainFunction,
        base_pos: tuple[int, ...],
        overlay_pos: tuple[int, ...],
        gate_pos: tuple[int, ...],
    ):
        self.base = base
        self.overlay = overlay
        self.gate = gate
        self.base_pos = tuple(int(i) for i in base_pos)
        self.overlay_pos = tuple(int(i) for i in overlay_pos)
        self.gate_pos = tuple(int(i) for i in gate_pos)

    def __call__(self, *args, key=None, **kwargs):
        def _align_axiswise(a: Any, b: Any, /) -> tuple[Any, Any]:
            a_arr = jnp.asarray(a)
            b_arr = jnp.asarray(b)
            if (
                a_arr.ndim == 1
                and b_arr.ndim >= 2
                and int(a_arr.shape[0]) == int(b_arr.shape[0])
            ):
                shape = (int(a_arr.shape[0]),) + (1,) * (b_arr.ndim - 1)
                return a_arr.reshape(shape), b_arr
            if (
                b_arr.ndim == 1
                and a_arr.ndim >= 2
                and int(b_arr.shape[0]) == int(a_arr.shape[0])
            ):
                shape = (int(b_arr.shape[0]),) + (1,) * (a_arr.ndim - 1)
                return a_arr, b_arr.reshape(shape)
            if (
                a_arr.ndim == 1
                and b_arr.ndim >= 2
                and int(a_arr.shape[0]) == int(b_arr.shape[-1])
            ):
                shape = (1,) * (b_arr.ndim - 1) + (int(a_arr.shape[0]),)
                return a_arr.reshape(shape), b_arr
            if (
                b_arr.ndim == 1
                and a_arr.ndim >= 2
                and int(b_arr.shape[0]) == int(a_arr.shape[-1])
            ):
                shape = (1,) * (a_arr.ndim - 1) + (int(b_arr.shape[0]),)
                return a_arr, b_arr.reshape(shape)
            return a_arr, b_arr

        def _mul_aligned(a: Any, b: Any, /):
            a_arr = jnp.asarray(a)
            b_arr = jnp.asarray(b)
            try:
                return a_arr * b_arr
            except (TypeError, ValueError):
                a_fix, b_fix = _align_axiswise(a_arr, b_arr)
                return a_fix * b_fix

        def _add_aligned(a: Any, b: Any, /):
            a_arr = jnp.asarray(a)
            b_arr = jnp.asarray(b)
            try:
                return a_arr + b_arr
            except (TypeError, ValueError):
                a_fix, b_fix = _align_axiswise(a_arr, b_arr)
                return a_fix + b_fix

        def _sub_aligned(a: Any, b: Any, /):
            a_arr = jnp.asarray(a)
            b_arr = jnp.asarray(b)
            try:
                return a_arr - b_arr
            except (TypeError, ValueError):
                a_fix, b_fix = _align_axiswise(a_arr, b_arr)
                return a_fix - b_fix

        base_args = tuple(args[i] for i in self.base_pos)
        overlay_args = tuple(args[i] for i in self.overlay_pos)
        gate_args = tuple(args[i] for i in self.gate_pos)
        base_val = self.base.func(*base_args, key=key, **kwargs)
        overlay_val = self.overlay.func(*overlay_args, key=key, **kwargs)
        gate_val = self.gate.func(*gate_args, key=key, **kwargs)
        delta = _sub_aligned(overlay_val, base_val)
        gated_delta = _mul_aligned(gate_val, delta)
        return _add_aligned(base_val, gated_delta)


def blend_with_gate(
    base: DomainFunction,
    overlay: DomainFunction,
    gate: DomainFunction,
    /,
) -> DomainFunction:
    r"""Blend `base` toward `overlay` using gate `g`: ``base + g * (overlay - base)``.

    The returned function carries an optimized derivative hook:

    ``d^n(base) + d^n(g * (overlay - base))``

    which preserves fast derivative paths of the constituent functions.
    """
    if base.domain.labels == overlay.domain.labels:
        joined = base.domain
    else:
        joined = base.domain.join(overlay.domain)
    if joined.labels != gate.domain.labels:
        joined = joined.join(gate.domain)

    base_p = base.promote(joined)
    overlay_p = overlay.promote(joined)
    gate_p = gate.promote(joined)

    delta = overlay_p - base_p
    blended_expr = base_p + gate_p * delta

    deps = tuple(
        lbl
        for lbl in joined.labels
        if (lbl in base_p.deps) or (lbl in overlay_p.deps) or (lbl in gate_p.deps)
    )
    dep_index = {lbl: i for i, lbl in enumerate(deps)}
    base_pos = tuple(dep_index[lbl] for lbl in base_p.deps)
    overlay_pos = tuple(dep_index[lbl] for lbl in overlay_p.deps)
    gate_pos = tuple(dep_index[lbl] for lbl in gate_p.deps)

    blended = DomainFunction(
        domain=joined,
        deps=deps,
        func=_BlendWithGateCallable(
            base=base_p,
            overlay=overlay_p,
            gate=gate_p,
            base_pos=base_pos,
            overlay_pos=overlay_pos,
            gate_pos=gate_pos,
        ),
        metadata=blended_expr.metadata,
    )

    def _hook(
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        if backend not in ("ad", "jet"):
            return None

        def _derive(fn: DomainFunction, k: int, /) -> DomainFunction:
            from ._domain_ops import partial_n

            return partial_n(
                fn,
                var=var,
                axis=axis,
                order=int(k),
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            )

        n = int(order)
        return _derive(base_p, n) + nth_product_rule(
            gate_p,
            delta,
            var=var,
            order=n,
            derive=_derive,
        )

    return with_derivative_hook(blended, _hook)


def nth_product_rule(
    left: DomainFunction,
    right: DomainFunction,
    /,
    *,
    var: str | None = None,
    order: int,
    derive: Callable[[DomainFunction, int], DomainFunction],
) -> DomainFunction:
    if int(order) < 0:
        raise ValueError("order must be non-negative.")
    n = int(order)
    if n == 0:
        return left * right
    if var is not None:
        left_dep = left.depends_on(var)
        right_dep = right.depends_on(var)
        if not left_dep and not right_dep:
            return 0.0 * (left * right)
        if not left_dep:
            return left * derive(right, n)
        if not right_dep:
            return derive(left, n) * right

    left_cache: dict[int, DomainFunction] = {}
    right_cache: dict[int, DomainFunction] = {}

    def _d_left(k: int, /) -> DomainFunction:
        kk = int(k)
        cached = left_cache.get(kk)
        if cached is not None:
            return cached
        out_k = derive(left, kk)
        left_cache[kk] = out_k
        return out_k

    def _d_right(k: int, /) -> DomainFunction:
        kk = int(k)
        cached = right_cache.get(kk)
        if cached is not None:
            return cached
        out_k = derive(right, kk)
        right_cache[kk] = out_k
        return out_k

    out: DomainFunction | None = None
    for k in range(n + 1):
        term = float(comb(n, k)) * _d_left(k) * _d_right(n - k)
        out = term if out is None else (out + term)
    assert out is not None
    return out


def nth_quotient_rule(
    numerator: DomainFunction,
    denominator: DomainFunction,
    /,
    *,
    var: str | None = None,
    order: int,
    derive: Callable[[DomainFunction, int], DomainFunction],
) -> DomainFunction:
    n = int(order)
    if n < 0:
        raise ValueError("order must be non-negative.")
    if n == 0:
        return numerator / denominator
    if var is not None:
        num_dep = numerator.depends_on(var)
        den_dep = denominator.depends_on(var)
        if not num_dep and not den_dep:
            return 0.0 * (numerator / denominator)
        if not den_dep:
            return derive(numerator, n) / denominator

    numerator_cache: dict[int, DomainFunction] = {}
    denominator_cache: dict[int, DomainFunction] = {}

    def _d_num(k: int, /) -> DomainFunction:
        kk = int(k)
        cached = numerator_cache.get(kk)
        if cached is not None:
            return cached
        out_k = derive(numerator, kk)
        numerator_cache[kk] = out_k
        return out_k

    def _d_den(k: int, /) -> DomainFunction:
        kk = int(k)
        cached = denominator_cache.get(kk)
        if cached is not None:
            return cached
        out_k = derive(denominator, kk)
        denominator_cache[kk] = out_k
        return out_k

    # r = 1/denominator. Recurrence:
    # r_n = -(1/den) * sum_{k=1..n} C(n,k) den_k * r_{n-k}
    inv_den = 1.0 / denominator
    reciprocal_terms: list[DomainFunction] = [inv_den]
    for m in range(1, n + 1):
        acc: DomainFunction | None = None
        for k in range(1, m + 1):
            term = float(comb(m, k)) * _d_den(k) * reciprocal_terms[m - k]
            acc = term if acc is None else (acc + term)
        assert acc is not None
        reciprocal_terms.append((-1.0) * inv_den * acc)

    if var is not None and not numerator.depends_on(var):
        return numerator * reciprocal_terms[n]

    out: DomainFunction | None = None
    for k in range(n + 1):
        term = float(comb(n, k)) * _d_num(k) * reciprocal_terms[n - k]
        out = term if out is None else (out + term)
    assert out is not None
    return out
