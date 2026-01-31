#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import inspect
from collections.abc import Callable

from ._strict import StrictModule


class _KeyIterAdapter(StrictModule):
    func: Callable
    has_var_kwargs: bool
    has_key: bool
    has_iter: bool

    def __init__(
        self,
        func: Callable,
        /,
        *,
        has_var_kwargs: bool,
        has_key: bool,
        has_iter: bool,
    ):
        self.func = func
        self.has_var_kwargs = bool(has_var_kwargs)
        self.has_key = bool(has_key)
        self.has_iter = bool(has_iter)

    def __call__(self, *args, key=None, iter_=None, **kwargs):
        out_kwargs = kwargs
        if key is not None and (self.has_key or self.has_var_kwargs):
            out_kwargs = dict(out_kwargs)
            out_kwargs["key"] = key
        if iter_ is not None and (self.has_iter or self.has_var_kwargs):
            if out_kwargs is kwargs:
                out_kwargs = dict(out_kwargs)
            out_kwargs["iter_"] = iter_
        return self.func(*args, **out_kwargs)


def _ensure_special_kwonly_args(func: Callable, /) -> Callable:
    """Ensure that `func` can accept special keyword-only arguments.

    This adapter is used to normalize call signatures across user callables so that
    we can always pass `key=` and/or `iter_=` without relying on try/except
    or hasattr/getattr probing.
    """
    is_bound_method = inspect.ismethod(func) and func.__self__ is not None
    sig = inspect.signature(func)
    params = sig.parameters
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    has_key = "key" in params
    has_iter = "iter_" in params

    if has_key and params["key"].kind != inspect.Parameter.KEYWORD_ONLY:
        raise TypeError("`key` must be a keyword-only argument for `func`.")
    if has_iter and params["iter_"].kind != inspect.Parameter.KEYWORD_ONLY:
        raise TypeError("`iter_` must be a keyword-only argument for `func`.")

    if is_bound_method:

        def _wrapped(*args, key=None, iter_=None, **kwargs):
            out_kwargs = kwargs
            if key is not None and (has_key or has_var_kwargs):
                out_kwargs = dict(out_kwargs)
                out_kwargs["key"] = key
            if iter_ is not None and (has_iter or has_var_kwargs):
                if out_kwargs is kwargs:
                    out_kwargs = dict(out_kwargs)
                out_kwargs["iter_"] = iter_
            return func(*args, **out_kwargs)

        return _wrapped

    if has_key and has_iter:
        return func

    return _KeyIterAdapter(
        func,
        has_var_kwargs=has_var_kwargs,
        has_key=has_key,
        has_iter=has_iter,
    )
