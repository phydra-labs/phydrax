#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import abc
import inspect
from collections.abc import Callable, Mapping
from typing import Any, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp

from .._strict import StrictModule


if TYPE_CHECKING:
    pass



class _AbstractDomain(StrictModule):
    @property
    @abc.abstractmethod
    def labels(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def factor(self, label: str, /) -> "_AbstractUnaryDomain":
        raise NotImplementedError

    @abc.abstractmethod
    def equivalent(self, other: object, /) -> bool:
        raise NotImplementedError

    def join(self, other: "_AbstractDomain", /) -> "_AbstractDomain":
        from ._product_domain import ProductDomain

        if self is other:
            return self
        return ProductDomain(self, other)

    def __matmul__(self, other: "_AbstractDomain", /) -> "_AbstractDomain":
        return self.join(other)

    def restrict(self, labels: tuple[str, ...], /) -> "_AbstractDomain":
        if set(labels) == set(self.labels):
            return self
        if len(labels) == 1:
            return self.factor(labels[0])
        from ._product_domain import ProductDomain

        factors = tuple(self.factor(lbl) for lbl in labels)
        return ProductDomain(*factors)

    def drop(self, labels: str | tuple[str, ...], /) -> "_AbstractDomain":
        if isinstance(labels, str):
            drop_set = {labels}
        else:
            drop_set = set(labels)
        kept = tuple(lbl for lbl in self.labels if lbl not in drop_set)
        if not kept:
            raise ValueError("Cannot drop all labels from a domain.")
        return self.restrict(kept)

    def component(
        self,
        spec: Any = None,
        *,
        where: Mapping[str, Callable] | None = None,
        where_all: Any = None,
        weight_all: Any = None,
    ):
        from ._components import ComponentSpec, DomainComponent

        spec_ = spec if isinstance(spec, ComponentSpec) else ComponentSpec(spec)
        return DomainComponent(
            domain=self,
            spec=spec_,
            where=where,
            where_all=where_all,
            weight_all=weight_all,
        )

    def Function(self, *deps: str):
        from ._function import batch_aware_callable, DomainFunction

        if deps:
            for dep in deps:
                if dep not in self.labels:
                    raise ValueError(
                        f"Unknown dependency label {dep!r}; expected subset of {self.labels}."
                    )

        def decorator(func):
            if not callable(func):
                return DomainFunction(domain=self, deps=deps, func=func)
            if batch_aware_callable(func) is not None:
                return DomainFunction(domain=self, deps=deps, func=func)

            sig = inspect.signature(func)
            params = sig.parameters
            has_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            has_key = "key" in params
            has_iter = "iter_" in params
            accepted_kw_names = {
                name
                for name, p in params.items()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }

            if has_key and params["key"].kind != inspect.Parameter.KEYWORD_ONLY:
                raise TypeError("`key` must be a keyword-only argument for `func`.")
            if has_iter and params["iter_"].kind != inspect.Parameter.KEYWORD_ONLY:
                raise TypeError("`iter_` must be a keyword-only argument for `func`.")

            def _wrap_coord_separable(*args, key=None, iter_=None, **kwargs):
                out_kwargs = kwargs
                if key is not None and (has_key or has_var_kwargs):
                    out_kwargs = dict(out_kwargs)
                    out_kwargs["key"] = key
                if iter_ is not None and (has_iter or has_var_kwargs):
                    if out_kwargs is kwargs:
                        out_kwargs = dict(out_kwargs)
                    out_kwargs["iter_"] = iter_

                if not has_var_kwargs and out_kwargs:
                    # Differential runtime knobs may be threaded through
                    # composite residual expressions. Drop those for plain
                    # user functions that do not declare **kwargs.
                    out_kwargs = {
                        k: v
                        for k, v in out_kwargs.items()
                        if ((k in accepted_kw_names) or (k != "force_generic"))
                    }

                coord_indices = [
                    i for i, arg in enumerate(args) if isinstance(arg, tuple)
                ]
                if not coord_indices:
                    return func(*args, **out_kwargs)

                coord_values = tuple(
                    jnp.asarray(coord).reshape((-1,))
                    for i in coord_indices
                    for coord in args[i]
                )
                if not coord_values:
                    return func(*args, **out_kwargs)

                def call_point(*values):
                    new_args = list(args)
                    offset = 0
                    for i in coord_indices:
                        count = len(args[i])
                        new_args[i] = jnp.stack(values[offset : offset + count])
                        offset += count
                    return func(*new_args, **out_kwargs)

                mapped = call_point
                for position in reversed(range(len(coord_values))):
                    in_axes = tuple(
                        0 if index == position else None
                        for index in range(len(coord_values))
                    )
                    mapped = jax.vmap(mapped, in_axes=in_axes, out_axes=0)
                return mapped(*coord_values)

            return DomainFunction(domain=self, deps=deps, func=_wrap_coord_separable)

        return decorator

    def Model(
        self,
        *deps: str,
        structured: bool = False,
        input_mode: Literal["flat", "structured"] | None = None,
    ):
        """Wrap a neural model as a `DomainFunction`.

        By default each model chooses its own domain input mode. Dense and separable
        pointwise models use flat concatenation, while operator models may request
        structured tuple inputs. `structured=True` is a compatibility alias for
        `input_mode="structured"`.
        """
        from ._function import DomainFunction
        from ._model_function import _ConcatenatedModelCallable, StructuredCallable

        if structured and input_mode is not None:
            raise ValueError("Use either structured=True or input_mode=..., not both.")
        mode: Literal["flat", "structured"] | None = input_mode
        if structured:
            mode = "structured"
        if mode is not None and mode not in ("flat", "structured"):
            raise ValueError("input_mode must be either 'flat' or 'structured'.")

        deps_ = self.labels if not deps else deps
        for dep in deps_:
            if dep not in self.labels:
                raise ValueError(
                    f"Unknown dependency label {dep!r}; expected subset of {self.labels}."
                )

        def decorator(model):
            if structured:
                model = StructuredCallable(model)
            return DomainFunction(
                domain=self,
                deps=deps_,
                func=_ConcatenatedModelCallable(model, input_mode=mode),
            )

        return decorator

    def Parameter(
        self,
        init: Any,
        *,
        transform: Callable[[Any], Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        from ._function import _TrainableConstCallable, _UnaryCallable, DomainFunction

        if init is None:
            raise TypeError("Domain.Parameter requires init to be array-like, not None.")

        raw = jnp.asarray(init)
        if not jnp.issubdtype(raw.dtype, jnp.inexact):
            raw = raw.astype(float)

        if transform is None:
            return DomainFunction(
                domain=self,
                deps=(),
                func=_TrainableConstCallable(raw),
                metadata=metadata,
            )

        if not callable(transform):
            raise TypeError("Domain.Parameter transform must be a callable or None.")

        return DomainFunction(
            domain=self,
            deps=(),
            func=_UnaryCallable(_TrainableConstCallable(raw), transform),
            metadata=metadata,
        )


class _AbstractUnaryDomain(_AbstractDomain):
    @property
    @abc.abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @property
    def labels(self) -> tuple[str, ...]:
        return (self.label,)

    @property
    @abc.abstractmethod
    def var_dim(self) -> int:
        raise NotImplementedError

    def factor(self, label: str, /) -> "_AbstractUnaryDomain":
        if label != self.label:
            raise KeyError(f"Label {label!r} not in domain {self.labels}.")
        return self

    def relabel(self, label: str, /) -> "RelabeledDomain":
        return RelabeledDomain(self, label)


class RelabeledDomain(_AbstractUnaryDomain):
    base: _AbstractUnaryDomain
    _label: str

    def __init__(self, base: _AbstractUnaryDomain, label: str):
        self.base = base
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @property
    def var_dim(self) -> int:
        return self.base.var_dim

    def equivalent(self, other: object, /) -> bool:
        if isinstance(other, RelabeledDomain):
            return self.base.equivalent(other.base)
        return self.base.equivalent(other)
