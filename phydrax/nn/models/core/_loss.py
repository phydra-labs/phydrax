#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._callable import _ensure_special_kwonly_args
from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._base import _AbstractBaseModel, DomainInputMode, EvalKey


class ModelLossTerm(StrictModule, NonTrainableState):
    """A fixed scalar penalty attached to a model."""

    penalty: Callable
    weight: Array
    label: str | None

    def __init__(
        self,
        penalty: Callable[..., Any],
        /,
        *,
        weight: Any = 1.0,
        label: str | None = None,
    ):
        if not callable(penalty):
            raise TypeError("Model loss penalty must be callable.")
        self.penalty = _ensure_special_kwonly_args(penalty)
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)

    def __call__(
        self,
        model: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
    ) -> Array:
        value = self.penalty(model, key=key, iter_=iter_)
        return self.weight * jnp.asarray(value, dtype=float).reshape(())


class ModelWithLoss(StrictModule):
    """Callable model wrapper carrying additional scalar objective terms."""

    model: Any
    loss_terms: tuple[ModelLossTerm, ...]
    loss_identity: int
    in_size: Any
    out_size: Any
    _call_has_var_kwargs: bool
    _call_has_key: bool
    _call_has_iter: bool

    def __init__(
        self,
        model: Any,
        /,
        *,
        loss_terms: Sequence[ModelLossTerm] = (),
        loss_identity: int | None = None,
    ):
        if not callable(model):
            raise TypeError("ModelWithLoss requires a callable model.")
        terms = tuple(loss_terms)
        bad = tuple(t for t in terms if not isinstance(t, ModelLossTerm))
        if bad:
            raise TypeError(
                "loss_terms must contain ModelLossTerm instances; got "
                f"{tuple(type(t).__name__ for t in bad)!r}."
            )
        self.model = model
        self.loss_terms = terms
        self.loss_identity = int(id(terms) if loss_identity is None else loss_identity)
        self.in_size = getattr(model, "in_size", None)
        self.out_size = getattr(model, "out_size", None)
        sig = inspect.signature(model)
        params = sig.parameters
        self._call_has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        self._call_has_key = "key" in params
        self._call_has_iter = "iter_" in params
        if self._call_has_key and params["key"].kind != inspect.Parameter.KEYWORD_ONLY:
            raise TypeError("`key` must be a keyword-only argument for model calls.")
        if self._call_has_iter and params["iter_"].kind != inspect.Parameter.KEYWORD_ONLY:
            raise TypeError("`iter_` must be a keyword-only argument for model calls.")

    def __call__(
        self,
        x: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
        **kwargs: Any,
    ) -> Array:
        out_kwargs = kwargs
        if self._call_has_key or self._call_has_var_kwargs:
            out_kwargs = dict(out_kwargs)
            out_kwargs["key"] = key
        if iter_ is not None and (self._call_has_iter or self._call_has_var_kwargs):
            if out_kwargs is kwargs:
                out_kwargs = dict(out_kwargs)
            out_kwargs["iter_"] = iter_
        return self.model(x, **out_kwargs)

    def add_model_loss(
        self,
        penalty: Callable[..., Any],
        /,
        *,
        weight: Any = 1.0,
        label: str | None = None,
    ) -> "ModelWithLoss":
        term = ModelLossTerm(penalty, weight=weight, label=label)
        return ModelWithLoss(self.model, loss_terms=self.loss_terms + (term,))

    def supports_structured_input(self) -> bool:
        return _supports_structured_input(self.model)

    def supports_blockwise_input(self) -> bool:
        return _supports_blockwise_input(self.model)

    def supports_axis_batch_input(self) -> bool:
        return _supports_axis_batch_input(self.model)

    def __call_axis_batch__(
        self,
        batch: Any,
        deps: tuple[str, ...],
        /,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(self.model, _AbstractBaseModel):
            return self.model.__call_axis_batch__(
                batch, deps, key=key, iter_=iter_, **kwargs
            )
        raise TypeError("Wrapped model does not support axis-batch execution.")

    def warn_on_auto_fallback(self) -> bool:
        method = getattr(self.model, "warn_on_auto_fallback", None)
        if callable(method):
            return bool(method())
        return bool(getattr(self.model, "_warn_on_auto_fallback", False))

    def domain_input_mode(self) -> DomainInputMode:
        return _domain_input_mode(self.model)


def add_model_loss(
    model: Any,
    penalty: Callable[..., Any],
    /,
    *,
    weight: Any = 1.0,
    label: str | None = None,
) -> ModelWithLoss:
    """Return `model` wrapped with an additional scalar objective term.

    `penalty` is called as `penalty(model, key=..., iter_=...)` and must return a
    scalar JAX-compatible value.
    """
    term = ModelLossTerm(penalty, weight=weight, label=label)
    if isinstance(model, ModelWithLoss):
        return ModelWithLoss(model.model, loss_terms=model.loss_terms + (term,))
    return ModelWithLoss(model, loss_terms=(term,))


def _supports_structured_input(model: Any, /) -> bool:
    method = getattr(model, "supports_structured_input", None)
    if callable(method):
        return bool(method())
    return bool(getattr(model, "_supports_structured_input", False))


def _supports_blockwise_input(model: Any, /) -> bool:
    method = getattr(model, "supports_blockwise_input", None)
    if callable(method):
        return bool(method())
    return bool(getattr(model, "_supports_blockwise_input", False))


def _supports_axis_batch_input(model: Any, /) -> bool:
    if isinstance(model, ModelWithLoss):
        return model.supports_axis_batch_input()
    if isinstance(model, _AbstractBaseModel):
        return model.supports_axis_batch_input()
    return False


def _domain_input_mode(model: Any, /) -> DomainInputMode:
    method = getattr(model, "domain_input_mode", None)
    if callable(method):
        mode = method()
    else:
        mode = getattr(model, "_domain_input_mode", "flat")
    if mode not in ("flat", "structured"):
        raise ValueError("Model domain input mode must be either 'flat' or 'structured'.")
    return mode


def model_domain_metadata(
    model: Any,
    /,
) -> tuple[DomainInputMode, bool, bool, bool, bool] | None:
    """Return domain-call metadata for Phydrax model-like objects."""
    if isinstance(model, (_AbstractBaseModel, ModelWithLoss)):
        return (
            _domain_input_mode(model),
            _supports_structured_input(model),
            _supports_blockwise_input(model),
            _supports_axis_batch_input(model),
            bool(
                model.warn_on_auto_fallback()
                if callable(getattr(model, "warn_on_auto_fallback", None))
                else getattr(model, "_warn_on_auto_fallback", False)
            ),
        )
    return None


def model_loss_labels(model: Any, /) -> tuple[str, ...]:
    """Return static labels for all objective terms attached to `model`."""
    return _model_loss_labels(model, seen=set())


def model_loss_values(
    model: Any,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    iter_: Array | None = None,
) -> tuple[Array, ...]:
    """Evaluate all scalar objective terms attached to `model`."""
    counter = [0]
    return _model_loss_values(model, key=key, iter_=iter_, seen=set(), counter=counter)


def _model_loss_labels(model: Any, /, *, seen: set[int]) -> tuple[str, ...]:
    obj_id = _model_loss_seen_key(model)
    if obj_id in seen:
        return ()
    seen.add(obj_id)

    if isinstance(model, ModelWithLoss):
        labels = list(_model_loss_labels(model.model, seen=seen))
        for i, term in enumerate(model.loss_terms):
            labels.append(
                term.label
                if term.label is not None
                else f"{type(model.model).__name__}.model_loss_{i}"
            )
        return tuple(labels)

    labels: list[str] = []
    if _has_custom_model_loss(model):
        labels.append(f"{type(model).__name__}.__loss__")

    for child in _iter_children(model):
        labels.extend(_model_loss_labels(child, seen=seen))
    return tuple(labels)


def _model_loss_values(
    model: Any,
    /,
    *,
    key: Key[Array, ""],
    iter_: Array | None,
    seen: set[int],
    counter: list[int],
) -> tuple[Array, ...]:
    obj_id = _model_loss_seen_key(model)
    if obj_id in seen:
        return ()
    seen.add(obj_id)

    if isinstance(model, ModelWithLoss):
        values = list(
            _model_loss_values(
                model.model,
                key=key,
                iter_=iter_,
                seen=seen,
                counter=counter,
            )
        )
        for term in model.loss_terms:
            term_key = jr.fold_in(key, counter[0])
            counter[0] += 1
            values.append(jnp.asarray(term(model.model, key=term_key, iter_=iter_)))
        return tuple(values)

    values: list[Array] = []
    if _has_custom_model_loss(model):
        loss_fn = _ensure_special_kwonly_args(getattr(model, "__loss__"))
        term_key = jr.fold_in(key, counter[0])
        counter[0] += 1
        value = loss_fn(key=term_key, iter_=iter_)
        values.append(jnp.asarray(value, dtype=float).reshape(()))

    for child in _iter_children(model):
        values.extend(
            _model_loss_values(
                child,
                key=key,
                iter_=iter_,
                seen=seen,
                counter=counter,
            )
        )
    return tuple(values)


def _has_custom_model_loss(model: Any, /) -> bool:
    if isinstance(model, ModelWithLoss):
        return False
    loss_attr = getattr(type(model), "__loss__", None)
    if loss_attr is None:
        return callable(getattr(model, "__loss__", None))
    return loss_attr is not _AbstractBaseModel.__loss__


def _model_loss_seen_key(model: Any, /) -> int:
    if isinstance(model, ModelWithLoss):
        return model.loss_identity
    return id(model)


def _iter_children(value: Any, /):
    if value is None or isinstance(value, (str, bytes, int, float, complex, bool)):
        return
    if isinstance(value, ModelWithLoss):
        return
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            if field.name == "_strict_initialized":
                continue
            yield getattr(value, field.name)
        return
    if isinstance(value, Mapping):
        yield from value.values()
        return
    if isinstance(value, tuple | list):
        yield from value


__all__ = [
    "ModelLossTerm",
    "ModelWithLoss",
    "add_model_loss",
    "model_domain_metadata",
    "model_loss_labels",
    "model_loss_values",
]
