#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._strict import StrictModule
from ._evaluation import (
    BatchEvaluator,
    complete_batch_axes,
    evaluate_pointwise_callable,
    try_blockwise_evaluation,
)


if TYPE_CHECKING:
    from ..nn.models.core._binding import ModelBinding


class ConcatenatedModelEvaluator(StrictModule, BatchEvaluator):
    raw_model: Any
    domain_labels: tuple[str, ...]
    deps: tuple[str, ...]
    binding: ModelBinding

    def __init__(
        self,
        model: Callable,
        /,
        *,
        domain_labels: tuple[str, ...],
        deps: tuple[str, ...],
        binding: ModelBinding,
    ):
        from ..nn.models.core._base import _AbstractBaseModel
        from ..nn.models.core._binding import ModelBinding
        from ..nn.models.core._loss import ModelWithLoss

        if not callable(model):
            raise TypeError("Domain models must be callable.")
        if not isinstance(binding, ModelBinding):
            raise TypeError("Domain models require an explicit ModelBinding.")
        if binding.batch_mode == "axis" and not isinstance(
            model, (_AbstractBaseModel, ModelWithLoss)
        ):
            raise TypeError(
                "Axis-batch model bindings require a Phydrax model implementation."
            )
        self.raw_model = model
        self.domain_labels = tuple(domain_labels)
        self.deps = tuple(deps)
        self.binding = binding

    def emit_auto_fallback_warning(self, message: str, /) -> None:
        if self.binding.warn_on_fallback:
            warnings.warn(message, UserWarning, stacklevel=3)

    def _call_model(self, x: Any, /, *, key=None, iter_=None, **kwargs: Any):
        return self.binding.call(
            self.raw_model,
            x,
            key=key,
            iter_=iter_,
            kwargs=dict(kwargs),
        )

    def _structured_input(self, args: tuple[Any, ...], /) -> Any:
        def _as_array_or_tuple(value: Any):
            if isinstance(value, tuple):
                return tuple(jnp.asarray(item) for item in value)
            return jnp.asarray(value)

        if len(args) == 1:
            return _as_array_or_tuple(args[0])
        packed: list[Any] = []
        for value in args:
            if isinstance(value, tuple):
                packed.extend(jnp.asarray(item) for item in value)
            else:
                packed.append(jnp.asarray(value))
        return tuple(packed)

    def _call_blockwise(
        self,
        *args: Any,
        key=None,
        iter_=None,
        **kwargs: Any,
    ) -> Any:
        return self._call_model(
            self._structured_input(args),
            key=key,
            iter_=iter_,
            **kwargs,
        )

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_=None,
        **kwargs: Any,
    ) -> cx.Field:
        if self.binding.batch_mode == "axis":
            out = self.__call_axis_batch__(
                batch,
                self.deps,
                key=key,
                iter_=iter_,
                **kwargs,
            )
            if not isinstance(out, cx.Field):
                raise TypeError("Axis-batch model evaluation must return a Field.")
            return out

        if self.binding.batch_mode == "blockwise":
            out, reason = try_blockwise_evaluation(
                self._call_blockwise,
                self.deps,
                batch,
                key=key,
                iter_=iter_,
                **kwargs,
            )
            if out is not None:
                return complete_batch_axes(out, batch, self.domain_labels)
            if reason is not None:
                self.emit_auto_fallback_warning(
                    "Falling back to pointwise evaluation for DomainFunction model: "
                    + reason
                )

        return evaluate_pointwise_callable(
            self,
            deps=self.deps,
            domain_labels=self.domain_labels,
            points=batch,
            key=key,
            kwargs={"iter_": iter_, **kwargs},
        )

    def __call_axis_batch__(
        self,
        batch: Any,
        deps: tuple[str, ...],
        /,
        *,
        key=None,
        iter_=None,
        **kwargs: Any,
    ):
        from ..nn.models.core._base import _AbstractBaseModel
        from ..nn.models.core._loss import ModelWithLoss

        if isinstance(self.raw_model, ModelWithLoss):
            return self.raw_model.__call_axis_batch__(
                batch, deps, key=key, iter_=iter_, **kwargs
            )
        if isinstance(self.raw_model, _AbstractBaseModel):
            return self.raw_model.__call_axis_batch__(
                batch, deps, key=key, iter_=iter_, **kwargs
            )
        raise TypeError("Model callable does not support axis-batch execution.")

    def __call__(self, *args: Any, key=None, iter_=None, **kwargs: Any):
        if not args:
            raise ValueError("Model callable requires at least one positional input.")

        coordinate_positions = tuple(
            index for index, value in enumerate(args) if isinstance(value, tuple)
        )
        if coordinate_positions:
            coordinate_values = tuple(
                jnp.asarray(coordinate).reshape((-1,))
                for index in coordinate_positions
                for coordinate in args[index]
            )
            if coordinate_values:

                def call_point(*values: Any) -> Any:
                    point_args = list(args)
                    offset = 0
                    for index in coordinate_positions:
                        count = len(args[index])
                        point_args[index] = jnp.stack(values[offset : offset + count])
                        offset += count
                    return self(
                        *point_args,
                        key=key,
                        iter_=iter_,
                        **kwargs,
                    )

                mapped = call_point
                for position in reversed(range(len(coordinate_values))):
                    mapped = jax.vmap(
                        mapped,
                        in_axes=tuple(
                            0 if index == position else None
                            for index in range(len(coordinate_values))
                        ),
                        out_axes=0,
                    )
                return mapped(*coordinate_values)

        if self.binding.input_mode == "structured":
            return self._call_model(
                self._structured_input(args),
                key=key,
                iter_=iter_,
                **kwargs,
            )

        arrays: list[Any] = []
        for value in args:
            if isinstance(value, tuple):
                raise ValueError(
                    "Flat ModelBinding cannot pack tuple inputs; use a structured "
                    "or blockwise binding, or materialize the grid explicitly."
                )
            arrays.append(jnp.asarray(value))

        if len(arrays) == 1:
            x_in = arrays[0]
            return self._call_model(x_in, key=key, iter_=iter_, **kwargs)

        leading_shape: tuple[int, ...] | None = None
        for arr in arrays:
            if arr.ndim < 2:
                continue
            candidate = tuple(int(i) for i in arr.shape[:-1])
            if leading_shape is None:
                leading_shape = candidate
                continue
            if candidate != leading_shape:
                raise ValueError(
                    "Flat model packing requires batched inputs to share leading "
                    f"shape; got {candidate} and {leading_shape}."
                )

        if leading_shape is None:
            parts = [arr.reshape((-1,)) for arr in arrays]
            x_in = jnp.concatenate(parts, axis=0)
            return self._call_model(x_in, key=key, iter_=iter_, **kwargs)

        parts = []
        for arr in arrays:
            if arr.ndim == 0:
                part = jnp.broadcast_to(arr, leading_shape + (1,))
            elif tuple(int(i) for i in arr.shape) == leading_shape:
                part = arr.reshape(leading_shape + (1,))
            elif tuple(int(i) for i in arr.shape[:-1]) == leading_shape:
                part = arr.reshape(leading_shape + (int(arr.shape[-1]),))
            else:
                raise ValueError(
                    "Flat model packing could not align input with shape "
                    f"{arr.shape} to leading batch shape {leading_shape}."
                )
            parts.append(part)
        x_in = jnp.concatenate(parts, axis=-1)
        return self._call_model(x_in, key=key, iter_=iter_, **kwargs)
