#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import jax.numpy as jnp


ModelInputMode: TypeAlias = Literal["flat", "structured"]
ModelBatchMode: TypeAlias = Literal["pointwise", "blockwise", "axis"]


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """Explicit contract for packing and invoking a model on domain coordinates."""

    input_mode: ModelInputMode = "flat"
    batch_mode: ModelBatchMode = "pointwise"
    pass_key: bool = True
    pass_iter: bool = False
    warn_on_fallback: bool = False

    def __post_init__(self) -> None:
        if self.input_mode not in ("flat", "structured"):
            raise ValueError("ModelBinding.input_mode must be 'flat' or 'structured'.")
        if self.batch_mode not in ("pointwise", "blockwise", "axis"):
            raise ValueError(
                "ModelBinding.batch_mode must be 'pointwise', 'blockwise', or 'axis'."
            )

    def pack_point(self, args: tuple[Any, ...], /) -> Any:
        """Pack one model invocation according to this binding's input contract."""
        if not args:
            raise ValueError("Model callable requires at least one positional input.")
        if (
            self.batch_mode == "blockwise"
            and len(args) == 1
            and isinstance(args[0], tuple)
        ):
            return tuple(jnp.asarray(item) for item in args[0])
        if self.input_mode == "structured":
            if len(args) == 1:
                value = args[0]
                if isinstance(value, tuple):
                    return tuple(jnp.asarray(item) for item in value)
                return jnp.asarray(value)
            packed: list[Any] = []
            for value in args:
                if isinstance(value, tuple):
                    packed.extend(jnp.asarray(item) for item in value)
                else:
                    packed.append(jnp.asarray(value))
            return tuple(packed)

        arrays: list[Any] = []
        for value in args:
            if isinstance(value, tuple):
                raise ValueError(
                    "Flat ModelBinding cannot pack tuple inputs; use a structured "
                    "or blockwise binding, or materialize the grid explicitly."
                )
            arrays.append(jnp.asarray(value))
        if len(arrays) == 1:
            return arrays[0]

        leading_shape: tuple[int, ...] | None = None
        for array in arrays:
            if array.ndim < 2:
                continue
            candidate = tuple(int(size) for size in array.shape[:-1])
            if leading_shape is None:
                leading_shape = candidate
            elif candidate != leading_shape:
                raise ValueError(
                    "Flat model packing requires batched inputs to share leading "
                    f"shape; got {candidate} and {leading_shape}."
                )
        if leading_shape is None:
            return jnp.concatenate(tuple(array.reshape((-1,)) for array in arrays))

        parts: list[Any] = []
        for array in arrays:
            shape = tuple(int(size) for size in array.shape)
            if array.ndim == 0:
                part = jnp.broadcast_to(array, leading_shape + (1,))
            elif shape == leading_shape:
                part = array.reshape(leading_shape + (1,))
            elif shape[:-1] == leading_shape:
                part = array.reshape(leading_shape + (shape[-1],))
            else:
                raise ValueError(
                    "Flat model packing could not align input with shape "
                    f"{array.shape} to leading batch shape {leading_shape}."
                )
            parts.append(part)
        return jnp.concatenate(tuple(parts), axis=-1)

    @classmethod
    def pointwise(
        cls,
        input_mode: ModelInputMode = "flat",
        *,
        pass_key: bool = True,
        pass_iter: bool = False,
    ) -> "ModelBinding":
        return cls(
            input_mode=input_mode,
            batch_mode="pointwise",
            pass_key=pass_key,
            pass_iter=pass_iter,
        )

    @classmethod
    def blockwise(
        cls,
        input_mode: ModelInputMode = "structured",
        *,
        pass_key: bool = True,
        pass_iter: bool = False,
        warn_on_fallback: bool = False,
    ) -> "ModelBinding":
        return cls(
            input_mode=input_mode,
            batch_mode="blockwise",
            pass_key=pass_key,
            pass_iter=pass_iter,
            warn_on_fallback=warn_on_fallback,
        )

    @classmethod
    def axis(
        cls,
        input_mode: ModelInputMode = "structured",
        *,
        pass_key: bool = True,
        pass_iter: bool = False,
    ) -> "ModelBinding":
        return cls(
            input_mode=input_mode,
            batch_mode="axis",
            pass_key=pass_key,
            pass_iter=pass_iter,
        )

    def call(
        self,
        model: Any,
        x: Any,
        /,
        *,
        key: Any,
        iter_: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        call_kwargs = dict(kwargs)
        if self.pass_key:
            call_kwargs["key"] = key
        if self.pass_iter and iter_ is not None:
            call_kwargs["iter_"] = iter_
        return model(x, **call_kwargs)


__all__ = ["ModelBatchMode", "ModelBinding", "ModelInputMode"]
