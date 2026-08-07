#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


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
