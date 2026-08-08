#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._evaluation import BatchEvaluator
from ._function import DomainFunction
from ._structure import PointBatch


class _IndexedFieldEvaluator(StrictModule, BatchEvaluator, NonTrainableState):
    values: Array
    index_key: str
    owner: str

    def __init__(self, values: Array, index_key: str, owner: str, /):
        self.values = jax.lax.stop_gradient(jnp.asarray(values))
        self.index_key = str(index_key)
        self.owner = str(owner)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        del args, key, kwargs
        raise TypeError(f"{self.owner} requires structured batch evaluation.")

    def __call_batch__(
        self,
        batch: PointBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointBatch):
            raise TypeError(f"{self.owner} requires PointBatch evaluation.")
        index = batch.points.get(self.index_key)
        if not isinstance(index, cx.Field):
            raise ValueError(
                f"{self.owner} requires batch metadata {self.index_key!r}."
            )
        indices = jnp.asarray(index.data, dtype=jnp.int32)
        selected = self.values[indices]
        dims = index.dims + (None,) * max(selected.ndim - len(index.dims), 0)
        return cx.Field(selected, dims=dims)


def indexed_field(
    domain,
    values: ArrayLike,
    /,
    *,
    size: int,
    index_key: str,
    owner: str,
) -> DomainFunction:
    """Expose row-aligned finite values through batch index metadata."""
    array = jnp.asarray(values)
    if array.ndim == 0:
        raise ValueError(f"{owner} values must have a leading row axis.")
    if int(array.shape[0]) != int(size):
        raise ValueError(
            f"{owner} values require leading size {size}, got {array.shape[0]}."
        )
    return DomainFunction(
        domain=domain,
        deps=domain.labels,
        func=_IndexedFieldEvaluator(array, index_key, owner),
        metadata={},
    )


__all__ = ["indexed_field"]
