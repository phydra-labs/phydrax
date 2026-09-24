#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._differentiation import _identifier
from .._trainable import fixed_field
from ._array import AbstractArrayModel
from ._component import ModelExecutionContract, RandomnessContract


def _prng_key(value: Any, /) -> Array:
    if not isinstance(value, jax.Array):
        raise TypeError("key must be a JAX PRNG key array.")
    typed = jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
    raw = value.dtype == jnp.uint32 and value.ndim >= 1
    if not (typed or raw):
        raise TypeError("key must be a JAX PRNG key array.")
    return value


@final
class FrozenRealization(AbstractArrayModel):
    """Owner binding of one fixed randomness realization of a model.

    Every evaluation passes the bound `key` to `model` and ignores the caller's
    key, so the model's randomness (resampled noise, active dropout, or random
    structure) is frozen to the one realization identified by `realization_id`.
    The binding's contract is the model's contract with randomness
    `RandomnessContract("fixed-realization", realization_id=...)`, unless the
    model is already deterministic without an inference-state requirement.
    Implicit and certified owners admit a fixed realization only through this
    binding. The model's arrays keep their roles; the key is FIXED.
    """

    model: AbstractArrayModel
    key: Array = fixed_field()
    realization_id: str = eqx.field(static=True)
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = staticmethod(lambda wrapper: wrapper.model.input_binding())

    def __init__(
        self,
        model: AbstractArrayModel,
        key: Array,
        /,
        *,
        realization_id: str,
    ):
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("FrozenRealization requires an AbstractArrayModel.")
        self.model = model
        self.key = _prng_key(key)
        self.realization_id = _identifier(realization_id, "realization_id")
        self.in_size = model.in_size
        self.out_size = model.out_size

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.model(x, key=self.key)

    def model_execution_contract(self) -> ModelExecutionContract:
        """Return the model's contract with its randomness frozen to this realization."""
        contract = self.model.model_execution_contract()
        randomness = contract.randomness
        if (
            randomness is None
            or randomness.mode != "deterministic"
            or randomness.requires_inference_state
        ):
            randomness = RandomnessContract(
                "fixed-realization", realization_id=self.realization_id
            )
        return ModelExecutionContract(
            derivative=contract.derivative,
            execution=contract.execution,
            precision=contract.precision,
            randomness=randomness,
            ports=contract.ports,
            certificates=contract.certificates,
            semantic_provenance=contract.semantic_provenance,
        )


__all__ = ["FrozenRealization"]
