#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from .._strict import StrictModule
from ..typing import PRNGKey


_ADDRESS_NAMESPACE = b"phydrax-sample-address\0"


class SampleAddress(StrictModule):
    """Static semantic address for a reproducible random substream."""

    namespace: str = eqx.field(static=True)
    operation: str = eqx.field(static=True)
    target: tuple[str, ...] = eqx.field(static=True)
    role: str = eqx.field(static=True)
    token: int = eqx.field(static=True)

    def __init__(
        self,
        namespace: str,
        operation: str,
        /,
        *,
        target: str | Sequence[str] = (),
        role: str = "sample",
    ) -> None:
        namespace_ = _nonempty(namespace, "namespace")
        operation_ = _nonempty(operation, "operation")
        if isinstance(target, str):
            target_ = (_nonempty(target, "target"),)
        else:
            target_ = tuple(_nonempty(value, "target") for value in target)
        role_ = _nonempty(role, "role")
        self.namespace = namespace_
        self.operation = operation_
        self.target = target_
        self.role = role_
        self.token = _address_token(
            namespace_,
            operation_,
            *target_,
            role_,
        )


def _nonempty(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _address_token(*parts: str) -> int:
    digest = hashlib.sha256(_ADDRESS_NAMESPACE)
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:4], "little")


_WORD_LIMIT = 2**32


def _index_word(index: int | Array, /) -> Array:
    """One index as one exact `uint32` word; host integers are range-checked."""
    if isinstance(index, bool | np.bool_):
        raise TypeError("derive_key indices must be integers, not booleans.")
    if isinstance(index, int | np.integer):
        if not 0 <= index < _WORD_LIMIT:
            raise ValueError("derive_key host indices must lie in [0, 2**32).")
        return jnp.asarray(index, dtype=jnp.uint32)
    value = jnp.asarray(index)
    if value.ndim != 0:
        raise ValueError("derive_key indices must be scalars.")
    if not jnp.issubdtype(value.dtype, jnp.integer):
        raise TypeError("derive_key indices must have an integer dtype.")
    return value.astype(jnp.uint32)


def derive_key(
    root_key: PRNGKey,
    address: SampleAddress,
    /,
    *indices: int | Array,
) -> PRNGKey:
    """Derive a JAX key from a semantic address and runtime indices.

    Each index is folded as one `uint32` word, in order. Host integers must
    lie in `[0, 2**32)`; array indices must be integer scalars and are folded
    modulo `2**32`. A persistent particle identity is folded as its two words,
    `derive_key(root, address, step, id_hi, id_lo, event)`, so the key follows
    the particle rather than its storage slot.
    """
    key = jr.fold_in(root_key, jnp.asarray(address.token, dtype=jnp.uint32))
    for index in indices:
        key = jr.fold_in(key, _index_word(index))
    return key


__all__ = ["SampleAddress", "derive_key"]
