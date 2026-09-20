#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._strict import StrictModule


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
    ):
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


def derive_key(
    root_key: Key[Array, ""],
    address: SampleAddress,
    /,
    *indices: int | Array,
) -> Key[Array, ""]:
    """Derive a JAX key from a semantic address and runtime indices."""
    key = jr.fold_in(root_key, jnp.asarray(address.token, dtype=jnp.uint32))
    for index in indices:
        key = jr.fold_in(key, jnp.asarray(index, dtype=jnp.uint32))
    return key


__all__ = ["SampleAddress", "derive_key"]
