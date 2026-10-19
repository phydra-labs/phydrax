#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Stored-field reconstruction and content comparison for source records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from functools import cache
from inspect import Parameter, signature
from typing import Any

import jax
import numpy as np
from jax.experimental import multihost_utils

from .._array_archive import ArrayArchiveLimits
from .._fingerprint import (
    canonical_fingerprint,
    logical_array_value_collection_digest,
    tree_content_equal,
)
from .._model._artifacts import (
    register_artifact_value,
    registered_artifact_value_id,
)
from .._model._structure import (
    model_recipe_array_pairs,
    prepare_model_recipe,
)
from ..typing import checked


@cache
def register_meshing_source_types() -> None:
    """Register source identities after field declarations load."""
    # Full admission includes field declarations, which import this leaf.
    # Resolve registration after their class bodies load; reconstruction and
    # content comparison never resolve an upper-layer validator or constructor.
    from ._meshing_sources import register_meshing_source_artifacts

    register_meshing_source_artifacts()


def register_stored_artifact_types(
    groups: tuple[tuple[str, tuple[type, ...]], ...], /
) -> None:
    """Register admitted record types without changing existing artifact identities."""
    for namespace, owners in groups:
        for owner in owners:
            if registered_artifact_value_id(owner) is None:
                register_artifact_value(f"{namespace}:{owner.__name__}", owner)


def stored_record_content_id(value: Any, /, *, excluded: str) -> str:
    """Bind declared-order fields, static structure, and raw numerical banks."""
    register_meshing_source_types()
    content = tuple(
        (member.name, getattr(value, member.name))
        for member in fields(value)
        if member.name != excluded
    )
    recipe, arrays = prepare_model_recipe(content, prefix="stored_fields")
    return canonical_fingerprint(
        {
            "kind": "stored-record-content",
            "structure": recipe,
            "banks": logical_array_value_collection_digest(arrays),
        }
    )


def reconstruct_stored_fields[T](
    value: T,
    /,
    *,
    bindings: Mapping[str, Any] | None = None,
    array_equal: Callable[[jax.Array | np.ndarray, jax.Array | np.ndarray], bool]
    | None = None,
) -> T:
    """Revalidate a registered record with retained constructor inputs.

    Owners whose constructor inputs differ from their stored representation
    provide an explicit reconstruction method at their admission boundary.
    External inputs are supplied only by the caller's validated binding table.
    An admitted array comparator receives stored values first. Constructor,
    type/field and nonarray/static metadata checks remain unchanged.
    """
    owner = type(value)
    match value:
        case _ if is_dataclass(value):
            names = tuple(member.name for member in fields(value))
        case tuple():
            names = getattr(owner, "_fields", None)
        case _:
            names = None
    if (
        registered_artifact_value_id(owner) is None
        or not isinstance(names, tuple)
        or any(not isinstance(name, str) for name in names)
    ):
        raise TypeError(
            f"stored_fields.owner must be a registered dataclass or named tuple; got {owner.__name__}"
        )
    declared = set(names)
    supplied = {} if bindings is None else bindings
    hook = getattr(owner, "from_stored_fields", None)
    constructor = owner if hook is None else hook
    parameters = dict(signature(constructor).parameters)
    if hook is not None:
        parameters.pop(next(iter(parameters)))
    unknown = supplied.keys() - parameters.keys()
    if unknown:
        raise TypeError(
            f"{owner.__name__}.bindings must name replay inputs; got {sorted(unknown)}"
        )
    positional: list[Any] = [] if hook is None else [value]
    keywords: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            raise TypeError(
                f"{owner.__name__}.constructor must declare its inputs; got {name!r}"
            )
        if name in supplied:
            argument = supplied[name]
        elif name in declared:
            argument = object.__getattribute__(value, name)
        elif parameter.default is not Parameter.empty:
            continue
        else:
            raise TypeError(
                f"{owner.__name__}.{name} must have a retained field or explicit binding; got missing"
            )
        if parameter.kind is Parameter.POSITIONAL_ONLY:
            positional.append(argument)
        else:
            keywords[name] = argument
    rebuilt = constructor(*positional, **keywords)
    if not tree_content_equal(value, rebuilt, array_equal=array_equal):
        difference = next(
            (
                name
                for name in names
                if not tree_content_equal(
                    object.__getattribute__(value, name),
                    object.__getattribute__(rebuilt, name),
                    array_equal=array_equal,
                )
            ),
            "structure",
        )
        raise ValueError(
            f"stored_fields.{owner.__name__} must reproduce its stored content; got changed {difference}"
        )
    return rebuilt


def reconstruct_distributed_stored_fields[T](
    value: T,
    /,
    *,
    bindings: Mapping[str, Any] | None = None,
    limits: ArrayArchiveLimits,
) -> T:
    """Collectively authenticate a host view while retaining distributed leaves.

    All processes must restore records and visit their leaves in the same order.
    Gathering is admitted before the first collective, including retained bindings.
    """
    original_leaves, structure = jax.tree.flatten(value)
    binding_leaves, binding_structure = jax.tree.flatten(bindings)
    leaves = (*original_leaves, *binding_leaves)
    distributed = [
        leaf
        for leaf in leaves
        if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable
    ]
    if not distributed:
        return reconstruct_stored_fields(value, bindings=bindings)
    total_bytes = 0
    for index, leaf in enumerate(distributed):
        size = leaf.size * leaf.dtype.itemsize
        if size > limits.max_member_bytes:
            raise ValueError(
                f"stored_fields.{type(value).__name__}.array_bytes must fit the archive member limit; got {size} bytes at leaf {index}"
            )
        total_bytes += size
    if total_bytes > limits.max_aggregate_bytes:
        raise ValueError(
            f"stored_fields.{type(value).__name__}.bytes must fit the archive aggregate limit; got {total_bytes} bytes"
        )
    device = jax.local_devices()[0]
    host_leaves = [
        jax.device_put(multihost_utils.process_allgather(leaf, tiled=True), device)
        if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable
        else leaf
        for leaf in leaves
    ]
    host_value = jax.tree.unflatten(structure, host_leaves[: len(original_leaves)])
    host_bindings = jax.tree.unflatten(
        binding_structure, host_leaves[len(original_leaves) :]
    )
    rebuilt = reconstruct_stored_fields(host_value, bindings=host_bindings)
    rebuilt_leaves, rebuilt_structure = jax.tree.flatten(rebuilt)
    # Compare static node metadata bitwise, not PyTreeDef's scalar ``==``.
    if not tree_content_equal(
        jax.tree.unflatten(structure, [None] * len(original_leaves)),
        jax.tree.unflatten(rebuilt_structure, [None] * len(rebuilt_leaves)),
    ):
        raise ValueError(
            f"stored_fields.{type(value).__name__}.structure must retain bitwise node metadata; got changed structure"
        )
    for index, (original, restored) in enumerate(
        zip(original_leaves, rebuilt_leaves, strict=True)
    ):
        if isinstance(original, (jax.Array, np.ndarray)) and (
            not isinstance(restored, (jax.Array, np.ndarray))
            or restored.shape != original.shape
            or restored.dtype != original.dtype
        ):
            raise ValueError(
                f"stored_fields.{type(value).__name__}.leaves must retain their shapes and dtypes; got changed array metadata at leaf {index}"
            )
    return jax.tree.unflatten(structure, original_leaves)


@checked
def content_equal(first: Any, second: Any, /, *, limits: ArrayArchiveLimits) -> bool:
    """Compare record structure and bounded logical array content.

    Object sharing and NumPy mutability describe archive storage, not source
    identity. Array ordering remains the ordering of the model recipe.
    """
    pairs = model_recipe_array_pairs(first, second, limits=limits)
    return pairs is not None and all(
        left is right
        or logical_array_value_collection_digest({"value": left})
        == logical_array_value_collection_digest({"value": right})
        for left, right in pairs
    )
