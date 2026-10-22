#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Stored-field reconstruction and content comparison for source records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from functools import cache
from enum import Enum
from fractions import Fraction
from inspect import Parameter, signature
from typing import Any

import jax
import numpy as np
from jax.experimental import multihost_utils

from .._array_archive import ArrayArchiveLimits
from .._host_storage import declared_object_storage_bytes
from .._fingerprint import (
    _NumpyArraySeal,
    array_bytes_digest,
    canonical_fingerprint,
    logical_array_value_collection_digest,
    tree_content_equal,
)
from .._model._artifacts import (
    artifact_value_codec_for,
    register_artifact_value,
    registered_artifact_value_id,
)
from .._model._structure import (
    _model_owner_children,
    model_recipe_array_pairs,
    prepare_model_recipe,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_operands import (
    OwnerFields,
    require_coordinate_input_owners,
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


@dataclass(frozen=True, slots=True)
class _StoredRecordContentAdmission:
    owner: object
    content_id: str
    owner_fields: OwnerFields
    container_owners: tuple[tuple[object, type, tuple[object, ...]], ...]
    immutable_banks: tuple[jax.Array, ...]
    mutable_banks: tuple[np.ndarray, ...]
    mutable_write_flags: tuple[bool, ...]
    mutable_seals: tuple[_NumpyArraySeal, ...]


def _mutable_record_bank_seals(
    banks: tuple[np.ndarray, ...], budget: CoordinateEnclosureBudget, /
) -> tuple[_NumpyArraySeal, ...]:
    if not banks:
        return ()
    with budget.temporary_scope():
        budget.reserve(
            sum(1 + bank.size for bank in banks),
            256 + 128 * len(banks) + 2 * max(bank.nbytes for bank in banks),
        )
        return tuple(
            _NumpyArraySeal(tuple(bank.shape), bank.dtype.str, array_bytes_digest(bank))
            for bank in banks
        )


def _require_record_containers(
    owners: tuple[tuple[object, type, tuple[object, ...]], ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    with budget.temporary_scope():
        for owner, owner_type, children in owners:
            budget.reserve(1 + len(children), 128 + 16 * len(children))
            if type(owner) is not owner_type:
                return False
            if isinstance(owner, Mapping):
                count = 2 * len(owner)
            elif isinstance(owner, (list, set, frozenset)):
                count = len(owner)
            else:
                return False
            if count != len(children) or any(
                actual is not retained
                for actual, retained in zip(
                    _model_owner_children(owner), children, strict=True
                )
            ):
                return False
    return True


def stored_record_content_id(
    value: Any,
    /,
    *,
    excluded: str,
    budget: CoordinateEnclosureBudget | None = None,
    memoize: bool = True,
) -> str:
    """Bind declared fields, static structure and raw banks without changing identity.

    A native caller supplies its current budget. The bounded owner traversal
    admits recipe construction and the largest streamed array-copy lifetime
    before either is allocated. The canonical traversal charges one owner visit
    and one scalar scan per distinct numerical bank at the existing prices.
    Initial mutable seals share that scan's immutable byte chunks; retained memo
    storage and every later mutable scan pay their actual additional work.
    Standalone record-integrity callers retain the original unbudgeted contract.
    A request may reuse its full canonical scan only after checking every actual
    retained field owner and rescanning every mutable NumPy bank's complete
    logical bytes and metadata. Readonly flags confer no trust. Immutable JAX
    payloads need no second scan; content reuse never issues proof authority.
    With ``memoize=False``, every call performs the full budgeted canonical scan
    without consulting, changing or retaining memo entries or seal snapshots.
    """
    key = (id(value), excluded)
    if budget is not None and memoize:
        cached = budget.stored_record_content_cache.get(key)
        if cached is not None and cached.owner is value:
            budget.reserve(len(cached.immutable_banks))
            if any(bank.is_deleted() for bank in cached.immutable_banks):
                del budget.stored_record_content_cache[key]
                raise ValueError("stored_record_content_id must retain live actual JAX banks")
            if (
                require_coordinate_input_owners(cached.owner_fields, budget)
                and _require_record_containers(cached.container_owners, budget)
                and tuple(bank.flags.writeable for bank in cached.mutable_banks)
                == cached.mutable_write_flags
                and _mutable_record_bank_seals(cached.mutable_banks, budget)
                == cached.mutable_seals
            ):
                return cached.content_id
            # A changed actual field or bank invalidates this earned binding.
            # The next full scan, not its former digest, supplies the new one.
            del budget.stored_record_content_cache[key]
    # Constructors may hash their declared inputs before assigning the excluded
    # derived self-ID. Authenticate those inputs normally, but do not retain a
    # still-incomplete owner graph as a request-local memo.
    cacheable = memoize and (not excluded or hasattr(value, excluded))
    owner_fields: list[tuple[object, type, tuple[str, ...], tuple[object, ...]]] = []
    container_owners: list[tuple[object, type, tuple[object, ...]]] = []
    immutable_banks: list[jax.Array] = []
    mutable_banks: list[np.ndarray] = []
    mutable_seals: dict[int, _NumpyArraySeal] = {}
    register_meshing_source_types()

    def identify() -> str:
        nonlocal cacheable
        members = fields(value)
        if budget is not None:
            budget.reserve(0, 256 + 128 * len(members))
            budget.reserve(1 + len(members))
            if memoize:
                names = tuple(member.name for member in members if member.name != excluded)
                owner_fields.append(
                    (value, type(value), names, tuple(getattr(value, name) for name in names))
                )
        content = tuple(
            (member.name, getattr(value, member.name))
            for member in members
            if member.name != excluded
        )
        if budget is not None:
            pending: list[object] = [content]
            seen: dict[int, object] = {}
            largest_bank = 0
            while pending:
                owner = pending.pop()
                if seen.get(id(owner)) is owner:
                    continue
                # Recipe nodes, traversal indices, field-name/path strings and
                # the canonical JSON hash buffer coexist. Fixed declared bounds
                # cover these structural copies; input array payloads do not.
                budget.reserve(1, 128 + 16 * declared_object_storage_bytes(owner))
                seen[id(owner)] = owner
                if not (
                    owner is None
                    or type(owner) in (bool, int, float, complex, str, bytes, tuple, Fraction)
                    or isinstance(owner, (jax.Array, np.ndarray, Mapping, list, set, frozenset))
                    or (is_dataclass(owner) and not isinstance(owner, type))
                    or (
                        artifact_value_codec_for(owner) is not None
                        and hasattr(owner, "__dict__")
                    )
                ):
                    # Foreign owners without inspectable canonical fields
                    # retain the complete canonical traversal on every call.
                    cacheable = False
                if isinstance(owner, (jax.Array, np.ndarray)):
                    bank_bytes = 2 * owner.nbytes
                    budget.reserve(owner.size, max(0, bank_bytes - largest_bank))
                    largest_bank = max(largest_bank, bank_bytes)
                    if memoize:
                        if isinstance(owner, np.ndarray):
                            mutable_banks.append(owner)
                        else:
                            immutable_banks.append(owner)
                    continue
                if is_dataclass(owner) and not isinstance(owner, type):
                    members = fields(owner)
                    count = len(members)
                    if memoize:
                        names = tuple(member.name for member in members)
                        owner_fields.append(
                            (owner, type(owner), names, tuple(getattr(owner, name) for name in names))
                        )
                    children = _model_owner_children(owner)
                elif isinstance(owner, Mapping):
                    count = 2 * len(owner)
                    children = _model_owner_children(owner)
                elif isinstance(owner, (tuple, list, set, frozenset)):
                    count = len(owner)
                    children = _model_owner_children(owner)
                elif isinstance(owner, Fraction):
                    count, children = 2, (owner.numerator, owner.denominator)
                    if memoize:
                        owner_fields.append(
                            (owner, type(owner), ("numerator", "denominator"), children)
                        )
                elif isinstance(owner, Enum):
                    count, children = 1, (owner.value,)
                elif artifact_value_codec_for(owner) is not None and hasattr(owner, "__dict__"):
                    # Closed registered value codecs serialize these actual
                    # stored owners; never expand arbitrary dynamic attributes.
                    attributes = vars(owner)
                    if memoize:
                        owner_fields.append(
                            (owner, type(owner), tuple(attributes), tuple(attributes.values()))
                        )
                    count, children = 2 * len(attributes), (
                        child for pair in attributes.items() for child in pair
                    )
                else:
                    count, children = 0, ()
                budget.reserve(0, 128 + 64 * count)
                if memoize and isinstance(owner, (Mapping, list, set, frozenset)):
                    children = tuple(children)
                    container_owners.append((owner, type(owner), children))
                pending.extend(children)
        if budget is not None and cacheable and mutable_banks:
            budget.reserve(len(mutable_banks), 256 + 2048 * len(mutable_banks))
        recipe, arrays = prepare_model_recipe(content, prefix="stored_fields")
        return canonical_fingerprint(
            {
                "kind": "stored-record-content",
                "structure": recipe,
                "banks": logical_array_value_collection_digest(
                    arrays,
                    _numpy_seals=mutable_seals
                    if budget is not None and cacheable and mutable_banks else None,
                ),
            }
        )

    if budget is None:
        return identify()
    with budget.temporary_scope():
        content_id = identify()
    if cacheable:
        banks = tuple(mutable_banks)
        entry = _StoredRecordContentAdmission(
            value,
            content_id,
            tuple(owner_fields),
            tuple(container_owners),
            tuple(immutable_banks),
            banks,
            tuple(bank.flags.writeable for bank in banks),
            tuple(mutable_seals[id(bank)] for bank in banks),
        )
        # Own actual banks and their authentication snapshot through this
        # request's lifetime, without promoting a digest to proof authority.
        budget.retain_owner(entry)
        budget.stored_record_content_cache[key] = entry
    return content_id


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
    identity. Array ordering remains the ordering of the model recipe. One
    object is equal to itself without lowering or reading its content.
    """
    if first is second:
        return True
    pairs = model_recipe_array_pairs(first, second, limits=limits)
    return pairs is not None and all(
        left is right
        or logical_array_value_collection_digest({"value": left})
        == logical_array_value_collection_digest({"value": right})
        for left, right in pairs
    )
