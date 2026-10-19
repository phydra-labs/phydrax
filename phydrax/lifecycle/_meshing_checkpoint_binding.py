#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source recipes and numerical bindings shared by checkpoint and restart."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, is_dataclass
from types import MappingProxyType
from typing import Any, Literal, NoReturn, TypeAlias
from weakref import ref, ReferenceType, WeakSet

import equinox as eqx
import jax
import numpy as np

from .._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from .._fingerprint import (
    canonical_fingerprint,
    canonical_json,
    logical_array_value_collection_digest,
)
from .._model._structure import (
    _model_owner_children,
    model_recipe_array_inventory,
    ModelRecipeArray,
    prepare_model_recipe_catalog,
    validate_model_structure_recipe,
)
from .._strict import Strict, StrictModule
from ..typing import checked
from ._chunk_repository import digest, MAX_METADATA_VALUE_BYTES, RepositoryCorruptionError
from ._meshing_sources import (
    MeshingSourceClosure,
    MeshingSourceValidation,
    register_meshing_source_artifacts,
    validate_collective_mesh_source_epochs,
    validate_meshing_source_closure,
)


MeshingArrayRole: TypeAlias = Literal[
    "entity-ids",
    "geometry",
    "topology",
    "lineage",
    "epoch",
    "evidence",
    "state",
    "constraints",
    "allocator",
    "source-record",
    "pending-work",
]


def source_array_name(name: str, owner_index: int, bindings: Mapping[str, str], /) -> str:
    alias = bindings.get(name)
    if alias is not None:
        return source_logical_array_name(alias)
    return f"source-owner-{owner_index}:{name.replace('/', ':')}"


def source_logical_array_name(alias: str, /) -> str:
    return f"source-logical-{canonical_fingerprint(alias)}"


def _unique_json_object(pairs: list[tuple[str, Any]], /) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(
                f"checkpoint.metadata.members must be unique; got duplicate member {name!r}"
            )
        value[name] = item
    return value


def metadata_json(
    value: str,
    /,
    *,
    maximum_nesting: int = 16,
    maximum_bytes: int = MAX_METADATA_VALUE_BYTES,
) -> Any:
    if not isinstance(value, str) or len(value.encode("utf-8")) > maximum_bytes:
        raise RepositoryCorruptionError(
            f"checkpoint.metadata.text must be text within {maximum_bytes} bytes; got invalid type or byte limit exceeded"
        )
    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum_nesting:
                raise RepositoryCorruptionError(
                    f"checkpoint.metadata.nesting must not exceed {maximum_nesting}; got depth {depth}"
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise RepositoryCorruptionError(
                    "checkpoint.metadata.json must decode as balanced JSON without duplicate keys; got invalid JSON"
                )
    if depth or in_string:
        raise RepositoryCorruptionError(
            "checkpoint.metadata.json must decode as balanced JSON without duplicate keys; got invalid JSON"
        )
    try:
        return json.loads(value, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise RepositoryCorruptionError(
            "checkpoint.metadata.json must decode as balanced JSON without duplicate keys; got invalid JSON"
        ) from error


MESHING_SOURCE_PREFIX = "source-record"


class MeshingSourceBinding(StrictModule):
    """Bind a source recipe, its array roles, and its numerical content."""

    recipe_json: str | None = eqx.field(static=True)
    content_digest: str | None = eqx.field(static=True)
    array_roles: tuple[tuple[str, MeshingArrayRole], ...] = eqx.field(static=True)
    logical_array_names: tuple[str, ...] = eqx.field(static=True)
    logical_content_digest: str | None = eqx.field(static=True)
    logical_arrays: AdmittedMeshingSourceArrays | None
    array_values: Mapping[str, jax.Array | np.ndarray] | None = None


class _LogicalBankBinding(StrictModule):
    names: tuple[str, ...] = eqx.field(static=True)
    content_digest: str | None = eqx.field(static=True)
    arrays: AdmittedMeshingSourceArrays | None


@dataclass(frozen=True, slots=True)
class _SourcePublicationSnapshot:
    recipe_json: str
    content_digest: str
    owner_index: int
    array_bindings: tuple[tuple[str, str], ...]
    array_roles: tuple[tuple[str, MeshingArrayRole], ...]
    logical_array_names: tuple[str, ...]
    logical_content_digest: str | None
    arrays: Mapping[str, jax.Array]
    source_validation: MeshingSourceValidation | None
    owner_references: tuple[object, ...]
    limits: ArrayArchiveLimits
    maximum_chunk_bytes: int


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class _SourceRestoreAdmissionScope:
    process_id: int


_source_restore_admission_scope: ContextVar[_SourceRestoreAdmissionScope | None] = (
    ContextVar("meshing_source_restore_admission_scope", default=None)
)


@contextmanager
def meshing_source_restore_admission() -> Iterator[None]:
    """Bound earned source binding reuse to one cold restore invocation."""
    handle = _source_restore_admission_scope.set(
        _SourceRestoreAdmissionScope(os.getpid())
    )
    try:
        yield
    finally:
        _source_restore_admission_scope.reset(handle)


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False, init=False)
class _SourcePublicationAdmission:
    snapshot: _SourcePublicationSnapshot
    process_id: int
    restore_scope: ReferenceType[_SourceRestoreAdmissionScope] | None

    def __init__(self) -> NoReturn:
        raise TypeError(
            "source_binding.publication_admission must be issued by full source validation; got direct construction"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "source_binding.publication_admission must remain process-local; got serialization"
        )


_issued_source_publications: WeakSet[_SourcePublicationAdmission] = WeakSet()


@dataclass(slots=True)
class _SourcePublicationSlot:
    admission: _SourcePublicationAdmission | None = None


class AdmittedMeshingSourceArrays(Strict, Mapping[str, jax.Array]):
    """Retain immutable logical-bank values with one content admission."""

    _arrays: Mapping[str, jax.Array]
    content_digest: str
    maximum_chunk_bytes: int
    source_validation: MeshingSourceValidation | None
    _publication_slot: _SourcePublicationSlot

    @checked
    def __init__(
        self,
        arrays: Mapping[str, jax.Array],
        /,
        *,
        maximum_chunk_bytes: int,
        source_validation: MeshingSourceValidation | None = None,
    ) -> None:

        if isinstance(arrays, AdmittedMeshingSourceArrays):
            self._arrays = arrays._arrays
            self.content_digest = arrays.content_digest
        else:
            values = dict(sorted(arrays.items()))
            if any(
                not isinstance(name, str) or not name or not isinstance(value, jax.Array)
                for name, value in values.items()
            ):
                raise TypeError(
                    "source_binding.logical_arrays must map nonempty names to JAX arrays; "
                    f"got value types {tuple(type(value).__name__ for value in values.values())}"
                )
            self._arrays = MappingProxyType(values)
            self.content_digest = logical_array_value_collection_digest(
                values, maximum_chunk_bytes=maximum_chunk_bytes
            )
        if source_validation is not None:
            source_validation.require_binding(self._arrays, self.content_digest)
        self.maximum_chunk_bytes = maximum_chunk_bytes
        self.source_validation = source_validation
        # Public copies retain bank evidence, never whole-closure admission.
        self._publication_slot = _SourcePublicationSlot()

    def __getitem__(self, name: str, /) -> jax.Array:
        return self._arrays[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)


def admit_meshing_source_logical_arrays(
    arrays: Mapping[str, jax.Array] | None,
    /,
    *,
    maximum_chunk_bytes: int = 1 << 20,
) -> AdmittedMeshingSourceArrays | None:
    if isinstance(arrays, AdmittedMeshingSourceArrays):
        return arrays
    if arrays is None or not arrays:
        return None
    return AdmittedMeshingSourceArrays(arrays, maximum_chunk_bytes=maximum_chunk_bytes)


@checked
def meshing_source_recipe(
    recipe_json: str,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> dict[str, Any]:

    register_meshing_source_artifacts()
    # A semantic recipe edge can use a mapping object, its items list and an
    # item-pair list. The recipe validator admits semantic depth separately.
    recipe = metadata_json(
        recipe_json,
        maximum_nesting=4 * limits.max_manifest_nesting + 2,
        maximum_bytes=limits.max_manifest_bytes,
    )
    if not isinstance(recipe, dict) or canonical_json(recipe) != recipe_json:
        raise ValueError(
            "source_binding.recipe_json must use deterministic JSON encoding; got different bytes"
        )
    validate_model_structure_recipe(recipe, limits=limits)
    return recipe


def _validate_meshing_source_logical_bank(
    inventory: Sequence[ModelRecipeArray],
    array_bindings: Mapping[str, str],
    logical_arrays: Mapping[str, jax.Array] | None,
    logical_array_names: Sequence[str],
    logical_content_digest: str | None,
    /,
    *,
    maximum_chunk_bytes: int,
) -> _LogicalBankBinding:

    entries = {entry.name: entry for entry in inventory}
    if set(array_bindings) - set(entries):
        raise ValueError(
            "source_binding.array_bindings must reference declared recipe arrays; got undeclared recipe names"
        )
    if any(
        entries[name].backend != "jax" or not isinstance(alias, str) or not alias
        for name, alias in array_bindings.items()
    ):
        raise ValueError(
            "source_binding.array_bindings must name JAX recipe arrays with nonempty aliases; got incompatible backends or aliases"
        )
    names = tuple(sorted(logical_array_names))
    if len(set(names)) != len(names) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError(
            f"source_binding.logical_array_names must contain unique nonempty strings; got {names!r}"
        )
    admitted = admit_meshing_source_logical_arrays(
        logical_arrays, maximum_chunk_bytes=maximum_chunk_bytes
    )
    if admitted is not None:
        actual_names = tuple(sorted(admitted))
        if names and names != actual_names:
            raise ValueError(
                "source_binding.logical_array_names must match the admitted bank inventory; got changed inventory"
            )
        names = actual_names
        actual_digest = admitted.content_digest
        if logical_content_digest is not None and logical_content_digest != actual_digest:
            raise ValueError(
                "source_binding.logical_content_digest must match admitted numerical content; got changed logical content"
            )
        logical_content_digest = actual_digest if names else None
    if set(array_bindings.values()) - set(names):
        raise ValueError(
            "source_binding.array_bindings must refer to declared logical banks; "
            f"got undeclared aliases {sorted(set(array_bindings.values()) - set(names))}"
        )
    if names:
        if not isinstance(logical_content_digest, str):
            raise TypeError(
                "source_binding.logical_content_digest must be a digest for nonempty banks; "
                f"got {type(logical_content_digest).__name__}"
            )
        digest(logical_content_digest, "source_logical_content_digest")
    elif logical_content_digest is not None:
        raise ValueError(
            "source_binding.logical_content_digest must be absent for an empty bank; got a declared digest"
        )
    return _LogicalBankBinding(names, logical_content_digest, admitted)


def meshing_source_content_digest(
    values: Mapping[str, jax.Array | np.ndarray],
    array_bindings: Mapping[str, str],
    logical_arrays: Mapping[str, jax.Array] | None,
    logical_content_digest: str | None,
    /,
    *,
    maximum_chunk_bytes: int,
    owner_index: int,
    array_paths: Mapping[str, str],
) -> str:

    local_values: dict[str, jax.Array | np.ndarray] = {}
    for name, value in values.items():
        alias = array_bindings.get(name)
        if alias is not None:
            if logical_arrays is None or value is not logical_arrays[alias]:
                raise ValueError(
                    f"source_owner.{owner_index}.{array_paths[name]} must retain logical binding {alias!r} value identity; got a different value"
                )
        else:
            if isinstance(value, jax.Array) and not value.is_fully_addressable:
                raise ValueError(
                    f"source_owner.{owner_index}.{array_paths[name]} must declare a logical binding for nonaddressable data; got an unbound global array"
                )
            local_values[name] = value
    return canonical_fingerprint(
        {
            "local_content_digest": logical_array_value_collection_digest(
                local_values, maximum_chunk_bytes=maximum_chunk_bytes
            ),
            "logical_content_digest": logical_content_digest,
            "array_bindings": tuple(sorted(array_bindings.items())),
        }
    )


class _SourceRecipeAdmission(StrictModule):
    recipe_json: str = eqx.field(static=True)
    logical_arrays: AdmittedMeshingSourceArrays
    inventory: tuple[ModelRecipeArray, ...] = eqx.field(static=True)
    values: dict[str, jax.Array | np.ndarray]


def _admit_live_source_recipe(
    closure: Mapping[str, Any],
    recipe_json: str | None,
    logical_arrays: Mapping[str, jax.Array] | None,
    limits: ArrayArchiveLimits,
    maximum_chunk_bytes: int,
) -> _SourceRecipeAdmission:
    admitted = admit_meshing_source_logical_arrays(
        logical_arrays, maximum_chunk_bytes=maximum_chunk_bytes
    )
    if admitted is None:
        admitted = AdmittedMeshingSourceArrays(
            {}, maximum_chunk_bytes=maximum_chunk_bytes
        )
    if admitted.source_validation is None:
        proof = validate_collective_mesh_source_epochs(
            (closure,),
            source_logical_arrays=admitted,
            source_logical_content_digest=admitted.content_digest,
            maximum_chunk_bytes=maximum_chunk_bytes,
            limits=limits,
        )
        admitted = AdmittedMeshingSourceArrays(
            admitted, maximum_chunk_bytes=maximum_chunk_bytes, source_validation=proof
        )
    validate_meshing_source_closure(
        closure,
        limits=limits,
        source_logical_arrays=admitted,
        source_logical_content_digest=admitted.content_digest,
        maximum_chunk_bytes=maximum_chunk_bytes,
        source_validation=admitted.source_validation,
    )
    recipe, inventory, values = prepare_model_recipe_catalog(
        closure, prefix=MESHING_SOURCE_PREFIX, limits=limits
    )
    current_recipe_json = canonical_json(recipe)
    if recipe_json is not None and current_recipe_json != recipe_json:
        raise ValueError(
            "source_binding.recipe_json must match the retained source records; got source records changed from their recipe"
        )
    return _SourceRecipeAdmission(current_recipe_json, admitted, inventory, values)


def _source_owner_references(closure: Mapping[str, Any], /) -> tuple[object, ...]:
    """Retain container types and owners beyond recipe JSON scalar encoding."""
    pending: list[object] = [closure]
    seen: set[int] = set()
    retained: list[object] = []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, (jax.Array, np.ndarray)):
            retained.append(value)
            continue
        if isinstance(value, (Mapping, tuple, list, set, frozenset, slice)):
            retained.append(type(value))
        if (
            is_dataclass(value)
            and not isinstance(value, type)
            or isinstance(value, np.generic)
            or isinstance(value, float)
            and not np.isfinite(value)
        ):
            retained.append(value)
        pending.extend(_model_owner_children(value))
    return tuple(retained)


def _issue_source_publication(
    closure: Mapping[str, Any],
    recipe_json: str,
    content_digest: str,
    bank: AdmittedMeshingSourceArrays,
    array_roles: tuple[tuple[str, MeshingArrayRole], ...],
    logical_array_names: tuple[str, ...],
    logical_content_digest: str | None,
    /,
    *,
    owner_index: int,
    array_bindings: Mapping[str, str],
    limits: ArrayArchiveLimits,
    maximum_chunk_bytes: int,
) -> AdmittedMeshingSourceArrays:
    snapshot = _SourcePublicationSnapshot(
        recipe_json,
        content_digest,
        owner_index,
        tuple(sorted(array_bindings.items())),
        array_roles,
        logical_array_names,
        logical_content_digest,
        bank._arrays,
        bank.source_validation,
        _source_owner_references(closure),
        limits,
        maximum_chunk_bytes,
    )
    token = object.__new__(_SourcePublicationAdmission)
    object.__setattr__(token, "snapshot", snapshot)
    object.__setattr__(token, "process_id", os.getpid())
    scope = _source_restore_admission_scope.get()
    object.__setattr__(token, "restore_scope", None if scope is None else ref(scope))
    _issued_source_publications.add(token)
    retained = AdmittedMeshingSourceArrays(
        bank,
        maximum_chunk_bytes=bank.maximum_chunk_bytes,
        source_validation=bank.source_validation,
    )
    retained._publication_slot.admission = token
    return retained


@checked
def prepare_meshing_source_binding(
    closure: Mapping[str, Any] | None,
    recipe_json: str | None,
    content_digest: str | None,
    /,
    *,
    owner_index: int,
    array_bindings: Mapping[str, str],
    logical_arrays: Mapping[str, jax.Array] | None,
    logical_array_names: Sequence[str],
    logical_content_digest: str | None,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    maximum_chunk_bytes: int = 1 << 20,
) -> MeshingSourceBinding:
    if isinstance(closure, MeshingSourceClosure):
        closure = closure.to_records()
    if closure is None and recipe_json is None:
        if (
            content_digest is not None
            or array_bindings
            or logical_arrays
            or logical_array_names
            or logical_content_digest is not None
        ):
            raise ValueError(
                "source_binding.recipe_json must accompany source numerical content; got None"
            )
        return MeshingSourceBinding(None, None, (), (), None, None)

    register_meshing_source_artifacts()
    admission: _SourceRecipeAdmission | None = None
    if closure is not None:
        admission = _admit_live_source_recipe(
            closure, recipe_json, logical_arrays, limits, maximum_chunk_bytes
        )
        recipe_json = admission.recipe_json
        logical_arrays = admission.logical_arrays
    if not isinstance(recipe_json, str):
        raise TypeError(
            f"source_binding.recipe_json must contain JSON text; got {type(recipe_json).__name__}"
        )
    recipe = meshing_source_recipe(recipe_json, limits=limits)
    inventory = (
        admission.inventory
        if admission is not None
        else model_recipe_array_inventory(
            recipe, prefix=MESHING_SOURCE_PREFIX, limits=limits
        )
    )
    bank = _validate_meshing_source_logical_bank(
        inventory,
        array_bindings,
        logical_arrays,
        logical_array_names,
        logical_content_digest,
        maximum_chunk_bytes=maximum_chunk_bytes,
    )
    names = bank.names
    logical_content_digest = bank.content_digest
    logical_arrays = bank.arrays
    if admission is not None:
        values = admission.values
        actual_digest = meshing_source_content_digest(
            values,
            array_bindings,
            logical_arrays,
            logical_content_digest,
            maximum_chunk_bytes=maximum_chunk_bytes,
            owner_index=owner_index,
            array_paths={entry.name: entry.path for entry in inventory},
        )
        if content_digest is not None and content_digest != actual_digest:
            raise ValueError(
                "source_binding.content_digest must match the retained closure arrays; got source numerical content changed"
            )
        content_digest = actual_digest
    if not isinstance(content_digest, str):
        raise TypeError(
            f"source_binding.content_digest must identify the closure numerical content; got {type(content_digest).__name__}"
        )
    digest(content_digest, "source_content_digest")
    roles: tuple[tuple[str, MeshingArrayRole], ...] = tuple(
        sorted(
            {
                (
                    source_array_name(entry.name, owner_index, array_bindings),
                    "source-record",
                )
                for entry in inventory
            }
            | {(source_logical_array_name(name), "source-record") for name in names}
        )
    )
    if admission is not None and closure is not None and logical_arrays is not None:
        logical_arrays = _issue_source_publication(
            closure,
            recipe_json,
            content_digest,
            logical_arrays,
            roles,
            names,
            logical_content_digest,
            owner_index=owner_index,
            array_bindings=array_bindings,
            limits=limits,
            maximum_chunk_bytes=maximum_chunk_bytes,
        )
    elif (
        logical_arrays is not None
        and logical_arrays._publication_slot.admission is not None
    ):
        logical_arrays = AdmittedMeshingSourceArrays(
            logical_arrays,
            maximum_chunk_bytes=logical_arrays.maximum_chunk_bytes,
            source_validation=logical_arrays.source_validation,
        )
    return MeshingSourceBinding(
        recipe_json,
        content_digest,
        roles,
        names,
        logical_content_digest,
        logical_arrays,
        admission.values if admission is not None else None,
    )


def _source_publication_snapshot(
    logical_arrays: Mapping[str, jax.Array] | None,
    owner_index: int,
    array_bindings: Mapping[str, str],
    limits: ArrayArchiveLimits,
    maximum_chunk_bytes: int,
    /,
) -> _SourcePublicationSnapshot | None:
    if type(logical_arrays) is not AdmittedMeshingSourceArrays:
        return None
    token = logical_arrays._publication_slot.admission
    if (
        type(token) is not _SourcePublicationAdmission
        or token not in _issued_source_publications
        or token.process_id != os.getpid()
    ):
        return None
    active_scope = _source_restore_admission_scope.get()
    if active_scope is not None and active_scope.process_id != os.getpid():
        return None
    if token.restore_scope is None:
        if active_scope is not None:
            return None
    elif active_scope is None or token.restore_scope() is not active_scope:
        return None
    snapshot = token.snapshot
    if (
        snapshot.arrays is not logical_arrays._arrays
        or snapshot.source_validation is not logical_arrays.source_validation
        or snapshot.owner_index != owner_index
        or snapshot.array_bindings != tuple(sorted(array_bindings.items()))
        or snapshot.limits != limits
        or snapshot.maximum_chunk_bytes != maximum_chunk_bytes
    ):
        return None
    return snapshot


def _source_publication_content_matches(
    snapshot: _SourcePublicationSnapshot,
    closure: Mapping[str, Any],
    recipe_json: str | None,
    content_digest: str | None,
    current_recipe_json: str,
    current_digest: str,
    bank: _LogicalBankBinding,
    /,
) -> bool:
    references = _source_owner_references(closure)
    return (
        current_recipe_json == snapshot.recipe_json
        and (recipe_json is None or recipe_json == current_recipe_json)
        and current_digest == snapshot.content_digest
        and (content_digest is None or content_digest == current_digest)
        and bank.names == snapshot.logical_array_names
        and bank.content_digest == snapshot.logical_content_digest
        and len(references) == len(snapshot.owner_references)
        and all(
            current is retained
            for current, retained in zip(references, snapshot.owner_references)
        )
    )


def _reuse_source_publication(
    closure: Mapping[str, Any],
    recipe_json: str | None,
    content_digest: str | None,
    /,
    *,
    owner_index: int,
    array_bindings: Mapping[str, str],
    logical_arrays: Mapping[str, jax.Array] | None,
    logical_array_names: Sequence[str],
    logical_content_digest: str | None,
    limits: ArrayArchiveLimits,
    maximum_chunk_bytes: int,
) -> MeshingSourceBinding | None:
    if type(logical_arrays) is not AdmittedMeshingSourceArrays:
        return None
    # Every use admits the current catalog and hashes all live local buffers.
    # Expected admission failures defer to the full validator's error ordering.
    try:
        snapshot = _source_publication_snapshot(
            logical_arrays, owner_index, array_bindings, limits, maximum_chunk_bytes
        )
        if snapshot is None:
            return None
        recipe, inventory, values = prepare_model_recipe_catalog(
            closure, prefix=MESHING_SOURCE_PREFIX, limits=limits
        )
        current_recipe_json = canonical_json(recipe)
        meshing_source_recipe(current_recipe_json, limits=limits)
        bank = _validate_meshing_source_logical_bank(
            inventory,
            array_bindings,
            logical_arrays,
            logical_array_names,
            logical_content_digest,
            maximum_chunk_bytes=maximum_chunk_bytes,
        )
        if logical_arrays.source_validation is not None:
            logical_arrays.source_validation.require_binding(
                logical_arrays._arrays, logical_arrays.content_digest
            )
        current_digest = meshing_source_content_digest(
            values,
            array_bindings,
            logical_arrays,
            bank.content_digest,
            maximum_chunk_bytes=maximum_chunk_bytes,
            owner_index=owner_index,
            array_paths={entry.name: entry.path for entry in inventory},
        )
    except (TypeError, ValueError, RepositoryCorruptionError):
        return None
    if not _source_publication_content_matches(
        snapshot,
        closure,
        recipe_json,
        content_digest,
        current_recipe_json,
        current_digest,
        bank,
    ):
        return None
    return MeshingSourceBinding(
        current_recipe_json,
        current_digest,
        snapshot.array_roles,
        bank.names,
        bank.content_digest,
        logical_arrays,
        values,
    )


@checked
def authenticate_meshing_source_publication(
    closure: Mapping[str, Any] | None,
    recipe_json: str | None,
    content_digest: str | None,
    /,
    *,
    owner_index: int,
    array_bindings: Mapping[str, str],
    logical_arrays: Mapping[str, jax.Array] | None,
    logical_array_names: Sequence[str],
    logical_content_digest: str | None,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    maximum_chunk_bytes: int = 1 << 20,
) -> MeshingSourceBinding:
    """Authenticate live source content or perform full source admission."""
    records = (
        closure.to_records() if isinstance(closure, MeshingSourceClosure) else closure
    )
    if records is not None:
        reused = _reuse_source_publication(
            records,
            recipe_json,
            content_digest,
            owner_index=owner_index,
            array_bindings=array_bindings,
            logical_arrays=logical_arrays,
            logical_array_names=logical_array_names,
            logical_content_digest=logical_content_digest,
            limits=limits,
            maximum_chunk_bytes=maximum_chunk_bytes,
        )
        if reused is not None:
            return reused
    return prepare_meshing_source_binding(
        closure,
        recipe_json,
        content_digest,
        owner_index=owner_index,
        array_bindings=array_bindings,
        logical_arrays=logical_arrays,
        logical_array_names=logical_array_names,
        logical_content_digest=logical_content_digest,
        limits=limits,
        maximum_chunk_bytes=maximum_chunk_bytes,
    )


__all__ = ("MeshingArrayRole", "MeshingSourceBinding")
