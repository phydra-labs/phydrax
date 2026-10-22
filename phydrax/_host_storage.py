#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Declared host-owner storage bounds independent of native execution handles."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping
from dataclasses import is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from weakref import ReferenceType

import numpy as np
from jax import Array, Device, ShapeDtypeStruct
from jax.experimental.layout import Format, Layout
from jax.extend.core import Jaxpr
from jax.sharding import (
    AbstractMesh,
    ManualAxisType,
    Mesh,
    NamedSharding,
    PartitionSpec,
    SingleDeviceSharding,
)
from jax.tree_util import PyTreeDef


def immutable_host_array_layout(array: np.ndarray, /) -> tuple[object, ...] | None:
    """Bind a plain contiguous numeric view to its actual immutable bytes owner.

    A read-only flag, an owning ndarray, and a read-only view of mutable storage
    are not immutable. The returned layout retains the bytes owner as well as
    its identity; callers must also retain and check the exact ndarray owner.
    No content scan is needed to recheck this process-local layout.
    """
    if (
        type(array) is not np.ndarray
        or array.dtype.hasobject
        or array.dtype.kind not in "biufc"
        or not array.flags.c_contiguous
    ):
        return None
    owner: object = array
    while type(owner) is np.ndarray:
        if owner.flags.writeable:
            return None
        owner = owner.base
    if type(owner) is not bytes:
        return None
    return (
        array.dtype,
        array.shape,
        array.strides,
        array.__array_interface__["data"][0],
        array.nbytes,
        id(owner),
        owner,
    )


def immutable_host_array(array: np.ndarray, /) -> np.ndarray:
    """Snapshot a supported host bank once without changing its logical values.

    Existing byte-backed banks are shared only after proving their backing.
    Other numeric banks receive private immutable bytes in logical C order.
    Unsupported arrays, including object-valued failure evidence, are left to
    their existing owner policy rather than reinterpreted as numerical bytes.
    """
    if (
        type(array) is not np.ndarray
        or array.dtype.hasobject
        or array.dtype.kind not in "biufc"
    ):
        return array
    if immutable_host_array_layout(array) is not None:
        return array
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def declared_integer_storage_bytes(bit_length: int, /) -> int:
    """Bound integer storage using the declared bit length and fixed metadata."""
    if bit_length < 0:
        raise ValueError(f"integer.bit_length must be nonnegative; got {bit_length}.")
    return 128 + 4 * ((bit_length + 7) // 8)


def declared_object_storage_bytes(value: object, /) -> int:
    """Bound structural metadata without interpreter object-size introspection."""
    if isinstance(value, PyTreeDef):
        # Public tree nodes own structural descriptors independently of their
        # auxiliary numeric banks, which are traversed below.
        return 256 * value.num_nodes
    match value:
        case int():
            return declared_integer_storage_bytes(value.bit_length())
        case str():
            return 256 + 4 * len(value)
        case bytes() | bytearray():
            return 256 + len(value)
        case Mapping():
            return 256 + 64 * len(value)
        case tuple() | list() | set() | frozenset():
            return 256 + 16 * len(value)
        case np.generic():
            return 256 + value.dtype.itemsize
        case _:
            return 256


def retained_host_storage_fanout_upper(owner: object, /) -> int:
    """Bound a profile owner's expansion before its child bank is materialized."""
    fields = getattr(type(owner), "__dataclass_fields__", None)
    if isinstance(owner, Mapping):
        return 2 * len(owner)
    if isinstance(owner, (tuple, list, set, frozenset)):
        return len(owner)
    if fields is not None:
        return len(fields)
    if isinstance(owner, PyTreeDef):
        return owner.num_nodes + 2
    if isinstance(owner, Jaxpr):
        return len(owner.consts)
    if isinstance(owner, ShapeDtypeStruct):
        return len(owner.__slots__)
    if isinstance(owner, np.ndarray):
        return int(owner.base is not None) + (owner.size if owner.dtype.hasobject else 0)
    if isinstance(owner, (Fraction, Enum)):
        return 2
    if owner is None or owner is Ellipsis or isinstance(
        owner,
        (Array, np.generic, str, bytes, bytearray, bool, int, float, complex, np.dtype, type),
    ):
        return 0
    # Supported special owners (sharding/layout, Mesh, FormType/Spec,
    # memoryview and auxiliary namespaces) expand to at most eight children.
    # Unsupported owners still refuse in the canonical profiler.
    return 8


def retained_host_storage_profile(
    value: object,
    existing: Mapping[int, object],
    native_buffer: Callable[[ctypes.Array], bool],
    /,
    *,
    admit_owner: Callable[[object], None] | None = None,
) -> tuple[dict[int, object], int, int, int]:
    """Return new retained owners, declared bytes, logical JAX bytes and visits.

    Views share their payload owner's bound. Fixed structural allowances and
    integer bit lengths describe host storage; JAX bytes describe logical banks,
    not device or compiler allocation. Native-backed payloads are excluded.
    A cyclic retention graph or unsupported dynamic owner refuses explicitly.
    ``existing`` owns previously profiled graphs with strong references;
    identity alone without that retention lifetime cannot earn a profile reuse.
    """
    identifier = id(value)
    if identifier in existing and existing[identifier] is value:
        return {}, 0, 0, 0
    # Model structure imports native capabilities; resolve that cycle only when
    # the fully imported package requests host preparation.
    from ._model._structure import _model_owner_children
    from .exterior._form_type import FormType, FormValueSpec

    pending: list[tuple[object, bool, bool, bool]] = [(value, False, False, False)]
    visiting: set[int] = set()
    retained: dict[int, object] = {}
    declared = 0
    logical = 0
    visits = 0
    while pending:
        owner, leaving, device_bank, tree_aux = pending.pop()
        identifier = id(owner)
        if leaving:
            visiting.remove(identifier)
            retained[identifier] = owner
            continue
        if (identifier in existing and existing[identifier] is owner) or (
            identifier in retained and retained[identifier] is owner
        ):
            continue
        if identifier in visiting:
            raise ValueError(
                f"retained_host_storage_profile.owner must form an acyclic graph; got revisited {type(owner).__qualname__}."
            )
        if admit_owner is not None:
            admit_owner(owner)
        visits += 1
        declared += declared_object_storage_bytes(owner)
        children: tuple[object, ...] = ()
        match owner:
            case ShapeDtypeStruct():
                # Traverse stored abstract metadata, not the derived .sharding
                # property (which may resolve a PartitionSpec via ambient mesh)
                # or fictitious numerical storage for the described shape.
                children = tuple(
                    getattr(owner, name) for name in ShapeDtypeStruct.__slots__
                )
            case NamedSharding():
                children = (
                    owner.mesh, owner.spec, owner.memory_kind,
                    owner._logical_device_ids,
                )
            case SingleDeviceSharding():
                children = (owner._device, owner.memory_kind)
            case AbstractMesh():
                children = (vars(owner),)
            case PartitionSpec():
                children = (
                    owner.partitions, owner.unreduced, owner.reduced,
                    owner.unreduced_kind,
                )
            case ManualAxisType():
                children = (
                    owner.varying, owner.unreduced, owner.reduced,
                    owner.unreduced_kind,
                )
            case Format():
                children = (owner.layout, owner.sharding)
            case Layout():
                children = (vars(owner),)
            case _ if owner is PartitionSpec.UNCONSTRAINED:
                children = (vars(owner),)
            case Mesh():
                visiting.add(identifier)
                pending.append((owner, True, False, False))
                pending.append((owner.devices, False, True, False))
                pending.append((owner.axis_names, False, False, False))
                pending.append((owner.axis_types, False, False, False))
                continue
            case Device():
                pass
            case Jaxpr():
                children = tuple(owner.consts)
            case PyTreeDef():
                # node_data includes custom-node auxiliary metadata (including
                # Equinox static fields); children() covers every subtree.
                # Never treat this compiler-adjacent object as an opaque atom.
                visiting.add(identifier)
                pending.append((owner, True, False, tree_aux))
                pending.append((owner.node_data(), False, False, True))
                pending.extend(
                    (child, False, False, True) for child in owner.children()
                )
                continue
            case Array():
                declared += owner.nbytes
                logical += owner.nbytes
            case np.ndarray():
                if owner.dtype.hasobject:
                    if not device_bank:
                        raise TypeError(
                            f"retained_host_storage_profile.owner.dtype must exclude objects outside Mesh device topology; got {owner.dtype}."
                        )
                    for device in owner.flat:
                        match device:
                            case Device():
                                pending.append((device, False, False, False))
                            case _:
                                raise TypeError(
                                    f"retained_host_storage_profile.device must be a JAX Device; got {type(device).__qualname__}."
                                )
                    declared += owner.nbytes if owner.base is None else 0
                    if owner.base is not None:
                        pending.append((owner.base, False, True, False))
                elif owner.base is not None:
                    children = (owner.base,)
                else:
                    declared += owner.nbytes
            case memoryview():
                children = (owner.obj,)
            case ctypes.Array():
                if not native_buffer(owner):
                    declared += ctypes.sizeof(owner)
            case Fraction():
                children = (owner.numerator, owner.denominator)
            case slice():
                children = (owner.start, owner.stop, owner.step)
            case ReferenceType() if type(owner) is ReferenceType:
                # Weak edges own their shell and callback, never their referent.
                # Callback storage follows the ordinary strict owner policy.
                children = (() if owner.__callback__ is None else (owner.__callback__,))
            case Enum():
                children = (owner.value, owner.name)
            case np.generic():
                if owner.dtype.hasobject:
                    raise TypeError(
                        f"retained_host_storage_profile.owner.dtype must exclude objects; got {owner.dtype}."
                    )
            case FormType():
                children = (
                    owner.dimension,
                    owner.degree,
                    owner.twist,
                    owner.fiber_shape,
                    owner.ambient_dimension,
                    owner.form_type_id,
                )
            case FormValueSpec():
                children = (owner.form_type, owner.proxy, owner.value_spec_id)
            case _ if is_dataclass(owner) or isinstance(
                owner, (Mapping, tuple, list, set, frozenset)
            ):
                children = tuple(_model_owner_children(owner))
            case (
                None
                | str()
                | bytes()
                | bytearray()
                | bool()
                | int()
                | float()
                | complex()
                | np.dtype()
                | Path()
                | type()
            ):
                pass
            case _ if owner is Ellipsis:
                pass
            case _ if tree_aux and not callable(owner) and hasattr(owner, "__dict__"):
                # Public custom-node aux may contain ordinary instances such
                # as Equinox's missing-field sentinel. Inspect the whole owned
                # namespace, including any arrays, rather than trusting a
                # private type name or an assumed empty metadata object.
                children = (vars(owner),)
            case _:
                raise TypeError(
                    f"retained_host_storage_profile.owner must have a supported storage representation; got {type(owner).__qualname__}."
                )
        if children:
            visiting.add(identifier)
            pending.append((owner, True, False, tree_aux))
            pending.extend((child, False, False, tree_aux) for child in children)
        else:
            retained[identifier] = owner
    return retained, declared, logical, visits
