#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import dataclasses
import enum
import functools
import hashlib
import sys
from collections.abc import Mapping, Sequence
from inspect import getclosurevars
from types import CodeType, FunctionType, ModuleType
from typing import Any

import equinox as eqx
import jax
import numpy as np
from jax import core as jax_public_core
from jax.extend import core as jax_core

from ._fingerprint import canonical_fingerprint, canonical_json
from ._strict import StrictModule
from ._trainable import _global_reads, _hidden_arrays


RecordInput = Mapping[str, Any] | Sequence[tuple[str, Any]]
_ARRAY_TYPES = (jax.Array, jax.ShapeDtypeStruct, np.ndarray)
_AUTO_COMPILER_LAYOUT = jax_public_core.ShapedArray((), np.dtype("float32")).layout


def _type_id(value: Any, /) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _named_records(value: RecordInput, name: str, /) -> tuple[tuple[str, Any], ...]:
    raw = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    records: list[tuple[str, Any]] = []
    for record in raw:
        if not isinstance(record, (tuple, list)) or len(record) != 2:
            raise TypeError(f"{name} must be a mapping or a sequence of pairs.")
        key = _identifier(record[0], f"{name} key")
        records.append((key, record[1]))
    keys = tuple(key for key, _ in records)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} keys must be unique.")
    return tuple(sorted(records, key=lambda record: record[0]))


def _array_payload(value: Any, path: str, /) -> dict[str, Any]:
    if isinstance(value, jax.ShapeDtypeStruct):
        raise TypeError(f"Numeric identity cannot realize abstract array {path}.")
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError(f"Numeric identity cannot encode object-dtype array {path}.")
    return {
        "kind": "array",
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "storage_bytes": array.nbytes,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _compiler_fingerprint(payload: Any, /) -> str:
    # Compiler-owned records are already JSON-normalized: arrays, enums,
    # abstract types and programs have explicit payloads. Rewalking them with
    # the general array-aware normalizer would copy every large IR record.
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _compiler_sharding_payload(value: Any, path: str, /) -> Any:
    if value is None:
        return None
    # The unplaced, empty abstract mesh is JAX's ordinary ShapedArray default.
    # Device/mesh placement is executable state: never silently erase it.
    if (
        isinstance(value, jax.sharding.NamedSharding)
        and isinstance(value.mesh, jax.sharding.AbstractMesh)
        and value.mesh.empty
    ):
        spec = value.spec
        if spec.unreduced or spec.reduced or spec.unreduced_kind is not None:
            raise TypeError(f"Unsupported partition metadata at {path}.")
        return {
            "kind": "unplaced-named-sharding",
            "partitions": _static_payload(tuple(spec), f"{path}.partitions"),
            "memory_kind": value.memory_kind,
        }
    raise TypeError(
        f"Unsupported compiler sharding at {path}; placement needs an explicit identity."
    )


def _manual_axes_payload(value: jax.sharding.ManualAxisType | None, path: str, /) -> Any:
    if value is None:
        return None
    return {
        name: sorted(
            (_static_payload(axis, f"{path}.{name}") for axis in axes),
            key=canonical_json,
        )
        for name, axes in (
            ("varying", value.varying),
            ("unreduced", value.unreduced),
            ("reduced", value.reduced),
        )
    } | {
        "unreduced_kind": _static_payload(value.unreduced_kind, f"{path}.unreduced_kind")
    }


def _abstract_array_payload(value: Any, path: str, /) -> dict[str, Any]:
    if isinstance(value, jax.ShapeDtypeStruct):
        if value.format.layout is not None or value.is_ref:
            raise TypeError(f"Unsupported compiler layout or reference at {path}.")
        extra = {
            "manual_axes": _manual_axes_payload(value.manual_axis_type, path),
            # This JAX constructor field has no public accessor. Keep its
            # explicit owner adapter here rather than erasing memory placement.
            "memory_space": _static_payload(value._memory_space, f"{path}.memory_space"),
        }
    elif type(value) is jax_public_core.ShapedArray:
        default_layout = _AUTO_COMPILER_LAYOUT
        if value.layout is not default_layout:
            raise TypeError(f"Unsupported compiler layout at {path}.")
        extra = {
            "manual_axes": _manual_axes_payload(value.manual_axis_type, path),
            "memory_space": _static_payload(value.memory_space, f"{path}.memory_space"),
        }
    else:
        raise TypeError(
            f"Unsupported compiler abstract type at {path}: {_type_id(value)}."
        )
    if any(type(dimension) is not int for dimension in value.shape):
        raise TypeError(
            f"Symbolic compiler dimensions at {path} need an explicit identity."
        )
    return {
        "kind": "abstract-array",
        "type": _type_id(value),
        "shape": list(value.shape),
        "dtype": np.dtype(value.dtype).str,
        "weak_type": bool(value.weak_type),
        "sharding": _compiler_sharding_payload(value.sharding, f"{path}.sharding"),
        **extra,
    }


class _CompilerPayload:
    """One content-addressed compiler DAG; object keys never enter its payload."""

    def __init__(self) -> None:
        self.memo: dict[tuple[Any, bool, tuple[int, ...]], str] = {}
        self.programs: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.metadata_memo: dict[tuple[int, tuple[int, ...]], tuple[Any, Any]] = {}
        self.metadata_values: dict[Any, Any] = {}
        self.metadata_active: set[tuple[int, tuple[int, ...]]] = set()
        self.array_memo: dict[int, tuple[Any, Any]] = {}
        self.code_memo: dict[CodeType, str] = {}
        self.abstract_memo: dict[tuple[Any, ...], Any] = {}
        self.function_payloads: dict[tuple[FunctionType, tuple[int, ...]], Any] = {}
        self.callback_scope: list[Any] = []
        self.rules: dict[str, Any] = {}
        self.primitive_memo: dict[jax_core.Primitive, Any] = {}
        self.primitives = {
            primitive: name
            for name, primitive in sorted(vars(jax_core.primitives).items(), reverse=True)
            if isinstance(primitive, jax_core.Primitive)
        }
        for module in (jax.lax, jax.lax.linalg):
            self.primitives.update(
                {
                    primitive: f"{module.__name__}.{name}"
                    for name, primitive in sorted(vars(module).items(), reverse=True)
                    if not name.startswith("_")
                    and isinstance(primitive, jax_core.Primitive)
                }
            )
        from jax._src.callback import pure_callback_p
        from jax._src.lax.control_flow.conditionals import platform_index_p

        self.primitives[pure_callback_p] = "jax.pure_callback"
        from equinox.internal import unvmap_all_p, unvmap_any_p, unvmap_max_p

        self.primitives.update(
            {
                platform_index_p: "jax._src.lax.control_flow.conditionals.platform_index_p",
                unvmap_all_p: "equinox.internal.unvmap_all_p",
                unvmap_any_p: "equinox.internal.unvmap_any_p",
                unvmap_max_p: "equinox.internal.unvmap_max_p",
            }
        )

    @staticmethod
    def payload_key(value: Any, /) -> Any:
        """Freeze an already-owned JSON record without canonical/hash walks."""
        value_type = type(value)
        if value_type is dict:
            return (
                "mapping",
                tuple(
                    (key, _CompilerPayload.payload_key(item))
                    for key, item in sorted(value.items())
                ),
            )
        if value_type in (list, tuple):
            return (
                value_type,
                tuple(_CompilerPayload.payload_key(item) for item in value),
            )
        if value_type is float:
            return (float, value.hex())
        return (value_type, value)

    def callback_reference(self, value: Any, /) -> Any:
        """Reference an enclosing declared callback binder, never an object ID."""
        for depth, owner in enumerate(reversed(self.callback_scope)):
            if value is owner:
                return {"kind": "recursive-callback", "depth": depth}
        return None

    def callback_scope_key(self) -> tuple[int, ...]:
        return tuple(id(owner) for owner in self.callback_scope)

    def primitive_reference(self, primitive: jax_core.Primitive, path: str, /) -> Any:
        previous = self.primitive_memo.get(primitive)
        if previous is not None:
            return previous
        binding = self.primitives.get(primitive)
        if binding is None:
            from .nonlinear._domain_cond import _domain_call

            if primitive is not _domain_call:
                raise TypeError(
                    f"Foreign compiler primitive at {path}: "
                    f"{_type_id(primitive)} ({primitive.name}) needs an explicit identity."
                )
            binding = "phydrax.nonlinear._domain_cond._domain_call"
        payload = {
            "kind": "native-primitive",
            "binding": binding,
            "name": primitive.name,
            "multiple_results": primitive.multiple_results,
        }
        primitive_id = _compiler_fingerprint(payload)
        reference = {"metadata_id": primitive_id}
        self.metadata[primitive_id] = payload
        self.primitive_memo[primitive] = reference
        return reference

    def parameter(self, value: Any, path: str, /) -> Any:
        recursive = self.callback_reference(value)
        if recursive is not None:
            return recursive
        key = (id(value), self.callback_scope_key())
        previous = self.metadata_memo.get(key)
        if previous is not None:
            return previous[1]
        if value is None or isinstance(
            value,
            (
                str,
                bool,
                int,
                float,
                complex,
                np.generic,
                np.dtype,
                enum.Enum,
                type,
                bytes,
            ),
        ):
            payload = self.parameter_payload(value, path)
            self.metadata_memo[key] = (value, payload)
            return payload
        abstract_key = None
        if isinstance(value, (jax.ShapeDtypeStruct, jax_public_core.ShapedArray)):
            axes = value.manual_axis_type
            axes_key = (
                None
                if axes is None
                else (axes.varying, axes.unreduced, axes.reduced, axes.unreduced_kind)
            )
            placement = (
                (value._memory_space, value._dll, bool(value.is_ref))
                if isinstance(value, jax.ShapeDtypeStruct)
                else (value.memory_space, value.layout)
            )
            abstract_key = (
                type(value),
                tuple((type(dimension), dimension) for dimension in value.shape),
                value.dtype,
                bool(value.weak_type),
                value.sharding,
                axes_key,
                placement,
            )
            previous_abstract = self.abstract_memo.get(abstract_key)
            if previous_abstract is not None:
                self.metadata_memo[key] = (value, previous_abstract)
                return previous_abstract
        if key in self.metadata_active:
            raise TypeError(
                f"Recursive compiler metadata at {path} needs an explicit graph owner."
            )
        self.metadata_active.add(key)
        try:
            payload = self.parameter_payload(value, path)
        finally:
            self.metadata_active.remove(key)
        semantic_key = self.payload_key(payload)
        reference = self.metadata_values.get(semantic_key)
        if reference is None:
            if (
                isinstance(value, (Mapping, tuple, list, set, frozenset))
                and len(value) < 32
            ):
                # Tiny primitive parameter records are cheap inline. Their
                # schemas/captures/programs already refer to shared graph nodes.
                reference = payload
            else:
                fingerprint = _compiler_fingerprint(payload)
                reference = {"metadata_id": fingerprint}
                self.metadata[fingerprint] = payload
            self.metadata_values[semantic_key] = reference
        # Keep the source alive so visitation addresses cannot be recycled.
        self.metadata_memo[key] = (value, reference)
        if abstract_key is not None:
            self.abstract_memo[abstract_key] = reference
        return reference

    def array(self, value: Any, path: str, /) -> Any:
        key = id(value)
        previous = self.array_memo.get(key)
        if previous is not None:
            return previous[1]
        payload = _array_payload(value, path)
        semantic_key = self.payload_key(payload)
        reference = self.metadata_values.get(semantic_key)
        if reference is None:
            fingerprint = _compiler_fingerprint(payload)
            reference = {"metadata_id": fingerprint}
            self.metadata[fingerprint] = payload
            self.metadata_values[semantic_key] = reference
        self.array_memo[key] = (value, reference)
        return reference

    def parameter_payload(self, value: Any, path: str, /) -> Any:
        # JAX's explicit owner types, not general object introspection. These
        # interpreter-only adapters are localized here, like native primitives.
        if isinstance(value, jax_core.Primitive):
            return self.primitive_reference(value, path)
        from equinox._ad import _ClosureConvert, _TrivialClosureConvert
        from equinox._module._flatten import MISSING
        from jax._src.callback import _FlatCallback
        from jax._src.core import no_axis_name
        from jax._src.flattree import (
            Either,
            FTDict,
            FTFiltered,
            FTPyTree,
            FTSingleton,
            FTStatic,
            FTTuple,
        )
        from jax._src.interpreters.batching import AxisData
        from jax._src.sharding_impls import UNSPECIFIED

        if value is UNSPECIFIED:
            return {"kind": "unspecified-sharding"}
        if value is no_axis_name:
            return {"kind": "unnamed-batch-axis"}
        if value is MISSING:
            return {
                "kind": "missing-wrapper-field",
                "owner": "equinox._module._flatten.MISSING",
            }
        if isinstance(value, AxisData):
            return {
                "kind": "batch-axis",
                "name": self.parameter(value.name, f"{path}.name"),
                "size": self.parameter(value.size, f"{path}.size"),
                "spmd_name": self.parameter(value.spmd_name, f"{path}.spmd_name"),
                # Preserve stored mesh axes, not a context-dependent property.
                "explicit_mesh_axis": self.parameter(
                    value._ema, f"{path}.explicit_mesh_axis"
                ),
            }
        if isinstance(value, _ClosureConvert):
            # A static transformation argument binds this callable, including
            # its current owner-held captures. Jaxpr.consts may still refer to
            # the original trace after Equinox replaces the dynamic leaves.
            # Eqx.partition also transports non-executable fragments with a
            # missing program. Preserve that slot as missing, never invent an
            # empty executable; native combine reconstructs the callable.
            return {
                "kind": "equinox-closure-partition"
                if value.jaxpr is None
                else "bound-equinox-program",
                "type": _type_id(value),
                "program": None
                if value.jaxpr is None
                else self.program(value.jaxpr, path, dynamic_captures=True),
                "static_captures": self.parameter(value.consts, f"{path}.consts"),
                "input_dynamic_structure": self.parameter(
                    value.in_dynamic_struct, f"{path}.in_dynamic_struct"
                ),
                "output_dynamic_structure": self.parameter(
                    value.out_dynamic_struct, f"{path}.out_dynamic_struct"
                ),
                "input_static": self.parameter(value.in_static, f"{path}.in_static"),
                "output_static": self.parameter(value.out_static, f"{path}.out_static"),
            }
        if isinstance(value, (eqx.filter_custom_jvp, jax.custom_jvp)):
            if type(value) not in (eqx.filter_custom_jvp, jax.custom_jvp):
                raise TypeError(
                    f"Unsupported custom-JVP owner subclass at {path}: {_type_id(value)}."
                )
            self.callback_scope.append(value)
            try:
                if isinstance(value, eqx.filter_custom_jvp):
                    return {
                        "kind": "filtered-custom-jvp",
                        "binding_scope": "lexical-callback",
                        "function": self.parameter(value.fn, f"{path}.fn"),
                    }
                return {
                    "kind": "custom-jvp",
                    "binding_scope": "lexical-callback",
                    "function": self.parameter(value.fun, f"{path}.fun"),
                    "derivative": self.parameter(value.jvp, f"{path}.jvp"),
                    "nondiff_argnums": self.parameter(
                        value.nondiff_argnums, f"{path}.nondiff_argnums"
                    ),
                    "symbolic_zeros": value.symbolic_zeros,
                }
            finally:
                self.callback_scope.pop()
        if isinstance(value, _TrivialClosureConvert):
            return {
                "kind": "equinox-function-partition"
                if value.fn is None
                else "bound-equinox-function",
                "type": _type_id(value),
                "function": None
                if value.fn is None
                else self.parameter(value.fn, f"{path}.fn"),
                "input_dynamic_structure": self.parameter(
                    value.in_dynamic_struct, f"{path}.in_dynamic_struct"
                ),
                "input_static": self.parameter(value.in_static, f"{path}.in_static"),
            }

        if isinstance(value, (jax_core.Jaxpr, jax_core.ClosedJaxpr)):
            return self.program(value, path, dynamic_captures=False)
        if isinstance(value, (jax.ShapeDtypeStruct, jax_public_core.ShapedArray)):
            return _abstract_array_payload(value, path)
        if isinstance(value, jax_core.AbstractToken):
            return {"kind": "abstract-token"}
        if isinstance(value, jax.tree_util.PyTreeDef):
            node = value.node_data()
            return {
                "kind": "pytree-def",
                "node": None
                if node is None
                else {
                    "type": f"{node[0].__module__}.{node[0].__qualname__}",
                    "metadata": self.parameter(node[1], f"{path}.metadata"),
                },
                "children": [
                    self.parameter(child, f"{path}.children[{index}]")
                    for index, child in enumerate(value.children())
                ],
            }
        if isinstance(value, jax.sharding.Sharding):
            return _compiler_sharding_payload(value, path)
        if isinstance(value, (jax.sharding.Mesh, jax.sharding.AbstractMesh)):
            if not value.empty:
                raise TypeError(
                    f"Compiler mesh placement at {path} needs an explicit identity."
                )
            return {"kind": "empty-mesh", "type": _type_id(value)}
        if isinstance(value, _FlatCallback):
            return {
                "kind": "host-callback",
                "function": self.parameter(value.callback_func, f"{path}.function"),
                "inputs": self.parameter(value.in_tree, f"{path}.inputs"),
            }
        if isinstance(value, Either):
            return {
                "kind": "flat-either",
                "is_right": value.is_right,
                "value": self.parameter(value.val, path),
            }
        if isinstance(value, (FTFiltered, FTStatic)):
            return {"kind": _type_id(value), "value": self.parameter(value.val, path)}
        if isinstance(value, FTDict):
            return {
                "kind": "flat-dict",
                "keys": self.parameter(value.keys, path),
                "values": self.parameter(value._vals, path),
            }
        if isinstance(value, FTSingleton):
            return {"kind": "flat-singleton", "value": self.parameter(value.val, path)}
        if isinstance(value, FTTuple):
            return {"kind": "flat-tuple", "items": self.parameter(value.elts, path)}
        if isinstance(value, FTPyTree):
            return {
                "kind": "flat-pytree",
                "values": self.parameter(value.xs, path),
                "tree": self.parameter(value.tree, path),
            }
        if isinstance(value, type):
            return {"kind": "type", "name": f"{value.__module__}.{value.__qualname__}"}
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "items": [
                    [key, self.parameter(item, f"{path}.{key}")]
                    for key, item in _named_records(value, path)
                ],
            }
        if isinstance(value, (tuple, list)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "type": _type_id(value),
                "items": [
                    self.parameter(item, f"{path}[{index}]")
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, (set, frozenset)):
            return {
                "kind": "set",
                "items": sorted(
                    (self.parameter(item, path) for item in value), key=canonical_json
                ),
            }
        if isinstance(value, (jax.Array, np.ndarray)):
            return _array_payload(value, path)
        if isinstance(value, (FunctionType, functools.partial)):
            return self.bound_callable(value, path)
        if isinstance(value, StrictModule):
            return {
                "kind": "strict-module",
                "type": _type_id(value),
                "fields": [
                    [
                        field.name,
                        self.parameter(
                            object.__getattribute__(value, field.name),
                            f"{path}.{field.name}",
                        ),
                    ]
                    for field in dataclasses.fields(value)
                ],
            }
        if callable(value) and not isinstance(value, StrictModule):
            raise TypeError(
                f"Opaque compiler callable at {path}: {_type_id(value)} "
                "requires its explicit native owner adapter."
            )
        return _static_payload(value, path)

    def bound_callable(self, value: Any, path: str, /) -> Any:
        recursive = self.callback_reference(value)
        if recursive is not None:
            return recursive
        scope = self.callback_scope_key()
        if isinstance(value, functools.partial):
            self.callback_scope.append(value)
            try:
                return {
                    "kind": "partial",
                    "binding_scope": "lexical-callback",
                    "function": self.bound_callable(value.func, path),
                    "args": self.parameter(value.args, path),
                    "keywords": self.parameter(value.keywords, path),
                }
            finally:
                self.callback_scope.pop()
        if not isinstance(value, FunctionType):
            raise TypeError(
                f"Opaque compiler callback at {path} needs an explicit identity."
            )
        key = (value, scope)
        if key in self.function_payloads:
            return self.function_payloads[key]
        self.callback_scope.append(value)
        try:
            payload = self.bound_function(value, path)
        finally:
            self.callback_scope.pop()
        self.function_payloads[key] = payload
        return payload

    def bound_function(self, value: FunctionType, path: str, /) -> Any:
        code = value.__code__
        if code not in self.code_memo:
            self.code_memo[code] = _compiler_fingerprint(_code_payload(code, path))
        return {
            "kind": "bound-function",
            "binding_scope": "lexical-callback",
            "module": value.__module__,
            "qualname": value.__qualname__,
            "code": self.code_memo[code],
            "defaults": self.parameter(value.__defaults__, path),
            "keyword_defaults": self.parameter(value.__kwdefaults__, path),
            "closure": [
                [name, self.parameter(cell.cell_contents, f"{path}.{name}")]
                for name, cell in zip(
                    value.__code__.co_freevars, value.__closure__ or (), strict=True
                )
            ],
            "globals": [
                [
                    name,
                    (
                        _global_payload(referenced, f"{path}.globals.{name}")
                        if isinstance(referenced, (ModuleType, type))
                        else self.parameter(referenced, f"{path}.globals.{name}")
                    ),
                ]
                for name, referenced in _global_reads(value, include_imported=True)
            ],
        }

    def rule_owner(self, thunk: Any, path: str, /) -> Any:
        # Unwrap only JAX's owned memoization/thunk protocol. The cache and
        # live tracing context are interpreter state, not executable constants.
        if (
            thunk.f.__module__ != "jax._src.interpreters.partial_eval"
            or thunk.f.__qualname__ != "_memoize.<locals>.memoized"
        ):
            raise TypeError(f"Unsupported custom-JVP thunk at {path}.")
        function = getclosurevars(thunk.f).nonlocals["fn"]
        return getclosurevars(function).nonlocals["jvp"]

    def rule_selector(self, owner: Any, path: str, /) -> tuple[Any, str]:
        from jax._src.api_util import _prepend_static_args
        from jax._src.custom_derivatives import _flatten_jvp
        from jax._src.interpreters.batching import batch_custom_jvp_subtrace
        from jax._src.util import Unhashable

        transformations = []
        for generator, arguments in owner.transforms:
            if generator is _flatten_jvp.args[0]:
                # The last argument is an auxiliary output-tree store.
                arguments = arguments[:-1]
            elif generator is _prepend_static_args.args[0]:
                arguments = tuple(
                    tuple(
                        item.val if isinstance(item, Unhashable) else item
                        for item in group
                    )
                    for group in arguments
                )
            elif generator is batch_custom_jvp_subtrace.args[0]:
                tag, axis_data, input_dimensions = arguments
                if not isinstance(tag, jax_core.TraceTag):
                    raise TypeError(f"Unsupported custom-JVP batching tag at {path}.")
                arguments = (axis_data, input_dimensions)
            else:
                raise TypeError(f"Unsupported custom-JVP transformation at {path}.")
            transformations.append(
                (
                    generator.__module__,
                    generator.__qualname__,
                    self.parameter(arguments, path),
                )
            )
        source = owner.f
        lifted = None
        if (
            source.__module__ == "jax._src.custom_derivatives"
            and source.__qualname__ == "lift_jvp.<locals>.jvp"
        ):
            captures = getclosurevars(source).nonlocals
            source, selector = self.rule_selector(
                self.rule_owner(captures["jvp_jaxpr_fun"], path), path
            )
            lifted = {"num_consts": captures["num_consts"], "selector": selector}
        return source, _compiler_fingerprint(
            {
                "transformations": transformations,
                "parameters": self.parameter(owner.params, path),
                "input_type": self.parameter(owner.in_type, path),
                "lifted": lifted,
            }
        )

    def program(self, value: Any, path: str, /, *, dynamic_captures: bool) -> Any:
        key = (value, dynamic_captures, self.callback_scope_key())
        if key in self.memo:
            return {"program_id": self.memo[key]}
        # In JAX 0.11 Jaxpr and ClosedJaxpr are the same public class, with
        # attached consts. The context, not this alias, determines binding.
        program = value.jaxpr if isinstance(value, jax_core.ClosedJaxpr) else value
        if program.effects:
            raise TypeError(
                f"Effectful compiler program at {path} needs an explicit effect identity."
            )
        variables: dict[jax_core.Var, int] = {}
        variable_types: list[Any] = []

        def variable(var: jax_core.Var, /) -> Any:
            if isinstance(var, jax_core.DropVar):
                return {"kind": "drop", "abstract_type": self.parameter(var.aval, path)}
            if var not in variables:
                variables[var] = len(variables)
                variable_types.append(self.parameter(var.aval, path))
            return variables[var]

        def atom(item: Any, /) -> Any:
            if isinstance(item, jax_core.Literal):
                return {
                    "kind": "literal",
                    "abstract_type": self.parameter(item.aval, path),
                    "value": self.array(item.val, path),
                }
            if isinstance(item, jax_core.Var):
                return variable(item)
            raise TypeError(f"Unsupported compiler atom at {path}.")

        captures = [variable(var) for var in program.constvars]
        inputs = [variable(var) for var in program.invars]
        equations = []
        for index, equation in enumerate(program.eqns):
            equation_path = f"{path}.equations[{index}]"
            if equation.effects:
                raise TypeError(
                    f"Effectful compiler equation at {equation_path} needs an explicit effect identity."
                )
            primitive = equation.primitive
            primitive_reference = self.primitive_reference(primitive, equation_path)
            params = equation.params
            if primitive is jax_core.primitives.custom_jvp_call_p:
                params = dict(params)
                thunk = params.pop("jvp_jaxpr_fun")
                owner = self.rule_owner(thunk, equation_path)
                source, selector = self.rule_selector(owner, equation_path)
                # Source, bindings, and native transformations determine
                # every symbolic-zero branch without tracing 2**N masks.
                declaration = {
                    "kind": "bound-custom-jvp",
                    "source": self.bound_callable(source, equation_path),
                    "selector": selector,
                }
                declaration_id = _compiler_fingerprint(declaration)
                self.rules.setdefault(declaration_id, declaration)
                params["jvp_rule"] = {"declaration_id": declaration_id}
            equations.append(
                [
                    primitive_reference,
                    [atom(item) for item in equation.invars],
                    [variable(var) for var in equation.outvars],
                    self.parameter(params, equation_path),
                    [],
                ]
            )
        payload = {
            "kind": "jaxpr",
            "is_high": program.is_high,
            "capture_binding": "dynamic" if dynamic_captures else "static",
            "captures": captures,
            "inputs": inputs,
            "variable_types": variable_types,
            "variable_reference": "index",
            "equation_fields": (
                "primitive",
                "inputs",
                "outputs",
                "parameters",
                "effects",
            ),
            "equations": equations,
            "outputs": [atom(item) for item in program.outvars],
            "effects": [],
        }
        if not dynamic_captures:
            constants = value.consts
            if len(constants) != len(program.constvars):
                raise TypeError(
                    f"Unbound compiler constants at {path} require dynamic capture declaration."
                )
            payload["static_constants"] = self.parameter(constants, f"{path}.constants")
        fingerprint = _compiler_fingerprint(payload)
        self.memo[key] = fingerprint
        self.programs[fingerprint] = payload
        return {"program_id": fingerprint}

    def definitions(self) -> dict[str, Any]:
        """Finish one shared native traversal without copying nested graphs."""
        return {
            "metadata": [[key, self.metadata[key]] for key in sorted(self.metadata)],
            "programs": [[key, self.programs[key]] for key in sorted(self.programs)],
            "derivative_rules": [[key, self.rules[key]] for key in sorted(self.rules)],
        }


def execution_metadata_payload(
    value: Any, path: str = "execution", /, *, dynamic_captures: bool = False
) -> Any:
    """Canonical compiler metadata, never scientific identity from array shapes.

    The owner must declare externally supplied captures as dynamic; embedded
    programs retain their bound constant values. Shared programs are recorded
    once in a content-addressed DAG, independent of object IDs/debug locations.
    Unsupported placement/effects/callables/foreign primitives fail closed.
    """
    compiler = _CompilerPayload()
    if isinstance(value, (jax_core.Jaxpr, jax_core.ClosedJaxpr)):
        root = compiler.program(value, path, dynamic_captures=dynamic_captures)
        return {
            "kind": "compiler-program",
            "root": root,
            **compiler.definitions(),
        }
    if isinstance(
        value,
        (jax.ShapeDtypeStruct, jax_public_core.ShapedArray, jax.tree_util.PyTreeDef),
    ):
        root = compiler.parameter(value, path)
        return {
            "kind": "compiler-metadata",
            "root": root,
            **compiler.definitions(),
        }
    raise TypeError(f"Unsupported execution metadata at {path}: {_type_id(value)}.")


def _static_payload(value: Any, path: str, /) -> Any:
    if isinstance(value, np.generic):
        return _static_payload(value.item(), path)
    if isinstance(value, (jax.ShapeDtypeStruct, jax_core.Jaxpr, jax_core.ClosedJaxpr)):
        return execution_metadata_payload(value, path)
    if isinstance(value, _ARRAY_TYPES):
        raise TypeError(
            f"Static identity field {path} cannot contain a numeric array; array values belong to a numeric revision."
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Identity field {path} must be finite.")
        return value
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"Identity field {path} must be finite.")
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": value.str}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, jax.tree_util.PyTreeDef):
        return execution_metadata_payload(value, path)
    if isinstance(value, enum.Enum):
        return {
            "kind": "enum",
            "type": _type_id(value),
            "name": value.name,
        }
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _static_payload(value.start, f"{path}.start"),
            "stop": _static_payload(value.stop, f"{path}.stop"),
            "step": _static_payload(value.step, f"{path}.step"),
        }
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, StrictModule):
        return _static_module_payload(value, path)
    if isinstance(value, Mapping):
        records = _named_records(value, path)
        return {
            "kind": "mapping",
            "items": [
                [key, _static_payload(item, f"{path}.{key}")] for key, item in records
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _static_payload(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _static_payload(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if _is_plain_function(value):
        return _function_payload(value, path)
    if callable(value):
        raise TypeError(
            f"Opaque callable {path} requires explicit semantic and numeric IDs."
        )
    raise TypeError(f"Identity field {path} has unsupported type {type(value).__name__}.")


def _is_plain_function(value: Any, /) -> bool:
    """Return whether ``value`` is a stateless function importable by its name.

    Closures, lambdas, nested functions, methods, and rebound names are opaque:
    their behavior is not determined by module, name, and code alone. So is a
    function whose defaults or read module globals (transitively, through the
    same-module functions and the objects they reach) hold a numeric array:
    array values are numeric state its code does not determine.
    """
    if not isinstance(value, FunctionType) or value.__closure__ is not None:
        return False
    if value.__qualname__ != value.__name__:
        return False
    module = sys.modules.get(value.__module__)
    if module is None or vars(module).get(value.__name__) is not value:
        return False
    return not _hidden_arrays(value, eqx.is_array)


def _global_payload(value: Any, path: str, /) -> Any:
    """Identify one module global a plain function reads.

    Scalar and container constants contribute their values; modules, classes,
    and other objects contribute a reference. Numeric arrays never reach here
    (`_is_plain_function` refuses functions that read them).
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _global_payload(value.item(), path)
    if isinstance(value, float):
        # Hex keeps non-finite sentinels (inf, nan) exact and encodable.
        return {"kind": "float", "value": value.hex()}
    if isinstance(value, complex):
        return {"kind": "complex", "real": value.real.hex(), "imag": value.imag.hex()}
    if isinstance(value, (bytes, enum.Enum, np.dtype)):
        return _static_payload(value, path)
    if isinstance(value, ModuleType):
        return {"kind": "module", "name": value.__name__}
    if isinstance(value, type):
        return {"kind": "type", "name": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, FunctionType):
        return {
            "kind": "function-reference",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [
                _global_payload(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, (frozenset, set)):
        items = [_global_payload(item, path) for item in value]
        return {"kind": "set", "items": sorted(items, key=canonical_json)}
    if isinstance(value, Mapping):
        items = [
            [_global_payload(key, path), _global_payload(item, f"{path}[{key!r}]")]
            for key, item in value.items()
        ]
        return {"kind": "mapping", "items": sorted(items, key=canonical_json)}
    return {"kind": "reference", "type": _type_id(value)}


def _code_constant_payload(value: Any, path: str, /) -> Any:
    if isinstance(value, CodeType):
        return _code_payload(value, path)
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_code_constant_payload(item, path) for item in value],
        }
    if isinstance(value, frozenset):
        items = [_code_constant_payload(item, path) for item in value]
        return {"kind": "frozenset", "items": sorted(items, key=canonical_json)}
    return _static_payload(value, path)


def _code_payload(code: CodeType, path: str, /) -> dict[str, Any]:
    # File names and line tables are excluded so identity follows behavior,
    # not source position.
    return {
        "bytecode": code.co_code.hex(),
        "constants": [
            _code_constant_payload(value, f"{path}.constants") for value in code.co_consts
        ],
        "names": list(code.co_names),
        "variables": list(code.co_varnames),
        "free_variables": list(code.co_freevars),
        "cell_variables": list(code.co_cellvars),
        "argument_count": code.co_argcount,
        "positional_only_count": code.co_posonlyargcount,
        "keyword_only_count": code.co_kwonlyargcount,
        "flags": code.co_flags,
    }


def _function_payload(value: FunctionType, path: str, /) -> dict[str, Any]:
    return {
        "kind": "function",
        "module": value.__module__,
        "qualname": value.__qualname__,
        "code": canonical_fingerprint(_code_payload(value.__code__, f"{path}.code")),
        "defaults": _static_payload(value.__defaults__, f"{path}.defaults"),
        "keyword_defaults": _static_payload(
            value.__kwdefaults__, f"{path}.keyword_defaults"
        ),
        "globals": [
            [name, _global_payload(referenced, f"{path}.globals.{name}")]
            for name, referenced in _global_reads(value)
        ],
    }


def _static_module_payload(module: StrictModule, path: str, /) -> dict[str, Any]:
    return {
        "kind": "strict-module",
        "type": _type_id(module),
        "fields": [
            [
                field.name,
                _static_payload(
                    object.__getattribute__(module, field.name),
                    f"{path}.{field.name}",
                ),
            ]
            for field in dataclasses.fields(module)
        ],
    }


def _dynamic_payload(value: Any, path: str, /) -> tuple[Any, Any]:
    if isinstance(value, np.generic):
        return _dynamic_payload(value.item(), path)
    if isinstance(value, _ARRAY_TYPES):
        array = _array_payload(value, path)
        return (
            {
                "kind": "array",
                "shape": array["shape"],
                "dtype": array["dtype"],
            },
            array,
        )
    if value is None or isinstance(value, str):
        return value, None
    if isinstance(value, (bool, int)):
        return {"kind": type(value).__name__}, value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Identity field {path} must be finite.")
        return {"kind": "float"}, value
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"Identity field {path} must be finite.")
        return (
            {"kind": "complex"},
            {"real": value.real, "imag": value.imag},
        )
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": value.str}, None
    if isinstance(value, bytes):
        return _static_payload(value, path), None
    if isinstance(value, enum.Enum):
        return _static_payload(value, path), None
    if isinstance(value, slice) or value is Ellipsis:
        return _static_payload(value, path), None
    if isinstance(value, StrictModule):
        semantic_fields: list[list[Any]] = []
        numeric_fields: list[list[Any]] = []
        for field in dataclasses.fields(value):
            field_value = object.__getattribute__(value, field.name)
            field_path = f"{path}.{field.name}"
            if bool(field.metadata.get("static", False)):
                semantic_fields.append(
                    [field.name, _static_payload(field_value, field_path)]
                )
                continue
            semantic, numeric = _dynamic_payload(field_value, field_path)
            semantic_fields.append([field.name, semantic])
            numeric_fields.append([field.name, numeric])
        return (
            {
                "kind": "strict-module",
                "type": _type_id(value),
                "fields": semantic_fields,
            },
            {
                "kind": "strict-module",
                "fields": numeric_fields,
            },
        )
    if isinstance(value, Mapping):
        records = _named_records(value, path)
        semantic_items: list[list[Any]] = []
        numeric_items: list[list[Any]] = []
        for key, item in records:
            semantic, numeric = _dynamic_payload(item, f"{path}.{key}")
            semantic_items.append([key, semantic])
            numeric_items.append([key, numeric])
        return (
            {"kind": "mapping", "items": semantic_items},
            {"kind": "mapping", "items": numeric_items},
        )
    if isinstance(value, (tuple, list)):
        semantic_items = []
        numeric_items = []
        for index, item in enumerate(value):
            semantic, numeric = _dynamic_payload(item, f"{path}[{index}]")
            semantic_items.append(semantic)
            numeric_items.append(numeric)
        kind = "tuple" if isinstance(value, tuple) else "list"
        return (
            {"kind": kind, "items": semantic_items},
            {"kind": kind, "items": numeric_items},
        )
    if _is_plain_function(value):
        return _function_payload(value, path), None
    if callable(value):
        raise TypeError(
            f"Opaque callable {path} requires explicit semantic and numeric IDs."
        )
    raise TypeError(f"Identity field {path} has unsupported type {type(value).__name__}.")


def strict_module_payload(module: StrictModule, /) -> dict[str, Any]:
    """Return content-addressed semantic and numeric payloads for a StrictModule.

    Static fields contribute only to semantic content. Dynamic array/scalar values
    contribute to numeric content, while their structure contributes to semantics.
    Static numeric arrays and opaque callable fields are rejected.
    """
    if not isinstance(module, StrictModule):
        raise TypeError("strict_module_payload requires a StrictModule.")
    semantic, numeric = _dynamic_payload(module, "module")
    semantic_id = canonical_fingerprint(
        {"kind": "strict-module-semantic-content", "payload": semantic}
    )
    numeric_id = canonical_fingerprint(
        {
            "kind": "strict-module-numeric-content",
            "semantic_content_id": semantic_id,
            "payload": numeric,
        }
    )
    return {
        "semantic_payload": semantic,
        "numeric_payload": numeric,
        "semantic_content_id": semantic_id,
        "numeric_content_id": numeric_id,
    }


def callable_payload(
    value: Any,
    /,
    *,
    semantic_id: str | None = None,
    numeric_id: str | None = None,
) -> dict[str, Any]:
    """Identify a callable without falling back to its Python class or name.

    Callable StrictModules are content-addressed from their fields. A plain
    stateless function (module-level, importable by its name, without closure
    cells, and reading no numeric array through its defaults or module globals)
    is content-addressed from its module, name, position-independent code,
    static defaults, and the values of the scalar constants and references of
    the module globals it reads, unless the caller declares both identities.
    Every other callable (closure, lambda, nested function, method, partial,
    foreign callable object, or function reading array state) is opaque and
    requires both identities explicitly.
    """
    if not callable(value):
        raise TypeError("callable_payload requires a callable value.")
    if isinstance(value, StrictModule):
        if semantic_id is not None or numeric_id is not None:
            raise ValueError(
                "Content-addressed StrictModule callables do not accept ID overrides."
            )
        return strict_module_payload(value)
    if semantic_id is None and numeric_id is None and _is_plain_function(value):
        semantic = _function_payload(value, "callable")
        semantic_content_id = canonical_fingerprint(
            {"kind": "function-semantic-content", "payload": semantic}
        )
        return {
            "semantic_payload": semantic,
            "numeric_payload": None,
            "semantic_content_id": semantic_content_id,
            "numeric_content_id": canonical_fingerprint(
                {
                    "kind": "function-numeric-content",
                    "semantic_content_id": semantic_content_id,
                    "payload": None,
                }
            ),
        }
    if semantic_id is None or numeric_id is None:
        raise TypeError(
            "Opaque callables require explicit semantic_id and numeric_id values; "
            "a function is opaque when it closes over state or reads a numeric "
            "array from its defaults or module globals."
        )
    semantic_id_ = _identifier(semantic_id, "semantic_id")
    numeric_id_ = _identifier(numeric_id, "numeric_id")
    return {
        "semantic_payload": {
            "kind": "opaque-callable",
            "semantic_id": semantic_id_,
        },
        "numeric_payload": {
            "kind": "opaque-callable",
            "numeric_id": numeric_id_,
        },
        "semantic_content_id": semantic_id_,
        "numeric_content_id": numeric_id_,
    }


class SemanticProvenance(StrictModule):
    """Content-addressed semantics with separately named external resources."""

    content_json: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    resource_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)

    def __init__(
        self,
        content: Any,
        /,
        *,
        resource_ids: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    ) -> None:
        content_ = _static_payload(content, "semantic_content")
        resources = tuple(
            (name, _identifier(identifier, f"resource_ids[{name!r}]"))
            for name, identifier in _named_records(resource_ids, "resource_ids")
        )
        content_json = canonical_json(content_)
        content_id = canonical_fingerprint(
            {"kind": "semantic-content", "payload": content_}
        )
        self.content_json = content_json
        self.content_id = content_id
        self.resource_ids = resources
        self.semantic_id = canonical_fingerprint(
            {
                "kind": "semantic-provenance",
                "content_id": content_id,
                "resource_ids": [list(record) for record in resources],
            }
        )


class NumericRevision(StrictModule):
    """Dynamic numeric realization bound to one semantic provenance identity."""

    semantic_id: str = eqx.field(static=True)
    content_json: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)

    def __init__(
        self,
        semantic: SemanticProvenance | str,
        content: Any,
        /,
    ) -> None:
        semantic_id = (
            semantic.semantic_id
            if isinstance(semantic, SemanticProvenance)
            else _identifier(semantic, "semantic_id")
        )
        structure, realization = _dynamic_payload(content, "numeric_content")
        payload = {"structure": structure, "realization": realization}
        content_json = canonical_json(payload)
        content_id = canonical_fingerprint(
            {"kind": "numeric-content", "payload": payload}
        )
        self.semantic_id = semantic_id
        self.content_json = content_json
        self.content_id = content_id
        self.revision_id = canonical_fingerprint(
            {
                "kind": "numeric-revision",
                "semantic_id": semantic_id,
                "content_id": content_id,
            }
        )


def _shape_records(value: RecordInput, /) -> tuple[tuple[str, tuple[int, ...]], ...]:
    records: list[tuple[str, tuple[int, ...]]] = []
    for name, raw_shape in _named_records(value, "shapes"):
        if isinstance(raw_shape, _ARRAY_TYPES) or isinstance(raw_shape, (str, bytes)):
            raise TypeError("Executable shapes must be explicit integer sequences.")
        shape_values = tuple(raw_shape)
        if any(
            isinstance(size, bool) or not isinstance(size, (int, np.integer))
            for size in shape_values
        ):
            raise TypeError("Executable shape dimensions must be integers.")
        shape = tuple(shape_values)
        if any(size < 0 for size in shape):
            raise ValueError("Executable shape dimensions must be nonnegative.")
        records.append((name, shape))
    return tuple(records)


def _dtype_records(value: RecordInput, /) -> tuple[tuple[str, str], ...]:
    records: list[tuple[str, str]] = []
    for name, raw_dtype in _named_records(value, "dtypes"):
        if isinstance(raw_dtype, _ARRAY_TYPES):
            raise TypeError("Executable dtypes must be dtype specifications, not arrays.")
        records.append((name, np.dtype(raw_dtype).str))
    return tuple(records)


def _identifier_records(value: RecordInput, name: str, /) -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, _identifier(identifier, f"{name}[{key!r}]"))
        for key, identifier in _named_records(value, name)
    )


def _capacity_records(value: RecordInput, /) -> tuple[tuple[str, int], ...]:
    records: list[tuple[str, int]] = []
    for name, raw_capacity in _named_records(value, "capacities"):
        if isinstance(raw_capacity, bool) or not isinstance(
            raw_capacity, (int, np.integer)
        ):
            raise TypeError("Executable capacities must be integers.")
        capacity = int(raw_capacity)
        if capacity < 0:
            raise ValueError("Executable capacities must be nonnegative.")
        records.append((name, capacity))
    return tuple(records)


def _fact_records(value: RecordInput, name: str, /) -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, canonical_json(_static_payload(fact, f"{name}.{key}")))
        for key, fact in _named_records(value, name)
    )


def _static_callable_records(value: RecordInput, /) -> tuple[tuple[str, str, str], ...]:
    records: list[tuple[str, str, str]] = []
    for name, component in _named_records(value, "static_callables"):
        payload = callable_payload(component)
        records.append(
            (name, payload["semantic_content_id"], payload["numeric_content_id"])
        )
    return tuple(records)


class ExecutableSignature(StrictModule):
    """Static compilation identity with no dynamic numeric realization values."""

    shapes: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    dtypes: tuple[tuple[str, str], ...] = eqx.field(static=True)
    space_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    topology_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    capacities: tuple[tuple[str, int], ...] = eqx.field(static=True)
    algorithm_facts: tuple[tuple[str, str], ...] = eqx.field(static=True)
    backend_facts: tuple[tuple[str, str], ...] = eqx.field(static=True)
    static_callables: tuple[tuple[str, str, str], ...] = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        shapes: RecordInput = (),
        dtypes: RecordInput = (),
        space_ids: RecordInput = (),
        topology_ids: RecordInput = (),
        capacities: RecordInput = (),
        algorithm_facts: RecordInput = (),
        backend_facts: RecordInput = (),
        static_callables: RecordInput = (),
    ) -> None:
        """Build the signature from static facts.

        `static_callables` names callables compiled into the executable. Each is
        identified by `callable_payload`: weights held statically by a callable
        are part of the executable, so its semantic and numeric content IDs
        both enter the signature, while opaque callables must declare them.
        """
        shapes_ = _shape_records(shapes)
        dtypes_ = _dtype_records(dtypes)
        spaces_ = _identifier_records(space_ids, "space_ids")
        topology_ = _identifier_records(topology_ids, "topology_ids")
        capacities_ = _capacity_records(capacities)
        algorithms_ = _fact_records(algorithm_facts, "algorithm_facts")
        backend_ = _fact_records(backend_facts, "backend_facts")
        callables_ = _static_callable_records(static_callables)
        payload = {
            "kind": "executable-signature",
            "shapes": [[name, list(shape)] for name, shape in shapes_],
            "dtypes": [list(record) for record in dtypes_],
            "space_ids": [list(record) for record in spaces_],
            "topology_ids": [list(record) for record in topology_],
            "capacities": [list(record) for record in capacities_],
            "algorithm_facts": [list(record) for record in algorithms_],
            "backend_facts": [list(record) for record in backend_],
            "static_callables": [list(record) for record in callables_],
        }
        self.shapes = shapes_
        self.dtypes = dtypes_
        self.space_ids = spaces_
        self.topology_ids = topology_
        self.capacities = capacities_
        self.algorithm_facts = algorithms_
        self.backend_facts = backend_
        self.static_callables = callables_
        self.signature_id = canonical_fingerprint(payload)


_BINDING_FIELDS = frozenset(
    {"semantic_id", "numeric_revision_id", "executable_signature_id", "binding_id"}
)


class ArtifactBindingIdentity(StrictModule):
    """Semantic, numeric, and executable identity of one bound artifact model.

    A frozen or published model is identified by all three IDs together: its
    `SemanticProvenance`, the `NumericRevision` of its dynamic numeric content,
    and the `ExecutableSignature` of its static compilation. `binding_id`
    content-addresses the triple.
    """

    semantic_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    executable_signature_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        semantic: SemanticProvenance | str,
        numeric_revision: NumericRevision | str,
        executable_signature: ExecutableSignature | str,
        /,
    ) -> None:
        semantic_id = (
            semantic.semantic_id
            if isinstance(semantic, SemanticProvenance)
            else _identifier(semantic, "semantic_id")
        )
        if isinstance(numeric_revision, NumericRevision):
            if numeric_revision.semantic_id != semantic_id:
                raise ValueError(
                    "numeric_revision is bound to another semantic provenance."
                )
            numeric_id = numeric_revision.revision_id
        else:
            numeric_id = _identifier(numeric_revision, "numeric_revision_id")
        executable_id = (
            executable_signature.signature_id
            if isinstance(executable_signature, ExecutableSignature)
            else _identifier(executable_signature, "executable_signature_id")
        )
        self.semantic_id = semantic_id
        self.numeric_revision_id = numeric_id
        self.executable_signature_id = executable_id
        self.binding_id = canonical_fingerprint(
            {
                "kind": "artifact-binding-identity",
                "semantic_id": semantic_id,
                "numeric_revision_id": numeric_id,
                "executable_signature_id": executable_id,
            }
        )

    def to_record(self) -> dict[str, str]:
        """Return the JSON record read back by `from_record`."""
        return {
            "semantic_id": self.semantic_id,
            "numeric_revision_id": self.numeric_revision_id,
            "executable_signature_id": self.executable_signature_id,
            "binding_id": self.binding_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> "ArtifactBindingIdentity":
        """Rebuild a binding identity, failing closed on any field or ID mismatch."""
        if not isinstance(record, Mapping) or set(record) != _BINDING_FIELDS:
            raise ValueError("Artifact binding identity record fields do not match.")
        identity = cls(
            record["semantic_id"],
            record["numeric_revision_id"],
            record["executable_signature_id"],
        )
        if identity.binding_id != record["binding_id"]:
            raise ValueError("Artifact binding identity record is corrupt.")
        return identity


__all__ = [
    "ArtifactBindingIdentity",
    "callable_payload",
    "ExecutableSignature",
    "NumericRevision",
    "SemanticProvenance",
    "strict_module_payload",
]
