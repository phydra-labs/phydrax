#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array

from .._array_archive import (
    array_collection_digest,
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._complex_parameters import (
    ComplexImportPolicy,
    ComplexInterchangeEntry,
    ComplexInterchangeState,
    export_complex_parameters,
    import_complex_parameters,
)


ComplexOptimizerRouteKind: TypeAlias = Literal[
    "complex-vector",
    "cartesian-second-moment",
    "exact-real",
    "exact-discrete",
]

_ROUTE_KINDS = frozenset(
    {
        "complex-vector",
        "cartesian-second-moment",
        "exact-real",
        "exact-discrete",
    }
)


@dataclass(frozen=True, slots=True)
class ComplexOptimizerStateGroup:
    """One explicit optimizer leaf route with truthful complex geometry."""

    name: str
    kind: ComplexOptimizerRouteKind
    paths: tuple[str, ...]

    def __init__(
        self,
        name: str,
        kind: ComplexOptimizerRouteKind,
        paths: Sequence[str],
    ):
        name_ = str(name)
        paths_ = tuple(str(path) for path in paths)
        if not name_ or "/" in name_ or name_ in (".", ".."):
            raise ValueError(
                "Optimizer interchange group name must be safe and non-empty."
            )
        if kind not in _ROUTE_KINDS:
            raise ValueError("Unknown optimizer interchange route kind.")
        expected = 2 if kind in ("complex-vector", "cartesian-second-moment") else 1
        if len(paths_) != expected or any(not path for path in paths_):
            raise ValueError(f"Optimizer group {kind!r} requires {expected} paths.")
        if len(set(paths_)) != len(paths_):
            raise ValueError("Optimizer group paths must be unique.")
        object.__setattr__(self, "name", name_)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "paths", paths_)


class ComplexOptimizerStateLayout(StrictModule):
    """Exact optimizer treedef and explicit grouped leaf routes."""

    treedef: jax.tree_util.PyTreeDef = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    groups: tuple[ComplexOptimizerStateGroup, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)


def prepare_complex_optimizer_state_layout(
    template: Any,
    groups: Sequence[ComplexOptimizerStateGroup],
    /,
) -> ComplexOptimizerStateLayout:
    """Bind explicit optimizer routes to one exact array-only treedef."""
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    paths = tuple(jax.tree_util.keystr(path) or "<root>" for path, _ in path_leaves)
    arrays = tuple(jnp.asarray(leaf) for _, leaf in path_leaves)
    groups_ = tuple(groups)
    if not groups_ or not all(
        isinstance(group, ComplexOptimizerStateGroup) for group in groups_
    ):
        raise TypeError("groups must contain ComplexOptimizerStateGroup values.")
    names = tuple(group.name for group in groups_)
    if len(set(names)) != len(names):
        raise ValueError("Optimizer interchange group names must be unique.")
    routed = tuple(path for group in groups_ for path in group.paths)
    if len(set(routed)) != len(routed):
        raise ValueError("Optimizer leaves cannot appear in more than one route.")
    if set(routed) != set(paths):
        raise ValueError(
            "Optimizer routes must cover every leaf exactly; "
            f"missing={sorted(set(paths) - set(routed))}, "
            f"unknown={sorted(set(routed) - set(paths))}."
        )
    index = {path: position for position, path in enumerate(paths)}
    for group in groups_:
        values = tuple(arrays[index[path]] for path in group.paths)
        if group.kind in ("complex-vector", "cartesian-second-moment"):
            if values[0].shape != values[1].shape or values[0].dtype != values[1].dtype:
                raise ValueError(f"Optimizer group {group.name!r} components must match.")
        if group.kind != "exact-discrete" and any(
            not jnp.issubdtype(value.dtype, jnp.floating) for value in values
        ):
            raise TypeError(
                f"Optimizer group {group.name!r} requires real floating leaves."
            )
        if group.kind == "exact-discrete" and jnp.issubdtype(
            values[0].dtype, jnp.inexact
        ):
            raise TypeError("exact-discrete routes require integer or boolean leaves.")
    layout_id = canonical_fingerprint(
        {
            "kind": "complex-optimizer-state-layout",
            "paths": paths,
            "shapes": [list(value.shape) for value in arrays],
            "dtypes": [value.dtype.name for value in arrays],
            "groups": [
                {"name": group.name, "kind": group.kind, "paths": group.paths}
                for group in groups_
            ],
        }
    )
    return ComplexOptimizerStateLayout(
        treedef=treedef,
        paths=paths,
        shapes=tuple(tuple(int(size) for size in value.shape) for value in arrays),
        dtypes=tuple(value.dtype.name for value in arrays),
        groups=groups_,
        layout_id=layout_id,
    )


class ComplexOptimizerInterchangeEntry(StrictModule):
    name: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    values: tuple[Array, ...]
    entry_id: str = eqx.field(static=True)

    def __init__(self, group: ComplexOptimizerStateGroup, values: Sequence[Any], /):
        values_ = tuple(jnp.asarray(value) for value in values)
        expected = 1 if group.kind != "cartesian-second-moment" else 2
        if len(values_) != expected:
            raise ValueError(
                "Optimizer interchange entry payload cardinality is invalid."
            )
        self.name = group.name
        self.kind = group.kind
        self.paths = group.paths
        self.values = values_
        self.entry_id = canonical_fingerprint(
            {
                "kind": "complex-optimizer-interchange-entry",
                "name": group.name,
                "semantics": group.kind,
                "paths": group.paths,
                "values": array_tree_fingerprint(values_),
            }
        )


class ComplexOptimizerInterchangeState(StrictModule):
    entries: tuple[ComplexOptimizerInterchangeEntry, ...]
    layout_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: ComplexOptimizerStateLayout,
        entries: Sequence[ComplexOptimizerInterchangeEntry],
        /,
    ):
        entries_ = tuple(entries)
        expected = tuple(group.name for group in layout.groups)
        if tuple(entry.name for entry in entries_) != expected:
            raise ValueError("Optimizer interchange entries do not match their layout.")
        self.entries = entries_
        self.layout_id = layout.layout_id
        self.content_id = canonical_fingerprint(
            {
                "kind": "complex-optimizer-interchange-state",
                "layout": layout.layout_id,
                "entries": [entry.entry_id for entry in entries_],
            }
        )


class RNGInterchangeState(StrictModule):
    key_data: tuple[Array, ...]
    key_impls: tuple[str, ...] = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    def __init__(self, rng_state: Any, /):
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(rng_state)
        paths = tuple(jax.tree_util.keystr(path) or "<root>" for path, _ in path_leaves)
        keys = tuple(leaf for _, leaf in path_leaves)
        if not keys:
            raise ValueError("RNG interchange requires at least one typed key.")
        if any(
            not jax.dtypes.issubdtype(jnp.asarray(key).dtype, jax.dtypes.prng_key)
            for key in keys
        ):
            raise TypeError("RNG interchange accepts typed JAX keys only.")
        data = tuple(jr.key_data(key) for key in keys)
        implementations = tuple(str(jr.key_impl(key)) for key in keys)
        self.key_data = data
        self.key_impls = implementations
        self.paths = paths
        self.content_id = canonical_fingerprint(
            {
                "kind": "rng-interchange-state",
                "paths": paths,
                "implementations": implementations,
                "data": array_tree_fingerprint(data),
            }
        )


class PreparedComplexTrainingInterchange(StrictModule):
    parameter_template: Any
    optimizer_template: Any
    rng_template: Any
    auxiliary_template: Any
    optimizer_layout: ComplexOptimizerStateLayout = eqx.field(static=True)
    parameter_architecture_id: str = eqx.field(static=True)
    training_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)


class ComplexTrainingInterchangeState(StrictModule):
    parameters: ComplexInterchangeState
    optimizer: ComplexOptimizerInterchangeState
    rng: RNGInterchangeState
    auxiliary_state: Any
    step: Array
    training_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class ImportedComplexTrainingState:
    parameters: Any
    optimizer_state: Any
    rng_state: Any
    auxiliary_state: Any
    step: int


def prepare_complex_training_interchange(
    parameters: Any,
    optimizer_state: Any,
    rng_state: Any,
    auxiliary_state: Any,
    /,
    *,
    optimizer_groups: Sequence[ComplexOptimizerStateGroup],
    training_id: str,
) -> PreparedComplexTrainingInterchange:
    """Prepare complete training-state conversion against exact templates."""
    training = str(training_id)
    if not training:
        raise ValueError("training_id must be non-empty.")
    parameter_state = export_complex_parameters(parameters)
    optimizer_layout = prepare_complex_optimizer_state_layout(
        optimizer_state, optimizer_groups
    )
    rng_interchange = RNGInterchangeState(rng_state)
    auxiliary_leaves = tuple(jax.tree.leaves(auxiliary_state))
    if any(np.asarray(leaf).dtype.hasobject for leaf in auxiliary_leaves):
        raise TypeError("Training auxiliary state must be an array-only PyTree.")
    layout_id = canonical_fingerprint(
        {
            "kind": "complex-training-interchange-layout",
            "training": training,
            "parameters": parameter_state.architecture_id,
            "optimizer": optimizer_layout.layout_id,
            "rng_paths": rng_interchange.paths,
            "auxiliary": [
                [list(jnp.asarray(leaf).shape), jnp.asarray(leaf).dtype.name]
                for leaf in auxiliary_leaves
            ],
        }
    )
    return PreparedComplexTrainingInterchange(
        parameter_template=parameters,
        optimizer_template=optimizer_state,
        rng_template=rng_state,
        auxiliary_template=auxiliary_state,
        optimizer_layout=optimizer_layout,
        parameter_architecture_id=parameter_state.architecture_id,
        training_id=training,
        layout_id=layout_id,
    )


def _optimizer_entries(
    layout: ComplexOptimizerStateLayout,
    optimizer_state: Any,
    /,
) -> tuple[ComplexOptimizerInterchangeEntry, ...]:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(optimizer_state)
    if treedef != layout.treedef:
        raise ValueError("Optimizer state treedef changed after interchange preparation.")
    values = {
        jax.tree_util.keystr(path) or "<root>": jnp.asarray(value)
        for path, value in path_leaves
    }
    entries = []
    for group in layout.groups:
        routed = tuple(values[path] for path in group.paths)
        if group.kind == "complex-vector":
            payload = (jax.lax.complex(routed[0], routed[1]),)
        else:
            payload = routed
        entries.append(ComplexOptimizerInterchangeEntry(group, payload))
    return tuple(entries)


def export_complex_training_state(
    prepared: PreparedComplexTrainingInterchange,
    parameters: Any,
    optimizer_state: Any,
    rng_state: Any,
    auxiliary_state: Any,
    /,
    *,
    step: int,
) -> ComplexTrainingInterchangeState:
    """Export full in-memory complex training state without reseeding or repair."""
    if not isinstance(prepared, PreparedComplexTrainingInterchange):
        raise TypeError("prepared must be PreparedComplexTrainingInterchange.")
    if int(step) < 0:
        raise ValueError("step must be non-negative.")
    parameters_ = export_complex_parameters(parameters)
    if parameters_.architecture_id != prepared.parameter_architecture_id:
        raise ValueError("Training parameter architecture changed.")
    optimizer_ = ComplexOptimizerInterchangeState(
        prepared.optimizer_layout,
        _optimizer_entries(prepared.optimizer_layout, optimizer_state),
    )
    rng_ = RNGInterchangeState(rng_state)
    template_paths = tuple(
        jax.tree_util.keystr(path) or "<root>"
        for path, _ in jax.tree_util.tree_flatten_with_path(prepared.rng_template)[0]
    )
    if rng_.paths != template_paths:
        raise ValueError("Training RNG tree changed.")
    auxiliary_structure = jax.tree.structure(auxiliary_state)
    if auxiliary_structure != jax.tree.structure(prepared.auxiliary_template):
        raise ValueError("Training auxiliary state treedef changed.")
    auxiliary_ = jax.tree.map(jnp.asarray, auxiliary_state)
    step_ = jnp.asarray(int(step), dtype=jnp.int64)
    state_id = canonical_fingerprint(
        {
            "kind": "complex-training-interchange-state",
            "training": prepared.training_id,
            "layout": prepared.layout_id,
            "parameters": parameters_.state_id,
            "optimizer": optimizer_.content_id,
            "rng": rng_.content_id,
            "auxiliary": array_tree_fingerprint(auxiliary_),
            "step": int(step),
        }
    )
    return ComplexTrainingInterchangeState(
        parameters=parameters_,
        optimizer=optimizer_,
        rng=rng_,
        auxiliary_state=auxiliary_,
        step=step_,
        training_id=prepared.training_id,
        layout_id=prepared.layout_id,
        state_id=state_id,
    )


def _cast_like(value: Array, target: Array, policy: ComplexImportPolicy, /) -> Array:
    if value.shape != target.shape:
        raise ValueError("Optimizer interchange leaf shape changed.")
    if value.dtype.itemsize > target.dtype.itemsize and not policy.allow_precision_loss:
        raise ValueError("Optimizer interchange import would narrow precision.")
    result = jnp.asarray(value, dtype=target.dtype)
    if policy.preserve_sharding and isinstance(target, jax.Array):
        result = jax.device_put(result, target.sharding)
    return result


def _import_optimizer(
    prepared: PreparedComplexTrainingInterchange,
    state: ComplexOptimizerInterchangeState,
    policy: ComplexImportPolicy,
    /,
):
    layout = prepared.optimizer_layout
    if state.layout_id != layout.layout_id:
        raise ValueError("Complex optimizer layout identity mismatch.")
    template_path_leaves, treedef = jax.tree_util.tree_flatten_with_path(
        prepared.optimizer_template
    )
    templates = {
        jax.tree_util.keystr(path) or "<root>": jnp.asarray(value)
        for path, value in template_path_leaves
    }
    restored: dict[str, Array] = {}
    for group, entry in zip(layout.groups, state.entries, strict=True):
        if (entry.name, entry.kind, entry.paths) != (
            group.name,
            group.kind,
            group.paths,
        ):
            raise ValueError("Complex optimizer entry does not match its route.")
        if group.kind == "complex-vector":
            complex_value = entry.values[0]
            components = (jnp.real(complex_value), jnp.imag(complex_value))
        else:
            components = entry.values
        for path, component in zip(group.paths, components, strict=True):
            target = templates[path]
            if group.kind == "exact-discrete":
                if component.shape != target.shape or component.dtype != target.dtype:
                    raise ValueError("Exact discrete optimizer state changed.")
                restored[path] = (
                    jax.device_put(component, target.sharding)
                    if isinstance(target, jax.Array)
                    else component
                )
            else:
                restored[path] = _cast_like(component, target, policy)
    leaves = [
        restored[jax.tree_util.keystr(path) or "<root>"]
        for path, _ in template_path_leaves
    ]
    return jax.tree.unflatten(treedef, leaves)


def _import_rng(template: Any, state: RNGInterchangeState, /):
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    paths = tuple(jax.tree_util.keystr(path) or "<root>" for path, _ in path_leaves)
    if paths != state.paths or len(path_leaves) != len(state.key_data):
        raise ValueError("RNG interchange tree does not match its destination.")
    keys = []
    for (_, target), data, implementation in zip(
        path_leaves,
        state.key_data,
        state.key_impls,
        strict=True,
    ):
        if str(jr.key_impl(target)) != implementation:
            raise ValueError("RNG implementation changed across interchange.")
        keys.append(jr.wrap_key_data(data, impl=implementation))
    return jax.tree.unflatten(treedef, keys)


def import_complex_training_state(
    prepared: PreparedComplexTrainingInterchange,
    state: ComplexTrainingInterchangeState,
    /,
    *,
    policy: ComplexImportPolicy | None = None,
) -> ImportedComplexTrainingState:
    """Restore every continuation-owned state component against templates."""
    if not isinstance(prepared, PreparedComplexTrainingInterchange):
        raise TypeError("prepared must be PreparedComplexTrainingInterchange.")
    if not isinstance(state, ComplexTrainingInterchangeState):
        raise TypeError("state must be ComplexTrainingInterchangeState.")
    if state.training_id != prepared.training_id or state.layout_id != prepared.layout_id:
        raise ValueError("Complex training interchange identity mismatch.")
    policy_ = ComplexImportPolicy() if policy is None else policy
    if not isinstance(policy_, ComplexImportPolicy):
        raise TypeError("policy must be ComplexImportPolicy or None.")
    parameters = import_complex_parameters(
        prepared.parameter_template,
        state.parameters,
        policy=policy_,
    )
    optimizer = _import_optimizer(prepared, state.optimizer, policy_)
    rng = _import_rng(prepared.rng_template, state.rng)
    auxiliary = jax.tree.map(
        lambda value, target: _cast_like(
            jnp.asarray(value), jnp.asarray(target), policy_
        ),
        state.auxiliary_state,
        prepared.auxiliary_template,
    )
    return ImportedComplexTrainingState(
        parameters,
        optimizer,
        rng,
        auxiliary,
        int(state.step),
    )


def write_complex_training_checkpoint(
    path: str,
    state: ComplexTrainingInterchangeState,
    /,
):
    """Atomically write one checksum-protected, pickle-free full-state archive."""
    if not isinstance(state, ComplexTrainingInterchangeState):
        raise TypeError("state must be ComplexTrainingInterchangeState.")
    arrays: dict[str, Any] = {}
    parameter_entries = []
    for index, entry in enumerate(state.parameters.entries):
        name = f"parameters/{index:06d}"
        arrays[name] = entry.value
        parameter_entries.append(
            {
                "name": entry.name,
                "role": entry.role,
                "component_dtype": entry.component_dtype,
                "trainable": entry.trainable,
                "array": name,
            }
        )
    optimizer_entries = []
    for index, entry in enumerate(state.optimizer.entries):
        names = []
        for value_index, value in enumerate(entry.values):
            name = f"optimizer/{index:06d}/{value_index:02d}"
            arrays[name] = value
            names.append(name)
        optimizer_entries.append(
            {
                "name": entry.name,
                "kind": entry.kind,
                "paths": list(entry.paths),
                "arrays": names,
            }
        )
    rng_names = []
    for index, data in enumerate(state.rng.key_data):
        name = f"rng/{index:06d}"
        arrays[name] = data
        rng_names.append(name)
    auxiliary_spec = pack_array_tree("auxiliary", state.auxiliary_state, arrays)
    manifest = {
        "format": "phydrax-complex-training-interchange",
        "training_id": state.training_id,
        "layout_id": state.layout_id,
        "state_id": state.state_id,
        "step": int(state.step),
        "parameters": {
            "semantics": state.parameters.semantics,
            "provider_kind": state.parameters.provider_kind,
            "architecture_id": state.parameters.architecture_id,
            "metadata": state.parameters.metadata,
            "entries": parameter_entries,
        },
        "optimizer": {
            "layout_id": state.optimizer.layout_id,
            "entries": optimizer_entries,
        },
        "rng": {
            "paths": list(state.rng.paths),
            "implementations": list(state.rng.key_impls),
            "arrays": rng_names,
        },
        "auxiliary": auxiliary_spec,
        "array_collection_id": array_collection_digest(arrays),
    }
    return write_array_archive(path, manifest=manifest, arrays=arrays)


def read_complex_training_checkpoint(
    path: str,
    prepared: PreparedComplexTrainingInterchange,
    /,
) -> ComplexTrainingInterchangeState:
    """Read and identity-check a full-state archive against exact templates."""
    manifest, arrays = read_array_archive(path)
    expected = {
        "format",
        "training_id",
        "layout_id",
        "state_id",
        "step",
        "parameters",
        "optimizer",
        "rng",
        "auxiliary",
        "array_collection_id",
        "arrays",
    }
    if set(manifest) != expected:
        raise ValueError("Complex training checkpoint fields are not canonical.")
    if manifest["format"] != "phydrax-complex-training-interchange":
        raise ValueError("File is not a complex training checkpoint.")
    if (
        manifest["training_id"] != prepared.training_id
        or manifest["layout_id"] != prepared.layout_id
    ):
        raise ValueError("Complex training checkpoint layout mismatch.")
    if manifest["array_collection_id"] != array_collection_digest(arrays):
        raise ValueError("Complex training checkpoint collection checksum mismatch.")
    parameter_manifest = manifest["parameters"]
    parameter_entries = tuple(
        ComplexInterchangeEntry(
            entry["name"],
            arrays[entry["array"]],
            role=entry["role"],
            component_dtype=entry["component_dtype"],
            trainable=entry["trainable"],
        )
        for entry in parameter_manifest["entries"]
    )
    parameters = ComplexInterchangeState(
        parameter_manifest["semantics"],
        parameter_manifest["provider_kind"],
        parameter_manifest["architecture_id"],
        parameter_entries,
        metadata=parameter_manifest["metadata"],
    )
    optimizer_manifest = manifest["optimizer"]
    if optimizer_manifest["layout_id"] != prepared.optimizer_layout.layout_id:
        raise ValueError("Complex optimizer checkpoint layout mismatch.")
    optimizer_entries = tuple(
        ComplexOptimizerInterchangeEntry(
            group,
            tuple(arrays[name] for name in entry["arrays"]),
        )
        for group, entry in zip(
            prepared.optimizer_layout.groups,
            optimizer_manifest["entries"],
            strict=True,
        )
    )
    optimizer = ComplexOptimizerInterchangeState(
        prepared.optimizer_layout,
        optimizer_entries,
    )
    rng_manifest = manifest["rng"]
    rng = object.__new__(RNGInterchangeState)
    object.__setattr__(
        rng, "key_data", tuple(arrays[name] for name in rng_manifest["arrays"])
    )
    object.__setattr__(rng, "key_impls", tuple(rng_manifest["implementations"]))
    object.__setattr__(rng, "paths", tuple(rng_manifest["paths"]))
    object.__setattr__(
        rng,
        "content_id",
        canonical_fingerprint(
            {
                "kind": "rng-interchange-state",
                "paths": tuple(rng_manifest["paths"]),
                "implementations": tuple(rng_manifest["implementations"]),
                "data": array_tree_fingerprint(rng.key_data),
            }
        ),
    )
    auxiliary = unpack_array_tree(
        manifest["auxiliary"],
        arrays,
        prepared.auxiliary_template,
    )
    state = ComplexTrainingInterchangeState(
        parameters=parameters,
        optimizer=optimizer,
        rng=rng,
        auxiliary_state=auxiliary,
        step=jnp.asarray(manifest["step"], dtype=jnp.int64),
        training_id=prepared.training_id,
        layout_id=prepared.layout_id,
        state_id=manifest["state_id"],
    )
    recomputed = export_complex_training_state(
        prepared,
        import_complex_parameters(prepared.parameter_template, parameters),
        _import_optimizer(prepared, optimizer, ComplexImportPolicy()),
        _import_rng(prepared.rng_template, rng),
        auxiliary,
        step=int(state.step),
    )
    if recomputed.state_id != state.state_id:
        raise ValueError("Complex training checkpoint content identity mismatch.")
    return state


__all__ = [
    "ComplexOptimizerInterchangeState",
    "ComplexOptimizerRouteKind",
    "ComplexOptimizerStateGroup",
    "ComplexOptimizerStateLayout",
    "ComplexTrainingInterchangeState",
    "ImportedComplexTrainingState",
    "PreparedComplexTrainingInterchange",
    "RNGInterchangeState",
    "export_complex_training_state",
    "import_complex_training_state",
    "prepare_complex_optimizer_state_layout",
    "prepare_complex_training_interchange",
    "read_complex_training_checkpoint",
    "write_complex_training_checkpoint",
]
