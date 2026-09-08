#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only, bounded SONATA grouped circuits and population spike interchange.

This is an explicit-resource adapter, not a simulator configuration interpreter.
Only native templates are instantiated; foreign mechanisms are never executed.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO, StringIO
from math import isfinite, prod
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...interchange import (
    account_bounded_resource,
    AdapterError,
    AdapterReport,
    AdapterStatus,
    bounded_resource_from_bytes,
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
    ResourceManifest,
    ResourceReadError,
)
from ._cable import PreparedCableSolver
from ._neurons import AdaptiveExponentialIntegrateAndFire, LeakyIntegrateAndFire
from ._synapses import (
    ConductanceSynapse,
    CurrentSynapse,
    SynapseConnection,
    SynapseNetworkPlan,
)


_DEFAULT_LIMITS = ResourceLimits(64 * 1024 * 1024, 16, 2_000_000, 100_000, 100)
_BASE_PARAMETERS = (
    "capacitance_nF",
    "leak_conductance_uS",
    "resting_mV",
    "threshold_mV",
    "reset_mV",
    "refractory_ms",
)
_ADAPTATION_PARAMETERS = (
    "slope_mV",
    "adaptation_conductance_uS",
    "adaptation_time_constant_ms",
    "adaptation_increment_nA",
    "exponential_threshold_mV",
)
_MODEL_PARAMETERS = {
    "phydrax:LeakyIntegrateAndFire": _BASE_PARAMETERS,
    "phydrax:AdaptiveExponentialIntegrateAndFire": _BASE_PARAMETERS
    + _ADAPTATION_PARAMETERS,
    "phydrax:CurrentSynapse": ("time_constant_ms", "current_scale_nA"),
    "phydrax:ConductanceSynapse": (
        "time_constant_ms",
        "conductance_scale_uS",
        "reversal_mV",
    ),
}
_Scalar = str | int | float | bool
_Properties = tuple[tuple[str, _Scalar], ...]


@dataclass(frozen=True, slots=True)
class SONATAFilePair:
    """One HDF5 circuit resource and its space-separated type table."""

    h5_path: str | Path
    types_path: str | Path


@dataclass(frozen=True, slots=True)
class SONATACableBinding:
    """An explicitly supplied cable and exact (section, position, compartment) map.

    ``morphology`` must equal the node's morphology resource name. Its bytes must
    also be supplied in ``components``. The adapter does not infer section order
    or convert a foreign channel/template parameterization.
    """

    model: PreparedCableSolver
    morphology: str
    sites: tuple[tuple[int, float, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, PreparedCableSolver):
            raise TypeError("Cable bindings require a PreparedCableSolver.")
        if not isinstance(self.morphology, str) or not self.morphology:
            raise ValueError(
                "Cable bindings require an explicit morphology resource name."
            )
        count = self.model.morphology.plan.compartment_count
        seen = set()
        normalized = []
        for section, position, compartment in self.sites:
            section = _integer(section, "section id")
            compartment = _integer(compartment, "compartment index")
            position = _number(position, "section position")
            if not 0.0 <= position <= 1.0 or compartment >= count:
                raise ValueError(
                    "Cable site lies outside its section or native morphology."
                )
            if (section, position) in seen:
                raise ValueError(
                    "Cable site mapping is ambiguous: duplicate section/position."
                )
            seen.add((section, position))
            normalized.append((section, position, compartment))
        if not seen:
            raise ValueError("Cable bindings require an exact, nonempty site map.")
        object.__setattr__(self, "sites", tuple(normalized))


@dataclass(frozen=True, slots=True)
class SONATANode:
    population: str
    node_id: int
    type_id: int
    model: (
        LeakyIntegrateAndFire
        | AdaptiveExponentialIntegrateAndFire
        | PreparedCableSolver
        | None
    )
    properties: _Properties
    dynamics: _Properties
    cable_binding: SONATACableBinding | None = None

    @property
    def key(self) -> tuple[str, int]:
        return self.population, self.node_id

    @property
    def compartment_count(self) -> int:
        if isinstance(self.model, PreparedCableSolver):
            return self.model.morphology.plan.compartment_count
        return 1


@dataclass(frozen=True, slots=True)
class SONATAEdge:
    population: str
    edge_id: int
    type_id: int
    source: tuple[str, int]
    target: tuple[str, int]
    connection: SynapseConnection
    properties: _Properties
    dynamics: _Properties


@dataclass(frozen=True, slots=True)
class SONATASpikes:
    """Population-qualified recorded spikes, normalized to milliseconds.

    Arrays remain host NumPy arrays, with uint64 IDs and float64 timestamps.
    Import validates the declared order before normalizing to ``by_time``.
    """

    population: str
    node_ids: np.ndarray
    timestamps_ms: np.ndarray

    def __post_init__(self) -> None:
        ids = _ids(np.asarray(self.node_ids), "spike node_ids")
        times = np.asarray(self.timestamps_ms)
        if times.ndim != 1 or times.dtype.kind not in "fiu" or times.shape != ids.shape:
            raise ValueError("Spike timestamps and IDs must be matching numeric vectors.")
        if times.dtype.itemsize > 8 or (
            times.dtype.kind in "iu" and np.any(times > 2**53)
        ):
            raise ValueError("Spike timestamps cannot be narrowed exactly to float64.")
        times = times.astype(np.float64, copy=True)
        if not np.all(np.isfinite(times)) or np.any(times < 0):
            raise ValueError("Spike times must be finite and nonnegative.")
        order = np.lexsort((ids, times))
        ids = ids[order]
        times = times[order]
        ids.flags.writeable = False
        times.flags.writeable = False
        object.__setattr__(self, "node_ids", ids)
        object.__setattr__(self, "timestamps_ms", times)


@dataclass(frozen=True, slots=True)
class SONATAImport:
    nodes: tuple[SONATANode, ...]
    edges: tuple[SONATAEdge, ...]
    spikes: tuple[SONATASpikes, ...]
    synapse_plan: SynapseNetworkPlan
    report: AdapterReport
    resources: tuple[ResourceManifest, ...]
    components: tuple[tuple[str, BoundedResource], ...]


@dataclass(frozen=True, slots=True)
class SONATAExport:
    node_files: tuple[SONATAFilePair, ...]
    edge_files: tuple[SONATAFilePair, ...]
    spike_files: tuple[Path, ...]
    components: tuple[tuple[str, Path], ...]
    report: AdapterReport
    resources: tuple[ResourceManifest, ...]


def _unsupported(message: str) -> None:
    raise AdapterError(AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC, message)


def _integer(value, name: str) -> int:
    if isinstance(value, str):
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f"{name} must be an unsigned integer, without narrowing.")
        value = int(value)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an unsigned integer, not bool.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an unsigned integer, without narrowing.")
    result = int(value)
    if not 0 <= result <= np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must fit uint64.")
    return result


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite number, not bool.")
    if isinstance(value, (int, np.integer)) and abs(int(value)) > 2**53:
        raise ValueError(f"{name} cannot be narrowed exactly to float64.")
    if (
        isinstance(value, str)
        and value.lstrip("+-").isdecimal()
        and abs(int(value)) > 2**53
    ):
        raise ValueError(f"{name} cannot be narrowed exactly to float64.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _ids(value: np.ndarray, name: str) -> np.ndarray:
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a rank-one integer dataset.")
    if value.dtype.kind == "i" and np.any(value < 0):
        raise ValueError(f"{name} cannot contain negative IDs.")
    if value.dtype.itemsize > 8:
        raise ValueError(f"{name} cannot be narrowed to uint64.")
    return value.astype(np.uint64, copy=False)


def _scalar(value) -> _Scalar:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if not isinstance(value, (str, int, float, bool)):
        _unsupported(
            "Only scalar string, numeric and boolean SONATA properties are supported."
        )
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("SONATA properties cannot contain NaN or infinity.")
    return value


def _text(value, name: str) -> str:
    value = _scalar(value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string.")
    return value


class _Reader:
    """One aggregate byte/element budget across all explicitly supplied resources."""

    def __init__(self, root, limits: ResourceLimits, max_decoded_bytes: int):
        self.root = root
        self.limits = limits
        self.max_decoded_bytes = _integer(max_decoded_bytes, "max_decoded_bytes")
        if not self.max_decoded_bytes:
            raise ValueError("max_decoded_bytes must be positive.")
        self.decoded_bytes = 0
        self.elements = 0
        self.attributes = 0
        self.resources: list[ResourceManifest] = []
        self.cache: dict[str, BoundedResource] = {}

    def charge(self, size: int, elements: int = 0) -> None:
        self.decoded_bytes += size
        self.elements += elements
        if (
            self.decoded_bytes > self.max_decoded_bytes
            or self.elements > self.limits.max_nodes
        ):
            raise ResourceReadError(
                "limit", "SONATA aggregate decoded arrays exceed the configured limit."
            )

    def load(self, path) -> BoundedResource:
        key = str(path)
        if key not in self.cache:
            resource = read_bounded_resource(
                path, trusted_root=self.root, limits=self.limits
            )
            self.charge(len(resource.data))
            self.cache[key] = resource
        return self.cache[key]

    def record(self, resource, *, depth: int, nodes: int, attributes: int = 0) -> None:
        accounted = account_bounded_resource(
            resource, depth=depth, nodes=nodes, attributes=attributes, losses=0
        )
        self.resources.append(accounted.manifest)

    def hdf5(self, path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, _Scalar]]]:
        resource = self.load(path)
        self.charge(len(resource.data))
        arrays: dict[str, np.ndarray] = {}
        attributes: dict[str, dict[str, _Scalar]] = {}
        depth = 0
        count = 0
        attribute_count = 0
        seen = set()
        with h5py.File(BytesIO(resource.data), "r") as handle:
            stack = [("", handle, 0)]
            datasets = []
            while stack:
                name, obj, level = stack.pop()
                depth = max(depth, level)
                count += 1
                if depth > self.limits.max_depth or count > self.limits.max_nodes:
                    raise ResourceReadError(
                        "limit", "SONATA HDF5 structure exceeds its bounds."
                    )
                address = h5py.h5o.get_info(obj.id).addr
                if address in seen:
                    _unsupported("HDF5 hard-link aliases and cycles are unsupported.")
                seen.add(address)
                attributes[name] = {}
                for key in obj.attrs:
                    attribute_count += 1
                    self.attributes += 1
                    if self.attributes > self.limits.max_attributes:
                        raise ResourceReadError(
                            "limit", "SONATA attribute count exceeds its limit."
                        )
                    aid = obj.attrs.get_id(key)
                    size = prod(aid.shape)
                    dtype = aid.dtype
                    if (
                        h5py.check_dtype(ref=dtype) is not None
                        or dtype.fields is not None
                    ):
                        _unsupported(
                            "HDF5 reference and compound attributes are unsupported."
                        )
                    if dtype.hasobject:
                        if h5py.check_string_dtype(dtype) is None:
                            _unsupported("HDF5 object attributes are unsupported.")
                        # Variable strings reside in the bounded file's global heap.
                        # Reserve a whole source image per scalar before decoding.
                        self.charge(size * len(resource.data), size)
                    else:
                        self.charge(size * dtype.itemsize, size)
                    value = obj.attrs[key]
                    if name == "" and key in ("version", "magic"):
                        continue
                    allowed = (
                        (
                            name.startswith("spikes/")
                            and name.count("/") == 1
                            and key == "sorting"
                        )
                        or (name.endswith("/timestamps") and key == "units")
                        or (
                            name.endswith(("/source_node_id", "/target_node_id"))
                            and key == "node_population"
                        )
                    )
                    if not allowed:
                        _unsupported(f"Unsupported HDF5 attribute {name}@{key}.")
                    if np.asarray(value).ndim != 0:
                        _unsupported("Non-scalar SONATA attributes are unsupported.")
                    if key == "sorting" and h5py.check_enum_dtype(dtype) is not None:
                        enum = h5py.check_enum_dtype(dtype)
                        labels = [
                            label for label, code in enum.items() if code == int(value)
                        ]
                        if len(labels) != 1:
                            raise ValueError("Invalid SONATA spike sorting enum.")
                        value = labels[0]
                    attributes[name][key] = _scalar(value)
                if isinstance(obj, h5py.Group):
                    for key in obj:
                        if count + len(stack) >= self.limits.max_nodes:
                            raise ResourceReadError(
                                "limit", "SONATA HDF5 object count exceeds its limit."
                            )
                        self.charge(len(key.encode("utf-8")))
                        link = obj.get(key, getlink=True)
                        if not isinstance(link, h5py.HardLink):
                            _unsupported("HDF5 external and soft links are forbidden.")
                        child = obj[key]
                        stack.append((f"{name}/{key}".lstrip("/"), child, level + 1))
                elif isinstance(obj, h5py.Dataset):
                    if obj.shape is None:
                        _unsupported("HDF5 null-dataspace datasets are unsupported.")
                    if obj.is_virtual or obj.external:
                        _unsupported(
                            "HDF5 virtual datasets and external storage are forbidden."
                        )
                    dtype = obj.dtype
                    if (
                        dtype.hasobject
                        or dtype.fields is not None
                        or h5py.check_dtype(ref=dtype) is not None
                    ):
                        _unsupported(
                            "HDF5 variable-length/object/reference/compound datasets "
                            "are unsupported; use fixed-width UTF-8 strings."
                        )
                    if dtype.kind not in "biufS":
                        _unsupported("Unsupported HDF5 dataset dtype.")
                    plist = obj.id.get_create_plist()
                    for index in range(plist.get_nfilters()):
                        if plist.get_filter(index)[0] not in (1, 2, 3):
                            _unsupported(
                                "Only HDF5 deflate, shuffle and Fletcher32 filters are supported."
                            )
                    # Include uint64/float64 normalization before reading narrow arrays.
                    self.charge(obj.size * (dtype.itemsize + 8), obj.size)
                    datasets.append((name, obj))
                else:
                    _unsupported("HDF5 named datatypes are unsupported.")
            # No payload read occurs before every link, dtype, filter and bound is checked.
            for name, dataset in datasets:
                arrays[name] = dataset[()]
        self.record(
            resource,
            depth=depth,
            nodes=count + sum(array.size for array in arrays.values()),
            attributes=attribute_count,
        )
        return arrays, attributes

    def types(self, path, kind: str) -> dict[int, dict[str, _Scalar]]:
        resource = self.load(path)
        text = resource.data.decode("ascii", errors="strict")
        self.charge(len(text))
        rows = csv.reader(
            StringIO(text),
            delimiter=" ",
            quotechar='"',
            skipinitialspace=True,
            strict=True,
        )
        header = next(rows, None)
        if (
            not header
            or len(header) != len(set(header))
            or f"{kind}_type_id" not in header
        ):
            raise ValueError(
                "SONATA type CSV requires unique headers and its type ID column."
            )
        if any(not key or "/" in key or "\x00" in key for key in header):
            raise ValueError(
                "SONATA CSV property names must be nonempty HDF5 leaf names."
            )
        result = {}
        for row in rows:
            if not row:
                continue
            self.charge(sum(len(item) for item in row), len(row))
            if len(row) != len(header):
                raise ValueError("SONATA type CSV row width differs from its header.")
            values = dict(zip(header, row, strict=True))
            identifier = _integer(values.pop(f"{kind}_type_id"), f"{kind}_type_id")
            if identifier in result:
                raise ValueError("Duplicate SONATA type ID.")
            result[identifier] = {
                key: value for key, value in values.items() if value != "NULL"
            }
        self.record(resource, depth=1, nodes=len(result) + 1, attributes=len(header))
        return result

    def parameters(self, resource: BoundedResource) -> dict[str, _Scalar]:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate component JSON parameter.")
                result[key] = value
            return result

        # Native parameter components are deliberately flat; lexical depth is
        # checked before the JSON decoder can recurse into untrusted input.
        text = resource.data.decode("utf-8", errors="strict")
        quoted = escaped = False
        depth = maximum = 0
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                depth += 1
                maximum = max(maximum, depth)
                if maximum > self.limits.max_depth:
                    raise ResourceReadError(
                        "limit", "Component JSON nesting exceeds its limit."
                    )
                if maximum > 1 or char == "[":
                    _unsupported("Native dynamics JSON must be a flat scalar object.")
            elif char in "}]":
                depth -= 1
        value = json.loads(text, object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise TypeError("A SONATA dynamics component must be a flat JSON object.")
        result = {key: _scalar(item) for key, item in value.items()}
        self.charge(len(text), len(result))
        self.record(resource, depth=maximum, nodes=len(result) + 1)
        return result


def _populations(attributes, root: str) -> tuple[str, ...]:
    if root not in attributes:
        raise ValueError(f"Missing SONATA /{root} group.")
    if any(
        path and path != root and not path.startswith(root + "/") for path in attributes
    ):
        _unsupported(f"Only the /{root} content is supported in this explicit resource.")
    return tuple(
        sorted(
            path[len(root) + 1 :]
            for path in attributes
            if path.startswith(f"{root}/") and path.count("/") == 1
        )
    )


def _records(arrays, attributes, population_path, types, kind: str):
    prefix = population_path + "/"
    required = (f"{kind}_type_id", f"{kind}_group_id", f"{kind}_group_index")
    columns = {}
    for name in required:
        if prefix + name not in arrays:
            raise ValueError(f"Missing SONATA dataset {prefix + name}.")
        columns[name] = _ids(arrays[prefix + name], name)
    count = len(columns[required[0]])
    if any(len(column) != count for column in columns.values()):
        raise ValueError("SONATA population columns have inconsistent lengths.")
    id_name = prefix + f"{kind}_id"
    identifiers = (
        _ids(arrays[id_name], id_name)
        if id_name in arrays
        else np.arange(count, dtype=np.uint64)
    )
    if len(identifiers) != count or len(set(identifiers.tolist())) != count:
        raise ValueError("SONATA population IDs must be unique and match record count.")
    top_allowed = set(required) | {f"{kind}_id"}
    if kind == "edge":
        top_allowed |= {"source_node_id", "target_node_id"}
    for path in arrays:
        if (
            path.startswith(prefix)
            and "/" not in path[len(prefix) :]
            and path[len(prefix) :] not in top_allowed
        ):
            _unsupported(
                "SONATA record properties must be in indexed property groups, not flat population columns."
            )
    groups = {}
    for group_id in set(columns[f"{kind}_group_id"].tolist()):
        group_prefix = f"{prefix}{group_id}/"
        if group_prefix[:-1] not in attributes:
            raise ValueError("SONATA record references a missing property group.")
        group_arrays = {}
        group_size = None
        for path, value in arrays.items():
            if not path.startswith(group_prefix):
                continue
            name = path[len(group_prefix) :]
            if "/" in name and (
                not name.startswith("dynamics_params/") or name.count("/") != 1
            ):
                _unsupported("Unsupported nested SONATA property group.")
            if value.ndim != 1:
                _unsupported(
                    "SONATA per-record properties must be rank-one scalar arrays."
                )
            if group_size is None:
                group_size = len(value)
            elif group_size != len(value):
                raise ValueError(
                    "SONATA property group columns have inconsistent lengths."
                )
            group_arrays[name] = value
        groups[group_id] = (group_arrays, group_size)
    for row in range(count):
        type_id = int(columns[f"{kind}_type_id"][row])
        if type_id not in types:
            raise ValueError(f"Missing binding for SONATA {kind} type {type_id}.")
        props = dict(types[type_id])
        params = {}
        group_id = int(columns[f"{kind}_group_id"][row])
        index = int(columns[f"{kind}_group_index"][row])
        group_arrays, group_size = groups[group_id]
        # Empty groups carry no properties, but their logical rows still have a bound.
        logical_size = group_size if group_size is not None else count
        if index >= logical_size:
            raise ValueError("SONATA group index is out of bounds.")
        for name, values in group_arrays.items():
            if name.startswith("dynamics_params/"):
                params[name.removeprefix("dynamics_params/")] = _scalar(values[index])
            else:
                props[name] = _scalar(values[index])
        yield row, int(identifiers[row]), type_id, props, params


def _resolve_parameters(props, overrides, components, reader, parsed_components):
    reference = props.pop("dynamics_params", None)
    parameters = {}
    if reference is not None and reference != "NULL":
        reference = _text(reference, "dynamics_params resource")
        if reference not in components:
            _unsupported(f"Missing explicit component resource binding: {reference}.")
        if reference not in parsed_components:
            parsed_components[reference] = reader.parameters(components[reference])
        parameters.update(parsed_components[reference])
    parameters.update(overrides)
    return parameters


def _native_model(template, parameters, *, neuron: bool):
    templates = tuple(_MODEL_PARAMETERS)[:2] if neuron else tuple(_MODEL_PARAMETERS)[2:]
    if template not in templates:
        _unsupported(
            f"Unsupported native model template: {template!r}; foreign mechanisms are never executed."
        )
    names = _MODEL_PARAMETERS[template]
    required = set(names) - (
        {"refractory_ms", "exponential_threshold_mV"} if neuron else set()
    )
    if not required <= parameters.keys() or not parameters.keys() <= set(names):
        raise ValueError(
            f"Native template {template} requires exactly its documented parameter names."
        )
    values = {name: _number(value, name) for name, value in parameters.items()}
    if neuron:
        values.setdefault("refractory_ms", 0.0)
    if template == "phydrax:LeakyIntegrateAndFire":
        model = LeakyIntegrateAndFire(**values)
    elif template == "phydrax:AdaptiveExponentialIntegrateAndFire":
        values.setdefault("exponential_threshold_mV", -50.0)
        model = AdaptiveExponentialIntegrateAndFire(**values)
    elif template == "phydrax:CurrentSynapse":
        model = CurrentSynapse(*(values[name] for name in names))
    else:
        model = ConductanceSynapse(*(values[name] for name in names))
    if dict(_model_values(model)) != values:
        _unsupported(
            "Native parameter conversion loses precision; enable jax_enable_x64 before importing this circuit."
        )
    return model, tuple(sorted(values.items()))


def _model_values(model) -> _Properties:
    if model is None:
        return ()
    if isinstance(model, PreparedCableSolver):
        return (("runtime_id", model.runtime_id),)
    if isinstance(model, CurrentSynapse):
        return (
            ("current_scale_nA", model.current_scale_nA),
            ("time_constant_ms", model.time_constant_ms),
        )
    if isinstance(model, ConductanceSynapse):
        return (
            ("conductance_scale_uS", model.conductance_scale_uS),
            ("reversal_mV", model.reversal_mV),
            ("time_constant_ms", model.time_constant_ms),
        )
    values = {
        "capacitance_nF": float(model.capacitance_nF),
        "leak_conductance_uS": float(model.leak_conductance_uS),
        "resting_mV": float(model.resting_mV),
        "threshold_mV": float(model.threshold_mV),
        "reset_mV": float(model.reset_mV),
        "refractory_ms": float(model.refractory_ms),
    }
    if isinstance(model, AdaptiveExponentialIntegrateAndFire):
        values.update(
            {
                "slope_mV": float(model.slope_mV),
                "adaptation_conductance_uS": float(model.adaptation_conductance_uS),
                "adaptation_time_constant_ms": float(model.adaptation_time_constant_ms),
                "adaptation_increment_nA": float(model.adaptation_increment_nA),
                "exponential_threshold_mV": float(model.exponential_threshold_mV),
            }
        )
    return tuple(sorted(values.items()))


def _site(node: SONATANode, props, prefix: str) -> int:
    id_field = f"{prefix}_section_id"
    pos_field = f"{prefix}_section_pos"
    unsupported = [
        name
        for name in props
        if name.startswith(prefix + "_") and name not in (id_field, pos_field)
    ]
    if unsupported:
        _unsupported(f"Unsupported synaptic site coordinates: {unsupported}.")
    if (id_field in props) != (pos_field in props):
        raise ValueError("A synaptic section ID and position must be specified together.")
    if id_field not in props:
        if node.cable_binding is not None:
            _unsupported(
                "Cable synapses require an explicit exact section/position mapping."
            )
        return 0
    section = _integer(props[id_field], id_field)
    position = _number(props[pos_field], pos_field)
    if not 0.0 <= position <= 1.0:
        raise ValueError("Synaptic section positions must lie in [0, 1].")
    if node.cable_binding is None:
        _unsupported(
            "Section-based synaptic sites require an explicit cable binding, not a point or virtual node."
        )
    mapping = {
        (sid, pos): compartment for sid, pos, compartment in node.cable_binding.sites
    }
    if (section, position) not in mapping:
        _unsupported("Synaptic site has no exact native compartment mapping.")
    return mapping[section, position]


def _on_grid(value: float, dt: float, name: str) -> None:
    if value < 0 or not isfinite(value / dt) or value / dt > np.iinfo(np.int32).max:
        _unsupported(f"{name} cannot be represented by the native clock.")
    steps = round(value / dt)
    # Decimal-looking float inputs get only a representational ULP allowance,
    # never a physical tolerance that could silently round an off-grid event.
    if abs(steps * dt - value) > max(np.spacing(value), np.spacing(steps * dt)) * 2:
        _unsupported(f"{name} is off-grid for the requested clock dt_ms.")


def _semantic_id(nodes, edges, spikes) -> str:
    return canonical_fingerprint(
        {
            "kind": "sonata-native-circuit-semantics",
            "nodes": [
                [
                    node.population,
                    node.node_id,
                    node.type_id,
                    list(node.properties),
                    list(node.dynamics),
                    list(_model_values(node.model)),
                    None
                    if node.cable_binding is None
                    else [
                        node.cable_binding.model.runtime_id,
                        node.cable_binding.morphology,
                        list(node.cable_binding.sites),
                    ],
                ]
                for node in sorted(nodes, key=lambda value: value.key)
            ],
            "edges": [
                [
                    edge.population,
                    edge.edge_id,
                    edge.type_id,
                    edge.source,
                    edge.target,
                    list(edge.properties),
                    list(edge.dynamics),
                    list(_model_values(edge.connection.model)),
                    edge.connection.pre_cell,
                    edge.connection.pre_compartment,
                    edge.connection.post_cell,
                    edge.connection.post_compartment,
                    edge.connection.weight,
                    edge.connection.delay_ms,
                ]
                for edge in sorted(
                    edges, key=lambda value: (value.population, value.edge_id)
                )
            ],
            "spikes": [
                [spike.population, spike.node_ids.tolist(), spike.timestamps_ms.tolist()]
                for spike in sorted(spikes, key=lambda value: value.population)
            ],
        }
    )


def _report(source, target, source_id, target_id, *, stage="adapter"):
    return AdapterReport(
        AdapterStatus.LOSSLESS,
        source,
        target,
        source_id=source_id,
        target_id=target_id,
        stage=stage,
        preserved_fields=(
            "population-qualified node/edge identities",
            "parallel edges and physical delays",
            "resolved native model parameters",
            "resolved per-record metadata",
            "recorded spike population, ID and time",
        ),
        coordinate_mapping=(
            "SONATA timestamps s or ms -> native milliseconds",
            "explicit exact section/position -> native compartment",
        ),
        assumptions=(
            (
                "Native electrophysiology units: ms, mV, nA, uS, nF; "
                "syn_weight is dimensionless and model scale has physical units."
            ),
            (
                "Semantic, not byte/layout, equivalence; property-group layout, "
                "redundant type defaults and auxiliary edge indices are not "
                "executable semantics."
            ),
            (
                "Explicit cable bindings supply morphology and mechanism "
                "interpretation; no foreign model code is executed."
            ),
        ),
    )


def import_sonata(
    *,
    node_files: Sequence[SONATAFilePair],
    edge_files: Sequence[SONATAFilePair] = (),
    spike_files: Sequence[str | Path] = (),
    trusted_root: str | Path,
    components: Mapping[str, str | Path] | None = None,
    cable_bindings: Mapping[tuple[str, int], SONATACableBinding] | None = None,
    dt_ms: float = 0.1,
    execution: Literal["clock", "event"] = "event",
    limits: ResourceLimits = _DEFAULT_LIMITS,
    max_decoded_bytes: int = 256 * 1024 * 1024,
) -> SONATAImport:
    """Load real grouped SONATA files into native neurons, synapses and spikes.

    Precedence is type properties < indexed HDF5 properties, and component JSON
    dynamics < indexed HDF5 dynamics. Resources are explicit trusted-root-relative
    paths. No config traversal, environment expansion or template loading occurs.
    """
    dt = _number(dt_ms, "dt_ms")
    if dt <= 0 or execution not in ("event", "clock"):
        raise ValueError("dt_ms must be positive and execution must be event or clock.")
    reader = _Reader(trusted_root, limits, max_decoded_bytes)
    component_resources = {
        name: reader.load(path) for name, path in (components or {}).items()
    }
    bindings = dict(cable_bindings or {})
    parsed_components = {}
    nodes = []
    population_names = set()
    for files in node_files:
        types = reader.types(files.types_path, "node")
        arrays, attrs = reader.hdf5(files.h5_path)
        for population in _populations(attrs, "nodes"):
            if population in population_names:
                raise ValueError(
                    "A SONATA node population must be self-contained in one resource."
                )
            population_names.add(population)
            for _, identifier, type_id, props, overrides in _records(
                arrays, attrs, f"nodes/{population}", types, "node"
            ):
                params = _resolve_parameters(
                    props, overrides, component_resources, reader, parsed_components
                )
                model_type = props.get("model_type")
                binding = bindings.pop((population, identifier), None)
                if model_type == "virtual":
                    if (
                        params
                        or props.get("model_template", "NULL") != "NULL"
                        or binding is not None
                    ):
                        _unsupported(
                            "Virtual sources cannot have executable templates, dynamics or cable bindings."
                        )
                    props.pop("model_template", None)
                    model, dynamics = None, ()
                elif model_type == "point_neuron":
                    if binding is not None:
                        raise ValueError("A point neuron cannot have a cable binding.")
                    model, dynamics = _native_model(
                        props.get("model_template"), params, neuron=True
                    )
                elif model_type == "biophysical":
                    if binding is None:
                        _unsupported(
                            "Biophysical nodes require an explicitly supplied native cable binding."
                        )
                    if params:
                        _unsupported(
                            "Foreign cable dynamics overrides cannot be applied to a supplied native cable."
                        )
                    if (
                        props.get("morphology") != binding.morphology
                        or binding.morphology not in component_resources
                    ):
                        _unsupported(
                            "Cable morphology requires a matching explicit component resource and binding."
                        )
                    model, dynamics = binding.model, ()
                else:
                    _unsupported(f"Unsupported SONATA model_type: {model_type!r}.")
                nodes.append(
                    SONATANode(
                        population,
                        identifier,
                        type_id,
                        model,
                        tuple(sorted(props.items())),
                        dynamics,
                        binding,
                    )
                )
    if not nodes:
        raise ValueError("A native SONATA circuit requires at least one node.")
    if bindings:
        raise ValueError(
            "Cable bindings reference nonexistent population-qualified nodes."
        )
    nodes.sort(key=lambda node: node.key)
    lookup = {node.key: (index, node) for index, node in enumerate(nodes)}
    if len(nodes) > np.iinfo(np.int32).max:
        raise ValueError("Native cell indices cannot be narrowed to int32.")
    edges = []
    edge_populations = set()
    for files in edge_files:
        types = reader.types(files.types_path, "edge")
        arrays, attrs = reader.hdf5(files.h5_path)
        for population in _populations(attrs, "edges"):
            if population in edge_populations:
                raise ValueError(
                    "A SONATA edge population must be self-contained in one resource."
                )
            edge_populations.add(population)
            prefix = f"edges/{population}/"
            source_path, target_path = (
                prefix + "source_node_id",
                prefix + "target_node_id",
            )
            if source_path not in arrays or target_path not in arrays:
                raise ValueError("SONATA edges require source and target node datasets.")
            sources = _ids(arrays[source_path], source_path)
            targets = _ids(arrays[target_path], target_path)
            source_population = _text(
                attrs[source_path].get("node_population", ""), "source node_population"
            )
            target_population = _text(
                attrs[target_path].get("node_population", ""), "target node_population"
            )
            count = len(arrays[prefix + "edge_type_id"])
            if len(sources) != count or len(targets) != count:
                raise ValueError(
                    "SONATA edge endpoint columns have inconsistent lengths."
                )
            for row, identifier, type_id, props, overrides in _records(
                arrays, attrs, prefix[:-1], types, "edge"
            ):
                source, target = (
                    (source_population, int(sources[row])),
                    (target_population, int(targets[row])),
                )
                if source not in lookup or target not in lookup:
                    raise ValueError(
                        "SONATA edge references a missing population-qualified node."
                    )
                pre_index, pre = lookup[source]
                post_index, post = lookup[target]
                if post.model is None:
                    _unsupported("Virtual spike sources cannot receive synaptic inputs.")
                params = _resolve_parameters(
                    props, overrides, component_resources, reader, parsed_components
                )
                model, dynamics = _native_model(
                    props.get("model_template"), params, neuron=False
                )
                if "delay" not in props or "syn_weight" not in props:
                    raise ValueError(
                        "Native SONATA edges require explicit delay and syn_weight."
                    )
                delay, weight = (
                    _number(props["delay"], "delay"),
                    _number(props["syn_weight"], "syn_weight"),
                )
                if "nsyns" in props and _integer(props["nsyns"], "nsyns") != 1:
                    _unsupported(
                        "Aggregated nsyns are unsupported; represent each synapse as its own edge."
                    )
                if execution == "clock":
                    _on_grid(delay, dt, "Synaptic delay")
                props["delay"], props["syn_weight"] = delay, weight
                for name in ("afferent_section_id", "efferent_section_id", "nsyns"):
                    if name in props:
                        props[name] = _integer(props[name], name)
                for name in ("afferent_section_pos", "efferent_section_pos"):
                    if name in props:
                        props[name] = _number(props[name], name)
                connection = SynapseConnection(
                    json.dumps([population, identifier], separators=(",", ":")),
                    pre_index,
                    _site(pre, props, "efferent"),
                    post_index,
                    _site(post, props, "afferent"),
                    model,
                    delay_ms=delay,
                    weight=weight,
                )
                edges.append(
                    SONATAEdge(
                        population,
                        identifier,
                        type_id,
                        source,
                        target,
                        connection,
                        tuple(sorted(props.items())),
                        dynamics,
                    )
                )
    edges.sort(key=lambda edge: (edge.population, edge.edge_id))
    if len(edges) > np.iinfo(np.int32).max:
        raise ValueError("Native relation slots cannot be narrowed to int32.")
    spikes = []
    spike_populations = set()
    for path in spike_files:
        arrays, attrs = reader.hdf5(path)
        for population in _populations(attrs, "spikes"):
            if population in spike_populations:
                raise ValueError(
                    "Spike populations must be self-contained in one resource."
                )
            spike_populations.add(population)
            if population not in population_names:
                raise ValueError("Spike population has no node population binding.")
            prefix = f"spikes/{population}/"
            if prefix + "timestamps" not in arrays or prefix + "node_ids" not in arrays:
                raise ValueError("SONATA spikes require timestamps and node_ids.")
            ids = _ids(arrays[prefix + "node_ids"], "spike node_ids")
            times = arrays[prefix + "timestamps"]
            if (
                times.ndim != 1
                or times.dtype.kind != "f"
                or times.dtype.itemsize > 8
                or times.shape != ids.shape
            ):
                raise ValueError(
                    "SONATA timestamps must be floating vectors matching node_ids."
                )
            sorting = attrs[prefix[:-1]].get("sorting", "none")
            if sorting not in ("none", "by_id", "by_time"):
                raise ValueError("Invalid SONATA spike sorting declaration.")
            if sorting == "by_time" and np.any(times[1:] < times[:-1]):
                raise ValueError("SONATA spike timestamps violate by_time sorting.")
            if sorting == "by_id" and (
                np.any(ids[1:] < ids[:-1])
                or np.any((ids[1:] == ids[:-1]) & (times[1:] < times[:-1]))
            ):
                raise ValueError(
                    "SONATA spikes violate by_id sorting including timestamp secondary key."
                )
            units = attrs[prefix + "timestamps"].get("units")
            if units not in ("ms", "s"):
                _unsupported(
                    "SONATA spike timestamps require explicit 'ms' or 's' units."
                )
            reader.charge(ids.size * 32, ids.size)
            times = times.astype(np.float64) * (1000.0 if units == "s" else 1.0)
            if any((population, int(identifier)) not in lookup for identifier in ids):
                raise ValueError("A recorded spike references a missing node.")
            spike = SONATASpikes(population, ids, times)
            if execution == "clock":
                for time in spike.timestamps_ms:
                    _on_grid(float(time), dt, "Recorded spike time")
            spikes.append(spike)
    connections = tuple(edge.connection for edge in edges)
    plan = SynapseNetworkPlan(
        tuple(node.compartment_count for node in nodes),
        max(1, len(edges)),
        max((connection.delay_ms for connection in connections), default=0.0),
        dt,
        connections=connections,
        execution=execution,
    )
    # Every supplied component is accounted, including opaque morphology bytes.
    accounted = {manifest.content_sha256 for manifest in reader.resources}
    for resource in component_resources.values():
        if resource.manifest.content_sha256 not in accounted:
            reader.record(resource, depth=0, nodes=0)
    node_tuple, edge_tuple, spike_tuple = (
        tuple(nodes),
        tuple(edges),
        tuple(sorted(spikes, key=lambda spike: spike.population)),
    )
    semantic_id = _semantic_id(node_tuple, edge_tuple, spike_tuple)
    source_id = canonical_fingerprint(
        {"resources": [manifest.content_sha256 for manifest in reader.resources]}
    )
    return SONATAImport(
        node_tuple,
        edge_tuple,
        spike_tuple,
        plan,
        _report("SONATA", "phydrax-native-neural-circuit", source_id, semantic_id),
        tuple(reader.resources),
        tuple(component_resources.items()),
    )


def prepare_sonata_network(
    circuit: SONATAImport,
    /,
    *,
    queue_capacity: int,
    spike_capacity: int,
    recording_capacity: int,
    maximum_events_per_step: int = 128,
    root_subdivisions: int = 4,
    cable_detectors: Mapping[tuple[str, int], tuple[int | str, float, float]]
    | None = None,
    learning=None,
):
    """Lower imported native models and virtual spikes into neural execution.

    The imported node order is preserved, matching ``circuit.synapse_plan``.
    Spike records attached to virtual populations become external emissions;
    spike records attached to physical populations remain observations only.
    Cable models require an explicit detector compartment, upward threshold,
    and rearm voltage because SONATA morphology binding does not define those
    event semantics.
    """
    if not isinstance(circuit, SONATAImport):
        raise TypeError("circuit must be a SONATAImport.")
    from ._network import NeuralCellPlan, NeuralNetworkPlan, SpikeSource

    detector_map = dict(cable_detectors or {})
    cells = []
    cell_ids = {}
    for node in circuit.nodes:
        cell_id = json.dumps(node.key, separators=(",", ":"))
        cell_ids[node.key] = cell_id
        if node.model is None:
            cells.append(NeuralCellPlan(cell_id, SpikeSource()))
        elif isinstance(node.model, PreparedCableSolver):
            if node.key not in detector_map:
                _unsupported(
                    "Executing a SONATA cable node requires an explicit "
                    "compartment, threshold, and rearm binding."
                )
            compartment, threshold, rearm = detector_map.pop(node.key)
            detector = (
                node.model.morphology.plan.compartment_index(compartment)
                if isinstance(compartment, str)
                else compartment
            )
            cells.append(
                NeuralCellPlan(
                    cell_id,
                    node.model,
                    detector_compartment=detector,
                    threshold_mV=threshold,
                    rearm_mV=rearm,
                )
            )
        else:
            cells.append(NeuralCellPlan(cell_id, node.model))
    if detector_map:
        raise ValueError(
            "Cable detector bindings reference nonexistent or non-cable nodes."
        )

    virtual = {node.key for node in circuit.nodes if node.model is None}
    emissions = []
    for spike_group in circuit.spikes:
        for node_id, time_ms in zip(
            spike_group.node_ids, spike_group.timestamps_ms, strict=True
        ):
            key = spike_group.population, int(node_id)
            if key in virtual:
                emissions.append((float(time_ms), cell_ids[key]))
    emissions.sort(key=lambda event: (event[0], event[1]))
    return NeuralNetworkPlan(
        tuple(cells),
        circuit.synapse_plan,
        queue_capacity=queue_capacity,
        spike_capacity=spike_capacity,
        recording_capacity=recording_capacity,
        maximum_events_per_step=maximum_events_per_step,
        root_subdivisions=root_subdivisions,
        external_spikes=tuple(emissions),
        learning=learning,
    ).prepare()


class _BoundedBuffer(BytesIO):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise ResourceReadError("limit", "Export HDF5 exceeds its byte limit.")
        return super().write(data)


def _dataset(group, name, values):
    if values and isinstance(values[0], str):
        encoded = [value.encode("utf-8") for value in values]
        width = max(1, max(map(len, encoded)))
        group.create_dataset(name, data=np.asarray(encoded, dtype=f"S{width}"))
    elif values and type(values[0]) is int:
        if min(values) < 0 and max(values) > np.iinfo(np.int64).max:
            _unsupported(
                "Mixed-sign integer metadata cannot be exported without narrowing."
            )
        dtype = np.int64 if min(values) < 0 else np.uint64
        group.create_dataset(name, data=np.asarray(values, dtype=dtype))
    else:
        group.create_dataset(name, data=np.asarray(values))


def _write_population(group, records, kind):
    group.create_dataset(
        f"{kind}_id",
        data=np.asarray(
            [record.node_id if kind == "node" else record.edge_id for record in records],
            dtype=np.uint64,
        ),
    )
    group.create_dataset(
        f"{kind}_type_id",
        data=np.asarray([record.type_id for record in records], dtype=np.uint64),
    )
    schemas = {}
    schema_ids = {}
    group_ids, group_indices = [], []
    for record in records:
        signature = (
            tuple((key, type(value).__name__) for key, value in record.properties),
            tuple(key for key, _ in record.dynamics),
        )
        if signature not in schemas:
            schemas[signature] = []
            schema_ids[signature] = len(schema_ids)
        group_ids.append(schema_ids[signature])
        group_indices.append(len(schemas[signature]))
        schemas[signature].append(record)
    group.create_dataset(f"{kind}_group_id", data=np.asarray(group_ids, dtype=np.uint32))
    group.create_dataset(
        f"{kind}_group_index", data=np.asarray(group_indices, dtype=np.uint64)
    )
    for group_id, members in enumerate(schemas.values()):
        properties = group.create_group(str(group_id))
        for name, _ in members[0].properties:
            _dataset(
                properties, name, [dict(record.properties)[name] for record in members]
            )
        if members[0].dynamics:
            dynamics = properties.create_group("dynamics_params")
            for name, _ in members[0].dynamics:
                _dataset(
                    dynamics, name, [dict(record.dynamics)[name] for record in members]
                )
    if kind == "edge":
        sources = group.create_dataset(
            "source_node_id",
            data=np.asarray([record.source[1] for record in records], dtype=np.uint64),
        )
        targets = group.create_dataset(
            "target_node_id",
            data=np.asarray([record.target[1] for record in records], dtype=np.uint64),
        )
        sources.attrs["node_population"] = np.bytes_(records[0].source[0].encode("utf-8"))
        targets.attrs["node_population"] = np.bytes_(records[0].target[0].encode("utf-8"))


def export_sonata(
    circuit: SONATAImport,
    directory: str | Path,
    /,
    *,
    limits: ResourceLimits = _DEFAULT_LIMITS,
) -> SONATAExport:
    """Export the supported resolved circuit and spikes to a new directory.

    Group/type layout is canonicalized, not byte-preserved. Cable mechanism
    objects stay caller-owned: re-import still requires the same explicit native
    bindings. Referenced component bytes are copied without interpreting code.
    """
    if not isinstance(circuit, SONATAImport):
        raise TypeError("export_sonata requires a supported SONATAImport.")
    semantic_id = _semantic_id(circuit.nodes, circuit.edges, circuit.spikes)
    if semantic_id != circuit.report.target_id:
        raise ValueError(
            "SONATAImport semantics changed; import a new coherent circuit before exporting."
        )
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    manifests = []
    node_files, edge_files = [], []

    def save(name, data):
        resource = bounded_resource_from_bytes(
            data, limits=limits, source_path=str(target / name)
        )
        with (target / name).open("xb") as stream:
            stream.write(resource.data)
        manifests.append(resource.manifest)
        return target / name

    for kind, records, file_pairs in (
        ("node", circuit.nodes, node_files),
        ("edge", circuit.edges, edge_files),
    ):
        populations = {}
        for record in records:
            populations.setdefault(record.population, []).append(record)
        for index, (population, members) in enumerate(sorted(populations.items())):
            buffer = _BoundedBuffer(limits.max_bytes)
            with h5py.File(buffer, "w") as handle:
                handle.attrs["magic"] = np.uint32(0x0A7A)
                handle.attrs["version"] = np.asarray([0, 1], dtype=np.uint32)
                _write_population(
                    handle.create_group(f"{kind}s/{population}"), members, kind
                )
            h5_path = save(f"{kind}s_{index}.h5", buffer.getvalue())
            csv_data = f"{kind}_type_id\n" + "".join(
                f"{identifier}\n"
                for identifier in sorted({member.type_id for member in members})
            )
            types_path = save(f"{kind}_types_{index}.csv", csv_data.encode("ascii"))
            file_pairs.append(SONATAFilePair(h5_path, types_path))
    spike_paths = []
    if circuit.spikes:
        buffer = _BoundedBuffer(limits.max_bytes)
        with h5py.File(buffer, "w") as handle:
            handle.attrs["magic"] = np.uint32(0x0A7A)
            handle.attrs["version"] = np.asarray([0, 1], dtype=np.uint32)
            root = handle.create_group("spikes")
            for spike in circuit.spikes:
                group = root.create_group(spike.population)
                group.attrs.create(
                    "sorting",
                    2,
                    dtype=h5py.enum_dtype(
                        {"none": 0, "by_id": 1, "by_time": 2}, basetype="u1"
                    ),
                )
                group.create_dataset("node_ids", data=spike.node_ids)
                times = group.create_dataset("timestamps", data=spike.timestamps_ms)
                times.attrs["units"] = np.bytes_("ms")
        spike_paths.append(save("spikes.h5", buffer.getvalue()))
    component_paths = []
    for index, (name, resource) in enumerate(circuit.components):
        component_paths.append((name, save(f"component_{index}", resource.data)))
    output_id = canonical_fingerprint(
        {"resources": [manifest.content_sha256 for manifest in manifests]}
    )
    return SONATAExport(
        tuple(node_files),
        tuple(edge_files),
        tuple(spike_paths),
        tuple(component_paths),
        _report("phydrax-native-neural-circuit", "SONATA", semantic_id, output_id),
        tuple(manifests),
    )


def sonata_roundtrip_report(
    before: SONATAImport, after: SONATAImport, /
) -> AdapterReport:
    """Certify equal resolved model, connectivity, metadata and spike semantics."""
    source_id = _semantic_id(before.nodes, before.edges, before.spikes)
    target_id = _semantic_id(after.nodes, after.edges, after.spikes)
    if source_id != target_id:
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "SONATA semantic roundtrip changed models, connectivity, metadata or spikes.",
        )
    return _report(
        "phydrax-native-neural-circuit",
        "phydrax-native-neural-circuit",
        source_id,
        target_id,
        stage="roundtrip",
    )


__all__ = [
    "SONATACableBinding",
    "SONATAEdge",
    "SONATAExport",
    "SONATAFilePair",
    "SONATAImport",
    "SONATANode",
    "SONATASpikes",
    "export_sonata",
    "import_sonata",
    "prepare_sonata_network",
    "sonata_roundtrip_report",
]
