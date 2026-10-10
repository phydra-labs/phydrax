#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lossless native B-Rep persistence in the canonical Phydrax array archive.

The archive stores the complete `BRepModel`: exact carriers (lossless
hexadecimal payloads of the native carrier encoder), oriented topology, exact
p-curves, loops, shells, solids and occurrences, affine-normalized trim curves,
and `IntersectionCurve` generating surfaces, certified atlases and oriented
trim ranges. Query tessellation arrays retain their deviation/normal evidence
and exact vertex source bindings separately from source geometry and lineage.
Decoding is bounded by `ArrayArchiveLimits`, validates every defining closure,
and requires restored model, geometry and tessellation identities to match.
Publication is exclusive by default; replacement must be explicitly requested.
Exact definitions are content-addressed DAG records, and large defining tensors
are pickle-free array references. Only identical defining payloads share a node;
cycles, dangling references, orphan definitions and excessive reference depth
are refused before construction. Restored immutable definitions are memoized
within one resource, never borrowed from an ancestor process or source handle.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .._array_archive import (
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from .._physical import SpatialCoordinateContract
from .._publication import PublicationMode
from ..geometry._atlas import CurveTrimLoop, PolygonTrimLoop, TrimDomain
from ..geometry.brep._constructors import _NormalizedTrimCurve
from ..geometry.brep._intersection import (
    BranchRootEndpoint,
    CurveSurfaceIntersectionRoot,
    IntersectionCurvePointRoot,
    NativePeriodEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
    TripleSurfaceIntersectionRoot,
)
from ..geometry.brep._intersection_curve import (
    CurveTrimSegment,
    decode_geometry,
    encode_geometry,
    IntersectionCurve,
    IntersectionPCurve,
    original_trim_intersection_preparation,
    SurfaceRegion,
)
from ..geometry.brep._model import (
    BRepAssemblyContainer,
    BRepGeometry,
    BRepImportReport,
    BRepModel,
    BRepOccurrence,
    BRepTopology,
)
from ..geometry.brep._patches import AbstractCurve, AbstractSurfacePatch
from ..geometry.brep._root_bindings import (
    BRepCurveSurfaceLift,
    BRepPlacedVertex,
    BRepRootSupport,
    BRepVertexRoot,
)


_KIND = "phydrax-brep-archive"
_MANIFEST_FIELDS = frozenset(
    {
        "kind",
        "model_id",
        "geometry_id",
        "tessellation_id",
        "report",
        "patches",
        "physical_tags",
        "topology",
        "geometry",
        "trim_domains",
        "definitions",
        "arrays",
    }
)


_MAX_DEFINITIONS = 100_000
_ENCODE_CACHE: ContextVar[dict[tuple[str, int], Any] | None] = ContextVar(
    "brep_archive_encoding", default=None
)


class _DefinitionPayload(dict):
    """One decoded immutable definition, with per-resource construction memoization."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(payload)
        self.decoded: dict[str, Any] = {}


def _cached_definition(decoder: Any) -> Any:
    @wraps(decoder)
    def decode(payload: Any, *args: Any) -> Any:
        if not isinstance(payload, _DefinitionPayload):
            return decoder(payload, *args)
        key = decoder.__name__
        if key not in payload.decoded:
            payload.decoded[key] = decoder(payload, *args)
        return payload.decoded[key]

    return decode


def _cached_encoding(encoder: Any) -> Any:
    @wraps(encoder)
    def encode(value: Any, *args: Any) -> Any:
        cache = _ENCODE_CACHE.get()
        if cache is None or value is None:
            return encoder(value, *args)
        key = (encoder.__name__, id(value))
        if key not in cache:
            cache[key] = encoder(value, *args)
        return cache[key]

    return encode


def _encoding_scope(encoder: Any) -> Any:
    @wraps(encoder)
    def encode(*args: Any, **kwargs: Any) -> Any:
        token = _ENCODE_CACHE.set({})
        try:
            return encoder(*args, **kwargs)
        finally:
            _ENCODE_CACHE.reset(token)

    return encode


def _definition_id(payload: Any, /) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_definition_id(value: np.ndarray, /) -> str:
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _definition_references(value: Any, /) -> list[str]:
    if isinstance(value, dict):
        if set(value) == {"definition"}:
            if not isinstance(value["definition"], str):
                raise ValueError("Archive definition references have unexpected fields.")
            return [value["definition"]]
        return [
            child for item in value.values() for child in _definition_references(item)
        ]
    if isinstance(value, list):
        return [child for item in value for child in _definition_references(item)]
    return []


def _definition_children(
    definitions: Mapping[str, Any],
    limits: ArrayArchiveLimits,
    /,
) -> dict[str, list[str]]:
    """Validate semantic reference depth independently of JSON container depth."""
    children = {key: _definition_references(value) for key, value in definitions.items()}
    depths: dict[str, int] = {}
    active: set[str] = set()
    for root in definitions:
        if root in depths:
            continue
        stack = [(root, 0)]
        active.add(root)
        while stack:
            node, position = stack[-1]
            if position < len(children[node]):
                child = children[node][position]
                stack[-1] = (node, position + 1)
                if child not in definitions:
                    raise ValueError("Archive contains a dangling definition reference.")
                if child in active:
                    raise ValueError("Archive contains a cyclic definition reference.")
                if child not in depths:
                    if len(stack) >= limits.max_manifest_nesting:
                        raise ValueError(
                            "Archive defining reference depth exceeds its limit."
                        )
                    active.add(child)
                    stack.append((child, 0))
                continue
            depths[node] = 1 + max(
                (depths[child] for child in children[node]),
                default=0,
            )
            if depths[node] > limits.max_manifest_nesting:
                raise ValueError("Archive defining reference depth exceeds its limit.")
            active.remove(node)
            stack.pop()
    return children


def _factor_definitions(
    manifest: Mapping[str, Any],
    arrays: dict[str, np.ndarray],
    limits: ArrayArchiveLimits,
    /,
) -> dict[str, Any]:
    """Content-address only identical exact definitions; large tensors use NPY."""
    definitions: dict[str, Any] = {}
    memo: dict[int, Any] = {}

    def visit(value: Any, top: bool = False) -> Any:
        # Inline carrier wrappers are factored before wire-container admission.
        # Their depth is not the depth of the resulting definition DAG.
        if not isinstance(value, (dict, list)):
            return value
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, list):
            result = [visit(item) for item in value]
        else:
            if set(value) == {"dtype", "shape", "hex"} and len(value["hex"]) >= 32:
                array = np.asarray(
                    [float.fromhex(item) for item in value["hex"]],
                    dtype=np.dtype(value["dtype"]),
                ).reshape(tuple(value["shape"]))
                name = "definitions/arrays/" + _array_definition_id(array)
                arrays.setdefault(name, array)
                payload = {
                    "array_definition": name,
                    "dtype": value["dtype"],
                    "shape": value["shape"],
                }
            else:
                payload = {key: visit(item) for key, item in value.items()}
            is_definition = not top and (
                "kind" in value
                or "type" in value
                or set(value) == {"dtype", "shape", "hex"}
                or set(value) == {"patch", "parameter_box"}
            )
            if is_definition:
                identifier = _definition_id(payload)
                if identifier not in definitions:
                    if len(definitions) >= _MAX_DEFINITIONS:
                        raise ValueError(
                            "Archive definition entity count exceeds its limit."
                        )
                    definitions[identifier] = payload
                result = {"definition": identifier}
            else:
                result = payload
        memo[id(value)] = result
        return result

    factored = visit(dict(manifest), True)
    factored["definitions"] = {key: definitions[key] for key in sorted(definitions)}
    _definition_children(definitions, limits)
    return factored


def _expand_definitions(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    limits: ArrayArchiveLimits,
    /,
) -> tuple[dict[str, Any], set[str]]:
    """Validate the complete bounded DAG before constructing scientific objects."""
    definitions = manifest["definitions"]
    if not isinstance(definitions, dict) or len(definitions) > _MAX_DEFINITIONS:
        raise ValueError("Archive definition entity count exceeds its limit.")
    if list(definitions) != sorted(definitions):
        raise ValueError("Archive definition ordering is noncanonical.")

    children = _definition_children(definitions, limits)
    for identifier, payload in definitions.items():
        if _definition_id(payload) != identifier:
            raise ValueError(
                "Archive definition fingerprint does not match its defining payload."
            )
    body = {key: value for key, value in manifest.items() if key != "definitions"}
    pending = _definition_references(body)
    reachable: set[str] = set()
    while pending:
        identifier = pending.pop()
        if identifier not in definitions:
            raise ValueError("Archive contains a dangling definition reference.")
        if identifier not in reachable:
            reachable.add(identifier)
            pending.extend(children[identifier])
    if reachable != set(definitions):
        raise ValueError("Archive definitions do not match the exact support closure.")
    used_arrays: set[str] = set()
    memo: dict[str, _DefinitionPayload] = {}

    def expand(value: Any) -> Any:
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"definition"}:
            identifier = value["definition"]
            if identifier not in memo:
                memo[identifier] = expand(definitions[identifier])
            return memo[identifier]
        if "array_definition" in value:
            _record(value, {"array_definition", "dtype", "shape"}, "array definition")
            name = value["array_definition"]
            if not isinstance(name, str) or name not in arrays:
                raise ValueError("Archive lacks a defining array.")
            array = arrays[name]
            if (
                array.dtype != np.dtype(value["dtype"])
                or list(array.shape) != value["shape"]
            ):
                raise ValueError("Archive defining array shape or dtype changed.")
            if name != "definitions/arrays/" + _array_definition_id(array):
                raise ValueError("Archive defining array fingerprint changed.")
            used_arrays.add(name)
            payload = _DefinitionPayload(encode_geometry(array))
            payload.decoded["_decode_carrier"] = array
            return payload
        return _DefinitionPayload({key: expand(item) for key, item in value.items()})

    return expand(body), used_arrays


@dataclass(frozen=True, slots=True)
class BRepArchiveReceipt:
    """Identity of one published native B-Rep archive."""

    path: Path
    model_id: str
    geometry_id: str | None
    tessellation_id: str


def _hex(value: float, /) -> str:
    return float(value).hex()


def _unhex(value: Any, name: str, /) -> float:
    if not isinstance(value, str):
        raise ValueError(f"Archived {name} must be a hexadecimal float.")
    result = float.fromhex(value)
    if not np.isfinite(result):
        raise ValueError(f"Archived {name} must be finite.")
    return result


def _nested(value: Any, depth: int, name: str, /) -> Any:
    """Validate a nested integer tuple structure of exactly ``depth`` levels."""
    if depth == 0:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Archived {name} must contain integers.")
        return value
    if not isinstance(value, list):
        raise ValueError(f"Archived {name} must be nested lists.")
    return tuple(_nested(item, depth - 1, name) for item in value)


def _lists(value: Any, /) -> Any:
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    return value


def _record(value: Any, fields: set[str], name: str, /) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"Archived {name} has unexpected or missing fields.")
    return value


def _text(value: Any, name: str, /) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Archived {name} must be a string.")
    return value


def _boolean(value: Any, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Archived {name} must be boolean.")
    return value


@_cached_encoding
def _encode_carrier(value: Any, /) -> Any:
    if isinstance(value, IntersectionCurve):
        return value.payload()
    if isinstance(value, IntersectionPCurve):
        return _encode_trim_curve(value)
    return encode_geometry(value)


@_cached_definition
def _decode_carrier(payload: Any, /) -> Any:
    if isinstance(payload, Mapping):
        if payload.get("kind") == "intersection-curve":
            return _decode_intersection(payload)
        if payload.get("kind") == "intersection":
            return _decode_trim_curve(payload)
    # The defining closure must be identical after construction. In particular
    # extra fields, ignored array dtypes and coercible values are not admitted.
    try:
        value = decode_geometry(payload)
        if encode_geometry(value) != payload:
            raise ValueError("Archived geometry is not an exact defining payload.")
    except (TypeError, KeyError, OverflowError) as error:
        raise ValueError(f"Archived geometry is malformed: {error}") from error
    return value


@_cached_definition
def _decode_intersection(payload: Any, /) -> IntersectionCurve:
    _record(
        payload,
        {
            "kind",
            "first",
            "first_box",
            "second",
            "second_box",
            "chart_axes",
            "chart_start",
            "chart_end",
            "chart_pieces",
            "box_lower",
            "box_upper",
            "preconditioners",
            "contraction",
            "certified",
            "transition_shifts",
            "closed",
            "start_kind",
            "end_kind",
            "branch_id",
            "node_axes",
            "node_values",
            "node_pieces",
            "node_references",
            "node_period_shifts",
        },
        "intersection curve",
    )
    _boolean(payload["closed"], "intersection closed")
    certified = payload["certified"]
    if not isinstance(certified, list) or not all(
        isinstance(item, bool) for item in certified
    ):
        raise ValueError("Archived intersection certification must be boolean.")
    _nested(payload["chart_axes"], 1, "chart axes")
    _nested(payload["chart_pieces"], 2, "chart pieces")
    curve = IntersectionCurve.from_payload(payload)
    if curve.payload() != payload:
        raise ValueError("Archived intersection is not its exact defining closure.")
    return curve


# ----------------------------------------------------------------- encoding


@_cached_encoding
def _encode_region(region: SurfaceRegion, /) -> dict[str, Any]:
    return {
        "patch": _encode_carrier(region.patch),
        "parameter_box": encode_geometry(region.parameter_box),
    }


@_cached_definition
def _decode_region(record: Any, /) -> SurfaceRegion:
    _record(record, {"patch", "parameter_box"}, "surface region")
    return SurfaceRegion(
        _decode_carrier(record["patch"]), _decode_carrier(record["parameter_box"])
    )


@_cached_encoding
def _encode_root(root: Any, /) -> dict[str, Any] | None:
    if root is None:
        return None
    if isinstance(root, BRepCurveSurfaceLift):
        return {
            "kind": "curve-surface-lift",
            "lift_id": root.lift_id,
            "curve": _encode_carrier(root.curve),
            "pcurve": _encode_carrier(root.pcurve),
            "surface": _encode_region(root.surface),
            "first": _hex(root.first),
            "last": _hex(root.last),
        }
    identity = {"root_id": root.root_id}
    match root:
        case TrimIntersectionRoot():
            return identity | {
                "kind": "trim-root",
                "first": _encode_trim_curve(root.first),
                "second": _encode_trim_curve(root.second),
                "lower": encode_geometry(root.parameter_lower),
                "upper": encode_geometry(root.parameter_upper),
            }
        case CurveSurfaceIntersectionRoot():
            return identity | {
                "kind": "spatial-root",
                "curve": _encode_carrier(root.curve),
                "surface": _encode_region(root.surface),
                "lower": encode_geometry(root.parameter_lower),
                "upper": encode_geometry(root.parameter_upper),
            }
        case TripleSurfaceIntersectionRoot():
            return identity | {
                "kind": "triple-root",
                "first": _encode_region(root.first),
                "second": _encode_region(root.second),
                "third": _encode_region(root.third),
                "lower": encode_geometry(root.parameter_lower),
                "upper": encode_geometry(root.parameter_upper),
            }
        case IntersectionCurvePointRoot():
            return identity | {
                "kind": "branch-point-root",
                "curve": _encode_carrier(root.curve),
                "parameter": _hex(root.parameter),
            }
        case BRepRootSupport():
            return identity | {
                "kind": "face-root",
                "patch": _encode_carrier(root.patch),
                "root": _encode_root(root.root),
            }
        case BRepPlacedVertex():
            return identity | {
                "kind": "placed-vertex-root",
                "source_point": _encode_carrier(root.source_point),
                "rotation": _encode_carrier(root.rotation),
                "translation": _encode_carrier(root.translation),
                "source_entity_id": root.source_entity_id,
                "source_root": _encode_root(root.source_root),
            }
        case BRepVertexRoot():
            return identity | {
                "kind": "vertex-root",
                "primary": _encode_root(root.primary),
                "aliases": [_encode_root(alias) for alias in root.aliases],
                "spatial_root": _encode_root(root.spatial_root),
                "joint_root": _encode_root(root.joint_root),
                "source_edge_lifts": [
                    _encode_root(lift) for lift in root.source_edge_lifts
                ],
            }
        case BranchRootEndpoint():
            return identity | {
                "kind": "branch-endpoint-root",
                "root": _encode_root(root.root),
                "curve": _encode_carrier(root.curve),
                "chart": root.chart,
                "source_pcurve": None
                if root.source_pcurve is None
                else _encode_carrier(root.source_pcurve),
                "source_side": root.source_side,
                "source_first": None
                if root.source_first is None
                else _hex(root.source_first),
                "source_last": None
                if root.source_last is None
                else _hex(root.source_last),
                "periodic_shifts": list(root.periodic_shifts),
                "transforms": [
                    [_hex(scale), _hex(offset)]
                    for scale, offset in root.affine_transforms
                ],
            }
        case TrimRootEndpoint():
            return identity | {
                "kind": "endpoint-root",
                "root": _encode_root(root.root),
                "operand": root.operand,
                "offset": _hex(root.affine_parameter_offset),
                "scale": _hex(root.affine_parameter_scale),
                "transforms": [
                    [_hex(scale), _hex(offset)]
                    for scale, offset in root.affine_transforms
                ],
                "source_pcurve": None
                if root.source_pcurve is None
                else _encode_carrier(root.source_pcurve),
                "source_surface": None
                if root.source_surface is None
                else _encode_region(root.source_surface),
            }
        case NativePeriodEndpoint():
            # The owning codec retains the proved curve or surface period,
            # its original carrier, and the exact rational coefficients.
            return identity | {
                "kind": "native-period-endpoint",
                "definition": _encode_carrier(root),
            }
        case _:
            raise TypeError(f"{type(root).__name__} has no exact root archive encoding.")


@_cached_definition
def _decode_root(record: Any, /) -> Any:
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise ValueError("Archived root definitions must be records.")
    if record.get("kind") == "curve-surface-lift":
        _record(
            record,
            {"kind", "lift_id", "curve", "pcurve", "surface", "first", "last"},
            "curve surface lift",
        )
        lift = BRepCurveSurfaceLift(
            _decode_carrier(record["curve"]),
            _decode_carrier(record["pcurve"]),
            _decode_region(record["surface"]),
            _unhex(record["first"], "lift first"),
            _unhex(record["last"], "lift last"),
        )
        if lift.lift_id != record["lift_id"]:
            raise ValueError(
                "Restored lift identity does not match its source correspondence."
            )
        return lift
    common = {"kind", "root_id"}
    match record.get("kind"):
        case "trim-root":
            _record(record, common | {"first", "second", "lower", "upper"}, "trim root")
            root = TrimIntersectionRoot(
                _decode_trim_curve(record["first"]),
                _decode_trim_curve(record["second"]),
                parameter_lower=_decode_carrier(record["lower"]),
                parameter_upper=_decode_carrier(record["upper"]),
            )
        case "spatial-root":
            _record(
                record, common | {"curve", "surface", "lower", "upper"}, "spatial root"
            )
            root = CurveSurfaceIntersectionRoot(
                _decode_carrier(record["curve"]),
                _decode_region(record["surface"]),
                parameter_lower=_decode_carrier(record["lower"]),
                parameter_upper=_decode_carrier(record["upper"]),
            )
        case "triple-root":
            _record(
                record,
                common | {"first", "second", "third", "lower", "upper"},
                "triple root",
            )
            root = TripleSurfaceIntersectionRoot(
                _decode_region(record["first"]),
                _decode_region(record["second"]),
                _decode_region(record["third"]),
                parameter_lower=_decode_carrier(record["lower"]),
                parameter_upper=_decode_carrier(record["upper"]),
            )
        case "branch-point-root":
            _record(record, common | {"curve", "parameter"}, "branch point root")
            root = IntersectionCurvePointRoot(
                _decode_carrier(record["curve"]),
                _unhex(record["parameter"], "branch point parameter"),
            )
        case "face-root":
            _record(record, common | {"patch", "root"}, "face root")
            root = BRepRootSupport(
                _decode_carrier(record["patch"]), _decode_root(record["root"])
            )
        case "placed-vertex-root":
            _record(
                record,
                common
                | {
                    "source_point",
                    "rotation",
                    "translation",
                    "source_entity_id",
                    "source_root",
                },
                "placed vertex root",
            )
            source_root = _decode_root(record["source_root"])
            if source_root is not None and not isinstance(source_root, BRepVertexRoot):
                raise ValueError(
                    "A placed vertex must retain a complete canonical source vertex root."
                )
            source_point = _decode_carrier(record["source_point"])
            rotation = _decode_carrier(record["rotation"])
            translation = _decode_carrier(record["translation"])
            if any(
                not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64)
                for value in (source_point, rotation, translation)
            ):
                raise ValueError(
                    "Placed vertex source and pose arrays must preserve float64 construction data."
                )
            root = BRepPlacedVertex(
                source_point,
                rotation,
                translation,
                _text(record["source_entity_id"], "placed vertex source entity"),
                source_root=source_root,
            )
        case "vertex-root":
            _record(
                record,
                common
                | {
                    "primary",
                    "aliases",
                    "spatial_root",
                    "joint_root",
                    "source_edge_lifts",
                },
                "vertex root",
            )
            if not isinstance(record["aliases"], list) or not isinstance(
                record["source_edge_lifts"], list
            ):
                raise ValueError(
                    "Archived vertex aliases and source lifts must be explicit lists."
                )
            root = BRepVertexRoot(
                _decode_root(record["primary"]),
                aliases=tuple(_decode_root(alias) for alias in record["aliases"]),
                spatial_root=_decode_root(record["spatial_root"]),
                joint_root=_decode_root(record["joint_root"]),
                source_edge_lifts=tuple(
                    _decode_root(lift) for lift in record["source_edge_lifts"]
                ),
            )
        case "branch-endpoint-root":
            _record(
                record,
                common
                | {
                    "root",
                    "curve",
                    "chart",
                    "source_pcurve",
                    "source_side",
                    "source_first",
                    "source_last",
                    "periodic_shifts",
                    "transforms",
                },
                "branch endpoint root",
            )
            shifts = _nested(record["periodic_shifts"], 1, "periodic shifts")
            if len(shifts) != 4:
                raise ValueError("Branch endpoint periodic shifts need four integers.")
            transforms = record["transforms"]
            if not isinstance(transforms, list) or any(
                not isinstance(pair, list) or len(pair) != 2 for pair in transforms
            ):
                raise ValueError(
                    "Branch endpoint transforms must be ordered affine pairs."
                )
            root = BranchRootEndpoint(
                _decode_root(record["root"]),
                _decode_carrier(record["curve"]),
                _nested(record["chart"], 0, "branch chart"),
                source_pcurve=None
                if record["source_pcurve"] is None
                else _decode_carrier(record["source_pcurve"]),
                source_side=record["source_side"],
                source_first=None
                if record["source_first"] is None
                else _unhex(record["source_first"], "source first"),
                source_last=None
                if record["source_last"] is None
                else _unhex(record["source_last"], "source last"),
                periodic_shifts=shifts,
                affine_transforms=tuple(
                    (_unhex(scale, "affine scale"), _unhex(offset, "affine offset"))
                    for scale, offset in transforms
                ),
            )
        case "endpoint-root":
            _record(
                record,
                common
                | {
                    "root",
                    "operand",
                    "offset",
                    "scale",
                    "transforms",
                    "source_pcurve",
                    "source_surface",
                },
                "endpoint root",
            )
            transforms = record["transforms"]
            if not isinstance(transforms, list) or any(
                not isinstance(pair, list) or len(pair) != 2 for pair in transforms
            ):
                raise ValueError(
                    "Archived endpoint expressions require ordered affine pairs."
                )
            operand = _text(record["operand"], "root operand")
            if operand != "first" and operand != "second":
                raise ValueError(
                    "Archived endpoint root operands must be 'first' or 'second'."
                )
            if (record["source_pcurve"] is None) != (record["source_surface"] is None):
                raise ValueError(
                    "An archived endpoint lift requires both its p-curve and source surface."
                )
            source_pcurve = (
                None
                if record["source_pcurve"] is None
                else _decode_carrier(record["source_pcurve"])
            )
            source_surface = (
                None
                if record["source_surface"] is None
                else _decode_region(record["source_surface"])
            )
            if source_pcurve is not None and not isinstance(
                source_pcurve, (AbstractCurve, IntersectionPCurve)
            ):
                raise ValueError(
                    "An archived endpoint lift requires an exact native p-curve."
                )
            source_root = _decode_root(record["root"])
            if not isinstance(
                source_root,
                (
                    TrimIntersectionRoot,
                    CurveSurfaceIntersectionRoot,
                    IntersectionCurvePointRoot,
                ),
            ):
                raise ValueError(
                    "An archived scalar endpoint requires its actual canonical source root."
                )
            root = TrimRootEndpoint(
                source_root,
                operand,
                affine_parameter_offset=_unhex(record["offset"], "endpoint offset"),
                affine_parameter_scale=_unhex(record["scale"], "endpoint scale"),
                affine_transforms=tuple(
                    (_unhex(scale, "affine scale"), _unhex(offset, "affine offset"))
                    for scale, offset in transforms
                ),
                source_pcurve=source_pcurve,
                source_surface=source_surface,
            )
        case "native-period-endpoint":
            _record(record, common | {"definition"}, "native period endpoint")
            root = _decode_carrier(record["definition"])
            if not isinstance(root, NativePeriodEndpoint):
                raise ValueError(
                    "An archived native period endpoint must restore its exact scalar definition."
                )
        case kind:
            raise ValueError(f"Unsupported archived root kind {kind!r}.")
    if root.root_id != record["root_id"]:
        raise ValueError(
            "Restored root identity does not match its defining source system."
        )
    return root


def _decode_root_pairs(records: Any, /) -> tuple[tuple[Any, Any], ...]:
    if not isinstance(records, list) or any(
        not isinstance(pair, list) or len(pair) != 2 for pair in records
    ):
        raise ValueError("Archived endpoint inventories must contain exact pairs.")
    return tuple(tuple(_decode_root(value) for value in pair) for pair in records)


@_cached_encoding
def _encode_trim_curve(curve: Any, /) -> dict[str, Any]:
    match curve:
        case _NormalizedTrimCurve():
            return {
                "kind": "normalized",
                "curve": _encode_trim_curve(curve.curve),
                "lower": [_hex(value) for value in np.asarray(curve.lower)],
                "extent": [_hex(value) for value in np.asarray(curve.extent)],
                "patch": _encode_carrier(curve.patch),
                "start_root": _encode_root(curve.start_root),
                "end_root": _encode_root(curve.end_root),
                "start_endpoint": _encode_root(curve.start_endpoint),
                "end_endpoint": _encode_root(curve.end_endpoint),
            }
        case CurveTrimSegment():
            return {
                "kind": "segment",
                "curve": _encode_carrier(curve.curve),
                "first": _hex(curve.first),
                "last": _hex(curve.last),
                "reversed": curve.reversed,
                "first_root": _encode_root(curve.first_root),
                "last_root": _encode_root(curve.last_root),
            }
        case IntersectionPCurve():
            return {
                "kind": "intersection",
                "side": curve.side,
                "curve": curve.curve.payload(),
                "first": _hex(curve.first),
                "last": _hex(curve.last),
                "reversed": curve.reversed,
            }
        case _:
            raise TypeError(f"{type(curve).__name__} has no lossless archive encoding.")


def _encode_loop(
    loop: Any, name: str, arrays: dict[str, np.ndarray], /
) -> dict[str, Any]:
    match loop:
        case PolygonTrimLoop():
            arrays[name] = np.asarray(loop.vertices, dtype=np.float64)
            return {"kind": "polygon", "vertices": name}
        case CurveTrimLoop():
            return {
                "kind": "curves",
                "tolerance": _hex(loop.tolerance),
                "relative_closure_tolerance": _hex(loop.relative_closure_tolerance),
                "curves": [_encode_trim_curve(curve) for curve in loop.curves],
            }
        case _:
            raise TypeError(f"{type(loop).__name__} has no lossless archive encoding.")


def _encode_occurrence(occurrence: BRepOccurrence, /) -> dict[str, Any]:
    return {
        "path": list(occurrence.path),
        "solid": occurrence.solid,
        "rotation": [[_hex(value) for value in row] for row in occurrence.rotation],
        "translation": [_hex(value) for value in occurrence.translation],
    }


def _encode_geometry(
    geometry: BRepGeometry, arrays: dict[str, np.ndarray], /
) -> dict[str, Any]:
    arrays["geometry/vertex_points"] = np.asarray(
        geometry.vertex_points, dtype=np.float64
    )
    arrays["geometry/edge_ranges"] = np.asarray(geometry.edge_ranges, dtype=np.float64)
    return {
        "curves": [_encode_carrier(curve) for curve in geometry.curves],
        "pcurves": [_encode_carrier(curve) for curve in geometry.pcurves],
        "edge_curves": list(geometry.edge_curves),
        "edge_vertices": _lists(geometry.edge_vertices),
        "coedge_edges": list(geometry.coedge_edges),
        "coedge_senses": list(geometry.coedge_senses),
        "face_loops": _lists(geometry.face_loops),
        "shell_faces": _lists(geometry.shell_faces),
        "shell_orientations": _lists(geometry.shell_orientations),
        "solid_shells": _lists(geometry.solid_shells),
        "occurrences": [_encode_occurrence(item) for item in geometry.occurrences],
        "vertex_roots": [_encode_root(root) for root in geometry.vertex_roots],
        "edge_endpoint_roots": [
            [_encode_root(root) for root in pair] for pair in geometry.edge_endpoint_roots
        ],
        "coedge_endpoint_roots": [
            [_encode_root(root) for root in pair]
            for pair in geometry.coedge_endpoint_roots
        ],
        "assembly_containers": [
            {
                "path": list(container.path),
                "member_paths": [list(path) for path in container.member_paths],
                "child_paths": [list(path) for path in container.child_paths],
            }
            for container in geometry.assembly_containers
        ],
    }


def _encode_report(report: BRepImportReport, /) -> dict[str, Any]:
    return {
        "source_id": report.source_id,
        "source_digest": report.source_digest,
        "source_format": report.source_format,
        "coordinate_contract": report.coordinate_contract.to_dict(),
        "import_policy_id": report.import_policy_id,
        "counts": [
            report.num_solids,
            report.num_faces,
            report.num_edges,
            report.num_vertices,
            report.num_triangles,
            report.converted_surface_count,
        ],
        "linear_deflection": _hex(report.linear_deflection),
        "angular_deflection": _hex(report.angular_deflection),
        "trim_samples_per_edge": report.trim_samples_per_edge,
        "curve_surface_tolerance": _hex(report.curve_surface_tolerance),
        "curve_surface_scale": _hex(report.curve_surface_scale),
        "source_revision": report.source_revision,
    }


@_encoding_scope
def save_brep_archive(
    model: BRepModel,
    path: str | os.PathLike[str],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    mode: PublicationMode = "exclusive",
) -> BRepArchiveReceipt:
    """Publish lossless native geometry, exclusively unless replacement is requested."""
    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    arrays: dict[str, np.ndarray] = {
        "model/parameter_bounds": np.asarray(model.parameter_bounds, dtype=np.float64),
        "model/orientation": np.asarray(model.orientation, dtype=np.float64),
        "mesh/vertices": np.asarray(model.mesh_vertices, dtype=np.float64),
        "mesh/faces": np.asarray(model.mesh_faces, dtype=np.int32),
        "mesh/face_ids": np.asarray(model.triangle_face_ids, dtype=np.int32),
        "mesh/parameters": np.asarray(model.triangle_parameters, dtype=np.float64),
        "mesh/deviation_bounds": np.asarray(
            model.tessellation_deviation_bounds, dtype=np.float64
        ),
        "mesh/normal_bounds": np.asarray(
            model.tessellation_normal_bounds, dtype=np.float64
        ),
        "mesh/vertex_source_dimensions": np.asarray(
            model.mesh_vertex_source_dimensions, dtype=np.int32
        ),
        "mesh/vertex_source_indices": np.asarray(
            model.mesh_vertex_source_indices, dtype=np.int64
        ),
        "mesh/vertex_parameters": np.asarray(
            model.mesh_vertex_parameters, dtype=np.float64
        ),
        "mesh/chart_restriction_vertices": np.asarray(
            model.mesh_chart_restriction_vertices, dtype=np.int64
        ),
        "mesh/chart_restriction_edges": np.asarray(
            model.mesh_chart_restriction_edges, dtype=np.int64
        ),
        "mesh/chart_restriction_endpoint_parameters": np.asarray(
            model.mesh_chart_restriction_endpoint_parameters, dtype=np.float64
        ),
        "mesh/chart_restriction_parameters": np.asarray(
            model.mesh_chart_restriction_parameters, dtype=np.int64
        ),
        "mesh/coedge_deviation_bounds": np.asarray(
            model.coedge_deviation_bounds, dtype=np.float64
        ),
        "mesh/triangle_occurrence_ids": np.asarray(
            model.triangle_occurrence_ids, dtype=np.int32
        ),
        "mesh/vertex_occurrence_ids": np.asarray(
            model.vertex_occurrence_ids, dtype=np.int32
        ),
    }
    topology = model.topology
    trims = []
    for face, domain in enumerate(model.trim_domains):
        if domain is None:
            trims.append(None)
            continue
        trims.append(
            {
                "outer": _encode_loop(domain.outer, f"trim/{face:08d}/outer", arrays),
                "holes": [
                    _encode_loop(hole, f"trim/{face:08d}/hole{index:08d}", arrays)
                    for index, hole in enumerate(domain.holes)
                ],
            }
        )
    manifest = {
        "kind": _KIND,
        "model_id": model.model_id,
        "tessellation_id": model.tessellation_id,
        "geometry_id": None if model.geometry is None else model.geometry.geometry_id,
        "report": _encode_report(model.report),
        "patches": [encode_geometry(patch) for patch in model.patches],
        "physical_tags": list(model.physical_tags),
        "topology": {
            "face_edges": _lists(topology.face_edges),
            "edge_faces": _lists(topology.edge_faces),
            "face_wires": _lists(topology.face_wires),
            "solid_faces": _lists(topology.solid_faces),
            "solid_face_orientations": _lists(topology.solid_face_orientations),
            "num_vertices": topology.num_vertices,
        },
        "geometry": None
        if model.geometry is None
        else _encode_geometry(model.geometry, arrays),
        "trim_domains": trims,
    }
    manifest = _factor_definitions(manifest, arrays, limits)
    destination = write_array_archive(
        path, manifest=manifest, arrays=arrays, limits=limits, mode=mode
    )
    return BRepArchiveReceipt(
        destination,
        model.model_id,
        None if model.geometry is None else model.geometry.geometry_id,
        model.tessellation_id,
    )


# ----------------------------------------------------------------- decoding


def _array(arrays: Mapping[str, np.ndarray], name: Any, dtype: Any, /) -> np.ndarray:
    if not isinstance(name, str) or name not in arrays:
        raise ValueError(f"Archive lacks the array {name!r}.")
    value = arrays[name]
    if value.dtype != np.dtype(dtype):
        raise ValueError(f"Archived array {name!r} has an unexpected dtype.")
    return value


@_cached_definition
def _decode_trim_curve(record: Any, /) -> Any:
    if not isinstance(record, Mapping):
        raise ValueError("Archived trim curves must be records.")
    match record.get("kind"):
        case "normalized":
            _record(
                record,
                {
                    "kind",
                    "curve",
                    "lower",
                    "extent",
                    "patch",
                    "start_root",
                    "end_root",
                    "start_endpoint",
                    "end_endpoint",
                },
                "normalized trim",
            )
            lower, extent = record["lower"], record["extent"]
            if not all(
                isinstance(values, list) and len(values) == 2
                for values in (lower, extent)
            ):
                raise ValueError(
                    "Archived trim normalization requires two-dimensional vectors."
                )
            lower_ = [_unhex(value, "trim lower") for value in lower]
            extent_ = [_unhex(value, "trim extent") for value in extent]
            if any(value <= 0.0 for value in extent_):
                raise ValueError(
                    "Archived trim normalization must have positive extents."
                )
            patch = _decode_carrier(record["patch"])
            start_root, end_root = (
                _decode_root(record["start_root"]),
                _decode_root(record["end_root"]),
            )
            start_endpoint = _decode_root(record["start_endpoint"])
            end_endpoint = _decode_root(record["end_endpoint"])
            if (
                not isinstance(patch, AbstractSurfacePatch)
                or any(
                    root is not None and not isinstance(root, BRepVertexRoot)
                    for root in (start_root, end_root)
                )
                or any(
                    endpoint is not None
                    and not isinstance(
                        endpoint,
                        (TrimRootEndpoint, BranchRootEndpoint, NativePeriodEndpoint),
                    )
                    for endpoint in (start_endpoint, end_endpoint)
                )
            ):
                raise ValueError(
                    "Archived normalized trim root supports have invalid types."
                )
            return _NormalizedTrimCurve(
                _decode_trim_curve(record["curve"]),
                jnp.asarray(lower_),
                jnp.asarray(extent_),
                patch,
                start_root,
                end_root,
                start_endpoint,
                end_endpoint,
            )
        case "segment":
            _record(
                record,
                {"kind", "curve", "first", "last", "reversed", "first_root", "last_root"},
                "trim segment",
            )
            return CurveTrimSegment(
                _decode_carrier(record["curve"]),
                _unhex(record["first"], "first"),
                _unhex(record["last"], "last"),
                reversed=_boolean(record["reversed"], "trim reversed"),
                first_root=_decode_root(record["first_root"]),
                last_root=_decode_root(record["last_root"]),
            )
        case "intersection":
            _record(
                record,
                {"kind", "side", "curve", "first", "last", "reversed"},
                "intersection trim",
            )
            return IntersectionPCurve(
                _decode_intersection(record["curve"]),
                record["side"],
                first=_unhex(record["first"], "intersection first"),
                last=_unhex(record["last"], "intersection last"),
                reversed=_boolean(record["reversed"], "intersection reversed"),
            )
        case kind:
            raise ValueError(f"Unknown archived trim curve kind {kind!r}.")


def _decode_loop(record: Any, arrays: Mapping[str, np.ndarray], /) -> Any:
    if not isinstance(record, Mapping):
        raise ValueError("Archived trim loops must be records.")
    match record.get("kind"):
        case "polygon":
            _record(record, {"kind", "vertices"}, "polygon trim")
            return PolygonTrimLoop(_array(arrays, record["vertices"], np.float64))
        case "curves":
            _record(
                record,
                {
                    "kind",
                    "curves",
                    "tolerance",
                    "relative_closure_tolerance",
                },
                "curve trim loop",
            )
            curves = record["curves"]
            if not isinstance(curves, list):
                raise ValueError("Archived curve loops must list their curves.")
            return CurveTrimLoop(
                [_decode_trim_curve(curve) for curve in curves],
                tolerance=_unhex(record["tolerance"], "tolerance"),
                relative_closure_tolerance=_unhex(
                    record["relative_closure_tolerance"],
                    "relative_closure_tolerance",
                ),
            )
        case kind:
            raise ValueError(f"Unknown archived trim loop kind {kind!r}.")


def _decode_occurrence(record: Any, /) -> BRepOccurrence:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "solid",
        "rotation",
        "translation",
    }:
        raise ValueError("Archived occurrences are malformed.")
    rows = record["rotation"]
    translation = record["translation"]
    if (
        not isinstance(rows, list)
        or len(rows) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rows)
        or not isinstance(translation, list)
        or len(translation) != 3
    ):
        raise ValueError("Archived occurrence placements are malformed.")
    rotation = tuple(tuple(_unhex(value, "rotation") for value in row) for row in rows)
    path = record["path"]
    if not isinstance(path, list):
        raise ValueError("Archived occurrence paths are lists.")
    return BRepOccurrence(
        tuple(_text(item, "occurrence path") for item in path),
        _nested(record["solid"], 0, "occurrence solid"),
        np.asarray(rotation),
        np.asarray(
            (
                _unhex(translation[0], "translation"),
                _unhex(translation[1], "translation"),
                _unhex(translation[2], "translation"),
            )
        ),
    )


def _decode_container(record: Any, /) -> BRepAssemblyContainer:
    _record(record, {"path", "member_paths", "child_paths"}, "assembly container")

    def path(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ValueError("Archived container paths must be string lists.")
        return tuple(_text(item, "container path") for item in value)

    if not isinstance(record["member_paths"], list) or not isinstance(
        record["child_paths"], list
    ):
        raise ValueError("Archived assembly membership must be explicit path lists.")
    return BRepAssemblyContainer(
        path(record["path"]),
        tuple(path(value) for value in record["member_paths"]),
        tuple(path(value) for value in record["child_paths"]),
    )


def _decode_geometry(record: Any, arrays: Mapping[str, np.ndarray], /) -> BRepGeometry:
    _record(
        record,
        {
            "curves",
            "pcurves",
            "edge_curves",
            "edge_vertices",
            "coedge_edges",
            "coedge_senses",
            "face_loops",
            "shell_faces",
            "shell_orientations",
            "solid_shells",
            "occurrences",
            "assembly_containers",
            "vertex_roots",
            "edge_endpoint_roots",
            "coedge_endpoint_roots",
        },
        "exact geometry",
    )
    curves = record["curves"]
    pcurves = record["pcurves"]
    occurrences = record["occurrences"]
    containers = record["assembly_containers"]
    if not all(
        isinstance(value, list) for value in (curves, pcurves, occurrences, containers)
    ):
        raise ValueError("Archived carriers and occurrences must be lists.")
    if not isinstance(record["vertex_roots"], list):
        raise ValueError("Archived vertex root inventory must be an explicit list.")
    return BRepGeometry(
        vertex_points=_array(arrays, "geometry/vertex_points", np.float64),
        curves=tuple(_decode_carrier(curve) for curve in curves),
        edge_curves=_nested(record["edge_curves"], 1, "edge_curves"),
        edge_ranges=_array(arrays, "geometry/edge_ranges", np.float64),
        edge_vertices=_nested(record["edge_vertices"], 2, "edge_vertices"),
        pcurves=tuple(_decode_carrier(curve) for curve in pcurves),
        coedge_edges=_nested(record["coedge_edges"], 1, "coedge_edges"),
        coedge_senses=_nested(record["coedge_senses"], 1, "coedge_senses"),
        face_loops=_nested(record["face_loops"], 3, "face_loops"),
        shell_faces=_nested(record["shell_faces"], 2, "shell_faces"),
        shell_orientations=_nested(record["shell_orientations"], 2, "shell_orientations"),
        solid_shells=_nested(record["solid_shells"], 2, "solid_shells"),
        occurrences=tuple(_decode_occurrence(item) for item in occurrences),
        assembly_containers=tuple(_decode_container(item) for item in containers),
        vertex_roots=tuple(_decode_root(root) for root in record["vertex_roots"]),
        edge_endpoint_roots=_decode_root_pairs(record["edge_endpoint_roots"]),
        coedge_endpoint_roots=_decode_root_pairs(record["coedge_endpoint_roots"]),
    )


def _decode_report(record: Any, /) -> BRepImportReport:
    _record(
        record,
        {
            "source_id",
            "source_digest",
            "source_format",
            "coordinate_contract",
            "import_policy_id",
            "counts",
            "linear_deflection",
            "angular_deflection",
            "trim_samples_per_edge",
            "source_revision",
            "curve_surface_tolerance",
            "curve_surface_scale",
        },
        "import report",
    )
    counts = _nested(record["counts"], 1, "counts")
    if len(counts) != 6:
        raise ValueError("Archived import report counts are malformed.")
    contract = SpatialCoordinateContract.from_dict(record["coordinate_contract"])
    if contract.to_dict() != record["coordinate_contract"]:
        raise ValueError(
            "Archived coordinate contract is not its exact defining closure."
        )
    report = BRepImportReport(
        source_id=_text(record["source_id"], "source id"),
        source_digest=_text(record["source_digest"], "source digest"),
        source_format=_text(record["source_format"], "source format"),
        coordinate_contract=contract,
        import_policy_id=_text(record["import_policy_id"], "import policy id"),
        num_solids=counts[0],
        num_faces=counts[1],
        num_edges=counts[2],
        num_vertices=counts[3],
        num_triangles=counts[4],
        linear_deflection=_unhex(record["linear_deflection"], "linear_deflection"),
        angular_deflection=_unhex(record["angular_deflection"], "angular_deflection"),
        trim_samples_per_edge=record["trim_samples_per_edge"],
        converted_surface_count=counts[5],
        curve_surface_tolerance=_unhex(
            record["curve_surface_tolerance"], "curve surface tolerance"
        ),
        curve_surface_scale=_unhex(record["curve_surface_scale"], "curve surface scale"),
    )
    if report.source_revision != record["source_revision"]:
        raise ValueError("Archived source revision does not match its lineage.")
    return report


def _decode_topology(record: Any, /) -> BRepTopology:
    _record(
        record,
        {
            "face_edges",
            "edge_faces",
            "face_wires",
            "solid_faces",
            "solid_face_orientations",
            "num_vertices",
        },
        "topology",
    )
    return BRepTopology(
        face_edges=_nested(record["face_edges"], 2, "face_edges"),
        edge_faces=_nested(record["edge_faces"], 2, "edge_faces"),
        face_wires=_nested(record["face_wires"], 3, "face_wires"),
        solid_faces=_nested(record["solid_faces"], 2, "solid_faces"),
        solid_face_orientations=_nested(
            record["solid_face_orientations"], 2, "orientations"
        ),
        num_vertices=_nested(record["num_vertices"], 0, "num_vertices"),
    )


def load_brep_archive(
    path: str | os.PathLike[str],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> BRepModel:
    """Restore a native B-Rep archive under bounded decoding and identity checks."""
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be ArrayArchiveLimits.")
    # Restored copies of one branch share identical-query results while their
    # trim covers and model are rebuilt; a surrounding invocation is joined.
    with original_trim_intersection_preparation(()):
        return _load_brep_archive(path, limits)


def _load_brep_archive(
    path: str | os.PathLike[str], limits: ArrayArchiveLimits, /
) -> BRepModel:
    manifest, arrays = read_array_archive(path, limits=limits)
    if set(manifest) != _MANIFEST_FIELDS or manifest["kind"] != _KIND:
        raise ValueError("The file is not a native B-Rep archive.")
    manifest, definition_arrays = _expand_definitions(manifest, arrays, limits)
    patches = manifest["patches"]
    tags = manifest["physical_tags"]
    trims = manifest["trim_domains"]
    if not all(isinstance(value, list) for value in (patches, tags, trims)):
        raise ValueError("Archived patches, tags and trims must be lists.")
    report = _decode_report(manifest["report"])
    domains: list[TrimDomain | None] = []
    for record in trims:
        if record is None:
            domains.append(None)
            continue
        if not isinstance(record, Mapping) or not isinstance(record.get("holes"), list):
            raise ValueError("Archived trim domains are malformed.")
        _record(record, {"outer", "holes"}, "trim domain")
        domains.append(
            TrimDomain(
                _decode_loop(record["outer"], arrays),
                [_decode_loop(hole, arrays) for hole in record["holes"]],
            )
        )
    geometry_record = manifest["geometry"]
    model = BRepModel(
        patches=tuple(_decode_carrier(patch) for patch in patches),
        parameter_bounds=_array(arrays, "model/parameter_bounds", np.float64),
        orientation=_array(arrays, "model/orientation", np.float64),
        trim_domains=tuple(domains),
        topology=_decode_topology(manifest["topology"]),
        coordinate_contract=report.coordinate_contract,
        mesh_vertices=_array(arrays, "mesh/vertices", np.float64),
        mesh_faces=_array(arrays, "mesh/faces", np.int32),
        triangle_face_ids=_array(arrays, "mesh/face_ids", np.int32),
        triangle_parameters=_array(arrays, "mesh/parameters", np.float64),
        tessellation_deviation_bounds=_array(arrays, "mesh/deviation_bounds", np.float64),
        tessellation_normal_bounds=_array(arrays, "mesh/normal_bounds", np.float64),
        mesh_vertex_source_dimensions=_array(
            arrays, "mesh/vertex_source_dimensions", np.int32
        ),
        mesh_vertex_source_indices=_array(arrays, "mesh/vertex_source_indices", np.int64),
        mesh_vertex_parameters=_array(arrays, "mesh/vertex_parameters", np.float64),
        mesh_chart_restriction_vertices=_array(
            arrays, "mesh/chart_restriction_vertices", np.int64
        ),
        mesh_chart_restriction_edges=_array(
            arrays, "mesh/chart_restriction_edges", np.int64
        ),
        mesh_chart_restriction_endpoint_parameters=_array(
            arrays, "mesh/chart_restriction_endpoint_parameters", np.float64
        ),
        mesh_chart_restriction_parameters=_array(
            arrays, "mesh/chart_restriction_parameters", np.int64
        ),
        coedge_deviation_bounds=_array(
            arrays, "mesh/coedge_deviation_bounds", np.float64
        ),
        triangle_occurrence_ids=_array(arrays, "mesh/triangle_occurrence_ids", np.int32),
        vertex_occurrence_ids=_array(arrays, "mesh/vertex_occurrence_ids", np.int32),
        physical_tags=tuple(_text(tag, "physical tag") for tag in tags),
        report=report,
        geometry=None
        if geometry_record is None
        else _decode_geometry(geometry_record, arrays),
    )
    if (model.model_id, model.tessellation_id) != (
        manifest["model_id"],
        manifest["tessellation_id"],
    ):
        raise ValueError("Restored B-Rep identity does not match the archive.")
    geometry_id = None if model.geometry is None else model.geometry.geometry_id
    if geometry_id != manifest["geometry_id"]:
        raise ValueError("Restored exact geometry identity does not match the archive.")
    expected_arrays = {
        "model/parameter_bounds",
        "model/orientation",
        "mesh/vertices",
        "mesh/faces",
        "mesh/face_ids",
        "mesh/parameters",
        "mesh/deviation_bounds",
        "mesh/normal_bounds",
        "mesh/vertex_source_dimensions",
        "mesh/vertex_source_indices",
        "mesh/vertex_parameters",
        "mesh/chart_restriction_vertices",
        "mesh/chart_restriction_edges",
        "mesh/chart_restriction_endpoint_parameters",
        "mesh/chart_restriction_parameters",
        "mesh/coedge_deviation_bounds",
        "mesh/triangle_occurrence_ids",
        "mesh/vertex_occurrence_ids",
    }
    if geometry_record is not None:
        expected_arrays.update(("geometry/vertex_points", "geometry/edge_ranges"))
    for face, domain in enumerate(domains):
        if domain is None:
            continue
        if isinstance(domain.outer, PolygonTrimLoop):
            expected_arrays.add(f"trim/{face:08d}/outer")
        for index, hole in enumerate(domain.holes):
            if isinstance(hole, PolygonTrimLoop):
                expected_arrays.add(f"trim/{face:08d}/hole{index:08d}")
    expected_arrays.update(definition_arrays)
    if set(arrays) != expected_arrays:
        raise ValueError("Archive arrays do not match the exact model closure.")
    return model


__all__ = ["BRepArchiveReceipt", "load_brep_archive", "save_brep_archive"]
