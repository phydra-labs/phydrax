#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict, one-time conversion of XGBoost saved models to native trees."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .._schema import FeatureSchema, TargetSchema
from ..tree import TreeEnsemble
from ._contracts import (
    ConversionError,
    ConversionProvenance,
    ConversionResult,
    UnsupportedConversionError,
)


_SCHEMA_REVISION = "xgboost-saved-model-v2-v3.1"
_MAX_CATEGORY = 16_777_216
_INT32_MAX = 2_147_483_647
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT_TOKEN = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?")

_TOP_FIELDS = frozenset({"learner", "version"})
_LEARNER_FIELDS = frozenset(
    {
        "attributes",
        "feature_names",
        "feature_types",
        "gradient_booster",
        "learner_model_param",
        "objective",
    }
)
_LEARNER_PARAM_FIELDS = frozenset(
    {"base_score", "boost_from_average", "num_class", "num_feature", "num_target"}
)
_TREE_PARAM_FIELDS = frozenset(
    {"num_deleted", "num_feature", "num_nodes", "size_leaf_vector"}
)
_TREE_COMMON_FIELDS = frozenset(
    {
        "base_weights",
        "categories",
        "categories_nodes",
        "categories_segments",
        "categories_sizes",
        "default_left",
        "id",
        "left_children",
        "loss_changes",
        "parents",
        "right_children",
        "split_conditions",
        "split_indices",
        "split_type",
        "sum_hessian",
        "tree_param",
    }
)


class _UBJArray(list[Any]):
    """A logical JSON array retaining an optional UBJSON element marker."""

    __slots__ = ("marker",)

    def __init__(self, values: list[Any], marker: str | None = None):
        super().__init__(values)
        self.marker = marker


class _UBJSONDecoder:
    """Decoder for the strict UBJSON subset emitted by XGBoost's UBJWriter."""

    _PRIMITIVES = {
        "d": (">f", 4),
        "D": (">d", 8),
        "i": (">b", 1),
        "U": (">B", 1),
        "I": (">h", 2),
        "l": (">i", 4),
        "L": (">q", 8),
    }

    def __init__(self, data: bytes):
        self.data = data
        self.position = 0

    def decode(self) -> dict[str, Any]:
        value = self._value(0)
        if self.position != len(self.data):
            raise ConversionError("UBJSON artifact has trailing bytes.")
        if type(value) is not dict:
            raise ConversionError(
                "XGBoost saved-model UBJSON must contain a top-level object."
            )
        return value

    def _take(self, count: int) -> bytes:
        end = self.position + count
        if count < 0 or end > len(self.data):
            raise ConversionError("Truncated UBJSON artifact.")
        value = self.data[self.position : end]
        self.position = end
        return value

    def _marker(self) -> str:
        raw = self._take(1)
        try:
            return raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ConversionError("UBJSON marker is not ASCII.") from error

    def _expect(self, marker: str) -> None:
        actual = self._marker()
        if actual != marker:
            raise ConversionError(
                f"Malformed UBJSON: expected marker {marker!r}, found {actual!r}."
            )

    def _primitive(self, marker: str) -> int | float:
        specification = self._PRIMITIVES.get(marker)
        if specification is None:
            raise ConversionError(f"Unsupported UBJSON numeric marker {marker!r}.")
        fmt, size = specification
        return struct.unpack(fmt, self._take(size))[0]

    def _length(self) -> int:
        self._expect("L")
        value = self._primitive("L")
        if type(value) is not int or value < 0:
            raise ConversionError("UBJSON lengths must be non-negative int64 values.")
        return value

    def _string(self) -> str:
        size = self._length()
        raw = self._take(size)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ConversionError("UBJSON contains invalid UTF-8.") from error

    def _array(self, depth: int) -> _UBJArray:
        marker = self._marker()
        if marker == "$":
            element_marker = self._marker()
            if element_marker not in self._PRIMITIVES:
                raise ConversionError(
                    f"Unsupported UBJSON typed-array marker {element_marker!r}."
                )
            self._expect("#")
            count = self._length()
            _, size = self._PRIMITIVES[element_marker]
            if count > (len(self.data) - self.position) // size:
                raise ConversionError("Truncated UBJSON typed array.")
            return _UBJArray(
                [self._primitive(element_marker) for _ in range(count)],
                element_marker,
            )
        if marker != "#":
            raise ConversionError(
                "XGBoost UBJSON arrays must use its length-optimized encoding."
            )
        count = self._length()
        if count > len(self.data) - self.position:
            raise ConversionError("Impossible UBJSON array length.")
        return _UBJArray([self._value(depth + 1) for _ in range(count)])

    def _object(self, depth: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while True:
            if self.position >= len(self.data):
                raise ConversionError("Unterminated UBJSON object.")
            if self.data[self.position] == ord("}"):
                self.position += 1
                return result
            key = self._string()
            if key in result:
                raise ConversionError(f"Duplicate UBJSON object key {key!r}.")
            result[key] = self._value(depth + 1)

    def _value(self, depth: int) -> Any:
        if depth > 128:
            raise ConversionError("UBJSON nesting exceeds the supported bound.")
        marker = self._marker()
        if marker == "{":
            return self._object(depth)
        if marker == "[":
            return self._array(depth)
        if marker == "Z":
            return None
        if marker == "T":
            return True
        if marker == "F":
            return False
        if marker == "S":
            return self._string()
        if marker in self._PRIMITIVES:
            return self._primitive(marker)
        raise ConversionError(f"Unknown UBJSON construct {marker!r}.")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConversionError(f"Duplicate JSON object key {key!r}.")
        result[key] = value
    return result


def _json_constant(token: str) -> float:
    if token == "NaN":
        return math.nan
    raise ConversionError(f"Unsupported non-finite JSON number {token!r}.")


def _decode_json(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConversionError("JSON artifact is not valid UTF-8.") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_constant=_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ConversionError(
            f"Malformed XGBoost saved-model JSON at line {error.lineno}, column {error.colno}."
        ) from error
    if type(value) is not dict:
        raise ConversionError("XGBoost saved-model JSON must contain a top-level object.")
    return value


def _detect_binary_format(raw: bytes) -> str:
    stripped = raw.lstrip(b" \t\r\n")
    if not stripped or stripped[0] != ord("{"):
        raise ConversionError("Saved-model artifact must begin with an object.")
    remainder = stripped[1:].lstrip(b" \t\r\n")
    if remainder.startswith(b'"') or remainder.startswith(b"}"):
        return "json"
    if remainder.startswith(b"L"):
        return "ubjson"
    raise ConversionError("Artifact is neither XGBoost JSON nor XGBoost UBJSON.")


def _normalise_json(value: Any, path: str = "$", active: set[int] | None = None) -> Any:
    active_ = set() if active is None else active
    if type(value) is dict or isinstance(value, Mapping):
        identity = id(value)
        if identity in active_:
            raise ConversionError(f"Cyclic mapping at {path} is not JSON.")
        active_.add(identity)
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ConversionError(f"JSON object key at {path} is not a string.")
            result[key] = _normalise_json(item, f"{path}.{key}", active_)
        active_.remove(identity)
        return result
    if type(value) is list or isinstance(value, _UBJArray):
        identity = id(value)
        if identity in active_:
            raise ConversionError(f"Cyclic array at {path} is not JSON.")
        active_.add(identity)
        values = [
            _normalise_json(item, f"{path}[{index}]", active_)
            for index, item in enumerate(value)
        ]
        active_.remove(identity)
        if isinstance(value, _UBJArray):
            return _UBJArray(values, value.marker)
        return values
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if math.isinf(value):
            raise ConversionError(f"Infinite JSON number at {path} is unsupported.")
        return value
    raise ConversionError(f"Value at {path} has non-JSON type {type(value).__name__}.")


def _feed_canonical(hasher: Any, value: Any) -> None:
    if value is None:
        hasher.update(b"n")
    elif type(value) is bool:
        hasher.update(b"t" if value else b"f")
    elif type(value) is int:
        encoded = str(value).encode("ascii")
        hasher.update(b"i" + struct.pack(">Q", len(encoded)) + encoded)
    elif type(value) is float:
        hasher.update(b"d" + struct.pack(">d", value))
    elif type(value) is str:
        encoded = value.encode("utf-8")
        hasher.update(b"s" + struct.pack(">Q", len(encoded)) + encoded)
    elif type(value) is list or isinstance(value, _UBJArray):
        marker = value.marker if isinstance(value, _UBJArray) else ""
        hasher.update(b"a" + marker.encode("ascii") + struct.pack(">Q", len(value)))
        for item in value:
            _feed_canonical(hasher, item)
    elif type(value) is dict:
        hasher.update(b"o" + struct.pack(">Q", len(value)))
        for key in sorted(value):
            _feed_canonical(hasher, key)
            _feed_canonical(hasher, value[key])
    else:
        raise ConversionError("Internal canonicalization encountered a non-JSON value.")


def _logical_checksum(value: Any) -> str:
    hasher = hashlib.sha256()
    _feed_canonical(hasher, value)
    return hasher.hexdigest()


def _native_array_checksum(arrays: tuple[tuple[str, np.ndarray], ...]) -> str:
    hasher = hashlib.sha256()
    for name, value in arrays:
        name_bytes = name.encode("ascii")
        dtype = value.dtype.newbyteorder(">")
        contiguous = np.ascontiguousarray(value.astype(dtype, copy=False))
        hasher.update(struct.pack(">Q", len(name_bytes)))
        hasher.update(name_bytes)
        hasher.update(str(dtype).encode("ascii"))
        hasher.update(struct.pack(">Q", value.ndim))
        for size in value.shape:
            hasher.update(struct.pack(">Q", size))
        hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def _load_source(source: Any) -> tuple[dict[str, Any], str, str]:
    if isinstance(source, Mapping):
        document = _normalise_json(source)
        return document, "json-mapping", _logical_checksum(document)

    expected_format: str | None = None
    if type(source) is str:
        if source.lstrip().startswith("{"):
            raw = source.encode("utf-8")
            expected_format = "json"
        else:
            path = Path(source)
            suffix = path.suffix.lower()
            if suffix == ".json":
                expected_format = "json"
            elif suffix in {".ubj", ".ubjson"}:
                expected_format = "ubjson"
            else:
                raise UnsupportedConversionError(
                    "XGBoost saved-model paths must end in .json, .ubj, or .ubjson."
                )
            try:
                raw = path.read_bytes()
            except OSError as error:
                raise ConversionError(
                    f"Cannot read XGBoost artifact {path!s}."
                ) from error
    elif isinstance(source, os.PathLike):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".json":
            expected_format = "json"
        elif suffix in {".ubj", ".ubjson"}:
            expected_format = "ubjson"
        else:
            raise UnsupportedConversionError(
                "XGBoost saved-model paths must end in .json, .ubj, or .ubjson."
            )
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ConversionError(f"Cannot read XGBoost artifact {path!s}.") from error
    elif type(source) is bytes:
        raw = source
    else:
        raise TypeError(
            "source must be a saved-model mapping, JSON text/bytes, UBJSON bytes, or a path."
        )

    actual_format = _detect_binary_format(raw)
    if expected_format is not None and actual_format != expected_format:
        raise ConversionError(
            f"Artifact content is {actual_format}, but its source declares {expected_format}."
        )
    document = (
        _decode_json(raw) if actual_format == "json" else _UBJSONDecoder(raw).decode()
    )
    return _normalise_json(document), actual_format, hashlib.sha256(raw).hexdigest()


def _object(value: Any, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ConversionError(f"{path} must be an object.")
    return value


def _fields(
    value: dict[str, Any],
    required: frozenset[str],
    path: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    keys = frozenset(value)
    missing = required - keys
    if missing:
        raise ConversionError(f"{path} is missing required fields {sorted(missing)!r}.")
    unknown = keys - required - optional
    if unknown:
        raise UnsupportedConversionError(
            f"{path} contains unsupported schema fields {sorted(unknown)!r}."
        )


def _array(
    value: Any,
    path: str,
    *,
    marker: str | tuple[str, ...] | None = None,
    untyped: bool = False,
) -> list[Any]:
    if type(value) is not list and not isinstance(value, _UBJArray):
        raise ConversionError(f"{path} must be an array.")
    if isinstance(value, _UBJArray):
        if untyped and value.marker is not None:
            raise ConversionError(f"{path} must be an untyped UBJSON array.")
        if marker is not None:
            accepted = (marker,) if type(marker) is str else marker
            if value.marker not in accepted:
                raise ConversionError(
                    f"{path} has UBJSON element marker {value.marker!r}; expected {accepted!r}."
                )
    return value


def _string(value: Any, path: str) -> str:
    if type(value) is not str:
        raise ConversionError(f"{path} must be a string.")
    return value


def _decimal(value: Any, path: str, *, minimum: int = 0) -> int:
    text = _string(value, path)
    if not text or not text.isascii() or not text.isdecimal():
        raise ConversionError(f"{path} must be a canonical non-negative decimal string.")
    parsed = int(text)
    if str(parsed) != text or parsed < minimum:
        raise ConversionError(f"{path} is outside its supported range.")
    return parsed


def _integer(value: Any, path: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise ConversionError(f"{path} must be an integer.")
    if value < minimum or value > maximum:
        raise ConversionError(f"{path} is outside [{minimum}, {maximum}].")
    return value


def _float_string(value: Any, path: str) -> float:
    text = _string(value, path)
    if _FLOAT_TOKEN.fullmatch(text) is None:
        raise ConversionError(f"{path} must be a canonical finite floating-point string.")
    parsed = float(text)
    if not math.isfinite(parsed) or abs(parsed) > _FLOAT32_MAX:
        raise ConversionError(f"{path} must be finite float32 data.")
    return parsed


def _base_scores(value: Any, path: str) -> tuple[np.float32, ...]:
    text = _string(value, path)
    if text.startswith("["):
        if not text.endswith("]"):
            raise ConversionError(f"{path} has a malformed parameter vector.")
        body = text[1:-1]
        tokens = [] if not body else body.split(",")
    else:
        tokens = [text]
    if not tokens:
        raise ConversionError(f"{path} must contain at least one base score.")
    result: list[np.float32] = []
    for index, token in enumerate(tokens):
        if token != token.strip() or _FLOAT_TOKEN.fullmatch(token) is None:
            raise ConversionError(f"{path}[{index}] is not a canonical finite float.")
        parsed = float(token)
        if not math.isfinite(parsed) or abs(parsed) > _FLOAT32_MAX:
            raise ConversionError(f"{path}[{index}] is not finite float32 data.")
        converted = np.float32(parsed)
        result.append(converted)
    return tuple(result)


def _int_array(
    value: Any,
    path: str,
    *,
    minimum: int,
    maximum: int,
    marker: str | tuple[str, ...] | None,
    untyped: bool = False,
) -> np.ndarray:
    values = _array(value, path, marker=marker, untyped=untyped)
    result = np.empty((len(values),), dtype=np.int64)
    for index, item in enumerate(values):
        result[index] = _integer(
            item, f"{path}[{index}]", minimum=minimum, maximum=maximum
        )
    return result


def _float_array(
    value: Any,
    path: str,
    *,
    marker: str | None,
    untyped: bool = False,
) -> np.ndarray:
    values = _array(value, path, marker=marker, untyped=untyped)
    result = np.empty((len(values),), dtype=np.float32)
    for index, item in enumerate(values):
        if type(item) is not float:
            raise ConversionError(f"{path}[{index}] must be a floating-point number.")
        if math.isinf(item) or (math.isfinite(item) and abs(item) > _FLOAT32_MAX):
            raise ConversionError(f"{path}[{index}] is outside finite float32 range.")
        converted = np.float32(item)
        result[index] = converted
    return result


def _finite_float_parameter(
    objective: dict[str, Any],
    field: str,
    parameter: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    config = _object(objective[field], f"learner.objective.{field}")
    _fields(config, frozenset({parameter}), f"learner.objective.{field}")
    parsed = _float_string(config[parameter], f"learner.objective.{field}.{parameter}")
    if minimum is not None and parsed < minimum:
        raise UnsupportedConversionError(
            f"Objective parameter {parameter} is below its supported range."
        )
    if maximum is not None and (
        parsed > maximum or (not maximum_inclusive and parsed == maximum)
    ):
        raise UnsupportedConversionError(
            f"Objective parameter {parameter} is above its supported range."
        )
    return parsed


def _objective(
    value: Any, num_class: int, output_width: int
) -> tuple[str, str, str, str]:
    objective = _object(value, "learner.objective")
    if "name" not in objective:
        raise ConversionError("learner.objective is missing required field 'name'.")
    name = _string(objective["name"], "learner.objective.name")

    reg_loss = {
        "reg:squarederror": ("identity", "identity", "continuous"),
        "reg:logistic": ("sigmoid", "logit", "continuous"),
        "binary:logistic": ("sigmoid", "logit", "binary"),
        "binary:logitraw": ("identity", "identity", "binary"),
        "reg:gamma": ("exponential", "log", "continuous"),
    }
    name_only = {
        "reg:squaredlogerror": ("identity", "identity", "continuous"),
        "reg:absoluteerror": ("identity", "identity", "continuous"),
        "survival:cox": ("exponential", "log", "continuous"),
    }
    parameterized = {
        "reg:pseudohubererror": (
            "identity",
            "identity",
            "continuous",
            "pseudo_huber_param",
            "huber_slope",
            None,
            None,
            True,
        ),
        "count:poisson": (
            "exponential",
            "log",
            "count",
            "poisson_regression_param",
            "max_delta_step",
            0.0,
            None,
            True,
        ),
        "reg:tweedie": (
            "exponential",
            "log",
            "continuous",
            "tweedie_regression_param",
            "tweedie_variance_power",
            1.0,
            2.0,
            False,
        ),
    }

    if name in reg_loss:
        _fields(objective, frozenset({"name", "reg_loss_param"}), "learner.objective")
        scale = _finite_float_parameter(
            objective,
            "reg_loss_param",
            "scale_pos_weight",
            minimum=0.0,
        )
        del scale
        transform, margin_link, target_kind = reg_loss[name]
    elif name in name_only:
        _fields(objective, frozenset({"name"}), "learner.objective")
        transform, margin_link, target_kind = name_only[name]
    elif name in parameterized:
        (
            transform,
            margin_link,
            target_kind,
            field,
            parameter,
            minimum,
            maximum,
            inclusive,
        ) = parameterized[name]
        _fields(objective, frozenset({"name", field}), "learner.objective")
        parsed = _finite_float_parameter(
            objective,
            field,
            parameter,
            minimum=minimum,
            maximum=maximum,
            maximum_inclusive=inclusive,
        )
        if name == "reg:pseudohubererror" and parsed == 0.0:
            raise UnsupportedConversionError("Pseudo-Huber slope must be nonzero.")
    elif name == "survival:aft":
        _fields(objective, frozenset({"name", "aft_loss_param"}), "learner.objective")
        config = _object(objective["aft_loss_param"], "learner.objective.aft_loss_param")
        _fields(
            config,
            frozenset({"aft_loss_distribution", "aft_loss_distribution_scale"}),
            "learner.objective.aft_loss_param",
        )
        distribution = _string(
            config["aft_loss_distribution"],
            "learner.objective.aft_loss_param.aft_loss_distribution",
        )
        if distribution not in {"normal", "logistic", "extreme"}:
            raise UnsupportedConversionError(
                f"Unsupported AFT loss distribution {distribution!r}."
            )
        scale = _float_string(
            config["aft_loss_distribution_scale"],
            "learner.objective.aft_loss_param.aft_loss_distribution_scale",
        )
        if scale <= 0.0:
            raise UnsupportedConversionError("AFT distribution scale must be positive.")
        transform, margin_link, target_kind = "exponential", "log", "continuous"
    elif name == "multi:softprob":
        _fields(
            objective,
            frozenset({"name", "softmax_multiclass_param"}),
            "learner.objective",
        )
        config = _object(
            objective["softmax_multiclass_param"],
            "learner.objective.softmax_multiclass_param",
        )
        _fields(
            config,
            frozenset({"num_class"}),
            "learner.objective.softmax_multiclass_param",
        )
        objective_classes = _decimal(
            config["num_class"],
            "learner.objective.softmax_multiclass_param.num_class",
            minimum=2,
        )
        if objective_classes != num_class:
            raise ConversionError("Objective and learner num_class values disagree.")
        transform, margin_link, target_kind = "softmax", "identity", "multiclass"
    else:
        raise UnsupportedConversionError(
            f"XGBoost objective {name!r} has no exact native prediction transform."
        )

    if name.startswith("multi:"):
        if num_class < 2 or output_width != num_class:
            raise ConversionError(
                "Multiclass objective has inconsistent output dimensions."
            )
    elif num_class != 0:
        raise ConversionError("Non-multiclass objective has nonzero num_class.")
    if transform == "sigmoid" and output_width != 1:
        raise UnsupportedConversionError(
            "Native TreeEnsemble sigmoid prediction supports exactly one output."
        )
    if target_kind == "binary" and output_width != 1:
        raise ConversionError("Binary objective must have exactly one output margin.")
    return name, transform, margin_link, target_kind


def _margin_base_score(
    values: tuple[np.float32, ...], width: int, link: str
) -> np.ndarray:
    if len(values) == 1:
        expanded = np.full((width,), values[0], dtype=np.float32)
    elif len(values) == width:
        expanded = np.asarray(values, dtype=np.float32)
    else:
        raise ConversionError(
            "base_score must be scalar or have exactly one value per output."
        )
    if link == "identity":
        return expanded
    if link == "logit":
        if np.any((expanded < 0.0) | (expanded > 1.0)):
            raise ConversionError("Logistic base_score must lie in [0, 1].")
        epsilon = np.float32(1.0e-6)
        probability = np.clip(expanded, epsilon, np.float32(1.0) - epsilon)
        return np.log(probability / (np.float32(1.0) - probability)).astype(np.float32)
    if link == "log":
        if np.any(expanded <= 0.0):
            raise ConversionError("Log-link base_score must be positive.")
        return np.log(expanded).astype(np.float32)
    raise ConversionError(f"Internal unsupported base-score link {link!r}.")


def _empty_cats(value: Any) -> None:
    cats = _object(value, "learner.gradient_booster.model.cats")
    _fields(
        cats,
        frozenset({"enc", "feature_segments", "sorted_idx"}),
        "learner.gradient_booster.model.cats",
    )
    enc = _array(cats["enc"], "learner.gradient_booster.model.cats.enc", untyped=True)
    segments = _int_array(
        cats["feature_segments"],
        "learner.gradient_booster.model.cats.feature_segments",
        minimum=0,
        maximum=_INT32_MAX,
        marker="l",
    )
    sorted_indices = _int_array(
        cats["sorted_idx"],
        "learner.gradient_booster.model.cats.sorted_idx",
        minimum=0,
        maximum=_INT32_MAX,
        marker="l",
    )
    if enc or segments.size or sorted_indices.size:
        raise UnsupportedConversionError(
            "XGBoost dataframe category recoding metadata is unsupported; "
            "convert an artifact trained on already encoded category IDs."
        )


class _TreeData:
    __slots__ = (
        "categories",
        "categorical_features",
        "default_left",
        "feature_index",
        "gain",
        "leaf_value",
        "left_child",
        "node_count",
        "right_child",
        "split_kind",
        "threshold",
        "cover",
        "vector_leaf",
    )

    def __init__(
        self,
        *,
        categories: dict[int, tuple[int, ...]],
        categorical_features: set[int],
        default_left: np.ndarray,
        feature_index: np.ndarray,
        gain: np.ndarray,
        leaf_value: np.ndarray,
        left_child: np.ndarray,
        node_count: int,
        right_child: np.ndarray,
        split_kind: np.ndarray,
        threshold: np.ndarray,
        cover: np.ndarray,
        vector_leaf: bool,
    ):
        self.categories = categories
        self.categorical_features = categorical_features
        self.default_left = default_left
        self.feature_index = feature_index
        self.gain = gain
        self.leaf_value = leaf_value
        self.left_child = left_child
        self.node_count = node_count
        self.right_child = right_child
        self.split_kind = split_kind
        self.threshold = threshold
        self.cover = cover
        self.vector_leaf = vector_leaf


def _tree(
    value: Any,
    tree_index: int,
    *,
    num_feature: int,
    output_width: int,
    output_group: int,
) -> _TreeData:
    path = f"learner.gradient_booster.model.trees[{tree_index}]"
    tree = _object(value, path)
    tree_param = _object(tree.get("tree_param"), f"{path}.tree_param")
    _fields(tree_param, _TREE_PARAM_FIELDS, f"{path}.tree_param")
    size_leaf_vector = _decimal(
        tree_param["size_leaf_vector"], f"{path}.tree_param.size_leaf_vector", minimum=1
    )
    vector_leaf = size_leaf_vector != 1
    required = _TREE_COMMON_FIELDS | (
        frozenset({"leaf_weights"}) if vector_leaf else frozenset()
    )
    _fields(tree, required, path)

    identifier = _integer(tree["id"], f"{path}.id", minimum=0, maximum=_INT32_MAX)
    if identifier != tree_index:
        raise ConversionError(f"{path}.id must equal its zero-based tree position.")
    node_count = _decimal(
        tree_param["num_nodes"], f"{path}.tree_param.num_nodes", minimum=1
    )
    if node_count > _INT32_MAX:
        raise UnsupportedConversionError("Tree node count exceeds native int32 capacity.")
    deleted = _decimal(tree_param["num_deleted"], f"{path}.tree_param.num_deleted")
    if deleted != 0:
        raise UnsupportedConversionError(
            "Trees retaining deleted/pruned node slots are not supported by this schema gate."
        )
    tree_features = _decimal(
        tree_param["num_feature"], f"{path}.tree_param.num_feature", minimum=1
    )
    if tree_features != num_feature:
        raise ConversionError(f"{path} and learner num_feature values disagree.")
    if vector_leaf and size_leaf_vector != output_width:
        raise UnsupportedConversionError(
            f"{path} vector-leaf width does not equal the learner output width."
        )

    left = _int_array(
        tree["left_children"],
        f"{path}.left_children",
        minimum=-1,
        maximum=_INT32_MAX,
        marker="l",
    )
    right = _int_array(
        tree["right_children"],
        f"{path}.right_children",
        minimum=-1,
        maximum=_INT32_MAX,
        marker="l",
    )
    parents = _int_array(
        tree["parents"], f"{path}.parents", minimum=-1, maximum=_INT32_MAX, marker="l"
    )
    indices = _int_array(
        tree["split_indices"],
        f"{path}.split_indices",
        minimum=0,
        maximum=_INT32_MAX,
        marker=("l", "L"),
    )
    conditions = _float_array(
        tree["split_conditions"], f"{path}.split_conditions", marker="d"
    )
    defaults = _int_array(
        tree["default_left"],
        f"{path}.default_left",
        minimum=0,
        maximum=1,
        marker="U",
    )
    split_type = _int_array(
        tree["split_type"],
        f"{path}.split_type",
        minimum=0,
        maximum=1,
        marker="U",
    )
    gain = _float_array(tree["loss_changes"], f"{path}.loss_changes", marker="d")
    cover = _float_array(tree["sum_hessian"], f"{path}.sum_hessian", marker="d")
    base_weights = _float_array(tree["base_weights"], f"{path}.base_weights", marker="d")

    arrays = (
        left,
        right,
        parents,
        indices,
        conditions,
        defaults,
        split_type,
        gain,
        cover,
    )
    if any(array.size != node_count for array in arrays):
        raise ConversionError(
            f"All per-node arrays in {path} must have length num_nodes."
        )
    if np.any(~np.isfinite(gain)) or np.any(~np.isfinite(cover)):
        raise ConversionError(f"Training-statistic arrays in {path} must be finite.")
    if np.any(cover < 0.0):
        raise ConversionError(f"sum_hessian in {path} must be non-negative.")

    is_leaf = left == -1
    if not np.any(is_leaf):
        raise ConversionError(f"{path} has no leaf.")
    if parents[0] != -1:
        raise ConversionError(f"{path} root parent must be -1.")

    if vector_leaf:
        if base_weights.size != node_count * output_width:
            raise UnsupportedConversionError(
                f"{path}.base_weights does not use one split-weight vector per node."
            )
        leaf_weights = _float_array(
            tree["leaf_weights"], f"{path}.leaf_weights", marker="d"
        )
        leaf_count = int(np.count_nonzero(is_leaf))
        if leaf_weights.size != leaf_count * output_width:
            raise ConversionError(
                f"{path}.leaf_weights has inconsistent vector-leaf shape."
            )
        leaf_mapping = right[is_leaf]
        if sorted(int(index) for index in leaf_mapping) != list(range(leaf_count)):
            raise ConversionError(
                f"{path} vector-leaf indices are not a complete permutation."
            )
    else:
        if base_weights.size != node_count:
            raise ConversionError(f"{path}.base_weights must have length num_nodes.")
        leaf_weights = np.empty((0,), dtype=np.float32)
        if np.any(right[is_leaf] != -1):
            raise ConversionError(
                f"Scalar leaves in {path} must have two invalid children."
            )

    if np.any(~np.isfinite(base_weights)) or np.any(~np.isfinite(leaf_weights)):
        raise ConversionError(
            f"Prediction and training weights in {path} must be finite."
        )
    if np.any(split_type[is_leaf] != 0):
        raise UnsupportedConversionError(
            f"Leaves in {path} cannot be categorical splits."
        )

    internal = ~is_leaf
    if np.any(right[internal] < 0) or np.any(left[internal] < 0):
        raise ConversionError(f"Internal nodes in {path} must have two children.")
    if np.any(right[internal] >= node_count) or np.any(left[internal] >= node_count):
        raise ConversionError(f"Internal child index in {path} is out of range.")
    if np.any(right[internal] == left[internal]):
        raise ConversionError(f"Internal nodes in {path} must have distinct children.")
    if np.any(indices[internal] >= num_feature):
        raise ConversionError(f"Internal split feature in {path} is out of range.")

    visited = {0}
    stack = [0]
    expected_parent = np.full((node_count,), -2, dtype=np.int64)
    expected_parent[0] = -1
    while stack:
        node = stack.pop()
        if is_leaf[node]:
            continue
        for child in (int(left[node]), int(right[node])):
            if child in visited:
                raise ConversionError(
                    f"{path} routing graph has a cycle or shared child."
                )
            visited.add(child)
            expected_parent[child] = node
            stack.append(child)
    if len(visited) != node_count:
        raise ConversionError(f"{path} contains unreachable active nodes.")
    if not np.array_equal(parents, expected_parent):
        raise ConversionError(f"{path}.parents does not agree with child links.")

    category_nodes = _int_array(
        tree["categories_nodes"],
        f"{path}.categories_nodes",
        minimum=0,
        maximum=node_count - 1,
        marker="l",
    )
    category_segments = _int_array(
        tree["categories_segments"],
        f"{path}.categories_segments",
        minimum=0,
        maximum=_INT32_MAX,
        marker="L",
    )
    category_sizes = _int_array(
        tree["categories_sizes"],
        f"{path}.categories_sizes",
        minimum=1,
        maximum=_INT32_MAX,
        marker="L",
    )
    category_values = _int_array(
        tree["categories"],
        f"{path}.categories",
        minimum=0,
        maximum=_MAX_CATEGORY - 1,
        marker="l",
    )
    category_count = category_nodes.size
    if category_segments.size != category_count or category_sizes.size != category_count:
        raise UnsupportedConversionError(
            f"Categorical split segment arrays in {path} have inconsistent cardinality."
        )
    expected_nodes = np.flatnonzero(split_type == 1).astype(np.int64)
    if not np.array_equal(category_nodes, expected_nodes):
        raise UnsupportedConversionError(
            f"Categorical node IDs in {path} disagree with split_type."
        )

    categories: dict[int, tuple[int, ...]] = {}
    offset = 0
    for position, node in enumerate(category_nodes):
        begin = int(category_segments[position])
        size = int(category_sizes[position])
        if begin != offset or begin + size > category_values.size:
            raise UnsupportedConversionError(
                f"Categorical segment {position} in {path} is not contiguous and valid."
            )
        selected = tuple(int(item) for item in category_values[begin : begin + size])
        if tuple(sorted(set(selected))) != selected:
            raise UnsupportedConversionError(
                f"Categorical values for node {int(node)} in {path} are not strictly ordered."
            )
        categories[int(node)] = selected
        offset += size
    if offset != category_values.size:
        raise UnsupportedConversionError(
            f"Categorical segments in {path} leave unused values."
        )

    threshold = np.zeros((node_count,), dtype=np.float32)
    native_left = left.astype(np.int32)
    native_right = right.astype(np.int32)
    native_default = defaults.astype(bool)
    native_feature = indices.astype(np.int32)
    native_split_kind = split_type.astype(np.int8)
    categorical_features: set[int] = set()
    for node in range(node_count):
        if is_leaf[node]:
            if not vector_leaf and not np.isfinite(conditions[node]):
                raise ConversionError(f"Scalar leaf value in {path} must be finite.")
            continue
        if split_type[node] == 0:
            condition = conditions[node]
            if not np.isfinite(condition):
                raise ConversionError(f"Numerical threshold in {path} must be finite.")
            threshold[node] = condition
            native_split_kind[node] = np.int8(2)
        else:
            categorical_features.add(int(indices[node]))
            native_left[node] = np.int32(right[node])
            native_right[node] = np.int32(left[node])
            native_default[node] = not bool(defaults[node])

    leaf_value = np.zeros((node_count, output_width), dtype=np.float32)
    if vector_leaf:
        for node in np.flatnonzero(is_leaf):
            begin = int(right[node]) * output_width
            leaf_value[node, :] = leaf_weights[begin : begin + output_width]
            native_right[node] = -1
    else:
        leaf_value[is_leaf, output_group] = conditions[is_leaf]

    data = _TreeData(
        categorical_features=categorical_features,
        categories=categories,
        default_left=native_default,
        feature_index=native_feature,
        gain=gain,
        leaf_value=leaf_value,
        left_child=native_left,
        node_count=node_count,
        right_child=native_right,
        split_kind=native_split_kind,
        threshold=threshold,
        cover=cover,
        vector_leaf=vector_leaf,
    )
    return data


def from_xgboost_artifact(source: Any, /) -> ConversionResult:
    """Convert an XGBoost JSON/UBJSON saved model without importing XGBoost.

    The accepted inference domain is a dense float32-compatible feature matrix using
    NaN for missing values and finite integral codes for categorical features. The
    returned model contains all prediction arrays and never retains or calls the source.
    """

    document, source_format, artifact_sha256 = _load_source(source)
    _fields(document, _TOP_FIELDS, "saved model")

    version_values = _array(document["version"], "version", untyped=True)
    if len(version_values) != 3:
        raise ConversionError("version must be an exact [major, minor, patch] triplet.")
    version = tuple(
        _integer(item, f"version[{index}]", minimum=0, maximum=_INT32_MAX)
        for index, item in enumerate(version_values)
    )
    if version[0] not in {2, 3}:
        raise UnsupportedConversionError(
            f"XGBoost saved-model version {version!r} is outside the audited 2.x/3.x gate."
        )

    learner = _object(document["learner"], "learner")
    _fields(learner, _LEARNER_FIELDS, "learner")
    learner_param = _object(learner["learner_model_param"], "learner.learner_model_param")
    _fields(learner_param, _LEARNER_PARAM_FIELDS, "learner.learner_model_param")

    num_feature = _decimal(
        learner_param["num_feature"], "learner.learner_model_param.num_feature", minimum=1
    )
    if num_feature > _INT32_MAX:
        raise UnsupportedConversionError("num_feature exceeds native int32 capacity.")
    num_class = _decimal(
        learner_param["num_class"], "learner.learner_model_param.num_class"
    )
    num_target = _decimal(
        learner_param["num_target"], "learner.learner_model_param.num_target", minimum=1
    )
    if num_class > 1 and num_target > 1:
        raise UnsupportedConversionError(
            "Simultaneous multiclass and multi-target output dimensions are ambiguous."
        )
    output_width = max(num_class, num_target, 1)
    boost_from_average = _decimal(
        learner_param["boost_from_average"],
        "learner.learner_model_param.boost_from_average",
    )
    if boost_from_average not in {0, 1}:
        raise UnsupportedConversionError("boost_from_average must be 0 or 1.")

    objective_name, transform, margin_link, target_kind = _objective(
        learner["objective"], num_class, output_width
    )
    source_base_score = _base_scores(
        learner_param["base_score"], "learner.learner_model_param.base_score"
    )
    if version < (3, 1, 0) and len(source_base_score) != 1:
        raise UnsupportedConversionError(
            "Vector base_score serialization is supported only for XGBoost 3.1+."
        )
    base_score = _margin_base_score(source_base_score, output_width, margin_link)

    attributes = _object(learner["attributes"], "learner.attributes")
    for key, value in attributes.items():
        if type(key) is not str or type(value) is not str:
            raise ConversionError("learner.attributes must map strings to strings.")

    feature_name_values = _array(
        learner["feature_names"], "learner.feature_names", untyped=True
    )
    feature_names = tuple(
        _string(item, f"learner.feature_names[{index}]")
        for index, item in enumerate(feature_name_values)
    )
    if feature_names and len(feature_names) != num_feature:
        raise ConversionError("feature_names must be empty or have length num_feature.")
    if any(not name for name in feature_names) or len(set(feature_names)) != len(
        feature_names
    ):
        raise ConversionError("feature_names must be non-empty and unique when present.")
    source_feature_names = feature_names
    if not feature_names:
        feature_names = tuple(f"feature_{index}" for index in range(num_feature))

    feature_type_values = _array(
        learner["feature_types"], "learner.feature_types", untyped=True
    )
    feature_types = tuple(
        _string(item, f"learner.feature_types[{index}]")
        for index, item in enumerate(feature_type_values)
    )
    if feature_types and len(feature_types) != num_feature:
        raise ConversionError("feature_types must be empty or have length num_feature.")
    feature_kind_map = {
        "c": "categorical",
        "float": "continuous",
        "i": "ordinal",
        "int": "ordinal",
        "q": "continuous",
    }
    unknown_types = sorted(set(feature_types) - set(feature_kind_map))
    if unknown_types:
        raise UnsupportedConversionError(
            f"Unsupported XGBoost feature types {unknown_types!r}."
        )

    gradient_booster = _object(learner["gradient_booster"], "learner.gradient_booster")
    _fields(
        gradient_booster,
        frozenset({"model", "name"}),
        "learner.gradient_booster",
    )
    booster_name = _string(gradient_booster["name"], "learner.gradient_booster.name")
    if booster_name != "gbtree":
        raise UnsupportedConversionError(
            f"Only the direct gbtree saved-model layout is supported, not {booster_name!r}."
        )
    booster_model = _object(gradient_booster["model"], "learner.gradient_booster.model")
    model_required = frozenset(
        {"gbtree_model_param", "iteration_indptr", "tree_info", "trees"}
    )
    model_optional = frozenset({"cats", "weight_drop"})
    _fields(
        booster_model,
        model_required,
        "learner.gradient_booster.model",
        model_optional,
    )
    cats_required = version >= (3, 1, 0)
    if cats_required and "cats" not in booster_model:
        raise ConversionError(
            "XGBoost 3.1+ gbtree model is missing required cats metadata."
        )
    if not cats_required and "cats" in booster_model:
        raise UnsupportedConversionError(
            "cats metadata is not part of the audited pre-3.1 saved-model schema."
        )
    if "cats" in booster_model:
        _empty_cats(booster_model["cats"])

    booster_param = _object(
        booster_model["gbtree_model_param"],
        "learner.gradient_booster.model.gbtree_model_param",
    )
    _fields(
        booster_param,
        frozenset({"num_parallel_tree", "num_trees"}),
        "learner.gradient_booster.model.gbtree_model_param",
    )
    num_trees = _decimal(
        booster_param["num_trees"],
        "learner.gradient_booster.model.gbtree_model_param.num_trees",
    )
    if num_trees > _INT32_MAX:
        raise UnsupportedConversionError("num_trees exceeds native int32 capacity.")
    num_parallel_tree = _decimal(
        booster_param["num_parallel_tree"],
        "learner.gradient_booster.model.gbtree_model_param.num_parallel_tree",
        minimum=1,
    )

    tree_values = _array(
        booster_model["trees"], "learner.gradient_booster.model.trees", untyped=True
    )
    if len(tree_values) != num_trees:
        raise ConversionError("num_trees does not match the trees array length.")
    tree_info = _int_array(
        booster_model["tree_info"],
        "learner.gradient_booster.model.tree_info",
        minimum=0,
        maximum=output_width - 1,
        marker=None,
        untyped=True,
    )
    if tree_info.size != num_trees:
        raise ConversionError("tree_info must contain one output group per tree.")
    iteration_indptr = _int_array(
        booster_model["iteration_indptr"],
        "learner.gradient_booster.model.iteration_indptr",
        minimum=0,
        maximum=num_trees,
        marker=None,
        untyped=True,
    )
    if iteration_indptr.size == 0 or iteration_indptr[0] != 0:
        raise ConversionError("iteration_indptr must begin at zero.")
    if iteration_indptr[-1] != num_trees:
        raise ConversionError("iteration_indptr must end at num_trees.")
    if np.any(np.diff(iteration_indptr) <= 0):
        if not (num_trees == 0 and iteration_indptr.size == 1):
            raise ConversionError("iteration_indptr must be strictly increasing.")

    if "weight_drop" in booster_model:
        supplied_weights = _float_array(
            booster_model["weight_drop"],
            "learner.gradient_booster.model.weight_drop",
            marker=None,
            untyped=True,
        )
        if supplied_weights.size == 0:
            tree_weights = np.ones((num_trees,), dtype=np.float32)
        elif supplied_weights.size == num_trees:
            if np.any(~np.isfinite(supplied_weights)):
                raise UnsupportedConversionError("DART tree weights must be finite.")
            tree_weights = supplied_weights
        else:
            raise UnsupportedConversionError(
                "DART weight_drop must be empty or contain exactly one weight per tree."
            )
    else:
        tree_weights = np.ones((num_trees,), dtype=np.float32)
    dart_weighted = bool(num_trees and not np.all(tree_weights == np.float32(1.0)))

    tree_data = [
        _tree(
            tree,
            index,
            num_feature=num_feature,
            output_width=output_width,
            output_group=int(tree_info[index]),
        )
        for index, tree in enumerate(tree_values)
    ]
    vector_layouts = {tree.vector_leaf for tree in tree_data}
    if len(vector_layouts) > 1:
        raise UnsupportedConversionError("Scalar and vector-leaf trees cannot be mixed.")
    vector_leaf = vector_layouts == {True}

    for begin, end in zip(iteration_indptr[:-1], iteration_indptr[1:], strict=True):
        begin_ = int(begin)
        end_ = int(end)
        if vector_leaf:
            if end_ - begin_ != num_parallel_tree:
                raise UnsupportedConversionError(
                    "Vector-leaf iteration span does not match num_parallel_tree."
                )
            if np.any(tree_info[begin_:end_] != 0):
                raise ConversionError(
                    "Vector-leaf trees must use tree_info output group zero."
                )
        else:
            expected_groups = tuple(
                group for group in range(output_width) for _ in range(num_parallel_tree)
            )
            actual_groups = tuple(int(group) for group in tree_info[begin_:end_])
            if actual_groups != expected_groups:
                raise UnsupportedConversionError(
                    "Scalar-tree iteration has noncanonical parallel/output group ordering."
                )

    categorical_features: set[int] = set()
    for tree in tree_data:
        categorical_features.update(tree.categorical_features)
    if feature_types:
        declared_categorical = {
            index
            for index, feature_type in enumerate(feature_types)
            if feature_type == "c"
        }
        if not categorical_features.issubset(declared_categorical):
            raise UnsupportedConversionError(
                "Categorical split nodes disagree with learner feature_types metadata."
            )
        feature_kinds = tuple(feature_kind_map[item] for item in feature_types)
    else:
        feature_kinds = tuple(
            "categorical" if index in categorical_features else "continuous"
            for index in range(num_feature)
        )

    tree_capacity = max(1, num_trees)
    node_capacity = max((tree.node_count for tree in tree_data), default=1)
    category_capacity = max(
        (
            len(categories)
            for tree in tree_data
            for categories in tree.categories.values()
        ),
        default=1,
    )
    node_shape = (tree_capacity, node_capacity)
    feature_index = np.zeros(node_shape, dtype=np.int32)
    threshold = np.zeros(node_shape, dtype=np.float32)
    left_child = np.zeros(node_shape, dtype=np.int32)
    right_child = np.zeros(node_shape, dtype=np.int32)
    default_left = np.zeros(node_shape, dtype=bool)
    split_kind = np.zeros(node_shape, dtype=np.int8)
    category_values = np.zeros(node_shape + (category_capacity,), dtype=np.float32)
    category_mask = np.zeros(node_shape + (category_capacity,), dtype=bool)
    leaf_value = np.zeros(node_shape + (output_width,), dtype=np.float32)
    node_mask = np.zeros(node_shape, dtype=bool)
    leaf_mask = np.zeros(node_shape, dtype=bool)
    tree_mask = np.zeros((tree_capacity,), dtype=bool)
    native_tree_weight = np.zeros((tree_capacity,), dtype=np.float32)
    node_gain = np.zeros(node_shape, dtype=np.float32)
    node_cover = np.zeros(node_shape, dtype=np.float32)

    for index, tree in enumerate(tree_data):
        count = tree.node_count
        feature_index[index, :count] = tree.feature_index
        threshold[index, :count] = tree.threshold
        left_child[index, :count] = tree.left_child
        right_child[index, :count] = tree.right_child
        default_left[index, :count] = tree.default_left
        split_kind[index, :count] = tree.split_kind
        leaf_value[index, :count, :] = tree.leaf_value
        node_mask[index, :count] = True
        leaf_mask[index, :count] = tree.left_child == -1
        tree_mask[index] = True
        native_tree_weight[index] = tree_weights[index]
        node_gain[index, :count] = tree.gain
        node_cover[index, :count] = tree.cover
        for node, categories in tree.categories.items():
            size = len(categories)
            category_values[index, node, :size] = categories
            category_mask[index, node, :size] = True

    feature_schema = FeatureSchema(
        feature_names,
        kinds=feature_kinds,
        layout_id="xgboost-positional-float32-v1",
    )
    target_names = (
        ()
        if output_width == 1
        else tuple(f"target_{index}" for index in range(output_width))
    )
    target_schema = TargetSchema(target_kind, names=target_names)
    version_text = ".".join(str(component) for component in version)
    objective_json = json.dumps(
        learner["objective"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    attributes_json = json.dumps(
        attributes, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    source_base = tuple(float(value) for value in source_base_score)
    margin_base = tuple(float(value) for value in base_score)
    weights_configuration = tuple(float(value) for value in tree_weights)
    provenance = ConversionProvenance(
        source="xgboost",
        source_version=version_text,
        source_model="gbtree",
        configuration={
            "aggregation": "ordered float32 tree sum",
            "array_sha256": _native_array_checksum(
                (
                    ("base_score", base_score),
                    ("category_mask", category_mask),
                    ("category_values", category_values),
                    ("default_left", default_left),
                    ("feature_index", feature_index),
                    ("leaf_mask", leaf_mask),
                    ("leaf_value", leaf_value),
                    ("left_child", left_child),
                    ("node_cover", node_cover),
                    ("node_gain", node_gain),
                    ("node_mask", node_mask),
                    ("right_child", right_child),
                    ("split_kind", split_kind),
                    ("threshold", threshold),
                    ("tree_mask", tree_mask),
                    ("tree_weight", native_tree_weight),
                )
            ),
            "artifact_sha256": artifact_sha256,
            "attributes": attributes_json,
            "base_score_margin": margin_base,
            "base_score_source": source_base,
            "boost_from_average": boost_from_average,
            "booster": booster_name,
            "categorical_features": tuple(sorted(categorical_features)),
            "base_margin_policy": "external per-row base_margin overrides are unsupported",
            "categorical_input_domain": (
                "finite integral float32 codes in [0,16777216); string/category recoding rejected"
            ),
            "categorical_routing": "selected categories route right; native children swapped",
            "converter_schema": _SCHEMA_REVISION,
            "converted_at": datetime.now(timezone.utc).isoformat(),
            "dart_weighted": dart_weighted,
            "dense_input_domain": "finite float32 values with NaN missing sentinels",
            "feature_types": feature_types,
            "iteration_indptr": tuple(int(value) for value in iteration_indptr),
            "missing_routing": "NaN follows persisted default_left child",
            "num_class": num_class,
            "num_feature": num_feature,
            "num_parallel_tree": num_parallel_tree,
            "num_target": num_target,
            "num_trees": num_trees,
            "numeric_routing": "XGBoost strict float32 '<' encoded by predecessor thresholds",
            "objective": objective_name,
            "objective_config": objective_json,
            "objective_transform": transform,
            "output_width": output_width,
            "source_format": source_format,
            "semantic_notes": (
                "feature names are metadata; split_indices remain positional",
                "sparse absence must be normalized to NaN before native prediction",
                "prediction is fully native and never imports or calls XGBoost",
            ),
            "tree_info": tuple(int(value) for value in tree_info),
            "tree_weights": weights_configuration,
            "vector_leaf": vector_leaf,
        },
        feature_names=source_feature_names,
        class_labels=(),
        license_id="Apache-2.0",
    )

    model = TreeEnsemble(
        feature_index=feature_index,
        threshold=threshold,
        left_child=left_child,
        right_child=right_child,
        default_left=default_left,
        split_kind=split_kind,
        category_values=category_values,
        category_mask=category_mask,
        leaf_value=leaf_value,
        node_mask=node_mask,
        leaf_mask=leaf_mask,
        tree_mask=tree_mask,
        tree_weight=native_tree_weight,
        node_gain=node_gain,
        node_cover=node_cover,
        base_score=base_score,
        feature_schema=feature_schema,
        target_schema=target_schema,
        objective_transform=transform,
        aggregation="sum",
        input_dtype="float32",
        out_size="scalar" if output_width == 1 else output_width,
        max_steps=node_capacity,
    )
    return ConversionResult(model, provenance)


__all__ = ["from_xgboost_artifact"]
