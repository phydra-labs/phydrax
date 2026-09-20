#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded structural decoding for external text, JSON, and XML resources."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
from dataclasses import dataclass
from typing import Any

from ._external_resource import (
    account_bounded_resource,
    BoundedResource,
    ResourceReadError,
)


@dataclass(frozen=True, slots=True)
class DecodedText:
    """Decoded text and updated structural resource evidence."""

    value: str
    resource: BoundedResource


@dataclass(frozen=True, slots=True)
class DecodedJSON:
    """Finite JSON value and updated structural resource evidence."""

    value: Any
    resource: BoundedResource


@dataclass(frozen=True, slots=True)
class DecodedXML:
    """XML root and updated structural resource evidence."""

    root: ET.Element
    resource: BoundedResource


def decode_text_resource(
    resource: BoundedResource,
    /,
    *,
    encoding: str = "utf-8",
    allow_utf8_bom: bool = False,
    max_line_bytes: int,
) -> DecodedText:
    """Decode bounded text while enforcing encoding and line-size policy."""

    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if type(max_line_bytes) is not int or max_line_bytes <= 0:
        raise ValueError("max_line_bytes must be a positive integer.")
    if encoding not in {"ascii", "utf-8", "utf-8-sig"}:
        raise ValueError("Only ASCII and UTF-8 text encodings are admitted.")
    if resource.data.startswith(b"\xef\xbb\xbf") and not allow_utf8_bom:
        raise ResourceReadError("policy", "UTF-8 byte-order marks are not admitted.")
    if any(len(line) > max_line_bytes for line in resource.data.splitlines()):
        raise ResourceReadError("limit", "Resource line exceeds its byte limit.")
    try:
        value = resource.data.decode(encoding)
    except UnicodeDecodeError as error:
        raise ResourceReadError(
            "malformed", "Resource text encoding is invalid."
        ) from error
    nodes = value.count("\n") + (1 if value else 0)
    accounted = account_bounded_resource(
        resource,
        depth=1 if value else 0,
        nodes=nodes,
        attributes=0,
        losses=resource.manifest.observed_losses,
    )
    return DecodedText(value, accounted)


def decode_json_resource(resource: BoundedResource, /) -> DecodedJSON:
    """Decode finite duplicate-free JSON under the resource structural limits."""

    text = decode_text_resource(
        resource,
        encoding="utf-8",
        allow_utf8_bom=False,
        max_line_bytes=resource.manifest.limits.max_bytes,
    ).value
    _validate_json_nesting(text, resource.manifest.limits.max_depth)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ResourceReadError("malformed", "Resource JSON is invalid.") from error
    depth, nodes, attributes = _json_counts(value)
    accounted = account_bounded_resource(
        resource,
        depth=depth,
        nodes=nodes,
        attributes=attributes,
        losses=resource.manifest.observed_losses,
    )
    return DecodedJSON(value, accounted)


def decode_xml_resource(resource: BoundedResource, /) -> DecodedXML:
    """Decode XML only after a non-building structural and entity preflight."""

    text = decode_text_resource(
        resource,
        encoding="utf-8",
        allow_utf8_bom=True,
        max_line_bytes=resource.manifest.limits.max_bytes,
    ).value
    counts = _preflight_xml(resource.data, resource)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ResourceReadError("malformed", "Resource XML is invalid.") from error
    accounted = account_bounded_resource(
        resource,
        depth=counts[0],
        nodes=counts[1],
        attributes=counts[2],
        losses=resource.manifest.observed_losses,
    )
    return DecodedXML(root, accounted)


def _validate_json_nesting(payload: str, maximum: int, /) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in payload:
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
            if depth > maximum:
                raise ResourceReadError("limit", "Resource JSON exceeds its depth limit.")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ResourceReadError("malformed", "Resource JSON nesting is invalid.")
    if depth != 0 or in_string:
        raise ResourceReadError("malformed", "Resource JSON nesting is invalid.")


def _json_counts(value: Any, /) -> tuple[int, int, int]:
    maximum_depth = 0
    nodes = 0
    attributes = 0
    pending = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        nodes += 1
        maximum_depth = max(maximum_depth, depth)
        if isinstance(current, dict):
            attributes += len(current)
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
        elif isinstance(current, float) and not math.isfinite(current):
            raise ResourceReadError("malformed", "Resource JSON numbers must be finite.")
        elif current is not None and not isinstance(current, (bool, int, float, str)):
            raise ResourceReadError(
                "malformed", "Resource JSON contains an invalid value."
            )
    return maximum_depth, nodes, attributes


def _preflight_xml(
    data: bytes,
    resource: BoundedResource,
    /,
) -> tuple[int, int, int]:
    limits = resource.manifest.limits
    depth = 0
    maximum_depth = 0
    nodes = 0
    attributes = 0
    parser = expat.ParserCreate(namespace_separator="}")

    def start_element(_name: str, values: dict[str, str]) -> None:
        nonlocal depth, maximum_depth, nodes, attributes
        depth += 1
        maximum_depth = max(maximum_depth, depth)
        nodes += 1
        attributes += len(values)
        if maximum_depth > limits.max_depth:
            raise ResourceReadError("limit", "Resource XML exceeds its depth limit.")
        if nodes > limits.max_nodes:
            raise ResourceReadError("limit", "Resource XML exceeds its node limit.")
        if attributes > limits.max_attributes:
            raise ResourceReadError("limit", "Resource XML exceeds its attribute limit.")

    def end_element(_name: str) -> None:
        nonlocal depth
        depth -= 1

    def reject_declaration(*_arguments: object) -> None:
        raise ResourceReadError(
            "policy", "XML declarations with external semantics are disabled."
        )

    parser.StartElementHandler = start_element
    parser.EndElementHandler = end_element
    parser.StartDoctypeDeclHandler = reject_declaration
    parser.EntityDeclHandler = reject_declaration
    parser.ExternalEntityRefHandler = lambda *_arguments: 0
    try:
        parser.Parse(data, True)
    except ResourceReadError:
        raise
    except expat.ExpatError as error:
        raise ResourceReadError("malformed", "Resource XML is invalid.") from error
    if depth != 0:
        raise ResourceReadError("malformed", "Resource XML nesting is invalid.")
    return maximum_depth, nodes, attributes


def _unique_json_object(pairs: list[tuple[str, object]], /) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"Duplicate JSON member {name!r} is forbidden.")
        value[name] = item
    return value


def _reject_json_constant(value: str, /) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")


__all__ = [
    "DecodedJSON",
    "DecodedText",
    "DecodedXML",
    "decode_json_resource",
    "decode_text_resource",
    "decode_xml_resource",
]
