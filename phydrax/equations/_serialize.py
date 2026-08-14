#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from typing import Any

from ._ir import (
    PDECondition,
    PDECoordinate,
    PDEEquation,
    PDEExpression,
    PDEField,
    PDEParameter,
    PDEProblemIR,
    PDERegion,
)
from ._validate import validate_pde_ir


def _flatten_associative(
    expression: PDEExpression,
    operator: str,
    /,
) -> tuple[PDEExpression, ...]:
    flattened: list[PDEExpression] = []
    for argument in expression.args:
        if argument.op == operator:
            flattened.extend(_flatten_associative(argument, operator))
        else:
            flattened.append(argument)
    return tuple(flattened)


def _canonical_expression(expression: PDEExpression, /) -> dict[str, Any]:
    args = (
        _flatten_associative(expression, expression.op)
        if expression.op in ("add", "multiply")
        else expression.args
    )
    encoded_args = [_canonical_expression(argument) for argument in args]
    if expression.op in ("add", "multiply"):
        encoded_args.sort(
            key=lambda value: json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        )
    result: dict[str, Any] = {"op": expression.op}
    if encoded_args:
        result["args"] = encoded_args
    if expression.value is not None:
        result["value"] = float(expression.value)
    if expression.symbol is not None:
        result["symbol"] = expression.symbol
    if expression.coordinate is not None:
        result["coordinate"] = expression.coordinate
    if expression.axis is not None:
        result["axis"] = int(expression.axis)
    if expression.order != 1:
        result["order"] = int(expression.order)
    if expression.region is not None:
        result["region"] = expression.region
    if expression.physical_dimension:
        result["physical_dimension"] = list(expression.physical_dimension)
    return result


def pde_ir_to_dict(problem: PDEProblemIR, /) -> dict[str, Any]:
    """Return the canonical schema dictionary for a validated PDE problem."""
    validate_pde_ir(problem)
    return {
        "coordinates": [
            {
                "name": item.name,
                "kind": item.kind,
                "size": item.size,
                "physical_dimension": list(item.physical_dimension),
                "bounds": None if item.bounds is None else list(item.bounds),
                "periodic": item.periodic,
            }
            for item in sorted(problem.coordinates, key=lambda value: value.name)
        ],
        "fields": [
            {
                "name": item.name,
                "representation": item.representation,
                "components": item.components,
                "coordinates": list(item.coordinates),
                "physical_dimension": list(item.physical_dimension),
                "scale": list(item.scale),
                "component_names": list(item.component_names),
            }
            for item in sorted(problem.fields, key=lambda value: value.name)
        ],
        "parameters": [
            {
                "name": item.name,
                "value": item.value,
                "components": item.components,
                "physical_dimension": list(item.physical_dimension),
                "scale": list(item.scale),
                "functional": item.functional,
            }
            for item in sorted(problem.parameters, key=lambda value: value.name)
        ],
        "regions": [
            {
                "name": item.name,
                "kind": item.kind,
                "coordinates": list(item.coordinates),
                "component": item.component,
            }
            for item in sorted(problem.regions, key=lambda value: value.name)
        ],
        "equations": [
            {
                "name": item.name,
                "lhs": _canonical_expression(item.lhs),
                "rhs": _canonical_expression(item.rhs),
            }
            for item in sorted(problem.equations, key=lambda value: value.name)
        ],
        "conditions": [
            {
                "name": item.name,
                "kind": item.kind,
                "expression": _canonical_expression(item.expression),
                "target": _canonical_expression(item.target),
                "region": item.region,
                "coordinate": item.coordinate,
            }
            for item in sorted(problem.conditions, key=lambda value: value.name)
        ],
        "nondimensionalization": [
            [name, float(value)] for name, value in sorted(problem.nondimensionalization)
        ],
        "metadata": [list(item) for item in sorted(problem.metadata)],
    }


def pde_ir_to_json(problem: PDEProblemIR, /, *, indent: int | None = None) -> str:
    """Serialize a problem with deterministic ordering and numeric spelling."""
    return json.dumps(
        pde_ir_to_dict(problem),
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        ensure_ascii=True,
        allow_nan=False,
    )


def pde_ir_hash(problem: PDEProblemIR, /) -> str:
    return hashlib.sha256(pde_ir_to_json(problem).encode("utf-8")).hexdigest()


def _expression_from_dict(value: dict[str, Any], /) -> PDEExpression:
    allowed = {
        "op",
        "args",
        "value",
        "symbol",
        "coordinate",
        "axis",
        "order",
        "region",
        "physical_dimension",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown PDE expression fields {sorted(unknown)}.")
    if "op" not in value:
        raise ValueError("Serialized PDE expression is missing op.")
    return PDEExpression(
        value["op"],
        tuple(_expression_from_dict(argument) for argument in value.get("args", ())),
        value=None if value.get("value") is None else float(value["value"]),
        symbol=value.get("symbol"),
        coordinate=value.get("coordinate"),
        axis=value.get("axis"),
        order=int(value.get("order", 1)),
        region=value.get("region"),
        physical_dimension=tuple(value.get("physical_dimension", ())),
    )


def pde_ir_from_dict(value: dict[str, Any], /) -> PDEProblemIR:
    """Load, type, and fully validate a canonical PDE problem dictionary."""
    allowed = {
        "coordinates",
        "fields",
        "parameters",
        "regions",
        "equations",
        "conditions",
        "nondimensionalization",
        "metadata",
    }
    missing = allowed - set(value)
    if missing:
        raise ValueError(f"Serialized PDE IR is missing {sorted(missing)}.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown PDE IR fields {sorted(unknown)}.")
    coordinates = tuple(
        PDECoordinate(
            item["name"],
            item["kind"],
            size=int(item.get("size", 1)),
            physical_dimension=tuple(item.get("physical_dimension", ())),
            bounds=(
                None
                if item.get("bounds") is None
                else (float(item["bounds"][0]), float(item["bounds"][1]))
            ),
            periodic=bool(item.get("periodic", False)),
        )
        for item in value["coordinates"]
    )
    fields = tuple(
        PDEField(
            item["name"],
            representation=item.get("representation", "scalar"),
            components=int(item.get("components", 1)),
            coordinates=tuple(item.get("coordinates", ())),
            physical_dimension=tuple(item.get("physical_dimension", ())),
            scale=tuple(item.get("scale", (1.0,))),
            component_names=tuple(item.get("component_names", ())),
        )
        for item in value["fields"]
    )
    parameters = tuple(
        PDEParameter(
            item["name"],
            value=(
                None
                if item.get("value") is None
                else (
                    tuple(item["value"])
                    if isinstance(item["value"], list)
                    else float(item["value"])
                )
            ),
            components=int(item.get("components", 1)),
            physical_dimension=tuple(item.get("physical_dimension", ())),
            scale=tuple(item.get("scale", (1.0,))),
            functional=bool(item.get("functional", False)),
        )
        for item in value.get("parameters", ())
    )
    regions = tuple(
        PDERegion(
            item["name"],
            item["kind"],
            tuple(item["coordinates"]),
            component=item.get("component"),
        )
        for item in value.get("regions", ())
    )
    equations = tuple(
        PDEEquation(
            item["name"],
            _expression_from_dict(item["lhs"]),
            _expression_from_dict(item["rhs"]),
        )
        for item in value.get("equations", ())
    )
    conditions = tuple(
        PDECondition(
            item["name"],
            item["kind"],
            _expression_from_dict(item["expression"]),
            _expression_from_dict(item["target"]),
            region=item["region"],
            coordinate=item.get("coordinate"),
        )
        for item in value.get("conditions", ())
    )
    problem = PDEProblemIR(
        coordinates=coordinates,
        fields=fields,
        parameters=parameters,
        equations=equations,
        conditions=conditions,
        regions=regions,
        nondimensionalization=tuple(
            (str(name), float(scale))
            for name, scale in value.get("nondimensionalization", ())
        ),
        metadata=tuple(
            (str(name), str(content)) for name, content in value.get("metadata", ())
        ),
    )
    return validate_pde_ir(problem)


def pde_ir_from_json(payload: str, /) -> PDEProblemIR:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Serialized PDE IR root must be a JSON object.")
    return pde_ir_from_dict(value)


__all__ = [
    "pde_ir_from_dict",
    "pde_ir_from_json",
    "pde_ir_hash",
    "pde_ir_to_dict",
    "pde_ir_to_json",
]
