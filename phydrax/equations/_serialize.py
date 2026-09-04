#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from numbers import Integral
from typing import Any

from phydrax._fingerprint import canonical_fingerprint
from phydrax.units import DIMENSIONLESS, DimensionSignature

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


def _dimension_from_dict(value: Any, /) -> DimensionSignature:
    if not isinstance(value, Mapping):
        raise TypeError("Serialized PDE dimensions must be canonical mappings.")
    return DimensionSignature.from_dict(value)


def _literal_to_dict(
    value: int | float | Fraction,
    /,
) -> int | float | dict[str, int]:
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    return value


def _literal_from_dict(value: Any, /) -> int | float | Fraction:
    if isinstance(value, Mapping):
        if set(value) != {"numerator", "denominator"}:
            raise ValueError(
                "Serialized exact PDE literals require numerator and denominator."
            )
        numerator = value["numerator"]
        denominator = value["denominator"]
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, Integral)
            or isinstance(denominator, bool)
            or not isinstance(denominator, Integral)
        ):
            raise TypeError("Serialized exact PDE literal terms must be integers.")
        return Fraction(int(numerator), int(denominator))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Serialized PDE literals must be integers, fractions, or floats.")
    return value


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
        result["value"] = _literal_to_dict(expression.value)
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
    if not expression.dimension.is_dimensionless:
        result["dimension"] = expression.dimension.to_dict()
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
                "dimension": item.dimension.to_dict(),
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
                "dimension": item.dimension.to_dict(),
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
                "dimension": item.dimension.to_dict(),
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
    return canonical_fingerprint(pde_ir_to_dict(problem))


def _expression_from_dict(value: Mapping[str, Any], /) -> PDEExpression:
    if not isinstance(value, Mapping):
        raise TypeError("Serialized PDE expressions must be mappings.")
    allowed = {
        "op",
        "args",
        "value",
        "symbol",
        "coordinate",
        "axis",
        "order",
        "region",
        "dimension",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown PDE expression fields {sorted(unknown)}.")
    if "op" not in value:
        raise ValueError("Serialized PDE expression is missing op.")
    return PDEExpression(
        value["op"],
        tuple(_expression_from_dict(argument) for argument in value.get("args", ())),
        value=(
            None if value.get("value") is None else _literal_from_dict(value["value"])
        ),
        symbol=value.get("symbol"),
        coordinate=value.get("coordinate"),
        axis=value.get("axis"),
        order=int(value.get("order", 1)),
        region=value.get("region"),
        dimension=(
            DIMENSIONLESS
            if "dimension" not in value
            else _dimension_from_dict(value["dimension"])
        ),
    )


def _records(
    values: Any,
    fields: set[str],
    label: str,
    /,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"Serialized PDE {label} records must be a sequence.")
    records: list[Mapping[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError(f"Serialized PDE {label} records must be mappings.")
        missing = fields - set(value)
        unknown = set(value) - fields
        if missing or unknown:
            raise ValueError(
                f"Serialized PDE {label} records must use canonical fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        records.append(value)
    return tuple(records)


def pde_ir_from_dict(value: Mapping[str, Any], /) -> PDEProblemIR:
    """Load, type, and fully validate a canonical PDE problem dictionary."""
    if not isinstance(value, Mapping):
        raise TypeError("Serialized PDE IR must be a mapping.")
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
    coordinate_records = _records(
        value["coordinates"],
        {"name", "kind", "size", "dimension", "bounds", "periodic"},
        "coordinate",
    )
    field_records = _records(
        value["fields"],
        {
            "name",
            "representation",
            "components",
            "coordinates",
            "dimension",
            "scale",
            "component_names",
        },
        "field",
    )
    parameter_records = _records(
        value["parameters"],
        {"name", "value", "components", "dimension", "scale", "functional"},
        "parameter",
    )
    region_records = _records(
        value["regions"],
        {"name", "kind", "coordinates", "component"},
        "region",
    )
    equation_records = _records(
        value["equations"],
        {"name", "lhs", "rhs"},
        "equation",
    )
    condition_records = _records(
        value["conditions"],
        {"name", "kind", "expression", "target", "region", "coordinate"},
        "condition",
    )
    coordinates = tuple(
        PDECoordinate(
            item["name"],
            item["kind"],
            size=int(item.get("size", 1)),
            dimension=_dimension_from_dict(item["dimension"]),
            bounds=(
                None
                if item.get("bounds") is None
                else (float(item["bounds"][0]), float(item["bounds"][1]))
            ),
            periodic=bool(item.get("periodic", False)),
        )
        for item in coordinate_records
    )
    fields = tuple(
        PDEField(
            item["name"],
            representation=item.get("representation", "scalar"),
            components=int(item.get("components", 1)),
            coordinates=tuple(item.get("coordinates", ())),
            dimension=_dimension_from_dict(item["dimension"]),
            scale=tuple(item.get("scale", (1.0,))),
            component_names=tuple(item.get("component_names", ())),
        )
        for item in field_records
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
            dimension=_dimension_from_dict(item["dimension"]),
            scale=tuple(item.get("scale", (1.0,))),
            functional=bool(item.get("functional", False)),
        )
        for item in parameter_records
    )
    regions = tuple(
        PDERegion(
            item["name"],
            item["kind"],
            tuple(item["coordinates"]),
            component=item.get("component"),
        )
        for item in region_records
    )
    equations = tuple(
        PDEEquation(
            item["name"],
            _expression_from_dict(item["lhs"]),
            _expression_from_dict(item["rhs"]),
        )
        for item in equation_records
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
        for item in condition_records
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
