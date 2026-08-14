#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ._ir import PDEExpression, PDEProblemIR


@dataclass(frozen=True, slots=True)
class PDEValueType:
    """Inferred representation, component count, and physical dimension."""

    representation: str
    components: int
    physical_dimension: tuple[float, ...]

    @property
    def is_scalar(self) -> bool:
        return self.representation in ("scalar", "pseudoscalar") and self.components == 1


def _dimension_rank(problem: PDEProblemIR, /) -> int:
    dimensions = [coordinate.physical_dimension for coordinate in problem.coordinates]
    dimensions.extend(field.physical_dimension for field in problem.fields)
    dimensions.extend(parameter.physical_dimension for parameter in problem.parameters)
    return max((len(value) for value in dimensions), default=0)


def _pad_dimension(values: tuple[float, ...], rank: int, /) -> tuple[float, ...]:
    if len(values) not in (0, rank):
        raise ValueError(
            f"Physical dimensions must be empty or have the problem rank {rank}."
        )
    return (0.0,) * rank if not values else tuple(float(value) for value in values)


def _combine_dimensions(
    left: tuple[float, ...],
    right: tuple[float, ...],
    sign: float,
    /,
) -> tuple[float, ...]:
    return tuple(a + sign * b for a, b in zip(left, right, strict=True))


def _same_dimension(left: tuple[float, ...], right: tuple[float, ...], /) -> bool:
    return all(abs(a - b) <= 1e-12 for a, b in zip(left, right, strict=True))


def _require_finite(values: tuple[float, ...], name: str, /) -> None:
    if any(not isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must be finite.")


def _validate_expression_finite(expression: PDEExpression, /) -> None:
    _require_finite(expression.physical_dimension, "PDE expression physical dimensions")
    if expression.value is not None and not isfinite(float(expression.value)):
        raise ValueError("PDE expression value must be finite.")
    for argument in expression.args:
        _validate_expression_finite(argument)


def infer_expression_type(
    expression: PDEExpression,
    problem: PDEProblemIR,
    /,
) -> PDEValueType:
    """Infer and validate one expression recursively against a problem schema."""
    rank = _dimension_rank(problem)
    fields = {field.name: field for field in problem.fields}
    parameters = {parameter.name: parameter for parameter in problem.parameters}
    coordinates = {coordinate.name: coordinate for coordinate in problem.coordinates}
    regions = {region.name: region for region in problem.regions}

    def infer(node: PDEExpression) -> PDEValueType:
        op = node.op
        args = tuple(infer(argument) for argument in node.args)
        if op == "constant":
            if node.value is None or node.args or node.symbol is not None:
                raise ValueError("Constant expressions require only a numeric value.")
            return PDEValueType(
                "scalar", 1, _pad_dimension(node.physical_dimension, rank)
            )
        if op == "field":
            if node.symbol not in fields or node.args:
                raise ValueError(f"Unknown or malformed field reference {node.symbol!r}.")
            field = fields[node.symbol]
            return PDEValueType(
                field.representation,
                field.components,
                _pad_dimension(field.physical_dimension, rank),
            )
        if op == "parameter":
            if node.symbol not in parameters or node.args:
                raise ValueError(
                    f"Unknown or malformed parameter reference {node.symbol!r}."
                )
            parameter = parameters[node.symbol]
            representation = "scalar" if parameter.components == 1 else "vector"
            return PDEValueType(
                representation,
                parameter.components,
                _pad_dimension(parameter.physical_dimension, rank),
            )
        if op == "coordinate":
            if node.symbol not in coordinates or node.args:
                raise ValueError(
                    f"Unknown or malformed coordinate reference {node.symbol!r}."
                )
            coordinate = coordinates[node.symbol]
            representation = "scalar" if coordinate.size == 1 else "vector"
            return PDEValueType(
                representation,
                coordinate.size,
                _pad_dimension(coordinate.physical_dimension, rank),
            )
        if op in ("add", "multiply"):
            if len(args) < 2:
                raise ValueError(f"{op} requires at least two operands.")
            if op == "add":
                first = args[0]
                if any(
                    item.representation != first.representation
                    or item.components != first.components
                    or not _same_dimension(
                        item.physical_dimension, first.physical_dimension
                    )
                    for item in args[1:]
                ):
                    raise ValueError(
                        "Addition requires matching representations and dimensions."
                    )
                return first
            non_scalar = [item for item in args if not item.is_scalar]
            if len(non_scalar) > 1:
                raise ValueError(
                    "Multiplication supports at most one non-scalar operand."
                )
            result = non_scalar[0] if non_scalar else args[0]
            dimension = (0.0,) * rank
            for item in args:
                dimension = _combine_dimensions(dimension, item.physical_dimension, 1.0)
            return PDEValueType(result.representation, result.components, dimension)
        if op == "divide":
            if len(args) != 2 or not args[1].is_scalar:
                raise ValueError("Division requires one scalar denominator.")
            return PDEValueType(
                args[0].representation,
                args[0].components,
                _combine_dimensions(
                    args[0].physical_dimension,
                    args[1].physical_dimension,
                    -1.0,
                ),
            )
        if op == "negate":
            if len(args) != 1:
                raise ValueError("Negation requires one operand.")
            return args[0]
        if op == "power":
            if (
                len(args) != 2
                or not args[0].is_scalar
                or node.args[1].op != "constant"
                or node.args[1].value is None
                or not _same_dimension(args[1].physical_dimension, (0.0,) * rank)
            ):
                raise ValueError(
                    "Power requires a scalar base and dimensionless constant exponent."
                )
            exponent = float(node.args[1].value)
            return PDEValueType(
                args[0].representation,
                1,
                tuple(exponent * value for value in args[0].physical_dimension),
            )
        if op in ("sin", "cos", "exp", "log"):
            if (
                len(args) != 1
                or not args[0].is_scalar
                or not _same_dimension(args[0].physical_dimension, (0.0,) * rank)
            ):
                raise ValueError(f"{op} requires one dimensionless scalar operand.")
            return PDEValueType(args[0].representation, 1, (0.0,) * rank)
        if op == "sqrt":
            if len(args) != 1 or not args[0].is_scalar:
                raise ValueError("sqrt requires one scalar operand.")
            return PDEValueType(
                args[0].representation,
                1,
                tuple(0.5 * value for value in args[0].physical_dimension),
            )
        if op == "component":
            if len(args) != 1 or args[0].components <= 1 or node.axis is None:
                raise ValueError("component requires a non-scalar operand and axis.")
            if node.axis < 0 or node.axis >= args[0].components:
                raise ValueError("Expression component axis is out of range.")
            representation = (
                "pseudoscalar"
                if args[0].representation in ("pseudovector", "pseudotensor")
                else "scalar"
            )
            return PDEValueType(representation, 1, args[0].physical_dimension)
        if op == "dot":
            if (
                len(args) != 2
                or args[0].components <= 1
                or args[0].components != args[1].components
            ):
                raise ValueError("dot requires two equal-size vector-like operands.")
            odd = (args[0].representation.startswith("pseudo")) ^ (
                args[1].representation.startswith("pseudo")
            )
            return PDEValueType(
                "pseudoscalar" if odd else "scalar",
                1,
                _combine_dimensions(
                    args[0].physical_dimension, args[1].physical_dimension, 1.0
                ),
            )
        if op in (
            "derivative",
            "gradient",
            "divergence",
            "curl",
            "laplacian",
        ):
            if len(args) != 1 or node.coordinate not in coordinates:
                raise ValueError(f"{op} requires one operand and a known coordinate.")
            coordinate = coordinates[node.coordinate]
            coordinate_dimension = _pad_dimension(coordinate.physical_dimension, rank)
            factor = 2.0 if op == "laplacian" else float(node.order)
            dimension = _combine_dimensions(
                args[0].physical_dimension,
                tuple(factor * value for value in coordinate_dimension),
                -1.0,
            )
            if op == "derivative":
                if node.axis is not None and not 0 <= node.axis < coordinate.size:
                    raise ValueError("Derivative coordinate axis is out of range.")
                return PDEValueType(args[0].representation, args[0].components, dimension)
            if op == "gradient":
                if not args[0].is_scalar:
                    raise ValueError("gradient requires a scalar field.")
                representation = (
                    "pseudovector"
                    if args[0].representation == "pseudoscalar"
                    else "vector"
                )
                return PDEValueType(representation, coordinate.size, dimension)
            if op == "divergence":
                if args[0].components != coordinate.size:
                    raise ValueError("divergence vector size must match coordinate size.")
                representation = (
                    "pseudoscalar"
                    if args[0].representation == "pseudovector"
                    else "scalar"
                )
                return PDEValueType(representation, 1, dimension)
            if op == "curl":
                if coordinate.size != 3 or args[0].components != 3:
                    raise ValueError("curl requires a three-dimensional vector field.")
                representation = (
                    "vector"
                    if args[0].representation == "pseudovector"
                    else "pseudovector"
                )
                return PDEValueType(representation, 3, dimension)
            return PDEValueType(args[0].representation, args[0].components, dimension)
        if op == "integral":
            if len(args) != 1 or node.region not in regions:
                raise ValueError("integral requires one operand and a known region.")
            dimension = args[0].physical_dimension
            for coordinate_name in regions[node.region].coordinates:
                if coordinate_name not in coordinates:
                    raise ValueError(
                        f"Region {node.region!r} references unknown coordinate "
                        f"{coordinate_name!r}."
                    )
                coordinate_dimension = _pad_dimension(
                    coordinates[coordinate_name].physical_dimension, rank
                )
                dimension = _combine_dimensions(
                    dimension,
                    tuple(
                        coordinates[coordinate_name].size * value
                        for value in coordinate_dimension
                    ),
                    1.0,
                )
            return PDEValueType(args[0].representation, args[0].components, dimension)
        raise ValueError(f"Unsupported PDE expression operation {op!r}.")

    return infer(expression)


def validate_pde_ir(problem: PDEProblemIR, /) -> PDEProblemIR:
    """Validate references, dimensions, shapes, regions, and schema invariants."""
    collections = (
        ("coordinates", tuple(item.name for item in problem.coordinates)),
        ("fields", tuple(item.name for item in problem.fields)),
        ("parameters", tuple(item.name for item in problem.parameters)),
        ("equations", tuple(item.name for item in problem.equations)),
        ("conditions", tuple(item.name for item in problem.conditions)),
        ("regions", tuple(item.name for item in problem.regions)),
    )
    for label, names in collections:
        if len(set(names)) != len(names):
            raise ValueError(f"PDE IR {label} must have unique names.")
    symbols = tuple(item.name for item in problem.fields) + tuple(
        item.name for item in problem.parameters
    )
    if len(set(symbols)) != len(symbols):
        raise ValueError("PDE field and parameter symbols must not collide.")
    for coordinate in problem.coordinates:
        _require_finite(
            coordinate.physical_dimension,
            "PDE coordinate physical dimensions",
        )
        if coordinate.bounds is not None:
            _require_finite(coordinate.bounds, "PDE coordinate bounds")
            if coordinate.bounds[1] <= coordinate.bounds[0]:
                raise ValueError("PDE coordinate upper bound must exceed lower bound.")
    for field in problem.fields:
        _require_finite(field.physical_dimension, "PDE field physical dimensions")
        _require_finite(field.scale, "PDE field scale")
        if any(value <= 0.0 for value in field.scale):
            raise ValueError("PDE field scales must be positive.")
    for parameter in problem.parameters:
        _require_finite(
            parameter.physical_dimension,
            "PDE parameter physical dimensions",
        )
        _require_finite(parameter.scale, "PDE parameter scale")
        if any(value <= 0.0 for value in parameter.scale):
            raise ValueError("PDE parameter scales must be positive.")
        if parameter.value is not None:
            parameter_values = (
                parameter.value
                if isinstance(parameter.value, tuple)
                else (parameter.value,)
            )
            _require_finite(parameter_values, "PDE parameter value")
    coordinate_names = {item.name for item in problem.coordinates}
    for field in problem.fields:
        unknown = set(field.coordinates) - coordinate_names
        if unknown:
            raise ValueError(
                f"PDE field {field.name!r} references unknown coordinates {sorted(unknown)}."
            )
    for region in problem.regions:
        unknown = set(region.coordinates) - coordinate_names
        if unknown:
            raise ValueError(
                f"PDE region {region.name!r} references unknown coordinates {sorted(unknown)}."
            )
    region_by_name = {region.name: region for region in problem.regions}
    for equation in problem.equations:
        _validate_expression_finite(equation.lhs)
        _validate_expression_finite(equation.rhs)
        left = infer_expression_type(equation.lhs, problem)
        right = infer_expression_type(equation.rhs, problem)
        if (
            left.representation != right.representation
            or left.components != right.components
            or not _same_dimension(left.physical_dimension, right.physical_dimension)
        ):
            raise ValueError(
                f"PDE equation {equation.name!r} equates incompatible values."
            )
    for condition in problem.conditions:
        _validate_expression_finite(condition.expression)
        _validate_expression_finite(condition.target)
        if condition.region not in region_by_name:
            raise ValueError(
                f"PDE condition {condition.name!r} references unknown region."
            )
        region = region_by_name[condition.region]
        if condition.kind != region.kind and not (
            condition.kind == "boundary" and region.kind == "interior"
        ):
            raise ValueError(
                f"PDE condition {condition.name!r} kind does not match its region."
            )
        if (
            condition.coordinate is not None
            and condition.coordinate not in coordinate_names
        ):
            raise ValueError(
                f"PDE condition {condition.name!r} references unknown coordinate."
            )
        value = infer_expression_type(condition.expression, problem)
        target = infer_expression_type(condition.target, problem)
        if (
            value.representation != target.representation
            or value.components != target.components
            or not _same_dimension(value.physical_dimension, target.physical_dimension)
        ):
            raise ValueError(
                f"PDE condition {condition.name!r} has incompatible target units."
            )
    if len({name for name, _ in problem.nondimensionalization}) != len(
        problem.nondimensionalization
    ):
        raise ValueError("Nondimensionalization keys must be unique.")
    nondimensionalization_values = tuple(
        float(value) for _, value in problem.nondimensionalization
    )
    _require_finite(
        nondimensionalization_values,
        "PDE nondimensionalization scales",
    )
    if any(value <= 0.0 for value in nondimensionalization_values):
        raise ValueError("Nondimensionalization scales must be positive.")
    if len({name for name, _ in problem.metadata}) != len(problem.metadata):
        raise ValueError("PDE metadata keys must be unique.")
    return problem


__all__ = ["PDEValueType", "infer_expression_type", "validate_pde_ir"]
