#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, prod

from phydrax.units import DIMENSIONLESS, DimensionSignature

from ..exterior._form_type import FormProxy, FormType, FormValueSpec
from ._ir import PDECoordinate, PDEExpression, PDEProblemIR, PDERepresentation


@dataclass(frozen=True, slots=True)
class PDEValueType:
    """Inferred representation, component count, and physical dimension."""

    representation: PDERepresentation
    components: int
    dimension: DimensionSignature
    form: FormValueSpec | None = None

    @property
    def is_scalar(self) -> bool:
        return self.representation in ("scalar", "pseudoscalar") and self.components == 1


def form_value_type(
    form: FormValueSpec, dimension: DimensionSignature, /
) -> PDEValueType:
    """Infer Cartesian parity from the declared exterior proxy, never its shape."""
    odd = (form.form_type.twist == "twisted") ^ (form.proxy in ("flux", "density"))
    representation: PDERepresentation
    match form.proxy:
        case "scalar" | "density":
            representation = "pseudoscalar" if odd else "scalar"
        case "circulation" | "flux":
            representation = "pseudovector" if odd else "vector"
        case "components":
            representation = "pseudotensor" if odd else "tensor"
    return PDEValueType(representation, prod(form.value_shape), dimension, form)


def _derived_form(
    source: FormValueSpec, target: FormType, /, *, hodge: bool = False
) -> FormValueSpec:
    proxy: FormProxy = "components"
    if source.proxy != "components":
        if target.degree == 0:
            proxy = "scalar"
        elif target.degree == target.dimension:
            proxy = "density"
        elif hodge:
            proxy = "flux" if source.proxy == "circulation" else "circulation"
        elif target.degree == 1:
            proxy = "circulation"
        elif target.degree == target.dimension - 1:
            proxy = "flux"
    return FormValueSpec(target, proxy=proxy)


def _exterior_value_type(
    node: PDEExpression,
    args: tuple[PDEValueType, ...],
    problem: PDEProblemIR,
    /,
) -> PDEValueType:
    binary = node.op in ("wedge", "interior_product", "lie_derivative")
    if len(args) != (2 if binary else 1):
        raise ValueError(f"{node.op} has an invalid operand count.")
    operand = args[-1] if node.op in ("interior_product", "lie_derivative") else args[0]
    if operand.form is None:
        raise ValueError(f"{node.op} requires an explicitly declared form.")
    source = operand.form
    form_type = source.form_type
    dimension = operand.dimension
    if node.op in ("exterior_derivative", "codifferential", "lie_derivative"):
        spaces = tuple(item for item in problem.coordinates if item.kind == "space")
        if node.coordinate is not None:
            spaces = tuple(item for item in spaces if item.name == node.coordinate)
        if not spaces or len({item.dimension for item in spaces}) != 1:
            raise ValueError("Exterior differentiation requires a known spatial scale.")
        dimension = dimension / spaces[0].dimension
    match node.op:
        case "exterior_derivative":
            spec = _derived_form(source, form_type.exterior_derivative_type())
        case "codifferential":
            spec = _derived_form(source, form_type.codifferential_type())
        case "hodge_star":
            spec = _derived_form(source, form_type.hodge_dual(), hodge=True)
        case "wedge":
            right = args[1].form
            if right is None:
                raise ValueError("wedge requires two explicitly declared forms.")
            target = form_type.wedge_type(right.form_type, product=node.product)
            spec = FormValueSpec(target, proxy="components")
            dimension = args[0].dimension * args[1].dimension
        case "interior_product" | "lie_derivative":
            vector = args[0]
            if (
                vector.representation != "vector"
                or vector.components != form_type.dimension
                or vector.form is not None
            ):
                raise ValueError(
                    "Contraction requires an explicit untwisted tangent vector."
                )
            target = (
                form_type.interior_type() if node.op == "interior_product" else form_type
            )
            spec = (
                FormValueSpec(target, proxy="components")
                if node.op == "interior_product"
                else source
            )
            dimension = dimension * vector.dimension
        case "trace":
            regions = {region.name: region for region in problem.regions}
            if node.region not in regions or regions[node.region].kind not in (
                "boundary",
                "interface",
            ):
                raise ValueError(
                    "trace requires a declared boundary or interface region."
                )
            spec = FormValueSpec(form_type.trace_type(), proxy="components")
        case _:
            raise ValueError(f"Unsupported exterior operation {node.op!r}.")
    return form_value_type(spec, dimension)


def _is_zero(node: PDEExpression, /) -> bool:
    if node.op == "negate" and len(node.args) == 1:
        return _is_zero(node.args[0])
    return node.op == "constant" and node.value == 0 and node.dimension.is_dimensionless


def _same_form(left: FormValueSpec | None, right: FormValueSpec | None, /) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left.value_spec_id == right.value_spec_id


def _same_value(left: PDEValueType, right: PDEValueType, /) -> bool:
    return (
        left.representation == right.representation
        and left.components == right.components
        and left.dimension == right.dimension
        and _same_form(left.form, right.form)
    )


def _require_finite(values: tuple[float, ...], name: str, /) -> None:
    if any(not isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must be finite.")


def _validate_expression_finite(expression: PDEExpression, /) -> None:
    if expression.value is not None and not isfinite(float(expression.value)):
        raise ValueError("PDE expression value must be finite.")
    for argument in expression.args:
        _validate_expression_finite(argument)


def _differential_value_type(
    node: PDEExpression, operand: PDEValueType, coordinate: PDECoordinate, /
) -> PDEValueType:
    factor = 2 if node.op == "laplacian" else node.order
    dimension = operand.dimension / coordinate.dimension**factor
    form = operand.form
    if form is not None:
        if coordinate.kind != "space" and node.op != "derivative":
            raise ValueError("Form vector operators require spatial coordinates.")
        expected = {"gradient": "scalar", "curl": "circulation", "divergence": "flux"}
        if node.op in expected:
            if (
                form.proxy != expected[node.op]
                or coordinate.size != form.form_type.dimension
            ):
                raise ValueError(
                    f"{node.op} requires its declared exterior proxy and dimension."
                )
            if node.op == "curl" and coordinate.size != 3:
                raise ValueError("curl requires a three-dimensional circulation proxy.")
            return form_value_type(
                _derived_form(form, form.form_type.exterior_derivative_type()), dimension
            )
        if node.op == "derivative":
            if node.axis is not None and not 0 <= node.axis < coordinate.size:
                raise ValueError("Derivative coordinate axis is out of range.")
            if coordinate.kind == "space" and coordinate.size > 1 and node.axis is None:
                raise ValueError(
                    "A form partial derivative requires an explicit spatial axis."
                )
        return PDEValueType(operand.representation, operand.components, dimension, form)
    representation: PDERepresentation
    match node.op:
        case "derivative":
            if node.axis is not None and not 0 <= node.axis < coordinate.size:
                raise ValueError("Derivative coordinate axis is out of range.")
            return PDEValueType(operand.representation, operand.components, dimension)
        case "gradient":
            if not operand.is_scalar:
                raise ValueError("gradient requires a scalar field.")
            representation = (
                "pseudovector" if operand.representation == "pseudoscalar" else "vector"
            )
            return PDEValueType(representation, coordinate.size, dimension)
        case "divergence":
            if operand.components != coordinate.size:
                raise ValueError("divergence vector size must match coordinate size.")
            representation = (
                "pseudoscalar" if operand.representation == "pseudovector" else "scalar"
            )
            return PDEValueType(representation, 1, dimension)
        case "curl":
            if coordinate.size != 3 or operand.components != 3:
                raise ValueError("curl requires a three-dimensional vector field.")
            representation = (
                "vector" if operand.representation == "pseudovector" else "pseudovector"
            )
            return PDEValueType(representation, 3, dimension)
        case "laplacian":
            return PDEValueType(operand.representation, operand.components, dimension)
        case _:
            raise ValueError(f"Unsupported differential operation {node.op!r}.")


def infer_expression_type(
    expression: PDEExpression,
    problem: PDEProblemIR,
    /,
) -> PDEValueType:
    """Infer and validate one expression recursively against a problem schema."""
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
            return PDEValueType("scalar", 1, node.dimension)
        if op == "field":
            if node.symbol not in fields or node.args:
                raise ValueError(f"Unknown or malformed field reference {node.symbol!r}.")
            field = fields[node.symbol]
            return PDEValueType(
                field.representation,
                field.components,
                field.dimension,
                field.form,
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
                parameter.dimension,
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
                coordinate.dimension,
            )
        if op in (
            "exterior_derivative",
            "codifferential",
            "hodge_star",
            "wedge",
            "interior_product",
            "lie_derivative",
            "trace",
        ):
            return _exterior_value_type(node, args, problem)
        if op in ("add", "multiply"):
            if len(args) < 2:
                raise ValueError(f"{op} requires at least two operands.")
            if op == "add":
                nonzero = tuple(
                    item
                    for item, argument in zip(args, node.args, strict=True)
                    if not (
                        _is_zero(argument)
                        and any(value.form is not None for value in args)
                    )
                )
                first = nonzero[0] if nonzero else args[0]
                if any(
                    item.representation != first.representation
                    or item.components != first.components
                    or item.dimension != first.dimension
                    or not _same_form(item.form, first.form)
                    for item in nonzero
                ):
                    raise ValueError(
                        "Addition requires matching representations and dimensions."
                    )
                return first
            non_scalar = [
                item for item in args if item.form is not None or not item.is_scalar
            ]
            if len(non_scalar) > 1:
                raise ValueError(
                    "Multiplication supports at most one non-scalar operand."
                )
            result = non_scalar[0] if non_scalar else args[0]
            dimension = DIMENSIONLESS
            for item in args:
                dimension = dimension * item.dimension
            return PDEValueType(
                result.representation, result.components, dimension, result.form
            )
        if op == "divide":
            if len(args) != 2 or not args[1].is_scalar or args[1].form is not None:
                raise ValueError("Division requires one scalar denominator.")
            return PDEValueType(
                args[0].representation,
                args[0].components,
                args[0].dimension / args[1].dimension,
                args[0].form,
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
                or not args[1].dimension.is_dimensionless
            ):
                raise ValueError(
                    "Power requires a scalar base and dimensionless constant exponent."
                )
            if args[0].form is not None:
                raise ValueError(
                    "Power requires an explicit scalar-value view, not a form."
                )
            exponent = node.args[1].value
            base_dimension = args[0].dimension
            if base_dimension.is_dimensionless:
                dimension = DIMENSIONLESS
            elif isinstance(exponent, (int, Fraction)):
                dimension = base_dimension**exponent
            else:
                raise TypeError(
                    "A dimensionful base requires an exact integer or Fraction exponent."
                )
            return PDEValueType(args[0].representation, 1, dimension)
        if op in ("sin", "cos", "exp", "log"):
            if (
                len(args) != 1
                or not args[0].is_scalar
                or not args[0].dimension.is_dimensionless
            ):
                raise ValueError(f"{op} requires one dimensionless scalar operand.")
            if args[0].form is not None:
                raise ValueError(
                    "Scalar functions require an explicit scalar-value view."
                )
            return PDEValueType(args[0].representation, 1, DIMENSIONLESS)
        if op == "sqrt":
            if len(args) != 1 or not args[0].is_scalar:
                raise ValueError("sqrt requires one scalar operand.")
            if args[0].form is not None:
                raise ValueError("sqrt requires an explicit scalar-value view.")
            return PDEValueType(
                args[0].representation,
                1,
                args[0].dimension ** Fraction(1, 2),
            )
        if op == "component":
            if len(args) != 1 or args[0].components <= 1 or node.axis is None:
                raise ValueError("component requires a non-scalar operand and axis.")
            if args[0].form is not None:
                raise ValueError(
                    "Form components require an explicit component-view conversion."
                )
            if node.axis < 0 or node.axis >= args[0].components:
                raise ValueError("Expression component axis is out of range.")
            representation = (
                "pseudoscalar"
                if args[0].representation in ("pseudovector", "pseudotensor")
                else "scalar"
            )
            return PDEValueType(representation, 1, args[0].dimension)
        if op == "dot":
            if (
                len(args) != 2
                or args[0].components <= 1
                or args[0].components != args[1].components
            ):
                raise ValueError("dot requires two equal-size vector-like operands.")
            if any(item.form is not None for item in args):
                raise ValueError("dot does not implicitly reinterpret exterior proxies.")
            odd = (args[0].representation.startswith("pseudo")) ^ (
                args[1].representation.startswith("pseudo")
            )
            return PDEValueType(
                "pseudoscalar" if odd else "scalar",
                1,
                args[0].dimension * args[1].dimension,
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
            return _differential_value_type(node, args[0], coordinates[node.coordinate])
        if op == "integral":
            if len(args) != 1 or node.region not in regions:
                raise ValueError("integral requires one operand and a known region.")
            dimension = args[0].dimension
            for coordinate_name in regions[node.region].coordinates:
                if coordinate_name not in coordinates:
                    raise ValueError(
                        f"Region {node.region!r} references unknown coordinate {coordinate_name!r}."
                    )
                coordinate = coordinates[coordinate_name]
                dimension = dimension * coordinate.dimension**coordinate.size
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
        if coordinate.bounds is not None:
            _require_finite(coordinate.bounds, "PDE coordinate bounds")
            if coordinate.bounds[1] <= coordinate.bounds[0]:
                raise ValueError("PDE coordinate upper bound must exceed lower bound.")
    for field in problem.fields:
        _require_finite(field.scale, "PDE field scale")
        if any(value <= 0.0 for value in field.scale):
            raise ValueError("PDE field scales must be positive.")
    for parameter in problem.parameters:
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
        if field.form is not None:
            spatial_size = sum(
                item.size
                for item in problem.coordinates
                if item.name in field.coordinates and item.kind == "space"
            )
            if spatial_size != field.form.form_type.dimension:
                raise ValueError(
                    "PDE form dimension must equal its spatial coordinate dimension."
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
        neutral_zero = (_is_zero(equation.lhs) and right.form is not None) or (
            _is_zero(equation.rhs) and left.form is not None
        )
        if not neutral_zero and not _same_value(left, right):
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
        neutral_zero = (_is_zero(condition.expression) and target.form is not None) or (
            _is_zero(condition.target) and value.form is not None
        )
        if not neutral_zero and not _same_value(value, target):
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
