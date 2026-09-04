#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal, TypeAlias

from phydrax.units import DIMENSIONLESS, DimensionSignature


PDERepresentation = Literal[
    "scalar",
    "pseudoscalar",
    "vector",
    "pseudovector",
    "tensor",
    "pseudotensor",
]
PDECoordinateKind = Literal["space", "time"]
PDEConditionKind = Literal["initial", "boundary", "interface"]
PDERegionKind = Literal["interior", "boundary", "interface", "initial"]
PDELiteral: TypeAlias = int | float | Fraction
PDEExpressionOp = Literal[
    "constant",
    "coordinate",
    "field",
    "parameter",
    "add",
    "multiply",
    "divide",
    "negate",
    "power",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "component",
    "dot",
    "derivative",
    "gradient",
    "divergence",
    "curl",
    "laplacian",
    "integral",
]


_VALID_OPS = {
    "constant",
    "coordinate",
    "field",
    "parameter",
    "add",
    "multiply",
    "divide",
    "negate",
    "power",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "component",
    "dot",
    "derivative",
    "gradient",
    "divergence",
    "curl",
    "laplacian",
    "integral",
}


def _finite_float(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _finite_values(
    values: tuple[float, ...] | list[float],
    name: str,
    /,
) -> tuple[float, ...]:
    return tuple(_finite_float(value, name) for value in values)


def _literal(value: PDELiteral, name: str, /) -> PDELiteral:
    if isinstance(value, bool) or not isinstance(value, (Real, Fraction)):
        raise TypeError(f"{name} must be an integer, fraction, or float.")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return int(value)
    return _finite_float(value, name)


def _exact_integer_literal(value: PDELiteral | None, /) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else None
    return int(value) if value.is_integer() else None


@dataclass(frozen=True, slots=True)
class PDECoordinate:
    """One labeled spatial or temporal coordinate group."""

    name: str
    kind: PDECoordinateKind
    size: int = 1
    dimension: DimensionSignature = DIMENSIONLESS
    bounds: tuple[float, float] | None = None
    periodic: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.kind not in ("space", "time"):
            raise ValueError("PDE coordinates require a name and valid kind.")
        if int(self.size) <= 0:
            raise ValueError("PDE coordinate size must be positive.")
        object.__setattr__(self, "size", int(self.size))
        if not isinstance(self.dimension, DimensionSignature):
            raise TypeError("PDE coordinate dimension must be a DimensionSignature.")
        if self.bounds is not None:
            bounds = _finite_values(
                [self.bounds[0], self.bounds[1]],
                "PDE coordinate bounds",
            )
            if bounds[1] <= bounds[0]:
                raise ValueError("PDE coordinate upper bound must exceed lower bound.")
            object.__setattr__(self, "bounds", bounds)


@dataclass(frozen=True, slots=True)
class PDEField:
    """Named unknown or observed field with physical representation metadata."""

    name: str
    representation: PDERepresentation = "scalar"
    components: int = 1
    coordinates: tuple[str, ...] = ()
    dimension: DimensionSignature = DIMENSIONLESS
    scale: tuple[float, ...] = (1.0,)
    component_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PDE field name must not be empty.")
        if self.representation not in (
            "scalar",
            "pseudoscalar",
            "vector",
            "pseudovector",
            "tensor",
            "pseudotensor",
        ):
            raise ValueError("Unknown PDE field representation.")
        if int(self.components) <= 0:
            raise ValueError("PDE field components must be positive.")
        object.__setattr__(self, "components", int(self.components))
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        if not isinstance(self.dimension, DimensionSignature):
            raise TypeError("PDE field dimension must be a DimensionSignature.")
        scales = _finite_values(list(self.scale), "PDE field scale")
        if len(scales) not in (1, self.components) or any(
            value <= 0.0 for value in scales
        ):
            raise ValueError(
                "PDE field scale must be positive and scalar or per-component."
            )
        object.__setattr__(self, "scale", scales)
        names = tuple(self.component_names)
        if names and (len(names) != self.components or len(set(names)) != len(names)):
            raise ValueError("PDE field component names must uniquely cover components.")
        object.__setattr__(self, "component_names", names)


@dataclass(frozen=True, slots=True)
class PDEParameter:
    """Named scalar, vector, or field-valued coefficient in a PDE problem."""

    name: str
    value: float | tuple[float, ...] | None = None
    components: int = 1
    dimension: DimensionSignature = DIMENSIONLESS
    scale: tuple[float, ...] = (1.0,)
    functional: bool = False

    def __post_init__(self) -> None:
        if not self.name or int(self.components) <= 0:
            raise ValueError(
                "PDE parameters require a name and positive component count."
            )
        object.__setattr__(self, "components", int(self.components))
        if not isinstance(self.dimension, DimensionSignature):
            raise TypeError("PDE parameter dimension must be a DimensionSignature.")
        scales = _finite_values(list(self.scale), "PDE parameter scale")
        if len(scales) not in (1, self.components) or any(item <= 0.0 for item in scales):
            raise ValueError(
                "PDE parameter scale must be positive and scalar or per-component."
            )
        object.__setattr__(self, "scale", scales)
        if self.value is not None:
            value = (
                _finite_float(self.value, "PDE parameter value")
                if isinstance(self.value, (int, float))
                else _finite_values(list(self.value), "PDE parameter value")
            )
            if isinstance(value, tuple) and len(value) != self.components:
                raise ValueError("PDE parameter value must match its component count.")
            object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class PDEExpression:
    """One typed node in a validated, string-free PDE expression DAG."""

    op: PDEExpressionOp
    args: tuple["PDEExpression", ...] = ()
    value: PDELiteral | None = None
    symbol: str | None = None
    coordinate: str | None = None
    axis: int | None = None
    order: int = 1
    region: str | None = None
    dimension: DimensionSignature = DIMENSIONLESS

    def __post_init__(self) -> None:
        if self.op not in _VALID_OPS:
            raise ValueError(f"Unknown PDE expression operation {self.op!r}.")
        object.__setattr__(self, "args", tuple(self.args))
        if not isinstance(self.dimension, DimensionSignature):
            raise TypeError("PDE expression dimension must be a DimensionSignature.")
        if self.op != "constant" and not self.dimension.is_dimensionless:
            raise ValueError("Only PDE constant expressions may declare a dimension.")
        if self.value is not None:
            object.__setattr__(
                self,
                "value",
                _literal(self.value, "PDE expression value"),
            )
        if int(self.order) <= 0:
            raise ValueError("PDE derivative order must be positive.")
        object.__setattr__(self, "order", int(self.order))
        if self.axis is not None:
            object.__setattr__(self, "axis", int(self.axis))

    @classmethod
    def constant(
        cls,
        value: PDELiteral,
        /,
        *,
        dimension: DimensionSignature = DIMENSIONLESS,
    ) -> "PDEExpression":
        return cls(
            "constant",
            value=value,
            dimension=dimension,
        )

    @classmethod
    def field(cls, name: str, /) -> "PDEExpression":
        return cls("field", symbol=str(name))

    @classmethod
    def parameter(cls, name: str, /) -> "PDEExpression":
        return cls("parameter", symbol=str(name))

    @classmethod
    def coordinate_value(cls, name: str, /) -> "PDEExpression":
        return cls("coordinate", symbol=str(name))

    def derivative(
        self,
        coordinate: str,
        /,
        *,
        axis: int | None = None,
        order: int = 1,
    ) -> "PDEExpression":
        return PDEExpression(
            "derivative",
            (self,),
            coordinate=str(coordinate),
            axis=axis,
            order=order,
        )

    def gradient(self, coordinate: str, /) -> "PDEExpression":
        return PDEExpression("gradient", (self,), coordinate=str(coordinate))

    def divergence(self, coordinate: str, /) -> "PDEExpression":
        return PDEExpression("divergence", (self,), coordinate=str(coordinate))

    def curl(self, coordinate: str, /) -> "PDEExpression":
        return PDEExpression("curl", (self,), coordinate=str(coordinate))

    def laplacian(self, coordinate: str, /) -> "PDEExpression":
        return PDEExpression("laplacian", (self,), coordinate=str(coordinate))

    def integrate(self, region: str, /) -> "PDEExpression":
        return PDEExpression("integral", (self,), region=str(region))

    def component(self, axis: int, /) -> "PDEExpression":
        return PDEExpression("component", (self,), axis=int(axis))

    def dot(self, other: Any, /) -> "PDEExpression":
        return PDEExpression("dot", (self, as_expression(other)))

    def sin(self) -> "PDEExpression":
        return PDEExpression("sin", (self,))

    def cos(self) -> "PDEExpression":
        return PDEExpression("cos", (self,))

    def exp(self) -> "PDEExpression":
        return PDEExpression("exp", (self,))

    def log(self) -> "PDEExpression":
        return PDEExpression("log", (self,))

    def sqrt(self) -> "PDEExpression":
        return PDEExpression("sqrt", (self,))

    def __add__(self, other: Any) -> "PDEExpression":
        return PDEExpression("add", (self, as_expression(other)))

    def __radd__(self, other: Any) -> "PDEExpression":
        return PDEExpression("add", (as_expression(other), self))

    def __sub__(self, other: Any) -> "PDEExpression":
        return PDEExpression("add", (self, -as_expression(other)))

    def __rsub__(self, other: Any) -> "PDEExpression":
        return PDEExpression("add", (as_expression(other), -self))

    def __mul__(self, other: Any) -> "PDEExpression":
        return PDEExpression("multiply", (self, as_expression(other)))

    def __rmul__(self, other: Any) -> "PDEExpression":
        return PDEExpression("multiply", (as_expression(other), self))

    def __truediv__(self, other: Any) -> "PDEExpression":
        return PDEExpression("divide", (self, as_expression(other)))

    def __rtruediv__(self, other: Any) -> "PDEExpression":
        return PDEExpression("divide", (as_expression(other), self))

    def __pow__(self, power: Any) -> "PDEExpression":
        return PDEExpression("power", (self, as_expression(power)))

    def __neg__(self) -> "PDEExpression":
        return PDEExpression("negate", (self,))


def as_expression(value: Any, /) -> PDEExpression:
    if isinstance(value, PDEExpression):
        return value
    if isinstance(value, (Real, Fraction)) and not isinstance(value, bool):
        return PDEExpression.constant(value)
    raise TypeError(
        "PDE expressions only accept integer, fraction, or float constants "
        "and PDEExpression nodes."
    )


@dataclass(frozen=True, slots=True)
class PDEEquation:
    """Named equality between two PDE expression DAGs."""

    name: str
    lhs: PDEExpression
    rhs: PDEExpression = field(default_factory=lambda: PDEExpression.constant(0.0))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PDE equation name must not be empty.")

    @property
    def residual(self) -> PDEExpression:
        return self.lhs - self.rhs


@dataclass(frozen=True, slots=True)
class PDERegion:
    """Named integration or restriction region with explicit coordinates."""

    name: str
    kind: PDERegionKind
    coordinates: tuple[str, ...]
    component: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.kind not in (
            "interior",
            "boundary",
            "interface",
            "initial",
        ):
            raise ValueError("PDE regions require a name and valid kind.")
        coordinates = tuple(str(value) for value in self.coordinates)
        if not coordinates:
            raise ValueError("PDE regions require explicit coordinate labels.")
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True, slots=True)
class PDECondition:
    """Initial, boundary, or interface restriction on an expression."""

    name: str
    kind: PDEConditionKind
    expression: PDEExpression
    target: PDEExpression = field(default_factory=lambda: PDEExpression.constant(0.0))
    region: str = ""
    coordinate: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.kind not in ("initial", "boundary", "interface"):
            raise ValueError("PDE conditions require a name and valid kind.")
        if not self.region:
            raise ValueError("PDE conditions require an explicit region identifier.")

    @property
    def residual(self) -> PDEExpression:
        return self.expression - self.target


@dataclass(frozen=True, slots=True)
class PDEProblemIR:
    """Serializable PDE problem intermediate representation."""

    coordinates: tuple[PDECoordinate, ...]
    fields: tuple[PDEField, ...]
    parameters: tuple[PDEParameter, ...] = ()
    equations: tuple[PDEEquation, ...] = ()
    conditions: tuple[PDECondition, ...] = ()
    regions: tuple[PDERegion, ...] = ()
    nondimensionalization: tuple[tuple[str, float], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for attribute in (
            "coordinates",
            "fields",
            "parameters",
            "equations",
            "conditions",
            "regions",
            "metadata",
        ):
            object.__setattr__(self, attribute, tuple(getattr(self, attribute)))
        nondimensionalization = tuple(
            (
                name,
                _finite_float(value, "PDE nondimensionalization scale"),
            )
            for name, value in self.nondimensionalization
        )
        if any(value <= 0.0 for _, value in nondimensionalization):
            raise ValueError("Nondimensionalization scales must be positive.")
        object.__setattr__(
            self,
            "nondimensionalization",
            nondimensionalization,
        )

    @property
    def canonical_hash(self) -> str:
        from ._serialize import pde_ir_hash

        return pde_ir_hash(self)


__all__ = [
    "PDECondition",
    "PDEConditionKind",
    "PDECoordinate",
    "PDECoordinateKind",
    "PDEEquation",
    "PDEExpression",
    "PDEExpressionOp",
    "PDELiteral",
    "PDEField",
    "PDEParameter",
    "PDEProblemIR",
    "PDERegion",
    "PDERegionKind",
    "PDERepresentation",
    "as_expression",
]
