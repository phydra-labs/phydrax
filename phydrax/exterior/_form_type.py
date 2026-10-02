"""Scientific types and Cartesian proxy layouts for differential forms."""

from __future__ import annotations

from math import comb
from typing import Any, final, Literal, Mapping

from .._fingerprint import canonical_fingerprint
from .._strict import Strict
from .._validation import nonnegative_integer, positive_integer
from ..typing import checked


type FormTwist = Literal["untwisted", "twisted"]
type FormProxy = Literal["scalar", "circulation", "flux", "density", "components"]
type FiberProduct = Literal["scalar", "matrix"]
type FormPullbackRule = Literal[
    "identity", "covariant", "contravariant", "density", "exterior"
]


@final
class FormType(Strict):
    """Immutable form identity; coefficients use ambient lexicographic blades."""

    dimension: int
    degree: int
    twist: FormTwist
    fiber_shape: tuple[int, ...]
    ambient_dimension: int
    form_type_id: str

    def __init__(
        self,
        dimension: int,
        degree: int,
        /,
        *,
        twist: FormTwist = "untwisted",
        fiber_shape: tuple[int, ...] = (),
        ambient_dimension: int | None = None,
    ) -> None:
        from ..typing import parse

        dimension_ = nonnegative_integer(dimension, "dimension")
        degree_ = nonnegative_integer(degree, "degree")
        ambient = (
            dimension_
            if ambient_dimension is None
            else nonnegative_integer(ambient_dimension, "ambient_dimension")
        )
        if not degree_ <= dimension_ <= ambient:
            raise ValueError("Require 0 <= degree <= dimension <= ambient_dimension.")
        twist_ = parse(twist, FormTwist, "twist")
        fiber = tuple(positive_integer(size, "fiber_shape") for size in fiber_shape)
        identity = canonical_fingerprint(
            {
                "dimension": dimension_,
                "degree": degree_,
                "twist": twist_,
                "fiber_shape": fiber,
                "ambient_dimension": ambient,
            }
        )
        self.dimension = dimension_
        self.degree = degree_
        self.twist = twist_
        self.fiber_shape = fiber
        self.ambient_dimension = ambient
        self.form_type_id = identity

    @property
    def component_count(self) -> int:
        return comb(self.ambient_dimension, self.degree)

    def __eq__(self, other: object, /) -> bool:
        return isinstance(other, FormType) and self.form_type_id == other.form_type_id

    def __hash__(self) -> int:
        return hash(self.form_type_id)

    @property
    def value_shape(self) -> tuple[int, ...]:
        return (self.component_count, *self.fiber_shape)

    def _derive(
        self,
        degree: int,
        /,
        *,
        dimension: int | None = None,
        twist: FormTwist | None = None,
        fiber_shape: tuple[int, ...] | None = None,
        ambient_dimension: int | None = None,
    ) -> FormType:
        return FormType(
            self.dimension if dimension is None else dimension,
            degree,
            twist=self.twist if twist is None else twist,
            fiber_shape=self.fiber_shape if fiber_shape is None else fiber_shape,
            ambient_dimension=self.ambient_dimension
            if ambient_dimension is None
            else ambient_dimension,
        )

    def hodge_dual(self) -> FormType:
        if self.ambient_dimension != self.dimension:
            raise ValueError("Hodge dual requires intrinsic coefficient coordinates.")
        return self._derive(
            self.dimension - self.degree,
            twist="twisted" if self.twist == "untwisted" else "untwisted",
        )

    def exterior_derivative_type(self) -> FormType:
        if self.degree == self.dimension:
            raise ValueError("The exterior derivative of a top form has no valid degree.")
        return self._derive(self.degree + 1)

    def codifferential_type(self) -> FormType:
        if self.degree == 0:
            raise ValueError("The codifferential of a zero form has no valid degree.")
        return self._derive(self.degree - 1)

    def interior_type(self) -> FormType:
        if self.degree == 0:
            raise ValueError("Interior product of a zero form has no valid degree.")
        return self._derive(self.degree - 1)

    @checked
    def wedge_type(
        self, other: FormType, /, *, product: FiberProduct = "scalar"
    ) -> FormType:
        from ..typing import parse

        if (self.dimension, self.ambient_dimension) != (
            other.dimension,
            other.ambient_dimension,
        ):
            raise ValueError("Wedge factors must share intrinsic and ambient dimensions.")
        if self.degree + other.degree > self.dimension:
            raise ValueError("Wedge-product degree exceeds the dimension.")
        product_ = parse(product, FiberProduct, "product")
        match product_:
            case "scalar":
                if self.fiber_shape or other.fiber_shape:
                    raise ValueError("Scalar wedge requires scalar fibers.")
                fiber = ()
            case "matrix":
                if (
                    len(self.fiber_shape) != 2
                    or len(other.fiber_shape) != 2
                    or self.fiber_shape[1] != other.fiber_shape[0]
                ):
                    raise ValueError("Matrix wedge requires composable rank-two fibers.")
                fiber = (self.fiber_shape[0], other.fiber_shape[1])
        twist: FormTwist = "untwisted" if self.twist == other.twist else "twisted"
        return self._derive(self.degree + other.degree, twist=twist, fiber_shape=fiber)

    def trace_type(self) -> FormType:
        if self.dimension == 0 or self.degree >= self.dimension:
            raise ValueError("This degree has no nonzero hypersurface trace.")
        ambient = (
            self.ambient_dimension - 1
            if self.ambient_dimension == self.dimension
            else self.ambient_dimension
        )
        return self._derive(
            self.degree, dimension=self.dimension - 1, ambient_dimension=ambient
        )

    def with_twist(self, twist: FormTwist, /) -> FormType:
        return self._derive(self.degree, twist=twist)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "degree": self.degree,
            "twist": self.twist,
            "fiber_shape": list(self.fiber_shape),
            "ambient_dimension": self.ambient_dimension,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], /) -> FormType:
        required = {"dimension", "degree", "twist", "fiber_shape", "ambient_dimension"}
        if set(payload) != required:
            raise ValueError(
                "FormType payload requires exactly dimension, degree, twist, fiber_shape and ambient_dimension."
            )
        return cls(
            payload["dimension"],
            payload["degree"],
            twist=payload["twist"],
            fiber_shape=tuple(payload["fiber_shape"]),
            ambient_dimension=payload["ambient_dimension"],
        )


@final
class FormValueSpec(Strict):
    """Form identity plus an explicit physical-value proxy."""

    form_type: FormType
    proxy: FormProxy
    value_spec_id: str

    @checked
    def __init__(self, form_type: FormType, /, *, proxy: FormProxy) -> None:
        from ..typing import parse

        proxy_ = parse(proxy, FormProxy, "proxy")
        n, k = form_type.dimension, form_type.degree
        match proxy_:
            case "scalar":
                valid = k == 0
            case "circulation":
                valid = k == 1
            case "flux":
                valid = k == n - 1
            case "density":
                valid = k == n
            case "components":
                valid = True
        if not valid:
            raise ValueError("The requested proxy is incompatible with the form degree.")
        self.form_type = form_type
        self.proxy = proxy_
        self.value_spec_id = canonical_fingerprint(
            {"form_type": form_type.form_type_id, "proxy": proxy_}
        )

    @property
    def value_shape(self) -> tuple[int, ...]:
        match self.proxy:
            case "scalar" | "density":
                return self.form_type.fiber_shape
            case "circulation" | "flux":
                return (self.form_type.ambient_dimension, *self.form_type.fiber_shape)
            case "components":
                return self.form_type.value_shape

    def __eq__(self, other: object, /) -> bool:
        return (
            isinstance(other, FormValueSpec) and self.value_spec_id == other.value_spec_id
        )

    def __hash__(self) -> int:
        return hash(self.value_spec_id)

    @property
    def pullback_rule(self) -> FormPullbackRule:
        match self.proxy:
            case "scalar":
                return "identity"
            case "circulation":
                return "covariant"
            case "flux":
                return "contravariant"
            case "density":
                return "density"
            case "components":
                return "exterior"

    def to_dict(self) -> dict[str, Any]:
        return {"form_type": self.form_type.to_dict(), "proxy": self.proxy}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], /) -> FormValueSpec:
        if set(payload) != {"form_type", "proxy"}:
            raise ValueError(
                "FormValueSpec payload requires exactly form_type and proxy."
            )
        return cls(FormType.from_dict(payload["form_type"]), proxy=payload["proxy"])


__all__ = [
    "FiberProduct",
    "FormProxy",
    "FormPullbackRule",
    "FormTwist",
    "FormType",
    "FormValueSpec",
]
