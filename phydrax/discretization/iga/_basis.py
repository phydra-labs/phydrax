#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation._bspline_grid import BSplineGrid
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class SplineAxisPlan(StrictModule, NonTrainableState):
    """One fixed, nonperiodic, clamped H1 B-spline parameter axis."""

    name: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    knots: Array
    control_count: int = eqx.field(static=True)
    span_indices: Array
    span_bounds: Array
    axis_id: str = eqx.field(static=True)

    def __init__(
        self, name: str, knots: ArrayLike, /, *, degree: int, periodic: bool = False
    ):
        name_, degree_, knots_host = str(name), int(degree), np.asarray(knots)
        if not name_:
            raise ValueError("Spline axis name must be non-empty.")
        if periodic:
            raise ValueError("Spline axes do not support periodic knots.")
        if degree_ < 1:
            raise ValueError("H1 spline axes require degree at least one.")
        if knots_host.ndim != 1 or knots_host.size < 2 * (degree_ + 1):
            raise ValueError("Spline knots must be a sufficiently long rank-1 array.")
        if not np.issubdtype(knots_host.dtype, np.number):
            raise TypeError("Spline knots must be numeric.")
        knots_host = knots_host.astype(float, copy=False)
        if np.any(~np.isfinite(knots_host)) or np.any(np.diff(knots_host) < 0):
            raise ValueError("Spline knots must be finite and nondecreasing.")
        control_count = knots_host.size - degree_ - 1
        lower, upper = knots_host[degree_], knots_host[control_count]
        if control_count < degree_ + 1 or upper <= lower:
            raise ValueError("Spline knots define an invalid parameter interval.")
        if not (
            np.all(knots_host[: degree_ + 1] == lower)
            and np.all(knots_host[control_count:] == upper)
        ):
            raise ValueError("Spline axes require clamped endpoint knots.")
        interior = knots_host[(knots_host > lower) & (knots_host < upper)]
        if interior.size and np.any(np.unique(interior, return_counts=True)[1] > degree_):
            raise ValueError(
                "H1 spline axes require interior knot multiplicity at most degree."
            )
        spans = np.flatnonzero(np.diff(knots_host) > 0)
        spans = spans[(spans >= degree_) & (spans < control_count)]
        if not spans.size:
            raise ValueError("Spline axis must contain at least one active span.")
        self.name, self.degree, self.knots, self.control_count = (
            name_,
            degree_,
            jnp.asarray(knots_host),
            int(control_count),
        )
        self.span_indices = jnp.asarray(spans, dtype=jnp.int32)
        self.span_bounds = jnp.asarray(
            np.stack((knots_host[spans], knots_host[spans + 1]), axis=-1)
        )
        self.axis_id = canonical_fingerprint(
            {
                "kind": "spline-axis-plan",
                "name": name_,
                "degree": degree_,
                "knots": array_tree_fingerprint(knots_host),
            }
        )

    @property
    def span_count(self) -> int:
        return int(self.span_indices.shape[0])

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return (float(self.knots[self.degree]), float(self.knots[self.control_count]))


class TensorSplineBasisSpec(StrictModule, NonTrainableState):
    """Fixed 1D--3D tensor-product H1 spline basis; degrees may be anisotropic."""

    axes: tuple[SplineAxisPlan, ...]
    axis_names: tuple[str, ...] = eqx.field(static=True)
    degrees: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    span_shape: tuple[int, ...] = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        axes: Sequence[SplineAxisPlan | BSplineGrid],
        /,
        *,
        axis_names: Sequence[str] | None = None,
    ):
        inputs = tuple(axes)
        if not 1 <= len(inputs) <= 3:
            raise TypeError(
                "TensorSplineBasisSpec requires one to three fixed spline axes."
            )
        if all(isinstance(axis, BSplineGrid) for axis in inputs):
            names = (
                ("x", "y", "z")[: len(inputs)]
                if axis_names is None
                else tuple(map(str, axis_names))
            )
            if len(names) != len(inputs):
                raise ValueError("axis_names must provide one name per spline grid.")
            axes_ = tuple(
                SplineAxisPlan(name, grid.knots, degree=grid.degree)
                for name, grid in zip(names, inputs, strict=True)
            )
        elif all(isinstance(axis, SplineAxisPlan) for axis in inputs):
            if axis_names is not None:
                raise ValueError("Named spline axes already define axis_names.")
            axes_ = tuple(inputs)  # type: ignore[assignment]
        else:
            raise TypeError(
                "Tensor spline axes must be all BSplineGrid or all SplineAxisPlan values."
            )
        names, degrees = (
            tuple(axis.name for axis in axes_),
            tuple(axis.degree for axis in axes_),
        )
        if len(set(names)) != len(names):
            raise ValueError("Tensor spline axis names must be unique.")
        shape, spans = (
            tuple(axis.control_count for axis in axes_),
            tuple(axis.span_count for axis in axes_),
        )
        layout = canonical_fingerprint(
            {
                "kind": "tensor-spline-layout",
                "axis_names": list(names),
                "control_shape": list(shape),
                "order": "C",
            }
        )
        (
            self.axes,
            self.axis_names,
            self.degrees,
            self.control_shape,
            self.span_shape,
            self.layout_id,
        ) = axes_, names, degrees, shape, spans, layout
        self.basis_id = canonical_fingerprint(
            {
                "kind": "tensor-spline-basis",
                "axes": [axis.axis_id for axis in axes_],
                "degrees": list(degrees),
                "layout": layout,
            }
        )

    @property
    def degree(self) -> int | tuple[int, ...]:
        """Compatibility scalar for isotropic bases, otherwise per-axis degrees."""
        return self.degrees[0] if len(set(self.degrees)) == 1 else self.degrees

    @property
    def parametric_dimension(self) -> int:
        return len(self.axes)

    @property
    def coefficient_count(self) -> int:
        return prod(self.control_shape)

    @property
    def local_coefficient_count(self) -> int:
        return prod(degree + 1 for degree in self.degrees)

    @property
    def cell_count(self) -> int:
        return prod(self.span_shape)


class IsogeometricFieldSpec(StrictModule, NonTrainableState):
    """H1 scalar or vector tensor-spline field, polynomial or explicitly rational."""

    name: str = eqx.field(static=True)
    basis: TensorSplineBasisSpec
    conformity: str = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    mapping: str = eqx.field(static=True)
    weights: Array | None
    field_spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        basis: TensorSplineBasisSpec,
        /,
        *,
        conformity: str = "H1",
        component_shape: Sequence[int] = (),
        mapping: str = "identity",
        weights: ArrayLike | None = None,
    ):
        name_, components, conformity_, mapping_ = (
            str(name),
            tuple(int(x) for x in component_shape),
            str(conformity),
            str(mapping),
        )
        if not name_ or not isinstance(basis, TensorSplineBasisSpec):
            raise ValueError(
                "Isogeometric fields require a name and TensorSplineBasisSpec."
            )
        if conformity_ != "H1" or mapping_ != "identity":
            raise ValueError("IGA fields support only identity-mapped H1 conformity.")
        if any(x <= 0 for x in components):
            raise ValueError("Field component dimensions must be positive.")
        field_weights = None if weights is None else jnp.asarray(weights)
        if field_weights is not None:
            if field_weights.shape != basis.control_shape:
                raise ValueError(
                    "Field rational weights must match the field basis control shape."
                )
            if jnp.issubdtype(field_weights.dtype, jnp.complexfloating):
                raise TypeError("Field rational weights must be real.")
        (
            self.name,
            self.basis,
            self.conformity,
            self.component_shape,
            self.mapping,
            self.weights,
        ) = name_, basis, conformity_, components, mapping_, field_weights
        self.field_spec_id = canonical_fingerprint(
            {
                "kind": "isogeometric-field-spec",
                "name": name_,
                "basis": basis.basis_id,
                "conformity": conformity_,
                "component_shape": list(components),
                "mapping": mapping_,
                "weights": None
                if field_weights is None
                else array_tree_fingerprint(np.asarray(field_weights)),
            }
        )

    @property
    def is_rational(self) -> bool:
        return self.weights is not None


class IsogeometricQuadraturePolicy(StrictModule, NonTrainableState):
    """Explicit Gauss--Legendre quadrature with scalar or per-axis point counts."""

    points_per_axis: int | tuple[int, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, points_per_axis: int | Sequence[int], /):
        scalar = isinstance(points_per_axis, (int, np.integer))
        values = (
            (int(points_per_axis),) if scalar else tuple(int(x) for x in points_per_axis)
        )
        if not values or len(values) > 3 or any(x <= 0 for x in values):
            raise ValueError(
                "Quadrature points_per_axis must be one to three positive counts."
            )
        self.points_per_axis = values[0] if scalar else values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "isogeometric-quadrature-policy",
                "rule": "gauss-legendre",
                "points_per_axis": list(values),
            }
        )

    def count_for_axis(self, axis_index: int, dimension: int, /) -> int:
        values = (
            (self.points_per_axis,)
            if isinstance(self.points_per_axis, int)
            else self.points_per_axis
        )
        if len(values) == 1:
            return values[0]
        if len(values) != dimension:
            raise ValueError(
                "Anisotropic quadrature requires one count per parameter axis."
            )
        return values[axis_index]

    def axis_rule(
        self, axis: SplineAxisPlan, /, *, axis_index: int = 0, dimension: int = 1
    ) -> tuple[Array, Array]:
        count = self.count_for_axis(axis_index, dimension)
        nodes, weights = np.polynomial.legendre.leggauss(count)
        bounds = np.asarray(axis.span_bounds)
        half = 0.5 * (bounds[:, 1] - bounds[:, 0])
        points = 0.5 * (bounds[:, 0] + bounds[:, 1])[:, None] + half[:, None] * nodes
        return jnp.asarray(points), jnp.asarray(half[:, None] * weights)


__all__ = [
    "IsogeometricFieldSpec",
    "IsogeometricQuadraturePolicy",
    "SplineAxisPlan",
    "TensorSplineBasisSpec",
]
