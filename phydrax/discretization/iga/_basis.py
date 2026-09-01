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
    """One fixed, nonperiodic, clamped B-spline parameter axis."""

    name: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    knots: Array
    control_count: int = eqx.field(static=True)
    span_indices: Array
    span_bounds: Array
    axis_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        knots: ArrayLike,
        /,
        *,
        degree: int,
        periodic: bool = False,
    ):
        name_ = str(name)
        degree_ = int(degree)
        knots_host = np.asarray(knots)
        if not name_:
            raise ValueError("Spline axis name must be non-empty.")
        if periodic:
            raise ValueError("S1 spline axes do not support periodic knots.")
        if degree_ < 1:
            raise ValueError("S1 spline axes require degree at least one.")
        if knots_host.ndim != 1 or knots_host.size < 2 * (degree_ + 1):
            raise ValueError("Spline knots must be a sufficiently long rank-1 array.")
        if not np.issubdtype(knots_host.dtype, np.number):
            raise TypeError("Spline knots must be numeric.")
        knots_host = knots_host.astype(float, copy=False)
        if np.any(~np.isfinite(knots_host)) or np.any(np.diff(knots_host) < 0.0):
            raise ValueError("Spline knots must be finite and nondecreasing.")
        control_count = int(knots_host.size) - degree_ - 1
        if control_count < degree_ + 1:
            raise ValueError("Spline knots define too few control coefficients.")
        lower = knots_host[degree_]
        upper = knots_host[control_count]
        if not upper > lower:
            raise ValueError("Spline knots must define a nonempty parameter interval.")
        if not (
            np.all(knots_host[: degree_ + 1] == lower)
            and np.all(knots_host[control_count:] == upper)
        ):
            raise ValueError("S1 spline axes require clamped endpoint knots.")
        interior = knots_host[(knots_host > lower) & (knots_host < upper)]
        if interior.size:
            _, multiplicities = np.unique(interior, return_counts=True)
            if np.any(multiplicities > degree_):
                raise ValueError(
                    "H1 spline axes require interior knot multiplicity at most degree."
                )
        positive_spans = np.flatnonzero(np.diff(knots_host) > 0.0)
        positive_spans = positive_spans[
            (positive_spans >= degree_) & (positive_spans < control_count)
        ]
        if positive_spans.size == 0:
            raise ValueError("Spline axis must contain at least one active span.")
        span_bounds = np.stack(
            (knots_host[positive_spans], knots_host[positive_spans + 1]), axis=-1
        )
        self.name = name_
        self.degree = degree_
        self.knots = jnp.asarray(knots_host)
        self.control_count = control_count
        self.span_indices = jnp.asarray(positive_spans, dtype=jnp.int32)
        self.span_bounds = jnp.asarray(span_bounds)
        self.axis_id = canonical_fingerprint(
            {
                "kind": "spline-axis-plan",
                "name": name_,
                "degree": degree_,
                "knots": array_tree_fingerprint(knots_host),
                "control_count": control_count,
                "span_indices": array_tree_fingerprint(positive_spans),
            }
        )

    @property
    def span_count(self) -> int:
        return int(self.span_indices.shape[0])

    @property
    def parameter_interval(self) -> tuple[float, float]:
        return (float(self.knots[self.degree]), float(self.knots[self.control_count]))


class TensorSplineBasisSpec(StrictModule, NonTrainableState):
    """Fixed isotropic tensor-product spline basis."""

    axes: tuple[SplineAxisPlan, ...]
    axis_names: tuple[str, ...] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
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
        if not inputs or len(inputs) > 3:
            raise TypeError(
                "TensorSplineBasisSpec requires one to three fixed spline axes."
            )
        if all(isinstance(axis, BSplineGrid) for axis in inputs):
            grids = tuple(axis for axis in inputs if isinstance(axis, BSplineGrid))
            names = (
                ("x", "y", "z")[: len(grids)]
                if axis_names is None
                else tuple(str(name) for name in axis_names)
            )
            if len(names) != len(grids):
                raise ValueError("axis_names must provide one name per spline grid.")
            axes_: tuple[SplineAxisPlan, ...] = tuple(
                SplineAxisPlan(name, grid.knots, degree=grid.degree)
                for name, grid in zip(names, grids, strict=True)
            )
        elif all(isinstance(axis, SplineAxisPlan) for axis in inputs):
            if axis_names is not None:
                raise ValueError("Internal named spline axes already define axis_names.")
            axes_ = tuple(axis for axis in inputs if isinstance(axis, SplineAxisPlan))
        else:
            raise TypeError(
                "Tensor spline axes must be all BSplineGrid or all SplineAxisPlan values."
            )
        names = tuple(axis.name for axis in axes_)
        if len(set(names)) != len(names):
            raise ValueError("Tensor spline axis names must be unique.")
        degrees = tuple(axis.degree for axis in axes_)
        if len(set(degrees)) != 1:
            raise ValueError("S1 tensor splines require one isotropic degree.")
        control_shape = tuple(axis.control_count for axis in axes_)
        span_shape = tuple(axis.span_count for axis in axes_)
        layout_id = canonical_fingerprint(
            {
                "kind": "tensor-spline-layout",
                "axis_names": list(names),
                "control_shape": list(control_shape),
                "order": "C",
            }
        )
        self.axes = axes_
        self.axis_names = names
        self.degree = degrees[0]
        self.control_shape = control_shape
        self.span_shape = span_shape
        self.layout_id = layout_id
        self.basis_id = canonical_fingerprint(
            {
                "kind": "tensor-spline-basis",
                "axes": [axis.axis_id for axis in axes_],
                "degree": degrees[0],
                "layout": layout_id,
            }
        )

    @property
    def parametric_dimension(self) -> int:
        return len(self.axes)

    @property
    def coefficient_count(self) -> int:
        return prod(self.control_shape)

    @property
    def local_coefficient_count(self) -> int:
        return (self.degree + 1) ** self.parametric_dimension

    @property
    def cell_count(self) -> int:
        return prod(self.span_shape)


class IsogeometricFieldSpec(StrictModule, NonTrainableState):
    """Exact isoparametric scalar-H1 spline field specification."""

    name: str = eqx.field(static=True)
    basis: TensorSplineBasisSpec
    conformity: str = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    mapping: str = eqx.field(static=True)
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
    ):
        name_ = str(name)
        components = tuple(int(size) for size in component_shape)
        conformity_ = str(conformity)
        mapping_ = str(mapping)
        if not name_:
            raise ValueError("Isogeometric field name must be non-empty.")
        if not isinstance(basis, TensorSplineBasisSpec):
            raise TypeError("basis must be a TensorSplineBasisSpec.")
        if conformity_ != "H1":
            raise ValueError("S1 isogeometric fields support only H1 conformity.")
        if components:
            raise ValueError("S1 isogeometric fields support only scalar values.")
        if mapping_ != "identity":
            raise ValueError("S1 isogeometric fields support only identity mapping.")
        self.name = name_
        self.basis = basis
        self.conformity = conformity_
        self.component_shape = components
        self.mapping = mapping_
        self.field_spec_id = canonical_fingerprint(
            {
                "kind": "isogeometric-field-spec",
                "name": name_,
                "basis": basis.basis_id,
                "conformity": conformity_,
                "component_shape": list(components),
                "mapping": mapping_,
            }
        )


class IsogeometricQuadraturePolicy(StrictModule, NonTrainableState):
    """Explicit isotropic Gauss-Legendre quadrature for every nonzero span."""

    points_per_axis: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, points_per_axis: int, /):
        count = int(points_per_axis)
        if count <= 0:
            raise ValueError("Isogeometric quadrature points_per_axis must be positive.")
        self.points_per_axis = count
        self.policy_id = canonical_fingerprint(
            {
                "kind": "isogeometric-quadrature-policy",
                "rule": "gauss-legendre",
                "points_per_axis": count,
                "isotropic": True,
            }
        )

    def axis_rule(self, axis: SplineAxisPlan, /) -> tuple[Array, Array]:
        if not isinstance(axis, SplineAxisPlan):
            raise TypeError("axis must be a SplineAxisPlan.")
        nodes, weights = np.polynomial.legendre.leggauss(self.points_per_axis)
        bounds = np.asarray(axis.span_bounds)
        midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
        half_width = 0.5 * (bounds[:, 1] - bounds[:, 0])
        points = midpoint[:, None] + half_width[:, None] * nodes[None, :]
        scaled_weights = half_width[:, None] * weights[None, :]
        return jnp.asarray(points), jnp.asarray(scaled_weights)


__all__ = [
    "IsogeometricFieldSpec",
    "IsogeometricQuadraturePolicy",
    "SplineAxisPlan",
    "TensorSplineBasisSpec",
]
