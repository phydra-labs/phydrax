"""Independent carrier anchors and adjacent-order derivative comparisons."""

import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.geometry import (
    AbstractSurfacePatch,
    BSplineSurfacePatch,
    CircleCurve,
    OffsetSurface,
    RevolutionSurface,
)
from tests._support.coordinate_references import coordinate_reference_budget


type CarrierFamily = Literal["offset", "revolution", "bspline"]
type PatchEvaluator = Callable[[jax.Array], jax.Array]

# Empirical finite-difference constants from the live-source sweep/offset
# regressions in test_native_profile_construction.py::_assert_live_sweep_derivatives
# and test_normal_offset_of_rational_revolution_is_exact_parallel_body.
# Their centered 1e-6 step, 1e-9 absolute floor and 1e-8 relative ceiling
# include subtraction cancellation; the offset regression uses 1e-9 relative.
ADJACENT_STEP = 1e-6
ADJACENT_ABSOLUTE_ERROR = 1e-9
# Empirical binary64 closed-form comparison floor from
# test_native_profile_construction.py::_assert_live_sweep_derivatives:
# its analytic Rodrigues/JVP anchor and compiled/eager rational source checks
# use 1e-12 absolute error at order-one coordinate magnitude. Retain the same
# pre-existing floor for the rational and torus derivative anchors here.
ANCHOR_ABSOLUTE_ERROR = 1e-12


def derivative_carrier(family: CarrierFamily) -> AbstractSurfacePatch:
    """Build regular curved carriers with nonzero mixed parameter derivatives."""
    if family == "bspline":
        # Homogeneous bilinear numerators are u(1+v), v(1+u), uv;
        # denominator (1+u)(1+v), so every order has a rational anchor.
        return BSplineSurfacePatch(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.5, 0.0, 0.0], [0.5, 0.5, 0.25]],
            ],
            [[1.0, 2.0], [2.0, 4.0]],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            1,
            1,
        )
    meridian = CircleCurve((3.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5)
    revolution = RevolutionSurface(meridian, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    if family == "revolution":
        return revolution
    if family == "offset":
        with coordinate_reference_budget() as budget:
            return OffsetSurface(revolution, 0.125, budget=budget)
    raise ValueError(
        f"derivative_carrier.family must name a supported carrier; got {family!r}."
    )


def closed_form_partial(
    family: CarrierFamily, parameters: jax.Array, axes: tuple[int, ...]
) -> np.ndarray:
    """Differentiate the scalar rational graph or the torus formula analytically."""
    u, v = np.asarray(parameters)
    u_order, v_order = axes.count(0), axes.count(1)
    if family == "bspline":

        def rational_partial(value: np.float64, order: int) -> np.float64:
            if order == 0:
                return value / (1.0 + value)
            return np.float64(
                (-1) ** (order + 1) * math.factorial(order) / (1.0 + value) ** (order + 1)
            )

        first = rational_partial(u, u_order)
        second = rational_partial(v, v_order)
        return np.asarray(
            [
                first if v_order == 0 else 0.0,
                second if u_order == 0 else 0.0,
                first * second,
            ],
            dtype=np.float64,
        )
    radius = 0.625 if family == "offset" else 0.5
    radial = radius * math.cos(v + v_order * math.pi / 2.0)
    if v_order == 0:
        radial += 3.0
    return np.asarray(
        [
            radial * math.cos(u + u_order * math.pi / 2.0),
            radial * math.sin(u + u_order * math.pi / 2.0),
            radius * math.sin(v + v_order * math.pi / 2.0) if u_order == 0 else 0.0,
        ],
        dtype=np.float64,
    )


def parameter_partial(evaluate: PatchEvaluator, axes: tuple[int, ...]) -> PatchEvaluator:
    """Take directional partials without forming unused full derivative tensors."""
    if not axes:
        return evaluate
    previous = parameter_partial(evaluate, axes[:-1])
    direction = jnp.asarray(
        [1.0, 0.0] if axes[-1] == 0 else [0.0, 1.0], dtype=jnp.float64
    )

    def partial(parameters: jax.Array) -> jax.Array:
        return jax.jvp(previous, (parameters,), (direction,))[1]

    return partial


def assert_adjacent_order_partial(
    lower: PatchEvaluator,
    higher: PatchEvaluator,
    parameters: jax.Array,
    axis: int,
    *,
    rtol: float,
) -> None:
    """Compare order n with central differences of order n-1."""
    # Central-difference truncation is h^2 f'''/6; the named regression
    # constants above also cover roundoff amplified by the 1/(2h) subtraction.
    step = ADJACENT_STEP
    displacement = step * jnp.asarray(
        [1.0, 0.0] if axis == 0 else [0.0, 1.0], dtype=jnp.float64
    )
    difference = (lower(parameters + displacement) - lower(parameters - displacement)) / (
        2.0 * step
    )
    np.testing.assert_allclose(
        higher(parameters), difference, rtol=rtol, atol=ADJACENT_ABSOLUTE_ERROR
    )
