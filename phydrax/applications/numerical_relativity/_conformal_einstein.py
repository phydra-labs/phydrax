#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Four-dimensional vacuum metric conformal Einstein zero quantities."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein
from phydrax.linalg import inverse

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


class ConformalEinsteinState(StrictModule):
    """Unphysical metric, conformal fields, and rescaled Weyl tensor."""

    metric: Array
    conformal_factor: Array
    friedrich_scalar: Array
    schouten: Array
    rescaled_weyl: Array
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric: ArrayLike,
        conformal_factor: ArrayLike,
        friedrich_scalar: ArrayLike,
        schouten: ArrayLike,
        rescaled_weyl: ArrayLike,
        /,
        *,
        state_id: str,
    ):
        omega = jnp.asarray(conformal_factor)
        shape = tuple(omega.shape)
        metric_value = jnp.asarray(metric, dtype=omega.dtype)
        scalar = jnp.asarray(friedrich_scalar, dtype=omega.dtype)
        schouten_value = jnp.asarray(schouten, dtype=omega.dtype)
        weyl = jnp.asarray(rescaled_weyl, dtype=omega.dtype)
        if metric_value.shape != (4, 4) + shape:
            raise ValueError("metric must have shape (4, 4) + spatial_shape.")
        if scalar.shape != shape or schouten_value.shape != (4, 4) + shape:
            raise ValueError("Friedrich scalar or Schouten tensor shape is invalid.")
        if weyl.shape != (4, 4, 4, 4) + shape:
            raise ValueError(
                "rescaled_weyl must have shape (4, 4, 4, 4) + spatial_shape."
            )
        identifier = str(state_id)
        if not identifier:
            raise ValueError("state_id must be non-empty.")
        self.metric = metric_value
        self.conformal_factor = omega
        self.friedrich_scalar = scalar
        self.schouten = schouten_value
        self.rescaled_weyl = weyl
        self.spatial_shape = shape
        self.state_id = identifier


class ConformalEinsteinDerivativeData(StrictModule):
    """Caller-computed covariant derivatives and unphysical Riemann tensor."""

    conformal_gradient: Array
    conformal_hessian: Array
    friedrich_gradient: Array
    schouten_derivative: Array
    weyl_divergence: Array
    riemann: Array
    derivative_source_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: ConformalEinsteinState,
        /,
        *,
        conformal_gradient: ArrayLike,
        conformal_hessian: ArrayLike,
        friedrich_gradient: ArrayLike,
        schouten_derivative: ArrayLike,
        weyl_divergence: ArrayLike,
        riemann: ArrayLike,
        derivative_source_id: str,
    ):
        if not isinstance(state, ConformalEinsteinState):
            raise TypeError("state must be ConformalEinsteinState.")
        shape = state.spatial_shape
        gradient = jnp.asarray(conformal_gradient, dtype=state.metric.dtype)
        hessian = jnp.asarray(conformal_hessian, dtype=state.metric.dtype)
        scalar_gradient = jnp.asarray(friedrich_gradient, dtype=state.metric.dtype)
        schouten_gradient = jnp.asarray(schouten_derivative, dtype=state.metric.dtype)
        divergence = jnp.asarray(weyl_divergence, dtype=state.metric.dtype)
        curvature = jnp.asarray(riemann, dtype=state.metric.dtype)
        if gradient.shape != (4,) + shape or scalar_gradient.shape != (4,) + shape:
            raise ValueError("Conformal/scalar gradient shapes are invalid.")
        if hessian.shape != (4, 4) + shape:
            raise ValueError("Conformal Hessian shape is invalid.")
        if schouten_gradient.shape != (4, 4, 4) + shape:
            raise ValueError("Schouten derivative must have derivative,a,b axes.")
        if divergence.shape != (4, 4, 4) + shape:
            raise ValueError("Weyl divergence must have three tensor axes.")
        if curvature.shape != (4, 4, 4, 4) + shape:
            raise ValueError("Riemann tensor shape is invalid.")
        source = str(derivative_source_id)
        if not source:
            raise ValueError("derivative_source_id must be non-empty.")
        self.conformal_gradient = gradient
        self.conformal_hessian = hessian
        self.friedrich_gradient = scalar_gradient
        self.schouten_derivative = schouten_gradient
        self.weyl_divergence = divergence
        self.riemann = curvature
        self.derivative_source_id = source


class ConformalEinsteinSystem(StrictModule):
    """Vacuum metric conformal equations in canonical mostly-plus convention."""

    cosmological_constant: float = eqx.field(static=True)
    scalar_curvature_gauge: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        cosmological_constant: float,
        /,
        *,
        scalar_curvature_gauge: float,
        residual_tolerance: float = 1e-8,
    ):
        cosmological = float(cosmological_constant)
        scalar_curvature = float(scalar_curvature_gauge)
        tolerance = float(residual_tolerance)
        if not np.isfinite(cosmological) or cosmological >= 0.0:
            raise ValueError(
                "AdS conformal Einstein systems require a finite negative cosmological constant."
            )
        if (
            not np.isfinite(scalar_curvature)
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Conformal gauge and residual tolerance are invalid.")
        convention = (
            "dimension=4;signature=(-,+,+,+);"
            "R_abcd=Omega*d_abcd+2(g_a[c L_d]b-g_b[c L_d]a);"
            "lambda-6*Omega*s+3*gradOmegaSquared=0"
        )
        self.cosmological_constant = cosmological
        self.scalar_curvature_gauge = scalar_curvature
        self.residual_tolerance = tolerance
        self.convention = convention
        self.system_id = canonical_fingerprint(
            {
                "kind": "vacuum-metric-conformal-einstein-system",
                "cosmological_constant": cosmological,
                "scalar_curvature_gauge": scalar_curvature,
                "residual_tolerance": tolerance,
                "convention": convention,
            }
        )


class ConformalEinsteinZeroQuantities(StrictModule):
    conformal_hessian: Array
    friedrich_gradient: Array
    schouten_curl: Array
    weyl_divergence: Array
    scalar_constraint: Array
    riemann_decomposition: Array
    weyl_first_pair_antisymmetry: Array
    weyl_second_pair_antisymmetry: Array
    weyl_pair_exchange: Array
    weyl_trace: Array
    metric_inverse_residual: Array
    component_maxima: Array
    maximum_residual: Array
    finite: Array
    accepted: Array
    state_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    derivative_source_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _inverse_metric(metric: Array, /) -> tuple[Array, Array]:
    trailing = jnp.moveaxis(metric, (0, 1), (-2, -1))
    result = inverse(trailing)
    inverse_metric = jnp.moveaxis(result.value, (-2, -1), (0, 1))
    identity = ein.contract("ab...,bc...->ac...", metric, inverse_metric)
    expected = jnp.eye(4, dtype=metric.dtype).reshape((4, 4) + (1,) * (metric.ndim - 2))
    residual = jnp.max(jnp.abs(identity - expected))
    return inverse_metric, residual


def evaluate_conformal_einstein_zero_quantities(
    system: ConformalEinsteinSystem,
    state: ConformalEinsteinState,
    derivatives: ConformalEinsteinDerivativeData,
    /,
) -> ConformalEinsteinZeroQuantities:
    """Evaluate the full finite vacuum zero-quantity ledger pointwise."""
    if not isinstance(system, ConformalEinsteinSystem):
        raise TypeError("system must be ConformalEinsteinSystem.")
    if not isinstance(state, ConformalEinsteinState):
        raise TypeError("state must be ConformalEinsteinState.")
    if not isinstance(derivatives, ConformalEinsteinDerivativeData):
        raise TypeError("derivatives must be ConformalEinsteinDerivativeData.")
    inverse_metric, inverse_residual = _inverse_metric(state.metric)
    gradient_up = ein.contract(
        "ab...,b...->a...", inverse_metric, derivatives.conformal_gradient
    )
    hessian_zero = (
        derivatives.conformal_hessian
        + state.conformal_factor[None, None, ...] * state.schouten
        - state.friedrich_scalar[None, None, ...] * state.metric
    )
    friedrich_zero = derivatives.friedrich_gradient + ein.contract(
        "ab...,b...->a...", state.schouten, gradient_up
    )
    schouten_curl = jnp.zeros_like(derivatives.schouten_derivative)
    for first in range(4):
        for second in range(4):
            for third in range(4):
                weyl_gradient = sum(
                    gradient_up[contracted]
                    * state.rescaled_weyl[contracted, third, first, second]
                    for contracted in range(4)
                )
                schouten_curl = schouten_curl.at[first, second, third].set(
                    derivatives.schouten_derivative[first, second, third]
                    - derivatives.schouten_derivative[second, first, third]
                    - weyl_gradient
                )
    gradient_squared = ein.contract(
        "a...,a...->...", derivatives.conformal_gradient, gradient_up
    )
    scalar_zero = (
        system.cosmological_constant
        - 6.0 * state.conformal_factor * state.friedrich_scalar
        + 3.0 * gradient_squared
    )
    decomposition = jnp.zeros_like(derivatives.riemann)
    for first in range(4):
        for second in range(4):
            for third in range(4):
                for fourth in range(4):
                    schouten_wedge = (
                        state.metric[first, third] * state.schouten[fourth, second]
                        - state.metric[first, fourth] * state.schouten[third, second]
                        - state.metric[second, third] * state.schouten[fourth, first]
                        + state.metric[second, fourth] * state.schouten[third, first]
                    )
                    decomposition = decomposition.at[first, second, third, fourth].set(
                        derivatives.riemann[first, second, third, fourth]
                        - state.conformal_factor
                        * state.rescaled_weyl[first, second, third, fourth]
                        - schouten_wedge
                    )
    first_antisymmetry = state.rescaled_weyl + jnp.swapaxes(state.rescaled_weyl, 0, 1)
    second_antisymmetry = state.rescaled_weyl + jnp.swapaxes(state.rescaled_weyl, 2, 3)
    pair_exchange = state.rescaled_weyl - jnp.transpose(
        state.rescaled_weyl,
        (2, 3, 0, 1) + tuple(range(4, state.rescaled_weyl.ndim)),
    )
    trace = ein.contract("ac...,abcd...->bd...", inverse_metric, state.rescaled_weyl)
    maxima = jnp.stack(
        (
            jnp.max(jnp.abs(hessian_zero)),
            jnp.max(jnp.abs(friedrich_zero)),
            jnp.max(jnp.abs(schouten_curl)),
            jnp.max(jnp.abs(derivatives.weyl_divergence)),
            jnp.max(jnp.abs(scalar_zero)),
            jnp.max(jnp.abs(decomposition)),
            jnp.max(jnp.abs(first_antisymmetry)),
            jnp.max(jnp.abs(second_antisymmetry)),
            jnp.max(jnp.abs(pair_exchange)),
            jnp.max(jnp.abs(trace)),
            inverse_residual,
        )
    )
    maximum = jnp.max(maxima)
    finite = jnp.all(jnp.isfinite(maxima))
    return ConformalEinsteinZeroQuantities(
        conformal_hessian=hessian_zero,
        friedrich_gradient=friedrich_zero,
        schouten_curl=schouten_curl,
        weyl_divergence=derivatives.weyl_divergence,
        scalar_constraint=scalar_zero,
        riemann_decomposition=decomposition,
        weyl_first_pair_antisymmetry=first_antisymmetry,
        weyl_second_pair_antisymmetry=second_antisymmetry,
        weyl_pair_exchange=pair_exchange,
        weyl_trace=trace,
        metric_inverse_residual=inverse_residual,
        component_maxima=maxima,
        maximum_residual=maximum,
        finite=finite,
        accepted=finite & (maximum <= system.residual_tolerance),
        state_id=state.state_id,
        system_id=system.system_id,
        derivative_source_id=derivatives.derivative_source_id,
        claim="finite-vacuum-metric-conformal-einstein-zero-quantity-evaluation",
    )


def exact_ads_conformal_reference(
    system: ConformalEinsteinSystem,
    spatial_shape: Sequence[int] = (),
    /,
) -> tuple[ConformalEinsteinState, ConformalEinsteinDerivativeData]:
    """Return constant-curvature AdS with Omega=1 as an exact local control."""
    if not isinstance(system, ConformalEinsteinSystem):
        raise TypeError("system must be ConformalEinsteinSystem.")
    shape = tuple(spatial_shape)
    if any(value < 1 for value in shape):
        raise ValueError("spatial_shape must contain positive extents.")
    metric_matrix = np.diag((-1.0, 1.0, 1.0, 1.0))
    metric = jnp.broadcast_to(
        jnp.asarray(metric_matrix).reshape((4, 4) + (1,) * len(shape)),
        (4, 4) + shape,
    )
    omega = jnp.ones(shape)
    scalar_value = system.cosmological_constant / 6.0
    friedrich = jnp.full(shape, scalar_value)
    schouten = scalar_value * metric
    weyl = jnp.zeros((4, 4, 4, 4) + shape)
    curvature_scale = system.cosmological_constant / 3.0
    riemann = jnp.zeros_like(weyl)
    for first in range(4):
        for second in range(4):
            for third in range(4):
                for fourth in range(4):
                    riemann = riemann.at[first, second, third, fourth].set(
                        curvature_scale
                        * (
                            metric[first, third] * metric[second, fourth]
                            - metric[first, fourth] * metric[second, third]
                        )
                    )
    state = ConformalEinsteinState(
        metric,
        omega,
        friedrich,
        schouten,
        weyl,
        state_id=canonical_fingerprint(
            {
                "kind": "exact-ads-conformal-reference-state",
                "system": system.system_id,
                "shape": shape,
                "metric": array_tree_fingerprint(metric_matrix),
            }
        ),
    )
    derivatives = ConformalEinsteinDerivativeData(
        state,
        conformal_gradient=jnp.zeros((4,) + shape),
        conformal_hessian=jnp.zeros((4, 4) + shape),
        friedrich_gradient=jnp.zeros((4,) + shape),
        schouten_derivative=jnp.zeros((4, 4, 4) + shape),
        weyl_divergence=jnp.zeros((4, 4, 4) + shape),
        riemann=riemann,
        derivative_source_id="analytic-constant-curvature-ads",
    )
    return state, derivatives


__all__ = [
    "ConformalEinsteinDerivativeData",
    "ConformalEinsteinState",
    "ConformalEinsteinSystem",
    "ConformalEinsteinZeroQuantities",
    "evaluate_conformal_einstein_zero_quantities",
    "exact_ads_conformal_reference",
]
