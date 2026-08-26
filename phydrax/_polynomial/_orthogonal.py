#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from orthax import (
    chebyshev as _orthax_chebyshev,
    hermite as _orthax_hermite,
    hermite_e as _orthax_hermite_e,
    laguerre as _orthax_laguerre,
    legendre as _orthax_legendre,
)
from scipy.special import eval_legendre, roots_jacobi

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


OrthogonalFamily: TypeAlias = Literal[
    "chebyshev",
    "legendre",
    "hermite",
    "hermite_e",
    "laguerre",
]
LegendreRuleKind: TypeAlias = Literal["gauss", "radau", "lobatto"]


class OrthogonalRuleData(StrictModule, NonTrainableState):
    """Prepared one-dimensional rule with explicit family and measure semantics."""

    nodes: Array
    weights: Array
    exact_degree: int = eqx.field(static=True)
    family: str = eqx.field(static=True)
    node_rule: str = eqx.field(static=True)
    reference_domain: str = eqx.field(static=True)
    basis_measure: str = eqx.field(static=True)
    integration_measure: str = eqx.field(static=True)
    measure_mass: float = eqx.field(static=True)
    endpoint_policy: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        exact_degree: int,
        family: OrthogonalFamily,
        node_rule: str,
        reference_domain: str,
        basis_measure: str,
        integration_measure: str,
        measure_mass: float,
        endpoint_policy: str,
        backend: str,
    ):
        nodes_host = np.asarray(nodes)
        weights_host = np.asarray(weights)
        if (
            nodes_host.ndim != 1
            or weights_host.shape != nodes_host.shape
            or not nodes_host.size
        ):
            raise ValueError(
                "Orthogonal rule nodes and weights must be nonempty vectors."
            )
        if (
            np.any(~np.isfinite(nodes_host))
            or np.any(~np.isfinite(weights_host))
            or np.any(weights_host <= 0.0)
            or np.any(np.diff(nodes_host) <= 0.0)
        ):
            raise ValueError(
                "Orthogonal rule nodes must increase and weights must be finite and positive."
            )
        mass = float(measure_mass)
        tolerance = 64.0 * np.finfo(weights_host.dtype).eps * max(1, weights_host.size)
        if not np.isclose(np.sum(weights_host), mass, rtol=tolerance, atol=tolerance):
            raise ValueError("Orthogonal rule weights do not have their declared mass.")
        self.nodes = jnp.asarray(nodes_host)
        self.weights = jnp.asarray(weights_host)
        self.exact_degree = int(exact_degree)
        self.family = family
        self.node_rule = str(node_rule)
        self.reference_domain = str(reference_domain)
        self.basis_measure = str(basis_measure)
        self.integration_measure = str(integration_measure)
        self.measure_mass = mass
        self.endpoint_policy = str(endpoint_policy)
        self.backend = str(backend)
        self.rule_id = canonical_fingerprint(
            {
                "kind": "orthogonal-rule-v1",
                "family": family,
                "node_rule": self.node_rule,
                "reference_domain": self.reference_domain,
                "basis_measure": self.basis_measure,
                "integration_measure": self.integration_measure,
                "measure_mass": mass,
                "endpoint_policy": self.endpoint_policy,
                "exact_degree": self.exact_degree,
                "backend": self.backend,
                "data": array_tree_fingerprint((nodes_host, weights_host)),
            }
        )


def standard_series_value(
    family: OrthogonalFamily,
    coefficients: ArrayLike,
    x: ArrayLike,
    /,
) -> Array:
    """Evaluate one fixed-shape series in a classical standard normalization."""
    coefficients_ = jnp.asarray(coefficients)
    x_ = jnp.asarray(x)
    if coefficients_.ndim != 1:
        raise ValueError("Orthogonal series coefficients must be a rank-one array.")
    if x_.shape != ():
        raise ValueError("Orthogonal series scalar evaluation requires a scalar point.")
    if family == "chebyshev":
        return _orthax_chebyshev.chebval(x_, coefficients_)
    if family == "legendre":
        return _orthax_legendre.legval(x_, coefficients_)
    if family == "hermite":
        return _orthax_hermite.hermval(x_, coefficients_)
    if family == "hermite_e":
        return _orthax_hermite_e.hermeval(x_, coefficients_)
    if family == "laguerre":
        return _orthax_laguerre.lagval(x_, coefficients_)
    raise ValueError(f"Unsupported orthogonal polynomial family: {family!r}.")


def standard_vandermonde(
    family: OrthogonalFamily,
    nodes: ArrayLike,
    degree: int,
    /,
) -> Array:
    """Evaluate all standard-family modes through one declared degree."""
    nodes_ = jnp.asarray(nodes)
    degree_ = int(degree)
    if nodes_.ndim != 1 or degree_ < 0:
        raise ValueError(
            "Orthogonal Vandermonde nodes must be rank one and degree non-negative."
        )
    if family == "chebyshev":
        return _orthax_chebyshev.chebvander(nodes_, degree_)
    if family == "legendre":
        return _orthax_legendre.legvander(nodes_, degree_)
    if family == "hermite":
        return _orthax_hermite.hermvander(nodes_, degree_)
    if family == "hermite_e":
        return _orthax_hermite_e.hermevander(nodes_, degree_)
    if family == "laguerre":
        return _orthax_laguerre.lagvander(nodes_, degree_)
    raise ValueError(f"Unsupported orthogonal polynomial family: {family!r}.")


def standard_series_derivative_coefficients(
    family: OrthogonalFamily,
    coefficients: ArrayLike,
    order: int = 1,
    /,
    *,
    scale: ArrayLike = 1.0,
) -> Array:
    """Differentiate a fixed-capacity standard series without changing shape."""
    coefficients_ = jnp.asarray(coefficients)
    order_ = int(order)
    if coefficients_.ndim < 1 or order_ < 0:
        raise ValueError(
            "Orthogonal derivative coefficients need a leading mode axis and "
            "non-negative order."
        )
    if order_ == 0:
        return coefficients_
    if family == "chebyshev":
        derivative = _orthax_chebyshev.chebder(
            coefficients_,
            m=order_,
            scl=scale,
            axis=0,
        )
    elif family == "legendre":
        derivative = _orthax_legendre.legder(
            coefficients_,
            m=order_,
            scl=scale,
            axis=0,
        )
    elif family == "hermite":
        derivative = _orthax_hermite.hermder(
            coefficients_,
            m=order_,
            scl=scale,
            axis=0,
        )
    elif family == "hermite_e":
        derivative = _orthax_hermite_e.hermeder(
            coefficients_,
            m=order_,
            scl=scale,
            axis=0,
        )
    elif family == "laguerre":
        derivative = _orthax_laguerre.lagder(
            coefficients_,
            m=order_,
            scl=scale,
            axis=0,
        )
    else:
        raise ValueError(f"Unsupported orthogonal polynomial family: {family!r}.")
    padding = [(0, coefficients_.shape[0] - derivative.shape[0])]
    padding.extend((0, 0) for _ in coefficients_.shape[1:])
    return jnp.pad(derivative, tuple(padding))


def standard_derivative_matrix(
    family: OrthogonalFamily,
    count: int,
    order: int = 1,
    /,
    *,
    scale: ArrayLike = 1.0,
    dtype: Any = float,
) -> Array:
    """Return the fixed-capacity coefficient derivative matrix."""
    count_ = _node_count(count)
    return standard_series_derivative_coefficients(
        family,
        jnp.eye(count_, dtype=dtype),
        order,
        scale=scale,
    )


def standard_affine_coefficients(
    family: OrthogonalFamily,
    intercept: ArrayLike,
    slope: ArrayLike,
    /,
) -> Array:
    """Return standard-family coefficients for ``intercept + slope * x``."""
    intercept_, slope_ = jnp.broadcast_arrays(
        jnp.asarray(intercept),
        jnp.asarray(slope),
    )
    if family in ("chebyshev", "legendre", "hermite_e"):
        return jnp.stack((intercept_, slope_), axis=-1)
    if family == "hermite":
        return jnp.stack((intercept_, 0.5 * slope_), axis=-1)
    if family == "laguerre":
        return jnp.stack((intercept_ + slope_, -slope_), axis=-1)
    raise ValueError(f"Unsupported orthogonal polynomial family: {family!r}.")


def _node_count(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Orthogonal rule node count must be an integer.")
    count = int(value)
    if count < 1:
        raise ValueError("Orthogonal rule node count must be positive.")
    return count


def _canonical_arrays(nodes, weights, dtype, /) -> tuple[np.ndarray, np.ndarray]:
    nodes_host = np.asarray(nodes, dtype=float).reshape((-1,))
    weights_host = np.asarray(weights, dtype=float).reshape((-1,))
    order = np.argsort(nodes_host)
    dtype_ = np.dtype(dtype)
    return (
        np.asarray(nodes_host[order], dtype=dtype_),
        np.asarray(weights_host[order], dtype=dtype_),
    )


def legendre_rule_data(
    num_nodes: int,
    kind: LegendreRuleKind = "gauss",
    /,
    *,
    dtype=float,
) -> OrthogonalRuleData:
    """Return a canonical raw-Lebesgue Legendre Gauss, Radau, or Lobatto rule."""
    count = _node_count(num_nodes)
    if kind not in ("gauss", "radau", "lobatto"):
        raise ValueError("Legendre rule kind must be 'gauss', 'radau', or 'lobatto'.")
    if kind == "lobatto" and count < 2:
        raise ValueError("Legendre Lobatto rules require at least two nodes.")

    if kind == "gauss":
        nodes, weights = np.polynomial.legendre.leggauss(count)
        backend = "numpy"
        endpoint_policy = "none"
        exact_degree = 2 * count - 1
    elif count == 1:
        nodes, weights = np.asarray([-1.0]), np.asarray([2.0])
        backend = "analytic"
        endpoint_policy = "left"
        exact_degree = 0
    elif kind == "radau":
        interior, _ = roots_jacobi(count - 1, 0.0, 1.0)
        values = eval_legendre(count - 1, interior)
        nodes = np.concatenate((np.asarray([-1.0]), interior))
        weights = np.concatenate(
            (
                np.asarray([2.0 / count**2]),
                (1.0 - interior) / (count**2 * values**2),
            )
        )
        backend = "scipy"
        endpoint_policy = "left"
        exact_degree = 2 * count - 2
    elif count == 2:
        nodes, weights = np.asarray([-1.0, 1.0]), np.asarray([1.0, 1.0])
        backend = "analytic"
        endpoint_policy = "both"
        exact_degree = 1
    else:
        interior, _ = roots_jacobi(count - 2, 1.0, 1.0)
        values = eval_legendre(count - 1, interior)
        endpoint_weight = 2.0 / (count * (count - 1))
        nodes = np.concatenate((np.asarray([-1.0]), interior, np.asarray([1.0])))
        weights = np.concatenate(
            (
                np.asarray([endpoint_weight]),
                2.0 / (count * (count - 1) * values**2),
                np.asarray([endpoint_weight]),
            )
        )
        backend = "scipy"
        endpoint_policy = "both"
        exact_degree = 2 * count - 3

    nodes_, weights_ = _canonical_arrays(nodes, weights, dtype)
    if endpoint_policy in ("left", "both"):
        nodes_[0] = -1.0
    if endpoint_policy == "both":
        nodes_[-1] = 1.0
    return OrthogonalRuleData(
        nodes_,
        weights_,
        exact_degree=exact_degree,
        family="legendre",
        node_rule=kind,
        reference_domain="minus-one-one",
        basis_measure="lebesgue",
        integration_measure="lebesgue",
        measure_mass=2.0,
        endpoint_policy=endpoint_policy,
        backend=backend,
    )


def standard_normal_hermite_rule_data(
    num_nodes: int,
    /,
    *,
    dtype=float,
) -> OrthogonalRuleData:
    """Return a probabilists' Hermite rule normalized as a standard-normal expectation."""
    count = _node_count(num_nodes)
    nodes, raw_weights = np.polynomial.hermite_e.hermegauss(count)
    nodes_, weights_ = _canonical_arrays(
        nodes,
        raw_weights / np.sqrt(2.0 * np.pi),
        dtype,
    )
    return OrthogonalRuleData(
        nodes_,
        weights_,
        exact_degree=2 * count - 1,
        family="hermite_e",
        node_rule="gauss",
        reference_domain="real-line",
        basis_measure="exp-minus-x-squared-over-two",
        integration_measure="standard-normal",
        measure_mass=1.0,
        endpoint_policy="none",
        backend="numpy",
    )


__all__ = [
    "LegendreRuleKind",
    "OrthogonalFamily",
    "OrthogonalRuleData",
    "legendre_rule_data",
    "standard_derivative_matrix",
    "standard_affine_coefficients",
    "standard_normal_hermite_rule_data",
    "standard_series_derivative_coefficients",
    "standard_series_value",
    "standard_vandermonde",
]
