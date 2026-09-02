#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseCholesky,
    DenseLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    RHSLayout,
    solve,
)
from ._chart import ChartTransition
from ._connection import LeviCivitaConnection
from ._map import DifferentiableMap, Immersion
from ._metric import RiemannianMetric
from ._utils import _pointwise_array


def _solve_positive_metric(
    metric: Array,
    right_hand_side: Array,
    rhs_shape: tuple[int, ...],
    /,
) -> Array:
    operator = DenseLinearOperator(
        metric,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "asserted",
            },
        ),
    )
    result = solve(
        LinearSystem(operator),
        right_hand_side,
        policy=LinearSolvePolicy(DenseCholesky()),
        rhs_layout=RHSLayout(rhs_shape),
    )
    return eqx.error_if(
        result.value,
        jnp.any(~result.successful),
        "Induced Riemannian metric solve failed.",
    )


CoordinateMap = DifferentiableMap | Immersion | ChartTransition


class RiemannianMapGeometry(StrictModule):
    """One differentiable map with declared source and target metrics."""

    map: CoordinateMap
    source_metric: RiemannianMetric
    target_metric: RiemannianMetric

    def __init__(
        self,
        map: CoordinateMap,
        source_metric: RiemannianMetric,
        target_metric: RiemannianMetric,
        /,
    ):
        if not isinstance(map, (DifferentiableMap, Immersion, ChartTransition)):
            raise TypeError("map must be a differentiable coordinate map.")
        if not isinstance(source_metric, RiemannianMetric) or not isinstance(
            target_metric, RiemannianMetric
        ):
            raise TypeError("Riemannian map geometry requires two Riemannian metrics.")
        if not map.source.compatible_with(source_metric.chart):
            raise ValueError("Map source chart must match the source metric chart.")
        if not map.target.compatible_with(target_metric.chart):
            raise ValueError("Map target chart must match the target metric chart.")
        self.map = map
        self.source_metric = source_metric
        self.target_metric = target_metric

    def pullback_metric(self, coordinates: ArrayLike, /) -> Array:
        jacobian = self.map.jacobian(coordinates)
        target = self.target_metric(self.map(coordinates))
        return oe.contract("...ai,...ab,...bj->...ij", jacobian, target, jacobian)

    def energy_density(self, coordinates: ArrayLike, /) -> Array:
        inverse = self.source_metric.inverse(coordinates)
        return 0.5 * oe.contract(
            "...ij,...ij->...", inverse, self.pullback_metric(coordinates)
        )

    def distortion_tensor(self, coordinates: ArrayLike, /) -> Array:
        return self.pullback_metric(coordinates) - self.source_metric(coordinates)

    def isometry_residual(self, coordinates: ArrayLike, /) -> Array:
        return jnp.max(jnp.abs(self.distortion_tensor(coordinates)), axis=(-2, -1))

    def conformality_tensor(self, coordinates: ArrayLike, /) -> Array:
        source = self.source_metric(coordinates)
        pullback = self.pullback_metric(coordinates)
        scale = oe.contract(
            "...ij,...ij->...", self.source_metric.inverse(coordinates), pullback
        ) / float(self.map.source.dimension)
        return pullback - scale[..., None, None] * source

    def conformality_residual(self, coordinates: ArrayLike, /) -> Array:
        return jnp.max(jnp.abs(self.conformality_tensor(coordinates)), axis=(-2, -1))

    def volume_distortion(self, coordinates: ArrayLike, /) -> Array:
        _, source_logdet = jnp.linalg.slogdet(self.source_metric(coordinates))
        _, pullback_logdet = jnp.linalg.slogdet(self.pullback_metric(coordinates))
        return jnp.exp(0.5 * (pullback_logdet - source_logdet))

    def second_covariant_derivative(self, coordinates: ArrayLike, /) -> Array:
        source_connection = LeviCivitaConnection(self.source_metric)
        target_connection = LeviCivitaConnection(self.target_metric)

        def evaluate(point: Array) -> Array:
            mapped = self.map.map_function(point)
            jacobian = self.map.jacobian(point)
            hessian = self.map.hessian(point)
            source_coefficients = source_connection.coefficients(point)
            target_coefficients = target_connection.coefficients(mapped)
            source_correction = oe.contract("kij,ak->aij", source_coefficients, jacobian)
            target_correction = oe.contract(
                "abc,bi,cj->aij", target_coefficients, jacobian, jacobian
            )
            return hessian - source_correction + target_correction

        return _pointwise_array(evaluate, coordinates, self.map.source.dimension)

    def tension_field(self, coordinates: ArrayLike, /) -> Array:
        return oe.contract(
            "...ij,...aij->...a",
            self.source_metric.inverse(coordinates),
            self.second_covariant_derivative(coordinates),
        )

    def target_tangent_projector(self, coordinates: ArrayLike, /) -> Array:
        if self.map.source.dimension > self.map.target.dimension:
            raise ValueError("Tangent projection requires an immersion-sized map.")
        jacobian = self.map.jacobian(coordinates)
        target_metric = self.target_metric(self.map(coordinates))
        induced = self.pullback_metric(coordinates)
        right_hand_side = oe.contract(
            "...bj,...bc->...jc",
            jacobian,
            target_metric,
        )
        solved = _solve_positive_metric(
            induced,
            right_hand_side,
            (self.map.target.dimension,),
        )
        return oe.contract("...ai,...ic->...ac", jacobian, solved)

    def target_normal_projector(self, coordinates: ArrayLike, /) -> Array:
        tangent = self.target_tangent_projector(coordinates)
        identity = jnp.eye(self.map.target.dimension, dtype=tangent.dtype)
        return identity - tangent

    def second_fundamental_form(self, coordinates: ArrayLike, /) -> Array:
        return oe.contract(
            "...ab,...bij->...aij",
            self.target_normal_projector(coordinates),
            self.second_covariant_derivative(coordinates),
        )

    def mean_curvature_vector(self, coordinates: ArrayLike, /) -> Array:
        induced = self.pullback_metric(coordinates)
        second_form = self.second_fundamental_form(coordinates)
        right_hand_side = jnp.moveaxis(second_form, -3, -1)
        solved = _solve_positive_metric(
            induced,
            right_hand_side,
            (self.map.source.dimension, self.map.target.dimension),
        )
        return oe.contract("...iia->...a", solved) / float(self.map.source.dimension)


__all__ = ["RiemannianMapGeometry"]
