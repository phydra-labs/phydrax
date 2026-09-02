#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._geometry_precision import GeometryPrecisionPolicy
from .._strict import StrictModule
from ..linalg import (
    DenseCholesky,
    DenseLinearOperator,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ._manifold import AbstractRiemannianManifold


_SPD_SOLVE = LinearSolvePolicy(DenseCholesky(), failure=FailurePolicy("status"))


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _spd_solve(
    matrix: Array, right_hand_side: Array, identifier: str, /
) -> tuple[Array, Array]:
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"positive_definite": "asserted"},
        ),
        operator_id=f"{identifier}:operator",
    )
    result = solve(
        LinearSystem(operator, problem_id=f"{identifier}:system"),
        right_hand_side,
        policy=_SPD_SOLVE,
    )
    return result.value, jnp.all(result.diagnostics.converged)


class ManifoldTangentMeasureEvidence(StrictModule):
    metric: Array
    inverse_metric: Array
    tangent_projector: Array
    normal_projector: Array
    constraint_jacobian: Array
    normals: Array
    rank_margin: Array
    log_volume: Array
    orientation: Array
    valid: Array

    def __init__(
        self,
        *,
        metric: ArrayLike,
        inverse_metric: ArrayLike,
        tangent_projector: ArrayLike,
        normal_projector: ArrayLike,
        constraint_jacobian: ArrayLike,
        normals: ArrayLike,
        rank_margin: ArrayLike,
        log_volume: ArrayLike,
        orientation: ArrayLike,
        valid: ArrayLike,
    ):
        self.metric = jnp.asarray(metric)
        self.inverse_metric = jnp.asarray(inverse_metric)
        self.tangent_projector = jnp.asarray(tangent_projector)
        self.normal_projector = jnp.asarray(normal_projector)
        self.constraint_jacobian = jnp.asarray(constraint_jacobian)
        self.normals = jnp.asarray(normals)
        self.rank_margin = jnp.asarray(rank_margin)
        self.log_volume = jnp.asarray(log_volume)
        self.orientation = jnp.asarray(orientation)
        self.valid = jnp.asarray(valid, dtype=bool)


class RiemannianMapMeasureEvidence(StrictModule):
    ambient_point: Array
    metric: Array
    inverse_metric: Array
    tangent_projector: Array
    normal_projector: Array
    jacobian: Array
    rank_margin: Array
    log_volume: Array
    hausdorff_jacobian: Array
    orientation: Array
    valid: Array

    def __init__(
        self,
        *,
        ambient_point: ArrayLike,
        metric: ArrayLike,
        inverse_metric: ArrayLike,
        tangent_projector: ArrayLike,
        normal_projector: ArrayLike,
        jacobian: ArrayLike,
        rank_margin: ArrayLike,
        log_volume: ArrayLike,
        hausdorff_jacobian: ArrayLike,
        orientation: ArrayLike,
        valid: ArrayLike,
    ):
        self.ambient_point = jnp.asarray(ambient_point)
        self.metric = jnp.asarray(metric)
        self.inverse_metric = jnp.asarray(inverse_metric)
        self.tangent_projector = jnp.asarray(tangent_projector)
        self.normal_projector = jnp.asarray(normal_projector)
        self.jacobian = jnp.asarray(jacobian)
        self.rank_margin = jnp.asarray(rank_margin)
        self.log_volume = jnp.asarray(log_volume)
        self.hausdorff_jacobian = jnp.asarray(hausdorff_jacobian)
        self.orientation = jnp.asarray(orientation)
        self.valid = jnp.asarray(valid, dtype=bool)


class RegularLevelSetManifold(AbstractRiemannianManifold):
    """Declared regular level set ``constraint(x)=0`` in an ambient metric.

    The constraint count, ambient dimension and orientation are fixed for one smooth
    epoch. Rank loss is reported by ``local_geometry`` and never repaired.
    """

    constraint: Callable[[Array], Array]
    ambient_metric: Callable[[Array], Array]
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    codimension: int = eqx.field(static=True)
    orientation_sign: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    retraction_iterations: int = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)
    precision: GeometryPrecisionPolicy

    def __init__(
        self,
        constraint: Callable[[Array], Array],
        /,
        *,
        ambient_dimension: int,
        codimension: int,
        ambient_metric: Callable[[Array], Array] | None = None,
        orientation: int = 1,
        tolerance: float = 1e-8,
        retraction_iterations: int = 8,
        manifold_id: str = "regular-level-set",
        precision: GeometryPrecisionPolicy | None = None,
    ):
        if not callable(constraint):
            raise TypeError("constraint must be callable.")
        ambient = int(ambient_dimension)
        codim = int(codimension)
        if ambient < 1 or codim < 1 or codim >= ambient:
            raise ValueError("Require 0 < codimension < ambient_dimension.")
        orientation_ = int(orientation)
        if orientation_ not in (-1, 1):
            raise ValueError("orientation must be +1 or -1.")
        if float(tolerance) <= 0.0 or int(retraction_iterations) < 1:
            raise ValueError("tolerance and retraction_iterations must be positive.")
        identifier = str(manifold_id)
        if not identifier:
            raise ValueError("manifold_id must be non-empty.")
        self.constraint = constraint
        self.ambient_metric = (
            (lambda point: jnp.eye(ambient, dtype=point.dtype))
            if ambient_metric is None
            else ambient_metric
        )
        if not callable(self.ambient_metric):
            raise TypeError("ambient_metric must be callable.")
        self.manifold_id = identifier
        self.point_shape = (ambient,)
        self.codimension = codim
        self.orientation_sign = orientation_
        self.tolerance = float(tolerance)
        self.retraction_iterations = int(retraction_iterations)
        self.retraction_method = "fixed-metric-normal-newton"
        self.transport_method = "destination-tangent-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False
        self.precision = GeometryPrecisionPolicy() if precision is None else precision

    @property
    def scalar_field(self) -> str:
        dtype = self.precision.coordinate_dtype
        return "complex" if dtype is not None and dtype.startswith("complex") else "real"

    def _point(self, point: ArrayLike, /) -> Array:
        value = self.precision.compute(point)
        if value.shape[-1:] != self.point_shape:
            raise ValueError(f"Level-set point must end in shape {self.point_shape}.")
        return value

    def _single_geometry(self, point: Array, /) -> ManifoldTangentMeasureEvidence:
        constraint_value = jnp.asarray(self.constraint(point))
        if constraint_value.shape != (self.codimension,):
            raise ValueError("constraint must return the declared codimension.")
        jacobian = jax.jacfwd(self.constraint)(point)
        metric = jnp.asarray(self.ambient_metric(point), dtype=point.dtype)
        if metric.shape != (self.point_shape[0], self.point_shape[0]):
            raise ValueError("ambient_metric returned an incompatible shape.")
        identity = jnp.eye(self.point_shape[0], dtype=point.dtype)
        inverse_metric, metric_ok = _spd_solve(
            metric, identity, f"{self.manifold_id}:metric"
        )
        raised_normals = inverse_metric @ _adjoint(jacobian)
        normal_gram = jacobian @ raised_normals
        normal_inverse, normal_ok = _spd_solve(
            normal_gram,
            jnp.eye(self.codimension, dtype=point.dtype),
            f"{self.manifold_id}:normal-gram",
        )
        normal_projector = raised_normals @ normal_inverse @ jacobian
        tangent_projector = identity - normal_projector
        eigenvalues = jnp.linalg.eigvalsh(jnp.real(normal_gram))
        rank_margin = jnp.min(eigenvalues)
        metric_sign, metric_logdet = jnp.linalg.slogdet(metric)
        normal_sign, normal_logdet = jnp.linalg.slogdet(normal_gram)
        log_volume = 0.5 * jnp.real(metric_logdet - normal_logdet)
        finite = (
            jnp.all(jnp.isfinite(metric))
            & jnp.all(jnp.isfinite(jacobian))
            & jnp.isfinite(log_volume)
        )
        valid = (
            metric_ok
            & normal_ok
            & finite
            & (metric_sign > 0)
            & (normal_sign > 0)
            & (rank_margin > self.tolerance)
        )
        normals = _adjoint(jacobian)
        return ManifoldTangentMeasureEvidence(
            metric=metric,
            inverse_metric=inverse_metric,
            tangent_projector=tangent_projector,
            normal_projector=normal_projector,
            constraint_jacobian=jacobian,
            normals=normals,
            rank_margin=rank_margin,
            log_volume=log_volume,
            orientation=jnp.asarray(self.orientation_sign, dtype=point.real.dtype),
            valid=valid,
        )

    def local_geometry(self, point: ArrayLike, /) -> ManifoldTangentMeasureEvidence:
        value = self._point(point)
        if value.ndim == 1:
            return self._single_geometry(value)
        flat = value.reshape((-1, value.shape[-1]))
        evidence = jax.vmap(self._single_geometry)(flat)
        leading = value.shape[:-1]
        return jax.tree.map(
            lambda array: array.reshape(leading + array.shape[1:]), evidence
        )

    def contains(self, point: ArrayLike, /) -> Array:
        value = self._point(point)
        residual = jnp.max(
            jnp.abs(jax.vmap(self.constraint)(value.reshape((-1, value.shape[-1]))))
        )
        return jnp.all(self.local_geometry(value).valid) & (residual <= self.tolerance)

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        value = self._point(point)
        constraints = jax.vmap(self.constraint)(value.reshape((-1, value.shape[-1])))
        return jnp.max(jnp.abs(constraints))

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        value = self._point(point)
        vector = jnp.asarray(ambient_vector, dtype=value.dtype)
        if vector.shape != value.shape:
            raise ValueError("ambient_vector must match point shape.")
        projector = self.local_geometry(value).tangent_projector
        return contract("...ij,...j->...i", projector, vector)

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        value = self._point(point)
        cotangent = jnp.conj(jnp.asarray(ambient_cotangent, dtype=value.dtype))
        geometry = self.local_geometry(value)
        raised = contract("...ij,...j->...i", geometry.inverse_metric, cotangent)
        return contract("...ij,...j->...i", geometry.tangent_projector, raised)

    def inner(
        self, point: ArrayLike, left_tangent: ArrayLike, right_tangent: ArrayLike, /
    ) -> Array:
        value = self._point(point)
        left = self.project_tangent(value, left_tangent)
        right = self.project_tangent(value, right_tangent)
        metric = self.local_geometry(value).metric
        return jnp.real(contract("...i,...ij,...j->", jnp.conj(left), metric, right))

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        value = self._point(point)
        candidate = value + self.project_tangent(value, tangent_step)
        for _ in range(self.retraction_iterations):
            geometry = self.local_geometry(candidate)
            constraints = (
                jax.vmap(self.constraint)(
                    candidate.reshape((-1, candidate.shape[-1]))
                ).reshape(candidate.shape[:-1] + (self.codimension,))
                if candidate.ndim > 1
                else self.constraint(candidate)
            )
            raised_normals = geometry.inverse_metric @ _adjoint(
                geometry.constraint_jacobian
            )
            normal_gram = geometry.constraint_jacobian @ raised_normals
            multipliers, _ = _spd_solve(
                normal_gram,
                constraints,
                f"{self.manifold_id}:retraction-normal-gram",
            )
            correction = contract("...ij,...j->...i", raised_normals, multipliers)
            candidate = candidate - correction
        residual = self.constraint_residual(candidate)
        return eqx.error_if(
            candidate,
            (residual > self.tolerance) | ~jnp.all(self.local_geometry(candidate).valid),
            "Level-set retraction failed regularity or residual evidence.",
        )

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        del point, tangent_step
        return self.project_tangent(destination, tangent)


class ImmersedRiemannianManifoldAdapter(AbstractRiemannianManifold):
    """Declared immersion into a fixed ambient Riemannian manifold."""

    immersion: Callable[[Array], Array]
    ambient_metric: Callable[[Array], Array]
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    orientation_sign: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(
        self,
        immersion: Callable[[Array], Array],
        /,
        *,
        coordinate_dimension: int,
        ambient_dimension: int,
        ambient_metric: Callable[[Array], Array] | None = None,
        orientation: int = 1,
        rank_tolerance: float = 1e-8,
        manifold_id: str = "immersed-manifold",
    ):
        if not callable(immersion):
            raise TypeError("immersion must be callable.")
        coordinate = int(coordinate_dimension)
        ambient = int(ambient_dimension)
        if coordinate < 1 or ambient <= coordinate:
            raise ValueError("Require 0 < coordinate_dimension < ambient_dimension.")
        if int(orientation) not in (-1, 1) or float(rank_tolerance) <= 0.0:
            raise ValueError("orientation must be ±1 and rank_tolerance positive.")
        self.immersion = immersion
        self.ambient_metric = (
            (lambda point: jnp.eye(ambient, dtype=point.dtype))
            if ambient_metric is None
            else ambient_metric
        )
        self.manifold_id = str(manifold_id)
        self.point_shape = (coordinate,)
        self.ambient_dimension = ambient
        self.orientation_sign = int(orientation)
        self.rank_tolerance = float(rank_tolerance)
        self.retraction_method = "coordinate-addition"
        self.transport_method = "coordinate-identity"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    def _point(self, point: ArrayLike, /) -> Array:
        value = jnp.asarray(point)
        if value.shape[-1:] != self.point_shape:
            raise ValueError(
                f"Immersion coordinates must end in shape {self.point_shape}."
            )
        return value

    def _single_evidence(self, coordinates: Array, /) -> RiemannianMapMeasureEvidence:
        ambient = jnp.asarray(self.immersion(coordinates))
        if ambient.shape != (self.ambient_dimension,):
            raise ValueError("immersion returned an incompatible ambient shape.")
        jacobian = jax.jacfwd(self.immersion)(coordinates)
        ambient_metric = jnp.asarray(self.ambient_metric(ambient), dtype=ambient.dtype)
        pullback = _adjoint(jacobian) @ ambient_metric @ jacobian
        inverse, solved = _spd_solve(
            pullback,
            jnp.eye(self.point_shape[0], dtype=pullback.dtype),
            f"{self.manifold_id}:pullback",
        )
        tangent_projector = jacobian @ inverse @ _adjoint(jacobian) @ ambient_metric
        normal_projector = (
            jnp.eye(self.ambient_dimension, dtype=ambient.dtype) - tangent_projector
        )
        eigenvalues = jnp.linalg.eigvalsh(jnp.real(pullback))
        rank_margin = jnp.min(eigenvalues)
        sign, logdet = jnp.linalg.slogdet(pullback)
        log_volume = 0.5 * jnp.real(logdet)
        hausdorff = jnp.exp(log_volume)
        valid = (
            solved
            & (sign > 0)
            & (rank_margin > self.rank_tolerance)
            & jnp.all(jnp.isfinite(ambient))
            & jnp.isfinite(hausdorff)
        )
        return RiemannianMapMeasureEvidence(
            ambient_point=ambient,
            metric=pullback,
            inverse_metric=inverse,
            tangent_projector=tangent_projector,
            normal_projector=normal_projector,
            jacobian=jacobian,
            rank_margin=rank_margin,
            log_volume=log_volume,
            hausdorff_jacobian=hausdorff,
            orientation=jnp.asarray(self.orientation_sign, dtype=ambient.real.dtype),
            valid=valid,
        )

    def map_measure_evidence(
        self, coordinates: ArrayLike, /
    ) -> RiemannianMapMeasureEvidence:
        value = self._point(coordinates)
        if value.ndim == 1:
            return self._single_evidence(value)
        flat = value.reshape((-1, value.shape[-1]))
        evidence = jax.vmap(self._single_evidence)(flat)
        leading = value.shape[:-1]
        return jax.tree.map(
            lambda array: array.reshape(leading + array.shape[1:]), evidence
        )

    local_geometry = map_measure_evidence

    def induced_metric(self, coordinates: ArrayLike, /) -> Array:
        return self.map_measure_evidence(coordinates).metric

    def pushforward(self, coordinates: ArrayLike, tangent: ArrayLike, /) -> Array:
        evidence = self.map_measure_evidence(coordinates)
        return contract("...ai,...i->...a", evidence.jacobian, jnp.asarray(tangent))

    def pullback(self, coordinates: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        evidence = self.map_measure_evidence(coordinates)
        return contract(
            "...ai,...a->...i",
            jnp.conj(evidence.jacobian),
            jnp.asarray(ambient_cotangent),
        )

    def contains(self, point: ArrayLike, /) -> Array:
        return jnp.all(self.map_measure_evidence(point).valid)

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        evidence = self.map_measure_evidence(point)
        return jnp.maximum(self.rank_tolerance - jnp.min(evidence.rank_margin), 0.0)

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        value = self._point(point)
        vector = jnp.asarray(ambient_vector, dtype=value.dtype)
        if vector.shape != value.shape:
            raise ValueError("Coordinate tangent must match point shape.")
        return vector

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        evidence = self.map_measure_evidence(point)
        return contract(
            "...ij,...j->...i",
            evidence.inverse_metric,
            jnp.conj(jnp.asarray(ambient_cotangent)),
        )

    def inner(
        self, point: ArrayLike, left_tangent: ArrayLike, right_tangent: ArrayLike, /
    ) -> Array:
        evidence = self.map_measure_evidence(point)
        return jnp.real(
            contract(
                "...i,...ij,...j->",
                jnp.conj(jnp.asarray(left_tangent)),
                evidence.metric,
                jnp.asarray(right_tangent),
            )
        )

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        return self._point(point) + self.project_tangent(point, tangent_step)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        del point, tangent_step
        return self.project_tangent(destination, tangent)


__all__ = [
    "ImmersedRiemannianManifoldAdapter",
    "ManifoldTangentMeasureEvidence",
    "RegularLevelSetManifold",
    "RiemannianMapMeasureEvidence",
]
