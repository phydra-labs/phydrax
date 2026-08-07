#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule


def _basis_functions(parameter: Array, knots: Array, degree: int) -> Array:
    parameter_ = jnp.asarray(parameter, dtype=knots.dtype)
    basis = ((parameter_ >= knots[:-1]) & (parameter_ < knots[1:])).astype(knots.dtype)
    for order in range(1, degree + 1):
        output_count = basis.shape[0] - 1
        left_denominator = knots[order : order + output_count] - knots[:output_count]
        right_denominator = (
            knots[order + 1 : order + 1 + output_count] - knots[1 : 1 + output_count]
        )
        left = jnp.where(
            left_denominator != 0.0,
            (parameter_ - knots[:output_count]) / left_denominator * basis[:output_count],
            0.0,
        )
        right = jnp.where(
            right_denominator != 0.0,
            (knots[order + 1 : order + 1 + output_count] - parameter_)
            / right_denominator
            * basis[1 : output_count + 1],
            0.0,
        )
        basis = left + right
    basis_count = knots.shape[0] - degree - 1
    endpoint = jnp.isclose(parameter_, knots[-1])
    return jnp.where(
        endpoint,
        jax.nn.one_hot(basis_count - 1, basis_count, dtype=knots.dtype),
        basis,
    )


class AbstractSurfacePatch(StrictModule):
    """Pure-JAX parametric surface patch in native CAD coordinates."""

    @abstractmethod
    def evaluate(self, parameters: Array, /) -> Array:
        raise NotImplementedError


class PlanePatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array

    def __init__(self, origin: Array, first_axis: Array, second_axis: Array):
        self.origin = jnp.asarray(origin, dtype=float).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=float).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=float).reshape((3,))

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        return (
            self.origin
            + parameters_[..., :1] * self.first_axis
            + parameters_[..., 1:2] * self.second_axis
        )


class CylinderPatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    radius: Array

    def __init__(self, origin, first_axis, second_axis, axis, radius):
        self.origin = jnp.asarray(origin, dtype=float).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=float).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=float).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=float).reshape((3,))
        self.radius = jnp.asarray(radius, dtype=float).reshape(())

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        angle = parameters_[..., 0]
        height = parameters_[..., 1]
        radial = (
            jnp.cos(angle)[..., None] * self.first_axis
            + jnp.sin(angle)[..., None] * self.second_axis
        )
        return self.origin + self.radius * radial + height[..., None] * self.axis


class ConePatch(AbstractSurfacePatch):
    origin: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    reference_radius: Array
    semi_angle: Array

    def __init__(
        self,
        origin,
        first_axis,
        second_axis,
        axis,
        reference_radius,
        semi_angle,
    ):
        self.origin = jnp.asarray(origin, dtype=float).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=float).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=float).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=float).reshape((3,))
        self.reference_radius = jnp.asarray(reference_radius, dtype=float).reshape(())
        self.semi_angle = jnp.asarray(semi_angle, dtype=float).reshape(())

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.origin.dtype)
        angle = parameters_[..., 0]
        axial = parameters_[..., 1]
        radius = self.reference_radius + axial * jnp.tan(self.semi_angle)
        radial = (
            jnp.cos(angle)[..., None] * self.first_axis
            + jnp.sin(angle)[..., None] * self.second_axis
        )
        return self.origin + axial[..., None] * self.axis + radius[..., None] * radial


class SpherePatch(AbstractSurfacePatch):
    center: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    radius: Array

    def __init__(self, center, first_axis, second_axis, axis, radius):
        self.center = jnp.asarray(center, dtype=float).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=float).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=float).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=float).reshape((3,))
        self.radius = jnp.asarray(radius, dtype=float).reshape(())

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.center.dtype)
        longitude = parameters_[..., 0]
        latitude = parameters_[..., 1]
        equatorial = (
            jnp.cos(longitude)[..., None] * self.first_axis
            + jnp.sin(longitude)[..., None] * self.second_axis
        )
        direction = (
            jnp.cos(latitude)[..., None] * equatorial
            + jnp.sin(latitude)[..., None] * self.axis
        )
        return self.center + self.radius * direction


class TorusPatch(AbstractSurfacePatch):
    center: Array
    first_axis: Array
    second_axis: Array
    axis: Array
    major_radius: Array
    minor_radius: Array

    def __init__(
        self,
        center,
        first_axis,
        second_axis,
        axis,
        major_radius,
        minor_radius,
    ):
        self.center = jnp.asarray(center, dtype=float).reshape((3,))
        self.first_axis = jnp.asarray(first_axis, dtype=float).reshape((3,))
        self.second_axis = jnp.asarray(second_axis, dtype=float).reshape((3,))
        self.axis = jnp.asarray(axis, dtype=float).reshape((3,))
        self.major_radius = jnp.asarray(major_radius, dtype=float).reshape(())
        self.minor_radius = jnp.asarray(minor_radius, dtype=float).reshape(())

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.center.dtype)
        longitude = parameters_[..., 0]
        tube_angle = parameters_[..., 1]
        radial = (
            jnp.cos(longitude)[..., None] * self.first_axis
            + jnp.sin(longitude)[..., None] * self.second_axis
        )
        ring_radius = self.major_radius + self.minor_radius * jnp.cos(tube_angle)
        return (
            self.center
            + ring_radius[..., None] * radial
            + self.minor_radius * jnp.sin(tube_angle)[..., None] * self.axis
        )


class BSplineSurfacePatch(AbstractSurfacePatch):
    """Tensor-product rational B-spline surface with expanded knot vectors."""

    control_points: Array
    weights: Array
    u_knots: Array
    v_knots: Array
    u_degree: int = eqx.field(static=True)
    v_degree: int = eqx.field(static=True)

    def __init__(
        self,
        control_points: Array,
        weights: Array,
        u_knots: Array,
        v_knots: Array,
        u_degree: int,
        v_degree: int,
    ):
        control_points_ = jnp.asarray(control_points, dtype=float)
        weights_ = jnp.asarray(weights, dtype=float)
        u_knots_ = jnp.asarray(u_knots, dtype=float).reshape((-1,))
        v_knots_ = jnp.asarray(v_knots, dtype=float).reshape((-1,))
        if control_points_.ndim != 3 or control_points_.shape[-1] != 3:
            raise ValueError("control_points must have shape (num_u, num_v, 3).")
        if weights_.shape != control_points_.shape[:2]:
            raise ValueError("weights must match the control-point grid.")
        if int(u_degree) < 1 or int(v_degree) < 1:
            raise ValueError("B-spline degrees must be positive.")
        if u_knots_.shape[0] != control_points_.shape[0] + int(u_degree) + 1:
            raise ValueError(
                "u_knots length is inconsistent with control points and degree."
            )
        if v_knots_.shape[0] != control_points_.shape[1] + int(v_degree) + 1:
            raise ValueError(
                "v_knots length is inconsistent with control points and degree."
            )
        self.control_points = control_points_
        self.weights = weights_
        self.u_knots = u_knots_
        self.v_knots = v_knots_
        self.u_degree = int(u_degree)
        self.v_degree = int(v_degree)

    def _evaluate_one(self, parameters: Array) -> Array:
        u_basis = _basis_functions(parameters[0], self.u_knots, self.u_degree)
        v_basis = _basis_functions(parameters[1], self.v_knots, self.v_degree)
        coefficients = u_basis[:, None] * v_basis[None, :] * self.weights
        denominator = jnp.sum(coefficients)
        return (
            jnp.sum(coefficients[..., None] * self.control_points, axis=(0, 1))
            / denominator
        )

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.control_points.dtype)
        leading = parameters_.shape[:-1]
        values = jax.vmap(self._evaluate_one)(parameters_.reshape((-1, 2)))
        return values.reshape((*leading, 3))


class BSplineCurve(StrictModule):
    """Rational B-spline curve used for exact parametric trim evaluation."""

    control_points: Array
    weights: Array
    knots: Array
    degree: int = eqx.field(static=True)

    def __init__(self, control_points, weights, knots, degree):
        control_points_ = jnp.asarray(control_points, dtype=float)
        weights_ = jnp.asarray(weights, dtype=float).reshape((-1,))
        knots_ = jnp.asarray(knots, dtype=float).reshape((-1,))
        if control_points_.ndim != 2 or control_points_.shape[0] != weights_.shape[0]:
            raise ValueError("Curve control points and weights are inconsistent.")
        if knots_.shape[0] != control_points_.shape[0] + int(degree) + 1:
            raise ValueError("Curve knot vector length is inconsistent.")
        self.control_points = control_points_
        self.weights = weights_
        self.knots = knots_
        self.degree = int(degree)

    def _evaluate_one(self, parameter: Array) -> Array:
        basis = _basis_functions(parameter, self.knots, self.degree)
        coefficient = basis * self.weights
        return jnp.sum(coefficient[:, None] * self.control_points, axis=0) / jnp.sum(
            coefficient
        )

    def evaluate(self, parameters: Array, /) -> Array:
        parameters_ = jnp.asarray(parameters, dtype=self.control_points.dtype)
        return jax.vmap(self._evaluate_one)(parameters_.reshape((-1,))).reshape(
            (*parameters_.shape, self.control_points.shape[1])
        )


SurfacePatch: TypeAlias = (
    PlanePatch
    | CylinderPatch
    | ConePatch
    | SpherePatch
    | TorusPatch
    | BSplineSurfacePatch
)


def surface_differential(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    parameters_ = jnp.asarray(parameters, dtype=float)
    leading = parameters_.shape[:-1]
    values = jax.vmap(jax.jacfwd(lambda uv: patch.evaluate(uv)))(
        parameters_.reshape((-1, 2))
    )
    return values.reshape((*leading, 3, 2))


def surface_jacobian(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    differential = surface_differential(patch, parameters)
    return jnp.linalg.norm(
        jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
    )


def surface_normal(patch: AbstractSurfacePatch, parameters: Array, /) -> Array:
    differential = surface_differential(patch, parameters)
    normal = jnp.cross(differential[..., :, 0], differential[..., :, 1])
    return normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)


__all__ = [
    "AbstractSurfacePatch",
    "BSplineCurve",
    "BSplineSurfacePatch",
    "ConePatch",
    "CylinderPatch",
    "PlanePatch",
    "SpherePatch",
    "SurfacePatch",
    "TorusPatch",
    "surface_differential",
    "surface_jacobian",
    "surface_normal",
]
