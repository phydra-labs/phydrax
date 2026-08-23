#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._hypersurface import ProjectiveHypersurface


class HypersurfacePatchEvaluation(StrictModule):
    affine_coordinates: Array
    tangent_basis: Array
    induced_metric: Array
    polynomial_residual: Array
    smoothness_margin: Array
    residue_coefficient: Array
    valid: Array
    chart_index: int
    pivot_index: int

    def __init__(
        self,
        *,
        affine_coordinates: ArrayLike,
        tangent_basis: ArrayLike,
        induced_metric: ArrayLike,
        polynomial_residual: ArrayLike,
        smoothness_margin: ArrayLike,
        residue_coefficient: ArrayLike,
        valid: ArrayLike,
        chart_index: int,
        pivot_index: int,
    ):
        self.affine_coordinates = jnp.asarray(affine_coordinates)
        self.tangent_basis = jnp.asarray(tangent_basis)
        self.induced_metric = jnp.asarray(induced_metric)
        self.polynomial_residual = jnp.asarray(polynomial_residual)
        self.smoothness_margin = jnp.asarray(smoothness_margin)
        self.residue_coefficient = jnp.asarray(residue_coefficient)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.chart_index = int(chart_index)
        self.pivot_index = int(pivot_index)


class HypersurfacePatchGeometry(StrictModule):
    hypersurface: ProjectiveHypersurface
    tolerance: float

    def __init__(
        self, hypersurface: ProjectiveHypersurface, /, *, tolerance: float = 1e-9
    ):
        if not isinstance(hypersurface, ProjectiveHypersurface):
            raise TypeError("hypersurface must be a ProjectiveHypersurface.")
        self.hypersurface = hypersurface
        self.tolerance = float(tolerance)

    def evaluate(
        self,
        homogeneous_point: ArrayLike,
        /,
        *,
        chart_index: int | None = None,
        pivot_index: int | None = None,
    ) -> HypersurfacePatchEvaluation:
        point = jnp.asarray(homogeneous_point)
        expected = (self.hypersurface.projective_dimension + 1,)
        if point.shape != expected:
            raise ValueError(f"Homogeneous point must have shape {expected}.")
        owner = (
            int(jnp.argmax(jnp.abs(point))) if chart_index is None else int(chart_index)
        )
        scaled = point / point[owner]
        affine_axes = tuple(index for index in range(point.shape[0]) if index != owner)
        affine_complex = scaled[jnp.asarray(affine_axes)]
        convention = self.hypersurface.atlas.conventions[owner]
        affine = convention.to_real(affine_complex)

        def local_polynomial(local_real: Array) -> Array:
            return self.hypersurface.local_polynomial(owner, local_real)

        jacobian_complex = jax.jacfwd(local_polynomial)(affine)
        real_jacobian = jnp.stack(
            (jnp.real(jacobian_complex), jnp.imag(jacobian_complex)), axis=0
        )
        _, singular_values, vh = jnp.linalg.svd(real_jacobian, full_matrices=True)
        tangent = jnp.swapaxes(vh[2:, :], -1, -2)
        ambient_metric = self.hypersurface.atlas.metric(owner)(affine)
        induced = tangent.T @ ambient_metric @ tangent
        complex_gradient = jax.jacfwd(local_polynomial)(affine)
        complex_axis_gradient = (
            complex_gradient[: convention.complex_dimension]
            - 1j * complex_gradient[convention.complex_dimension :]
        ) / 2.0
        pivot = (
            int(jnp.argmax(jnp.abs(complex_axis_gradient)))
            if pivot_index is None
            else int(pivot_index)
        )
        denominator = complex_axis_gradient[pivot]
        residue = ((-1) ** pivot) / denominator
        polynomial_residual = jnp.abs(local_polynomial(affine))
        smoothness = jnp.min(singular_values)
        minimum_metric = jnp.min(jnp.linalg.eigvalsh(induced))
        valid = (
            jnp.all(jnp.isfinite(point))
            & jnp.isfinite(polynomial_residual)
            & (polynomial_residual <= self.tolerance)
            & (smoothness > self.tolerance)
            & (jnp.abs(denominator) > self.tolerance)
            & (minimum_metric > self.tolerance)
        )
        return HypersurfacePatchEvaluation(
            affine_coordinates=affine,
            tangent_basis=tangent,
            induced_metric=induced,
            polynomial_residual=polynomial_residual,
            smoothness_margin=smoothness,
            residue_coefficient=residue,
            valid=valid,
            chart_index=owner,
            pivot_index=pivot,
        )


class ResidueCanonicalSection(StrictModule):
    geometry: HypersurfacePatchGeometry

    def __init__(self, geometry: HypersurfacePatchGeometry, /):
        self.geometry = geometry

    def coefficient(
        self,
        homogeneous_point: ArrayLike,
        /,
        *,
        chart_index: int | None = None,
        pivot_index: int | None = None,
    ) -> Array:
        return self.geometry.evaluate(
            homogeneous_point,
            chart_index=chart_index,
            pivot_index=pivot_index,
        ).residue_coefficient

    def transition_residual(
        self,
        homogeneous_point: ArrayLike,
        left_chart: int,
        right_chart: int,
        /,
    ) -> Array:
        left = self.geometry.evaluate(homogeneous_point, chart_index=left_chart)
        right = self.geometry.evaluate(homogeneous_point, chart_index=right_chart)
        # Compare induced canonical volume magnitudes; phase/orientation remains explicit.
        left_volume = jnp.abs(left.residue_coefficient) ** 2 / jnp.linalg.det(
            left.induced_metric
        )
        right_volume = jnp.abs(right.residue_coefficient) ** 2 / jnp.linalg.det(
            right.induced_metric
        )
        return jnp.abs(jnp.log(left_volume) - jnp.log(right_volume))


__all__ = [
    "HypersurfacePatchEvaluation",
    "HypersurfacePatchGeometry",
    "ResidueCanonicalSection",
]
