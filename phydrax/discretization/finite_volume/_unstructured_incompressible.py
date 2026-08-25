#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cell_polynomial import PreparedCellPolynomialReconstruction
from ._unstructured import UnstructuredFiniteVolumeDiscretization


class UnstructuredCollocatedOperatorReport(StrictModule):
    minimum_projected_distance: Array
    maximum_projected_distance: Array
    maximum_nonorthogonality_degrees: Array


class PreparedUnstructuredCollocatedOperators(StrictModule, NonTrainableState):
    """Geometry-only collocated divergence, gradient, gauge, and Rhie--Chow actions."""

    discretization: UnstructuredFiniteVolumeDiscretization
    gradient: PreparedCellPolynomialReconstruction
    unit_normals: Array
    projected_distances: Array
    interior_faces: Array
    report: UnstructuredCollocatedOperatorReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        gradient: PreparedCellPolynomialReconstruction,
        /,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Collocated operators require unstructured FV geometry.")
        if not isinstance(gradient, PreparedCellPolynomialReconstruction):
            raise TypeError("gradient must be PreparedCellPolynomialReconstruction.")
        if gradient.basis.degree != 1:
            raise ValueError("Collocated operators require a degree-one gradient.")
        if gradient.discretization.prepared_id != discretization.prepared_id:
            raise ValueError("Collocated gradient belongs to a different geometry.")
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        interior = neighbour >= 0
        safe_neighbour = jnp.maximum(neighbour, 0)
        normals = discretization.area_vectors / discretization.face_measures[:, None]
        connector = (
            discretization.cell_centers[safe_neighbour]
            - discretization.cell_centers[owner]
        )
        owner_to_face = discretization.face_centers - discretization.cell_centers[owner]
        interior_distance = jnp.sum(connector * normals, axis=-1)
        boundary_distance = 2.0 * jnp.sum(owner_to_face * normals, axis=-1)
        distance = jnp.where(interior, interior_distance, boundary_distance)
        distance = eqx.error_if(
            distance,
            jnp.any(~jnp.isfinite(distance) | (distance <= 0.0)),
            "Collocated operators require positive normal-projected distances.",
        )
        connector_norm = jnp.linalg.norm(connector, axis=-1)
        cosine = jnp.sum(connector * normals, axis=-1) / jnp.where(
            connector_norm > 0.0, connector_norm, 1.0
        )
        angle = jnp.degrees(jnp.arccos(jnp.clip(cosine, 0.0, 1.0)))
        self.discretization = discretization
        self.gradient = gradient
        self.unit_normals = normals
        self.projected_distances = distance
        self.interior_faces = interior
        self.report = UnstructuredCollocatedOperatorReport(
            minimum_projected_distance=jnp.min(distance),
            maximum_projected_distance=jnp.max(distance),
            maximum_nonorthogonality_degrees=jnp.max(jnp.where(interior, angle, 0.0)),
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-collocated-operators",
                "geometry": discretization.prepared_id,
                "gradient": gradient.prepared_id,
            }
        )

    def validate_cell_scalar(self, value: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != (self.discretization.cell_count,):
            raise ValueError(
                f"{name} must have shape {(self.discretization.cell_count,)}."
            )
        return array

    def validate_cell_velocity(self, velocity: ArrayLike, /) -> Array:
        value = jnp.asarray(velocity)
        shape = (
            self.discretization.cell_count,
            self.discretization.cell_dimension,
        )
        if value.shape != shape:
            raise ValueError(f"Cell velocity must have shape {shape}.")
        return value

    def validate_face_scalar(self, value: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(value)
        shape = (self.discretization.face_measures.size,)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        return array

    def gauge_project(self, pressure: ArrayLike, /) -> Array:
        value = self.validate_cell_scalar(pressure, "Pressure")
        volumes = self.discretization.cell_volumes.astype(value.dtype)
        mean = jnp.sum(volumes * value) / jnp.sum(volumes)
        return value - mean

    def cell_gradient(self, pressure: ArrayLike, /) -> Array:
        value = self.validate_cell_scalar(pressure, "Pressure")
        coefficients = self.gradient.coefficients(value)
        lengths = self.gradient.characteristic_lengths.astype(value.dtype)
        return coefficients / lengths[:, None]

    def face_normal_gradient(self, pressure: ArrayLike, /) -> Array:
        value = self.validate_cell_scalar(pressure, "Pressure")
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        distance = self.projected_distances.astype(value.dtype)
        interior_gradient = (value[safe_neighbour] - value[owner]) / distance
        return jnp.where(self.interior_faces, interior_gradient, 0.0)

    def divergence(self, face_normal_velocity: ArrayLike, /) -> Array:
        velocity = self.validate_face_scalar(face_normal_velocity, "Face-normal velocity")
        measures = self.discretization.face_measures.astype(velocity.dtype)
        volumes = self.discretization.cell_volumes.astype(velocity.dtype)
        integrated = velocity * measures
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        divergence = jnp.zeros((self.discretization.cell_count,), dtype=velocity.dtype)
        divergence = divergence.at[owner].add(integrated)
        divergence = divergence.at[safe_neighbour].add(
            jnp.where(self.interior_faces, -integrated, 0.0)
        )
        return divergence / volumes

    def interpolate_normal_velocity(self, velocity: ArrayLike, /) -> Array:
        value = self.validate_cell_velocity(velocity)
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        average = 0.5 * (value[owner] + value[safe_neighbour])
        average = jnp.where(self.interior_faces[:, None], average, value[owner])
        return jnp.sum(average * self.unit_normals.astype(value.dtype), axis=-1)

    def interpolate_inverse_momentum(
        self, inverse_momentum_diagonal: ArrayLike, /
    ) -> Array:
        inverse = self.validate_cell_scalar(
            inverse_momentum_diagonal, "Inverse momentum diagonal"
        )
        inverse = eqx.error_if(
            inverse,
            jnp.any(~jnp.isfinite(inverse) | (inverse <= 0.0)),
            "Inverse momentum diagonal must be positive and finite.",
        )
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        average = 0.5 * (inverse[owner] + inverse[safe_neighbour])
        return jnp.where(self.interior_faces, average, inverse[owner])

    def rhie_chow_face_velocity(
        self,
        velocity: ArrayLike,
        pressure: ArrayLike,
        inverse_momentum_diagonal: ArrayLike,
        /,
    ) -> Array:
        value = self.validate_cell_velocity(velocity)
        pressure_ = self.validate_cell_scalar(pressure, "Pressure")
        face_inverse = self.interpolate_inverse_momentum(inverse_momentum_diagonal)
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        cell_gradient = self.cell_gradient(pressure_)
        average_gradient = 0.5 * (cell_gradient[owner] + cell_gradient[safe_neighbour])
        interpolated_normal_gradient = jnp.sum(
            average_gradient * self.unit_normals.astype(pressure_.dtype), axis=-1
        )
        two_point_gradient = self.face_normal_gradient(pressure_)
        correction = face_inverse * (two_point_gradient - interpolated_normal_gradient)
        correction = jnp.where(self.interior_faces, correction, 0.0)
        return self.interpolate_normal_velocity(value) - correction

    def laplacian(self, pressure: ArrayLike, /) -> Array:
        return self.divergence(self.face_normal_gradient(pressure))

    def weighted_laplacian(
        self, pressure: ArrayLike, face_inverse_momentum: ArrayLike, /
    ) -> Array:
        inverse = self.validate_face_scalar(
            face_inverse_momentum, "Face inverse momentum"
        )
        inverse = eqx.error_if(
            inverse,
            jnp.any(~jnp.isfinite(inverse) | (inverse <= 0.0)),
            "Face inverse momentum must be positive and finite.",
        )
        return self.divergence(inverse * self.face_normal_gradient(pressure))

    def positive_gauged_weighted_laplacian(
        self, pressure: ArrayLike, face_inverse_momentum: ArrayLike, /
    ) -> Array:
        value = self.validate_cell_scalar(pressure, "Pressure")
        volumes = self.discretization.cell_volumes.astype(value.dtype)
        mean = jnp.sum(volumes * value) / jnp.sum(volumes)
        projected = value - mean
        return -self.weighted_laplacian(projected, face_inverse_momentum) + mean

    def positive_gauged_laplacian(self, pressure: ArrayLike, /) -> Array:
        value = self.validate_cell_scalar(pressure, "Pressure")
        unit_inverse = jnp.ones(
            (self.discretization.face_measures.size,), dtype=value.dtype
        )
        return self.positive_gauged_weighted_laplacian(value, unit_inverse)


__all__ = [
    "PreparedUnstructuredCollocatedOperators",
    "UnstructuredCollocatedOperatorReport",
]
