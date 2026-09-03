#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleSetPlan
from ._rigid_body import PreparedRigidBodySet, RigidBodySetPlan


_FACTOR_COMPONENTS = 6
_SPATIAL_DIMENSION = 3


def _finite_array(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _stable_softplus(value: np.ndarray, /) -> np.ndarray:
    return np.logaddexp(0.0, value)


def _inverse_softplus(value: np.ndarray, /) -> np.ndarray:
    if np.any(value <= 0.0) or not np.all(np.isfinite(value)):
        raise ValueError("Positive factor values are outside the coordinate image.")
    return value + np.log(-np.expm1(-value))


def _lower_triangular_factors(coordinates: np.ndarray, /) -> np.ndarray:
    count = coordinates.shape[0]
    factors = np.zeros(
        (count, _SPATIAL_DIMENSION, _SPATIAL_DIMENSION),
        dtype=coordinates.dtype,
    )
    diagonal = _stable_softplus(coordinates[:, (0, 2, 5)])
    factors[:, 0, 0] = diagonal[:, 0]
    factors[:, 1, 0] = coordinates[:, 1]
    factors[:, 1, 1] = diagonal[:, 1]
    factors[:, 2, 0] = coordinates[:, 3]
    factors[:, 2, 1] = coordinates[:, 4]
    factors[:, 2, 2] = diagonal[:, 2]
    return factors


def _factor_coordinates(factors: np.ndarray, /) -> np.ndarray:
    return np.stack(
        (
            _inverse_softplus(factors[:, 0, 0]),
            factors[:, 1, 0],
            _inverse_softplus(factors[:, 1, 1]),
            factors[:, 2, 0],
            factors[:, 2, 1],
            _inverse_softplus(factors[:, 2, 2]),
        ),
        axis=-1,
    )


def _symmetric_eigenvalues(value: np.ndarray, /) -> np.ndarray:
    return np.linalg.eigvalsh(0.5 * (value + np.swapaxes(value, -1, -2)))


def _matrix_condition_numbers(eigenvalues: np.ndarray, /) -> np.ndarray:
    return eigenvalues[:, -1] / eigenvalues[:, 0]


class RigidInertialCoordinates(StrictModule):
    """Finite unconstrained coordinates bound to one inertial parameterization."""

    mass_coordinates: Array
    center_of_mass_offsets: Array
    covariance_coordinates: Array
    parameterization_id: str = eqx.field(static=True)
    coordinates_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass_coordinates: ArrayLike,
        center_of_mass_offsets: ArrayLike,
        covariance_coordinates: ArrayLike,
        /,
        *,
        parameterization_id: str,
    ):
        mass = _finite_array(mass_coordinates, "mass_coordinates")
        offsets = _finite_array(center_of_mass_offsets, "center_of_mass_offsets")
        covariance = _finite_array(covariance_coordinates, "covariance_coordinates")
        if mass.ndim != 1 or mass.size == 0:
            raise ValueError("mass_coordinates must be a nonempty rank-1 array.")
        count = mass.size
        if offsets.shape != (count, _SPATIAL_DIMENSION):
            raise ValueError("center_of_mass_offsets must have shape (N,3).")
        if covariance.shape != (count, _FACTOR_COMPONENTS):
            raise ValueError("covariance_coordinates must have shape (N,6).")
        identifier = str(parameterization_id).strip()
        if not identifier:
            raise ValueError("parameterization_id must be nonempty.")
        dtype = np.result_type(mass.dtype, offsets.dtype, covariance.dtype)
        mass = mass.astype(dtype, copy=False)
        offsets = offsets.astype(dtype, copy=False)
        covariance = covariance.astype(dtype, copy=False)
        self.mass_coordinates = jnp.asarray(mass)
        self.center_of_mass_offsets = jnp.asarray(offsets)
        self.covariance_coordinates = jnp.asarray(covariance)
        self.parameterization_id = identifier
        self.coordinates_id = canonical_fingerprint(
            {
                "kind": "rigid-inertial-coordinates",
                "parameterization": identifier,
                "values": array_tree_fingerprint(
                    {
                        "mass": mass,
                        "center_of_mass_offsets": offsets,
                        "covariance": covariance,
                    }
                ),
            }
        )


class RigidInertialParameters(StrictModule, NonTrainableState):
    """Positive mass, explicit COM, and realizable COM/body-origin inertia."""

    masses: Array
    center_of_mass_offsets: Array
    inertia_com: Array
    inertia_body_origin: Array
    pseudo_inertia_body_origin: Array
    parameterization_id: str = eqx.field(static=True)
    coordinates_id: str = eqx.field(static=True)
    parameters_id: str = eqx.field(static=True)

    def __init__(
        self,
        masses: ArrayLike,
        center_of_mass_offsets: ArrayLike,
        inertia_com: ArrayLike,
        /,
        *,
        parameterization_id: str,
        coordinates_id: str,
    ):
        mass = _finite_array(masses, "masses")
        offsets = _finite_array(center_of_mass_offsets, "center_of_mass_offsets")
        inertia = _finite_array(inertia_com, "inertia_com")
        dtype = np.result_type(mass.dtype, offsets.dtype, inertia.dtype)
        mass = mass.astype(dtype, copy=False)
        offsets = offsets.astype(dtype, copy=False)
        inertia = inertia.astype(dtype, copy=False)
        if mass.ndim != 1 or mass.size == 0:
            raise ValueError("masses must be a nonempty rank-1 array.")
        count = mass.size
        if offsets.shape != (count, _SPATIAL_DIMENSION):
            raise ValueError("center_of_mass_offsets must have shape (N,3).")
        if inertia.shape != (count, _SPATIAL_DIMENSION, _SPATIAL_DIMENSION):
            raise ValueError("inertia_com must have shape (N,3,3).")
        if np.any(mass <= 0.0):
            raise ValueError("masses must be strictly positive.")
        if not np.allclose(inertia, np.swapaxes(inertia, -1, -2)):
            raise ValueError("inertia_com must be symmetric.")
        inertia = 0.5 * (inertia + np.swapaxes(inertia, -1, -2))
        identity = np.eye(_SPATIAL_DIMENSION, dtype=inertia.dtype)
        traces = np.trace(inertia, axis1=-2, axis2=-1)
        central_second_moment = 0.5 * traces[:, None, None] * identity - inertia
        inertia_eigenvalues = _symmetric_eigenvalues(inertia)
        central_eigenvalues = _symmetric_eigenvalues(central_second_moment)
        if np.any(inertia_eigenvalues <= 0.0) or np.any(
            central_eigenvalues <= 0.0
        ):
            raise ValueError(
                "inertia_com must be SPD and obey strict principal-moment "
                "triangle inequalities."
            )
        outer_offsets = offsets[:, :, None] * offsets[:, None, :]
        squared_offsets = np.sum(offsets * offsets, axis=-1)
        parallel_axis = mass[:, None, None] * (
            squared_offsets[:, None, None] * identity - outer_offsets
        )
        inertia_body_origin = inertia + parallel_axis
        spatial_second_moment = (
            central_second_moment + mass[:, None, None] * outer_offsets
        )
        first_moment = mass[:, None] * offsets
        pseudo_inertia = np.zeros((count, 4, 4), dtype=inertia.dtype)
        pseudo_inertia[:, :3, :3] = spatial_second_moment
        pseudo_inertia[:, :3, 3] = first_moment
        pseudo_inertia[:, 3, :3] = first_moment
        pseudo_inertia[:, 3, 3] = mass
        if not np.all(np.isfinite(inertia_body_origin)) or not np.all(
            np.isfinite(pseudo_inertia)
        ):
            raise ValueError(
                "Realized inertial parameters exceed finite numeric range."
            )
        body_eigenvalues = _symmetric_eigenvalues(inertia_body_origin)
        body_triangle_margin = (
            np.trace(inertia_body_origin, axis1=-2, axis2=-1)
            - 2.0 * body_eigenvalues[:, -1]
        )
        pseudo_eigenvalues = _symmetric_eigenvalues(pseudo_inertia)
        if (
            np.any(body_eigenvalues <= 0.0)
            or np.any(body_triangle_margin <= 0.0)
            or np.any(pseudo_eigenvalues <= 0.0)
        ):
            raise ValueError(
                "Realized body-origin inertia must be physically realizable and "
                "pseudo-inertia must be SPD."
            )
        parameterization = str(parameterization_id).strip()
        coordinates = str(coordinates_id).strip()
        if not parameterization or not coordinates:
            raise ValueError("Parameter and coordinate identifiers must be nonempty.")
        inertia_body_origin = inertia_body_origin.astype(dtype, copy=False)
        pseudo_inertia = pseudo_inertia.astype(dtype, copy=False)
        self.masses = jnp.asarray(mass)
        self.center_of_mass_offsets = jnp.asarray(offsets)
        self.inertia_com = jnp.asarray(inertia)
        self.inertia_body_origin = jnp.asarray(inertia_body_origin)
        self.pseudo_inertia_body_origin = jnp.asarray(pseudo_inertia)
        self.parameterization_id = parameterization
        self.coordinates_id = coordinates
        self.parameters_id = canonical_fingerprint(
            {
                "kind": "rigid-inertial-parameters",
                "parameterization": parameterization,
                "coordinates": coordinates,
                "values": array_tree_fingerprint(
                    {
                        "masses": mass,
                        "center_of_mass_offsets": offsets,
                        "inertia_com": inertia,
                        "inertia_body_origin": inertia_body_origin,
                        "pseudo_inertia_body_origin": pseudo_inertia,
                    }
                ),
            }
        )

    @property
    def inertia_body(self) -> Array:
        """Body-origin inertia passed to the existing maximal rigid-body owner."""
        return self.inertia_body_origin


class RigidInertialEvaluation(StrictModule, NonTrainableState):
    """Validity, reconstruction, conditioning, and source-distance evidence."""

    parameters: RigidInertialParameters
    finite_mask: Array
    positive_mass_mask: Array
    inertia_spd_mask: Array
    triangle_inequality_mask: Array
    pseudo_inertia_spd_mask: Array
    body_origin_inertia_spd_mask: Array
    body_origin_triangle_inequality_mask: Array
    minimum_inertia_eigenvalue: Array
    minimum_triangle_margin: Array
    minimum_body_origin_inertia_eigenvalue: Array
    minimum_body_origin_triangle_margin: Array
    inertia_condition_number: Array
    pseudo_inertia_condition_number: Array
    body_origin_inertia_condition_number: Array
    mass_reconstruction_residual: Array
    center_of_mass_reconstruction_residual: Array
    inertia_reconstruction_residual: Array
    body_origin_reconstruction_residual: Array
    source_mass_residual: Array
    source_inertia_residual: Array
    valid: Array
    source_prepared_id: str = eqx.field(static=True)
    requires_repreparation: bool = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: RigidInertialParameters,
        source: PreparedRigidBodySet,
        /,
    ):
        if not isinstance(parameters, RigidInertialParameters):
            raise TypeError("parameters must be RigidInertialParameters.")
        if not isinstance(source, PreparedRigidBodySet):
            raise TypeError("source must be a PreparedRigidBodySet.")
        masses = np.asarray(parameters.masses)
        offsets = np.asarray(parameters.center_of_mass_offsets)
        inertia_com = np.asarray(parameters.inertia_com)
        inertia_body = np.asarray(parameters.inertia_body_origin)
        pseudo = np.asarray(parameters.pseudo_inertia_body_origin)
        count = masses.size
        identity = np.eye(_SPATIAL_DIMENSION, dtype=inertia_com.dtype)
        inertia_eigenvalues = _symmetric_eigenvalues(inertia_com)
        triangle_margin = np.trace(inertia_com, axis1=-2, axis2=-1) - 2.0 * np.max(
            inertia_eigenvalues, axis=-1
        )
        body_eigenvalues = _symmetric_eigenvalues(inertia_body)
        body_triangle_margin = (
            np.trace(inertia_body, axis1=-2, axis2=-1)
            - 2.0 * body_eigenvalues[:, -1]
        )
        pseudo_eigenvalues = _symmetric_eigenvalues(pseudo)
        reconstructed_masses = pseudo[:, 3, 3]
        reconstructed_offsets = pseudo[:, :3, 3] / reconstructed_masses[:, None]
        spatial_second_moment = pseudo[:, :3, :3]
        central_second_moment = spatial_second_moment - (
            pseudo[:, :3, 3, None] * pseudo[:, None, 3, :3]
        ) / reconstructed_masses[:, None, None]
        reconstructed_inertia_com = (
            np.trace(central_second_moment, axis1=-2, axis2=-1)[:, None, None]
            * identity
            - central_second_moment
        )
        reconstructed_inertia_body = (
            np.trace(spatial_second_moment, axis1=-2, axis2=-1)[:, None, None]
            * identity
            - spatial_second_moment
        )
        finite = (
            np.isfinite(masses)
            & np.all(np.isfinite(offsets), axis=-1)
            & np.all(np.isfinite(inertia_com), axis=(-2, -1))
            & np.all(np.isfinite(inertia_body), axis=(-2, -1))
            & np.all(np.isfinite(pseudo), axis=(-2, -1))
        )
        positive_mass = masses > 0.0
        inertia_spd = inertia_eigenvalues[:, 0] > 0.0
        triangle = triangle_margin > 0.0
        pseudo_spd = pseudo_eigenvalues[:, 0] > 0.0
        body_inertia_spd = body_eigenvalues[:, 0] > 0.0
        body_triangle = body_triangle_margin > 0.0
        mass_residual = np.abs(reconstructed_masses - masses)
        offset_residual = np.linalg.norm(reconstructed_offsets - offsets, axis=-1)
        inertia_residual = np.linalg.norm(
            reconstructed_inertia_com - inertia_com, axis=(-2, -1)
        )
        body_residual = np.linalg.norm(
            reconstructed_inertia_body - inertia_body, axis=(-2, -1)
        )
        source_masses = np.asarray(source.particles.safe_masses)
        source_inertia = np.asarray(source.inertia_body)
        if (
            source_masses.shape != (count,)
            or source_inertia.shape != inertia_body.shape
        ):
            raise ValueError("Source prepared data does not match parameter capacity.")
        source_mass_residual = np.abs(masses - source_masses)
        source_inertia_residual = np.linalg.norm(
            inertia_body - source_inertia, axis=(-2, -1)
        )
        valid = bool(
            np.all(finite)
            and np.all(positive_mass)
            and np.all(inertia_spd)
            and np.all(triangle)
            and np.all(pseudo_spd)
            and np.all(body_inertia_spd)
            and np.all(body_triangle)
            and np.all(np.isfinite(inertia_residual))
            and np.all(np.isfinite(body_residual))
        )
        self.parameters = parameters
        self.finite_mask = jnp.asarray(finite)
        self.positive_mass_mask = jnp.asarray(positive_mass)
        self.inertia_spd_mask = jnp.asarray(inertia_spd)
        self.triangle_inequality_mask = jnp.asarray(triangle)
        self.pseudo_inertia_spd_mask = jnp.asarray(pseudo_spd)
        self.body_origin_inertia_spd_mask = jnp.asarray(body_inertia_spd)
        self.body_origin_triangle_inequality_mask = jnp.asarray(body_triangle)
        self.minimum_inertia_eigenvalue = jnp.asarray(inertia_eigenvalues[:, 0])
        self.minimum_triangle_margin = jnp.asarray(triangle_margin)
        self.minimum_body_origin_inertia_eigenvalue = jnp.asarray(
            body_eigenvalues[:, 0]
        )
        self.minimum_body_origin_triangle_margin = jnp.asarray(body_triangle_margin)
        self.inertia_condition_number = jnp.asarray(
            _matrix_condition_numbers(inertia_eigenvalues)
        )
        self.pseudo_inertia_condition_number = jnp.asarray(
            _matrix_condition_numbers(pseudo_eigenvalues)
        )
        self.body_origin_inertia_condition_number = jnp.asarray(
            _matrix_condition_numbers(body_eigenvalues)
        )
        self.mass_reconstruction_residual = jnp.asarray(mass_residual)
        self.center_of_mass_reconstruction_residual = jnp.asarray(offset_residual)
        self.inertia_reconstruction_residual = jnp.asarray(inertia_residual)
        self.body_origin_reconstruction_residual = jnp.asarray(body_residual)
        self.source_mass_residual = jnp.asarray(source_mass_residual)
        self.source_inertia_residual = jnp.asarray(source_inertia_residual)
        self.valid = jnp.asarray(valid)
        self.source_prepared_id = source.prepared_id
        self.requires_repreparation = True
        self.evaluation_id = canonical_fingerprint(
            {
                "kind": "rigid-inertial-evaluation",
                "source": source.prepared_id,
                "parameters": parameters.parameters_id,
                "valid": valid,
                "requires_repreparation": True,
            }
        )


class RigidInertialParameterization(StrictModule, NonTrainableState):
    """Host-bound map from unconstrained coordinates to admissible 3-D inertia."""

    source: PreparedRigidBodySet
    body_count: int = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: PreparedRigidBodySet,
        /,
        *,
        parameterization_id: str | None = None,
    ):
        if not isinstance(source, PreparedRigidBodySet):
            raise TypeError("source must be a PreparedRigidBodySet.")
        if source.ambient_dimension != _SPATIAL_DIMENSION:
            raise ValueError(
                "Rigid inertial parameterization requires three dimensions."
            )
        generated = canonical_fingerprint(
            {
                "kind": "rigid-inertial-parameterization",
                "source": source.prepared_id,
                "body_count": source.capacity,
            }
        )
        if parameterization_id is None:
            identifier = generated
        else:
            user_identifier = str(parameterization_id).strip()
            if not user_identifier:
                raise ValueError("parameterization_id must be nonempty.")
            identifier = canonical_fingerprint(
                {
                    "kind": "rigid-inertial-parameterization",
                    "source": source.prepared_id,
                    "user_id": user_identifier,
                }
            )
        self.source = source
        self.body_count = source.capacity
        self.parameterization_id = identifier

    def coordinates(
        self,
        mass_coordinates: ArrayLike,
        center_of_mass_offsets: ArrayLike,
        covariance_coordinates: ArrayLike,
        /,
    ) -> RigidInertialCoordinates:
        return RigidInertialCoordinates(
            mass_coordinates,
            center_of_mass_offsets,
            covariance_coordinates,
            parameterization_id=self.parameterization_id,
        )

    def coordinates_from_prepared(
        self,
        center_of_mass_offsets: ArrayLike | None = None,
        /,
    ) -> RigidInertialCoordinates:
        masses = np.asarray(self.source.particles.safe_masses)
        inertia_body = np.asarray(self.source.inertia_body)
        offsets = (
            np.zeros((self.body_count, _SPATIAL_DIMENSION), dtype=inertia_body.dtype)
            if center_of_mass_offsets is None
            else _finite_array(center_of_mass_offsets, "center_of_mass_offsets")
        )
        if offsets.shape != (self.body_count, _SPATIAL_DIMENSION):
            raise ValueError("center_of_mass_offsets must have shape (N,3).")
        identity = np.eye(_SPATIAL_DIMENSION, dtype=inertia_body.dtype)
        outer_offsets = offsets[:, :, None] * offsets[:, None, :]
        parallel_axis = masses[:, None, None] * (
            np.sum(offsets * offsets, axis=-1)[:, None, None] * identity
            - outer_offsets
        )
        inertia_com = inertia_body - parallel_axis
        if not np.all(np.isfinite(inertia_com)) or not np.allclose(
            inertia_com, np.swapaxes(inertia_com, -1, -2)
        ):
            raise ValueError(
                "Prepared body-origin inertia and COM offsets are invalid."
            )
        inertia_com = 0.5 * (
            inertia_com + np.swapaxes(inertia_com, -1, -2)
        )
        central_second_moment = (
            0.5
            * np.trace(inertia_com, axis1=-2, axis2=-1)[:, None, None]
            * identity
            - inertia_com
        )
        covariance = central_second_moment / masses[:, None, None]
        covariance_eigenvalues = _symmetric_eigenvalues(covariance)
        if np.any(covariance_eigenvalues <= 0.0):
            raise ValueError(
                "Prepared data does not define physically realizable COM inertia."
            )
        factors = np.linalg.cholesky(covariance)
        mass_coordinates = _inverse_softplus(masses)
        covariance_coordinates = _factor_coordinates(factors)
        return self.coordinates(mass_coordinates, offsets, covariance_coordinates)

    def inverse(
        self,
        center_of_mass_offsets: ArrayLike | None = None,
        /,
    ) -> RigidInertialCoordinates:
        """Return coordinates reconstructing this prepared set for the given COMs."""
        return self.coordinates_from_prepared(center_of_mass_offsets)

    def evaluate(
        self,
        coordinates: RigidInertialCoordinates,
        /,
    ) -> RigidInertialEvaluation:
        if not isinstance(coordinates, RigidInertialCoordinates):
            raise TypeError("coordinates must be RigidInertialCoordinates.")
        if coordinates.parameterization_id != self.parameterization_id:
            raise ValueError("Coordinates belong to a different parameterization.")
        if (
            coordinates.mass_coordinates.shape != (self.body_count,)
            or coordinates.center_of_mass_offsets.shape
            != (self.body_count, _SPATIAL_DIMENSION)
            or coordinates.covariance_coordinates.shape
            != (self.body_count, _FACTOR_COMPONENTS)
        ):
            raise ValueError("Coordinate shapes do not match the prepared body set.")
        mass_coordinates = np.asarray(coordinates.mass_coordinates)
        covariance_coordinates = np.asarray(coordinates.covariance_coordinates)
        offsets = np.asarray(coordinates.center_of_mass_offsets)
        masses = _stable_softplus(mass_coordinates)
        factors = _lower_triangular_factors(covariance_coordinates)
        covariance = factors @ np.swapaxes(factors, -1, -2)
        identity = np.eye(_SPATIAL_DIMENSION, dtype=covariance.dtype)
        inertia_com = masses[:, None, None] * (
            np.trace(covariance, axis1=-2, axis2=-1)[:, None, None] * identity
            - covariance
        )
        parameters = RigidInertialParameters(
            masses,
            offsets,
            inertia_com,
            parameterization_id=self.parameterization_id,
            coordinates_id=coordinates.coordinates_id,
        )
        return RigidInertialEvaluation(parameters, self.source)

    def realize(
        self,
        coordinates: RigidInertialCoordinates,
        /,
    ) -> tuple[ParticleSetPlan, RigidBodySetPlan, RigidInertialEvaluation]:
        return realize_rigid_body_plans(self, coordinates)


def realize_rigid_body_plans(
    parameterization: RigidInertialParameterization,
    coordinates: RigidInertialCoordinates,
    /,
) -> tuple[ParticleSetPlan, RigidBodySetPlan, RigidInertialEvaluation]:
    """Build fresh plans; callers must explicitly prepare them before execution."""
    if not isinstance(parameterization, RigidInertialParameterization):
        raise TypeError("parameterization must be RigidInertialParameterization.")
    evaluation = parameterization.evaluate(coordinates)
    if not bool(np.asarray(evaluation.valid)):
        raise ValueError("Invalid inertial coordinates cannot be realized.")
    source = parameterization.source
    source_particle_plan = source.particles.plan
    source_rigid_plan = source.plan
    particle_plan_id = canonical_fingerprint(
        {
            "kind": "realized-rigid-inertial-particle-plan",
            "source": source.prepared_id,
            "evaluation": evaluation.evaluation_id,
        }
    )
    rigid_plan_id = canonical_fingerprint(
        {
            "kind": "realized-rigid-inertial-body-plan",
            "source": source.prepared_id,
            "evaluation": evaluation.evaluation_id,
        }
    )
    if (
        particle_plan_id == source_particle_plan.plan_id
        or rigid_plan_id == source_rigid_plan.plan_id
    ):
        raise ValueError("Realization must cross a fresh plan identity boundary.")
    particle_plan = ParticleSetPlan(
        source_particle_plan.particle_ids,
        evaluation.parameters.masses,
        ambient_dimension=_SPATIAL_DIMENSION,
        active_mask=source_particle_plan.active_mask,
        subsets=source_particle_plan.subsets,
        name=source_particle_plan.key.name,
        domain_labels=source_particle_plan.key.domain_labels,
        coordinate_dtype=source_particle_plan.coordinate_dtype,
        plan_id=particle_plan_id,
    )
    rigid_plan = RigidBodySetPlan(
        source_rigid_plan.material_ids,
        evaluation.parameters.inertia_body_origin,
        fixed_mask=source_rigid_plan.fixed_mask,
        name=source_rigid_plan.key.name,
        plan_id=rigid_plan_id,
    )
    return particle_plan, rigid_plan, evaluation


__all__ = [
    "RigidInertialCoordinates",
    "RigidInertialEvaluation",
    "RigidInertialParameterization",
    "RigidInertialParameters",
    "realize_rigid_body_plans",
]
