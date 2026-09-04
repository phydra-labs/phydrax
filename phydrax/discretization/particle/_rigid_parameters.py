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
from ._rigid_body import (
    PreparedRigidBodySet,
    RigidBodyReferenceFrameRebase,
    RigidBodySetPlan,
)


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


def _coordinate_image(dtype: np.dtype, /) -> tuple[float, float]:
    information = np.finfo(dtype)
    floor = float(information.eps)
    ceiling = float(np.cbrt(information.max) / 16.0)
    return floor, ceiling


def _stable_softplus(
    value: np.ndarray,
    positive_floor: float,
    finite_ceiling: float,
    /,
) -> np.ndarray:
    return np.minimum(
        positive_floor + np.logaddexp(0.0, value),
        finite_ceiling,
    )


def _inverse_softplus(
    value: np.ndarray,
    positive_floor: float,
    finite_ceiling: float,
    /,
) -> np.ndarray:
    shifted = value - positive_floor
    if (
        np.any(shifted <= 0.0)
        or np.any(value > finite_ceiling)
        or not np.all(np.isfinite(value))
    ):
        raise ValueError("Positive factor values are outside the coordinate image.")
    return shifted + np.log(-np.expm1(-shifted))


def _lower_triangular_factors(
    coordinates: np.ndarray,
    positive_floor: float,
    finite_ceiling: float,
    /,
) -> np.ndarray:
    count = coordinates.shape[0]
    factors = np.zeros(
        (count, _SPATIAL_DIMENSION, _SPATIAL_DIMENSION),
        dtype=coordinates.dtype,
    )
    diagonal = _stable_softplus(coordinates[:, (0, 2, 5)], positive_floor, finite_ceiling)
    factors[:, 0, 0] = diagonal[:, 0]
    factors[:, 1, 0] = np.clip(coordinates[:, 1], -finite_ceiling, finite_ceiling)
    factors[:, 1, 1] = diagonal[:, 1]
    factors[:, 2, 0] = np.clip(coordinates[:, 3], -finite_ceiling, finite_ceiling)
    factors[:, 2, 1] = np.clip(coordinates[:, 4], -finite_ceiling, finite_ceiling)
    factors[:, 2, 2] = diagonal[:, 2]
    return factors


def _factor_coordinates(
    factors: np.ndarray,
    positive_floor: float,
    finite_ceiling: float,
    /,
) -> np.ndarray:
    off_diagonal = factors[:, (1, 2, 2), (0, 0, 1)]
    if np.any(np.abs(off_diagonal) > finite_ceiling):
        raise ValueError("Covariance factors are outside the coordinate image.")
    return np.stack(
        (
            _inverse_softplus(factors[:, 0, 0], positive_floor, finite_ceiling),
            factors[:, 1, 0],
            _inverse_softplus(factors[:, 1, 1], positive_floor, finite_ceiling),
            factors[:, 2, 0],
            factors[:, 2, 1],
            _inverse_softplus(factors[:, 2, 2], positive_floor, finite_ceiling),
        ),
        axis=-1,
    )


def _symmetric_eigenvalues(value: np.ndarray, /) -> np.ndarray:
    return np.linalg.eigvalsh(0.5 * (value + np.swapaxes(value, -1, -2)))


def _matrix_condition_numbers(eigenvalues: np.ndarray, /) -> np.ndarray:
    information = np.finfo(eigenvalues.dtype)
    lower = np.maximum(np.abs(eigenvalues[:, 0]), information.tiny)
    upper = np.maximum(np.abs(eigenvalues[:, -1]), lower)
    with np.errstate(over="ignore", invalid="ignore"):
        condition = upper / lower
    return np.nan_to_num(
        condition,
        nan=information.max,
        posinf=information.max,
        neginf=information.max,
    )


def _finite_norm(value: np.ndarray, axes: tuple[int, ...], /) -> np.ndarray:
    absolute = np.abs(value)
    scale = np.max(absolute, axis=axes)
    expanded = scale
    for axis in sorted((item % value.ndim for item in axes)):
        expanded = np.expand_dims(expanded, axis=axis)
    normalized = np.divide(
        absolute,
        expanded,
        out=np.zeros_like(absolute),
        where=expanded > 0.0,
    )
    unit_norm = np.sqrt(np.sum(normalized * normalized, axis=axes))
    maximum = np.finfo(value.dtype).max
    return np.minimum(scale, maximum / np.maximum(unit_norm, 1.0)) * unit_norm


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
        if np.any(inertia_eigenvalues <= 0.0) or np.any(central_eigenvalues <= 0.0):
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
            raise ValueError("Realized inertial parameters exceed finite numeric range.")
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


class RigidInertialEvaluation(StrictModule, NonTrainableState):
    """Validity, reconstruction, conditioning, and source-distance evidence."""

    parameters: RigidInertialParameters
    finite_mask: Array
    evidence_finite_mask: Array
    coordinate_saturation_mask: Array
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
    positive_floor: float = eqx.field(static=True)
    finite_ceiling: float = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    requires_repreparation: bool = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: RigidInertialParameters,
        source: PreparedRigidBodySet,
        coordinate_saturation_mask: ArrayLike,
        /,
        *,
        positive_floor: float,
        finite_ceiling: float,
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
        saturation = np.asarray(coordinate_saturation_mask, dtype=bool)
        if saturation.shape != (count,):
            raise ValueError("coordinate_saturation_mask must have body shape.")
        identity = np.eye(_SPATIAL_DIMENSION, dtype=inertia_com.dtype)
        inertia_eigenvalues = _symmetric_eigenvalues(inertia_com)
        triangle_margin = np.trace(inertia_com, axis1=-2, axis2=-1) - 2.0 * np.max(
            inertia_eigenvalues, axis=-1
        )
        body_eigenvalues = _symmetric_eigenvalues(inertia_body)
        body_triangle_margin = np.trace(inertia_body, axis1=-2, axis2=-1) - 2.0 * np.max(
            body_eigenvalues, axis=-1
        )
        pseudo_eigenvalues = _symmetric_eigenvalues(pseudo)
        reconstructed_masses = pseudo[:, 3, 3]
        reconstructed_offsets = pseudo[:, :3, 3] / reconstructed_masses[:, None]
        spatial_second_moment = pseudo[:, :3, :3]
        central_second_moment = (
            spatial_second_moment
            - (pseudo[:, :3, 3, None] * pseudo[:, None, 3, :3])
            / reconstructed_masses[:, None, None]
        )
        reconstructed_inertia_com = (
            np.trace(central_second_moment, axis1=-2, axis2=-1)[:, None, None] * identity
            - central_second_moment
        )
        reconstructed_inertia_body = (
            np.trace(spatial_second_moment, axis1=-2, axis2=-1)[:, None, None] * identity
            - spatial_second_moment
        )
        parameter_finite = np.isfinite(masses) & np.all(np.isfinite(offsets), axis=-1)
        inertia_com_finite = np.all(np.isfinite(inertia_com), axis=(-2, -1))
        inertia_body_finite = np.all(np.isfinite(inertia_body), axis=(-2, -1))
        pseudo_inertia_finite = np.all(np.isfinite(pseudo), axis=(-2, -1))
        finite = (
            parameter_finite
            & inertia_com_finite
            & inertia_body_finite
            & pseudo_inertia_finite
        )
        positive_mass = masses >= positive_floor
        inertia_spd = inertia_eigenvalues[:, 0] > 0.0
        triangle = triangle_margin > 0.0
        central_eigenvalues = _symmetric_eigenvalues(
            0.5 * np.trace(inertia_com, axis1=-2, axis2=-1)[:, None, None] * identity
            - inertia_com
        )
        pseudo_spd = positive_mass & (central_eigenvalues[:, 0] > 0.0)
        body_inertia_spd = body_eigenvalues[:, 0] > 0.0
        body_triangle = body_triangle_margin > 0.0
        mass_residual = np.abs(reconstructed_masses - masses)
        offset_residual = _finite_norm(reconstructed_offsets - offsets, (-1,))
        inertia_residual = _finite_norm(reconstructed_inertia_com - inertia_com, (-2, -1))
        body_residual = _finite_norm(reconstructed_inertia_body - inertia_body, (-2, -1))
        source_masses = np.asarray(source.mass_properties.masses)
        source_inertia = np.asarray(source.mass_properties.inertia_com)
        if source_masses.shape != (count,) or source_inertia.shape != inertia_com.shape:
            raise ValueError("Source prepared data does not match parameter capacity.")
        source_mass_residual = np.abs(masses - source_masses)
        source_inertia_residual = _finite_norm(inertia_com - source_inertia, (-2, -1))
        inertia_condition = _matrix_condition_numbers(inertia_eigenvalues)
        pseudo_condition = _matrix_condition_numbers(pseudo_eigenvalues)
        body_condition = _matrix_condition_numbers(body_eigenvalues)
        evidence_finite = (
            np.isfinite(inertia_eigenvalues[:, 0])
            & np.isfinite(triangle_margin)
            & np.isfinite(body_eigenvalues[:, 0])
            & np.isfinite(body_triangle_margin)
            & np.isfinite(inertia_condition)
            & np.isfinite(pseudo_condition)
            & np.isfinite(body_condition)
            & np.isfinite(mass_residual)
            & np.isfinite(offset_residual)
            & np.isfinite(inertia_residual)
            & np.isfinite(body_residual)
            & np.isfinite(source_mass_residual)
            & np.isfinite(source_inertia_residual)
        )
        valid = bool(
            np.all(finite)
            and np.all(evidence_finite)
            and np.all(positive_mass)
            and np.all(inertia_spd)
            and np.all(triangle)
            and np.all(pseudo_spd)
            and np.all(body_inertia_spd)
            and np.all(body_triangle)
        )
        self.parameters = parameters
        self.finite_mask = jnp.asarray(finite)
        self.evidence_finite_mask = jnp.asarray(evidence_finite)
        self.coordinate_saturation_mask = jnp.asarray(saturation)
        self.positive_mass_mask = jnp.asarray(positive_mass)
        self.inertia_spd_mask = jnp.asarray(inertia_spd)
        self.triangle_inequality_mask = jnp.asarray(triangle)
        self.pseudo_inertia_spd_mask = jnp.asarray(pseudo_spd)
        self.body_origin_inertia_spd_mask = jnp.asarray(body_inertia_spd)
        self.body_origin_triangle_inequality_mask = jnp.asarray(body_triangle)
        self.minimum_inertia_eigenvalue = jnp.asarray(inertia_eigenvalues[:, 0])
        self.minimum_triangle_margin = jnp.asarray(triangle_margin)
        self.minimum_body_origin_inertia_eigenvalue = jnp.asarray(body_eigenvalues[:, 0])
        self.minimum_body_origin_triangle_margin = jnp.asarray(body_triangle_margin)
        self.inertia_condition_number = jnp.asarray(inertia_condition)
        self.pseudo_inertia_condition_number = jnp.asarray(pseudo_condition)
        self.body_origin_inertia_condition_number = jnp.asarray(body_condition)
        self.mass_reconstruction_residual = jnp.asarray(mass_residual)
        self.center_of_mass_reconstruction_residual = jnp.asarray(offset_residual)
        self.inertia_reconstruction_residual = jnp.asarray(inertia_residual)
        self.body_origin_reconstruction_residual = jnp.asarray(body_residual)
        self.source_mass_residual = jnp.asarray(source_mass_residual)
        self.source_inertia_residual = jnp.asarray(source_inertia_residual)
        self.valid = jnp.asarray(valid)
        self.positive_floor = positive_floor
        self.finite_ceiling = finite_ceiling
        self.source_prepared_id = source.prepared_id
        self.requires_repreparation = True
        evidence = array_tree_fingerprint(
            {
                "finite": finite,
                "evidence_finite": evidence_finite,
                "coordinate_saturation": saturation,
                "positive_mass": positive_mass,
                "inertia_spd": inertia_spd,
                "triangle_inequality": triangle,
                "pseudo_inertia_spd": pseudo_spd,
                "body_origin_inertia_spd": body_inertia_spd,
                "body_origin_triangle_inequality": body_triangle,
                "minimum_inertia_eigenvalue": inertia_eigenvalues[:, 0],
                "minimum_triangle_margin": triangle_margin,
                "minimum_body_origin_inertia_eigenvalue": body_eigenvalues[:, 0],
                "minimum_body_origin_triangle_margin": body_triangle_margin,
                "inertia_condition_number": inertia_condition,
                "pseudo_inertia_condition_number": pseudo_condition,
                "body_origin_inertia_condition_number": body_condition,
                "mass_reconstruction_residual": mass_residual,
                "center_of_mass_reconstruction_residual": offset_residual,
                "inertia_reconstruction_residual": inertia_residual,
                "body_origin_reconstruction_residual": body_residual,
                "source_mass_residual": source_mass_residual,
                "source_inertia_residual": source_inertia_residual,
                "valid": np.asarray(valid),
            }
        )
        self.evaluation_id = canonical_fingerprint(
            {
                "kind": "rigid-inertial-evaluation",
                "source": source.prepared_id,
                "parameters": parameters.parameters_id,
                "positive_floor": positive_floor,
                "finite_ceiling": finite_ceiling,
                "evidence": evidence,
                "requires_repreparation": True,
            }
        )


class RigidInertialParameterization(StrictModule, NonTrainableState):
    """Host-bound map from finite coordinates to admissible 3-D inertia."""

    source: PreparedRigidBodySet
    body_count: int = eqx.field(static=True)
    positive_floor: float = eqx.field(static=True)
    finite_ceiling: float = eqx.field(static=True)
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
            raise ValueError("Rigid inertial parameterization requires three dimensions.")
        source_masses = np.asarray(source.mass_properties.masses)
        source_inertia = np.asarray(source.mass_properties.inertia_com)
        dtype = source_masses.dtype
        default_floor, finite_ceiling = _coordinate_image(dtype)
        identity = np.eye(_SPATIAL_DIMENSION, dtype=dtype)
        central_second_moment = (
            0.5 * np.trace(source_inertia, axis1=-2, axis2=-1)[:, None, None] * identity
            - source_inertia
        )
        source_covariance = central_second_moment / source_masses[:, None, None]
        source_covariance_eigenvalues = _symmetric_eigenvalues(source_covariance)
        if np.all(source_covariance_eigenvalues > 0.0):
            source_factors = np.linalg.cholesky(source_covariance)
            source_scale = min(
                float(np.min(source_masses)),
                float(
                    np.min(
                        source_factors[
                            :,
                            np.arange(_SPATIAL_DIMENSION),
                            np.arange(_SPATIAL_DIMENSION),
                        ]
                    )
                ),
            )
            candidate_floor = 0.5 * source_scale
            positive_floor = (
                min(default_floor, candidate_floor)
                if candidate_floor > 0.0
                else float(np.nextafter(dtype.type(0.0), dtype.type(1.0)))
            )
        else:
            positive_floor = default_floor
        identity_values = {
            "source": source.prepared_id,
            "body_count": source.capacity,
            "positive_floor": positive_floor,
            "finite_ceiling": finite_ceiling,
        }
        generated = canonical_fingerprint(
            {
                "kind": "rigid-inertial-parameterization",
                **identity_values,
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
                    **identity_values,
                    "user_id": user_identifier,
                }
            )
        self.source = source
        self.body_count = source.capacity
        self.positive_floor = positive_floor
        self.finite_ceiling = finite_ceiling
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
        masses = np.asarray(self.source.mass_properties.masses)
        inertia_com = np.asarray(self.source.mass_properties.inertia_com)
        offsets = (
            np.zeros((self.body_count, _SPATIAL_DIMENSION), dtype=inertia_com.dtype)
            if center_of_mass_offsets is None
            else _finite_array(center_of_mass_offsets, "center_of_mass_offsets")
        )
        if offsets.shape != (self.body_count, _SPATIAL_DIMENSION):
            raise ValueError("center_of_mass_offsets must have shape (N,3).")
        if np.any(np.abs(offsets) > self.finite_ceiling):
            raise ValueError("center_of_mass_offsets are outside the coordinate image.")
        identity = np.eye(_SPATIAL_DIMENSION, dtype=inertia_com.dtype)
        central_second_moment = (
            0.5 * np.trace(inertia_com, axis1=-2, axis2=-1)[:, None, None] * identity
            - inertia_com
        )
        covariance = central_second_moment / masses[:, None, None]
        covariance_eigenvalues = _symmetric_eigenvalues(covariance)
        if np.any(covariance_eigenvalues <= 0.0):
            raise ValueError(
                "Prepared data does not define physically realizable COM inertia."
            )
        factors = np.linalg.cholesky(covariance)
        mass_coordinates = _inverse_softplus(
            masses, self.positive_floor, self.finite_ceiling
        )
        covariance_coordinates = _factor_coordinates(
            factors, self.positive_floor, self.finite_ceiling
        )
        return self.coordinates(mass_coordinates, offsets, covariance_coordinates)

    def inverse(
        self,
        center_of_mass_offsets: ArrayLike | None = None,
        /,
    ) -> RigidInertialCoordinates:
        """Return coordinates reconstructing this prepared COM mass property set."""

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
        raw_offsets = np.asarray(coordinates.center_of_mass_offsets)
        masses = _stable_softplus(
            mass_coordinates, self.positive_floor, self.finite_ceiling
        )
        factors = _lower_triangular_factors(
            covariance_coordinates, self.positive_floor, self.finite_ceiling
        )
        offsets = np.clip(raw_offsets, -self.finite_ceiling, self.finite_ceiling)
        raw_covariance = factors @ np.swapaxes(factors, -1, -2)
        identity = np.eye(_SPATIAL_DIMENSION, dtype=raw_covariance.dtype)
        covariance = raw_covariance
        inertia_com = masses[:, None, None] * (
            np.trace(covariance, axis1=-2, axis2=-1)[:, None, None] * identity
            - covariance
        )
        raw_mass_positive = np.logaddexp(0.0, mass_coordinates)
        raw_diagonal_positive = np.logaddexp(0.0, covariance_coordinates[:, (0, 2, 5)])
        saturation = (
            (raw_mass_positive <= self.positive_floor)
            | (raw_mass_positive >= self.finite_ceiling - self.positive_floor)
            | np.any(raw_diagonal_positive <= self.positive_floor, axis=-1)
            | np.any(
                raw_diagonal_positive >= self.finite_ceiling - self.positive_floor,
                axis=-1,
            )
            | np.any(np.abs(raw_offsets) > self.finite_ceiling, axis=-1)
            | np.any(
                np.abs(covariance_coordinates[:, (1, 3, 4)]) > self.finite_ceiling,
                axis=-1,
            )
        )
        parameters = RigidInertialParameters(
            masses,
            offsets,
            inertia_com,
            parameterization_id=self.parameterization_id,
            coordinates_id=coordinates.coordinates_id,
        )
        return RigidInertialEvaluation(
            parameters,
            self.source,
            saturation,
            positive_floor=self.positive_floor,
            finite_ceiling=self.finite_ceiling,
        )

    def realize(
        self,
        coordinates: RigidInertialCoordinates,
        /,
    ) -> RigidInertialRealization:
        return realize_rigid_body_plans(self, coordinates)


class RigidInertialRealization(StrictModule, NonTrainableState):
    """Fresh COM plans plus the mandatory old-origin reference-frame rebase."""

    particle_plan: ParticleSetPlan
    rigid_body_plan: RigidBodySetPlan
    evaluation: RigidInertialEvaluation
    reference_frame_rebase: RigidBodyReferenceFrameRebase
    rebase_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_plan: ParticleSetPlan,
        rigid_body_plan: RigidBodySetPlan,
        evaluation: RigidInertialEvaluation,
        reference_frame_rebase: RigidBodyReferenceFrameRebase,
        /,
    ):
        if not isinstance(particle_plan, ParticleSetPlan):
            raise TypeError("particle_plan must be a ParticleSetPlan.")
        if not isinstance(rigid_body_plan, RigidBodySetPlan):
            raise TypeError("rigid_body_plan must be a RigidBodySetPlan.")
        if not isinstance(evaluation, RigidInertialEvaluation):
            raise TypeError("evaluation must be a RigidInertialEvaluation.")
        if not isinstance(reference_frame_rebase, RigidBodyReferenceFrameRebase):
            raise TypeError(
                "reference_frame_rebase must be RigidBodyReferenceFrameRebase."
            )
        if (
            reference_frame_rebase.source_prepared_id != evaluation.source_prepared_id
            or reference_frame_rebase.target_particle_plan_id != particle_plan.plan_id
            or reference_frame_rebase.target_body_plan_id != rigid_body_plan.plan_id
        ):
            raise ValueError("Realization plans/evaluation do not match its rebase.")
        self.particle_plan = particle_plan
        self.rigid_body_plan = rigid_body_plan
        self.evaluation = evaluation
        self.reference_frame_rebase = reference_frame_rebase
        self.rebase_id = reference_frame_rebase.rebase_id
        self.realization_id = canonical_fingerprint(
            {
                "kind": "rigid-inertial-realization",
                "particle_plan": particle_plan.plan_id,
                "rigid_body_plan": rigid_body_plan.plan_id,
                "evaluation": evaluation.evaluation_id,
                "reference_frame_rebase": reference_frame_rebase.rebase_id,
            }
        )


def realize_rigid_body_plans(
    parameterization: RigidInertialParameterization,
    coordinates: RigidInertialCoordinates,
    /,
) -> RigidInertialRealization:
    """Build fresh COM plans and an identity-bound reference-frame rebase."""

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
        evaluation.parameters.inertia_com,
        fixed_mask=source_rigid_plan.fixed_mask,
        name=source_rigid_plan.key.name,
        plan_id=rigid_plan_id,
    )
    rebase = RigidBodyReferenceFrameRebase(
        source,
        particle_plan,
        rigid_plan,
        evaluation.parameters.center_of_mass_offsets,
    )
    return RigidInertialRealization(
        particle_plan,
        rigid_plan,
        evaluation,
        rebase,
    )


__all__ = [
    "RigidInertialCoordinates",
    "RigidInertialEvaluation",
    "RigidInertialParameterization",
    "RigidInertialRealization",
    "RigidInertialParameters",
    "realize_rigid_body_plans",
]
