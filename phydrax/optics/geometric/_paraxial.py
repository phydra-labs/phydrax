#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from numbers import Real

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.analytic import RigidFrame
from ._sequential import PreparedSequentialOptics, SequentialOpticsPlan


_COORDINATE_CONVENTION = "(u,v,nθu,nθv)"


class ParaxialOpticsStatus(IntEnum):
    """Status for an exact differential map or cached first-order evaluation."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    CHIEF_RAY_FAILURE = 2
    BRANCH_MARGIN_VIOLATION = 3
    NONFINITE_JACOBIAN = 4
    INVALID_DIFFERENTIAL_MAP = 5
    OUTSIDE_TRUST_ENVELOPE = 6
    NUMERICAL_FAILURE = 7


class DifferentialRayMap(StrictModule):
    """One fixed-branch ray Jacobian in canonical optical coordinates.

    This value is evidence, not a matrix-algebra abstraction. Coordinate order is
    exactly ``(u, v, nθu, nθv)`` in both declared reference frames.

    Encoding places the origin at local ``(u,v,0)``, sets
    ``theta=(nθ)/n``, and normalizes ``(tan(theta_u),tan(theta_v),1)``.
    Decoding intersects the output frame's ``w=0`` plane and returns
    ``n*atan2(d_tangent,d_longitudinal)`` for each momentum coordinate.
    """

    input_reference: Array
    output_reference: Array
    jacobian: Array
    branch_margin: Array
    finite: Array
    valid: Array
    status: Array
    input_frame_id: str = eqx.field(static=True)
    output_frame_id: str = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)


class ParaxialOpticsResult(StrictModule):
    """Affine ray coordinates and trust-envelope evidence."""

    coordinates: Array
    transverse_excursion: Array
    angular_excursion: Array
    within_envelope: Array
    finite: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _frame_id(frame: RigidFrame, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "rigid-frame-coordinate-chart",
            "arrays": array_tree_fingerprint(
                (np.asarray(frame.rotation), np.asarray(frame.translation))
            ),
        }
    )


def _validate_frame(frame: RigidFrame, name: str, /) -> None:
    if not isinstance(frame, RigidFrame):
        raise TypeError(f"{name} must be a RigidFrame.")
    if frame.dimension != 3:
        raise ValueError(f"{name} must be three-dimensional.")


def _real_scalar(value: float, name: str, /) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _host_coordinates(values: ArrayLike, name: str, /) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.dtype == np.dtype(bool)
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must contain real numeric data.")
    result = raw.astype(float)
    if result.shape != (4,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real array with shape (4,).")
    return result


def _encode_ray_coordinates(
    coordinates: Array,
    frame: RigidFrame,
    refractive_index: float,
    /,
) -> tuple[Array, Array]:
    local_origin = jnp.concatenate(
        (coordinates[..., :2], jnp.zeros_like(coordinates[..., :1])), axis=-1
    )
    angles = coordinates[..., 2:] / refractive_index
    local_direction = jnp.concatenate(
        (jnp.tan(angles), jnp.ones_like(angles[..., :1])), axis=-1
    )
    local_direction = local_direction / jnp.sqrt(
        jnp.sum(local_direction * local_direction, axis=-1, keepdims=True)
    )
    origin = frame.apply(local_origin)
    direction = local_direction @ frame.rotation.T
    return origin, direction


def _decode_ray_coordinates(
    origins: Array,
    directions: Array,
    frame: RigidFrame,
    refractive_index: float,
    /,
) -> tuple[Array, Array]:
    local_origin = frame.inverse_apply(origins)
    local_direction = directions @ frame.rotation
    longitudinal = local_direction[..., 2]
    safe_longitudinal = jnp.where(jnp.abs(longitudinal) > 0.0, longitudinal, 1.0)
    distance = -local_origin[..., 2] / safe_longitudinal
    point = local_origin + distance[..., None] * local_direction
    momenta = refractive_index * jnp.stack(
        (
            jnp.atan2(local_direction[..., 0], longitudinal),
            jnp.atan2(local_direction[..., 1], longitudinal),
        ),
        axis=-1,
    )
    coordinates = jnp.concatenate((point[..., :2], momenta), axis=-1)
    return coordinates, longitudinal


def linearize_sequential_optics(
    prepared: PreparedSequentialOptics,
    reference_coordinates: ArrayLike,
    /,
    *,
    input_frame: RigidFrame,
    output_frame: RigidFrame,
    input_refractive_index: float,
    output_refractive_index: float,
) -> DifferentialRayMap:
    """Differentiate one successful, margin-separated fixed sequential branch."""
    if not isinstance(prepared, PreparedSequentialOptics):
        raise TypeError("prepared must be a PreparedSequentialOptics.")
    _validate_frame(input_frame, "input_frame")
    _validate_frame(output_frame, "output_frame")
    input_index = _real_scalar(input_refractive_index, "input_refractive_index")
    output_index = _real_scalar(output_refractive_index, "output_refractive_index")
    if input_index <= 0.0 or output_index <= 0.0:
        raise ValueError("Input and output refractive indices must be positive.")
    if not np.isclose(
        input_index,
        float(np.asarray(prepared.refractive_indices[0])),
        rtol=1.0e-12,
        atol=1.0e-12,
    ) or not np.isclose(
        output_index,
        float(np.asarray(prepared.refractive_indices[-1])),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Differential coordinate indices must match the exact prescription."
        )
    reference_host = _host_coordinates(reference_coordinates, "reference_coordinates")
    reference = jnp.asarray(reference_host)

    def fixed_branch(coordinates: Array) -> Array:
        origin, direction = _encode_ray_coordinates(coordinates, input_frame, input_index)
        result, _ = prepared._trace(origin, direction)
        output, _ = _decode_ray_coordinates(
            result.rays.origins,
            result.rays.directions,
            output_frame,
            output_index,
        )
        return output

    input_origin, input_direction = _encode_ray_coordinates(
        reference, input_frame, input_index
    )
    chief_result, trace_margin = prepared._trace(input_origin, input_direction)
    output_reference, output_longitudinal = _decode_ray_coordinates(
        chief_result.rays.origins,
        chief_result.rays.directions,
        output_frame,
        output_index,
    )
    jacobian = jax.jacfwd(fixed_branch)(reference)
    output_plane_margin = output_longitudinal - prepared.incidence_tolerance
    branch_margin = jnp.minimum(trace_margin, output_plane_margin)
    finite = (
        jnp.all(jnp.isfinite(reference))
        & jnp.all(jnp.isfinite(output_reference))
        & jnp.all(jnp.isfinite(jacobian))
        & jnp.isfinite(branch_margin)
    )
    chief_success = chief_result.successful
    margin_valid = branch_margin > 0.0
    valid = chief_success & margin_valid & finite
    status = jnp.where(
        ~chief_success,
        int(ParaxialOpticsStatus.CHIEF_RAY_FAILURE),
        jnp.where(
            ~margin_valid,
            int(ParaxialOpticsStatus.BRANCH_MARGIN_VIOLATION),
            jnp.where(
                ~finite,
                int(ParaxialOpticsStatus.NONFINITE_JACOBIAN),
                int(ParaxialOpticsStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    return DifferentialRayMap(
        input_reference=reference,
        output_reference=output_reference,
        jacobian=jacobian,
        branch_margin=branch_margin,
        finite=finite,
        valid=valid,
        status=status,
        input_frame_id=_frame_id(input_frame),
        output_frame_id=_frame_id(output_frame),
        source_prepared_id=prepared.prepared_id,
        coordinate_convention=_COORDINATE_CONVENTION,
    )


class ParaxialOpticsPlan(StrictModule, NonTrainableState):
    """Preparation-time cache specification for one exact differential ray map."""

    sequential_plan: SequentialOpticsPlan
    input_frame: RigidFrame
    output_frame: RigidFrame
    chief_ray_coordinates: Array
    input_refractive_index: float = eqx.field(static=True)
    output_refractive_index: float = eqx.field(static=True)
    maximum_transverse_perturbation: float = eqx.field(static=True)
    maximum_angular_perturbation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sequential_plan: SequentialOpticsPlan,
        input_frame: RigidFrame,
        output_frame: RigidFrame,
        chief_ray_coordinates: ArrayLike,
        *,
        input_refractive_index: float,
        output_refractive_index: float,
        maximum_transverse_perturbation: float,
        maximum_angular_perturbation: float,
    ):
        if not isinstance(sequential_plan, SequentialOpticsPlan):
            raise TypeError("sequential_plan must be a SequentialOpticsPlan.")
        _validate_frame(input_frame, "input_frame")
        _validate_frame(output_frame, "output_frame")
        chief = _host_coordinates(chief_ray_coordinates, "chief_ray_coordinates")
        input_index = _real_scalar(input_refractive_index, "input_refractive_index")
        output_index = _real_scalar(output_refractive_index, "output_refractive_index")
        transverse = _real_scalar(
            maximum_transverse_perturbation, "maximum_transverse_perturbation"
        )
        angular = _real_scalar(
            maximum_angular_perturbation, "maximum_angular_perturbation"
        )
        if input_index <= 0.0 or output_index <= 0.0 or transverse < 0.0 or angular < 0.0:
            raise ValueError("Paraxial indices and trust-envelope bounds are invalid.")
        if not np.isclose(
            input_index,
            float(np.asarray(sequential_plan.refractive_indices[0])),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "input_refractive_index must match the sequential prescription."
            )
        if not np.isclose(
            output_index,
            float(np.asarray(sequential_plan.refractive_indices[-1])),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "output_refractive_index must match the sequential prescription."
            )
        self.sequential_plan = sequential_plan
        self.input_frame = input_frame
        self.output_frame = output_frame
        self.chief_ray_coordinates = jnp.asarray(chief)
        self.input_refractive_index = input_index
        self.output_refractive_index = output_index
        self.maximum_transverse_perturbation = transverse
        self.maximum_angular_perturbation = angular
        self.plan_id = canonical_fingerprint(
            {
                "kind": "paraxial-optics-plan",
                "sequential_plan_id": sequential_plan.plan_id,
                "input_frame_id": _frame_id(input_frame),
                "output_frame_id": _frame_id(output_frame),
                "chief_ray_coordinates": array_tree_fingerprint(chief),
                "input_refractive_index": input_index,
                "output_refractive_index": output_index,
                "maximum_transverse_perturbation": transverse,
                "maximum_angular_perturbation": angular,
                "coordinate_convention": _COORDINATE_CONVENTION,
            }
        )

    def prepare(
        self, prepared_sequential: PreparedSequentialOptics | None = None, /
    ) -> "PreparedParaxialOptics":
        """Prepare the exact prescription once and cache its chief-ray Jacobian."""
        exact = (
            self.sequential_plan.prepare()
            if prepared_sequential is None
            else prepared_sequential
        )
        if not isinstance(exact, PreparedSequentialOptics):
            raise TypeError("prepared_sequential must be a PreparedSequentialOptics.")
        if exact.source_plan_id != self.sequential_plan.plan_id:
            raise ValueError("prepared_sequential belongs to another exact prescription.")
        differential_map = linearize_sequential_optics(
            exact,
            self.chief_ray_coordinates,
            input_frame=self.input_frame,
            output_frame=self.output_frame,
            input_refractive_index=self.input_refractive_index,
            output_refractive_index=self.output_refractive_index,
        )
        return PreparedParaxialOptics(self, differential_map)


class PreparedParaxialOptics(StrictModule, NonTrainableState):
    """Cached affine map with explicit transverse and angular validity bounds."""

    __hash__ = object.__hash__

    differential_map: DifferentialRayMap
    input_refractive_index: float = eqx.field(static=True)
    maximum_transverse_perturbation: float = eqx.field(static=True)
    maximum_angular_perturbation: float = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ParaxialOpticsPlan, differential_map: DifferentialRayMap, /):
        if not isinstance(plan, ParaxialOpticsPlan):
            raise TypeError("plan must be a ParaxialOpticsPlan.")
        if not isinstance(differential_map, DifferentialRayMap):
            raise TypeError("differential_map must be a DifferentialRayMap.")
        if differential_map.source_prepared_id == "":
            raise ValueError("The differential map must identify its exact prescription.")
        if differential_map.input_frame_id != _frame_id(
            plan.input_frame
        ) or differential_map.output_frame_id != _frame_id(plan.output_frame):
            raise ValueError("The differential map belongs to different frame charts.")
        if not np.array_equal(
            np.asarray(differential_map.input_reference),
            np.asarray(plan.chief_ray_coordinates),
        ):
            raise ValueError("The differential map uses another chief ray.")
        self.differential_map = differential_map
        self.input_refractive_index = plan.input_refractive_index
        self.maximum_transverse_perturbation = plan.maximum_transverse_perturbation
        self.maximum_angular_perturbation = plan.maximum_angular_perturbation
        self.source_plan_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-paraxial-optics",
                "source_plan_id": plan.plan_id,
                "exact_prepared_id": differential_map.source_prepared_id,
                "map": array_tree_fingerprint(
                    (
                        np.asarray(differential_map.input_reference),
                        np.asarray(differential_map.output_reference),
                        np.asarray(differential_map.jacobian),
                        np.asarray(differential_map.branch_margin),
                    )
                ),
                "maximum_transverse_perturbation": plan.maximum_transverse_perturbation,
                "maximum_angular_perturbation": plan.maximum_angular_perturbation,
            }
        )

    def execute(self, coordinates: ArrayLike, /) -> ParaxialOpticsResult:
        """Evaluate the cached affine map and refuse points outside its envelope."""
        inputs = jnp.asarray(coordinates)
        if inputs.ndim < 1 or inputs.shape[-1] != 4:
            raise ValueError("coordinates must have shape B + (4,).")
        if jnp.issubdtype(inputs.dtype, jnp.complexfloating) or inputs.dtype == jnp.dtype(
            bool
        ):
            raise TypeError("Paraxial coordinates must be real numeric arrays.")
        dtype = jnp.result_type(inputs.dtype, self.differential_map.jacobian.dtype, 0.0)
        values = inputs.astype(dtype)
        reference = self.differential_map.input_reference.astype(dtype)
        delta = values - reference
        output = self.differential_map.output_reference.astype(dtype) + ein.contract(
            "ij,...j->...i", self.differential_map.jacobian.astype(dtype), delta
        )
        transverse_excursion = jnp.sqrt(jnp.sum(delta[..., :2] ** 2, axis=-1))
        angular_excursion = jnp.sqrt(
            jnp.sum((delta[..., 2:] / self.input_refractive_index) ** 2, axis=-1)
        )
        input_finite = jnp.all(jnp.isfinite(values), axis=-1)
        output_finite = jnp.all(jnp.isfinite(output), axis=-1)
        finite = input_finite & output_finite
        within_envelope = (
            input_finite
            & (transverse_excursion <= self.maximum_transverse_perturbation)
            & (angular_excursion <= self.maximum_angular_perturbation)
        )
        map_valid = self.differential_map.valid
        successful = map_valid & finite & within_envelope
        status = jnp.where(
            ~input_finite,
            int(ParaxialOpticsStatus.NONFINITE_INPUT),
            jnp.where(
                ~map_valid,
                int(ParaxialOpticsStatus.INVALID_DIFFERENTIAL_MAP),
                jnp.where(
                    ~within_envelope,
                    int(ParaxialOpticsStatus.OUTSIDE_TRUST_ENVELOPE),
                    jnp.where(
                        ~output_finite,
                        int(ParaxialOpticsStatus.NUMERICAL_FAILURE),
                        int(ParaxialOpticsStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return ParaxialOpticsResult(
            coordinates=output,
            transverse_excursion=transverse_excursion,
            angular_excursion=angular_excursion,
            within_envelope=within_envelope,
            finite=finite,
            status=status,
            successful=successful,
            plan_id=self.prepared_id,
        )


__all__ = [
    "DifferentialRayMap",
    "ParaxialOpticsPlan",
    "ParaxialOpticsResult",
    "ParaxialOpticsStatus",
    "PreparedParaxialOptics",
    "linearize_sequential_optics",
]
