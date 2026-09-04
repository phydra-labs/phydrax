#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import RigidFrame
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ..geometric._interface import OpticalRayState
from ..geometric._paraxial import (
    _COORDINATE_CONVENTION,
    _frame_id,
    DifferentialRayMap,
)


class BeamletStatus(IntEnum):
    """Terminal status for differential Gaussian-beamlet operations."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_CHIEF_RAY = 2
    DIFFERENTIAL_MAP_INVALID = 3
    FRAME_MISMATCH = 4
    TOPOLOGY_MISMATCH = 5
    INVARIANT_VIOLATION = 6
    CAUSTIC = 7
    SINGULAR_LAGRANGIAN_BLOCK = 8
    NO_VALID_BEAMLETS = 9
    PARTIAL_RECONSTRUCTION = 10
    COORDINATE_CONVENTION_MISMATCH = 11


class BeamletFrame(StrictModule, NonTrainableState):
    """A canonical rigid ray chart for deterministic beamlet transport."""

    frame: RigidFrame
    frame_id: str = eqx.field(static=True)

    def __init__(self, frame: RigidFrame, /):
        if not isinstance(frame, RigidFrame) or frame.dimension != 3:
            raise TypeError("frame must be a three-dimensional RigidFrame.")
        self.frame = frame
        self.frame_id = _frame_id(frame)


class GaussianBeamletState(StrictModule):
    """Chief rays plus a complex 4-by-2 transverse Lagrangian state.

    The first two rows of ``lagrangian_state`` are H and the last two are U.
    The phase-space convention is ``(u, v, n theta_u, n theta_v)`` and the
    field curvature is ``U H^{-1}``. Leading axes enumerate independent
    beamlets; no runtime scene or adaptive splitting state is retained.
    """

    chief_ray: OpticalRayState
    frame: BeamletFrame
    lagrangian_state: Array
    amplitudes: Array
    medium_wavenumbers: Array
    angular_frequency: Array
    reference_invariant: Array
    initial_determinant: Array
    determinant_phase: Array
    topology_index: Array
    valid: Array
    status: Array
    topology_id: str = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        chief_ray: OpticalRayState,
        frame: BeamletFrame,
        lagrangian_state: ArrayLike,
        amplitudes: ArrayLike,
        medium_wavenumbers: ArrayLike,
        angular_frequency: ArrayLike,
        /,
        *,
        topology_id: str,
        source_prepared_id: str,
        reference_invariant: ArrayLike | None = None,
        initial_determinant: ArrayLike | None = None,
        determinant_phase: ArrayLike | None = None,
        topology_index: ArrayLike | None = None,
        valid: ArrayLike | None = None,
        status: ArrayLike | None = None,
    ):
        if not isinstance(chief_ray, OpticalRayState) or not isinstance(
            frame, BeamletFrame
        ):
            raise TypeError("Expected an OpticalRayState and BeamletFrame.")
        origins = jnp.asarray(chief_ray.origins)
        directions = jnp.asarray(chief_ray.directions)
        if origins.shape != directions.shape or origins.shape[-1:] != (3,):
            raise ValueError("chief_ray origins and directions must have shape (..., 3).")
        leading = origins.shape[:-1]
        lagrangian = jnp.asarray(lagrangian_state)
        if lagrangian.shape != leading + (4, 2):
            raise ValueError("lagrangian_state must have shape chief_shape + (4, 2).")
        if not jnp.issubdtype(lagrangian.dtype, jnp.complexfloating):
            lagrangian = lagrangian.astype(jnp.result_type(lagrangian.dtype, 1j))
        amplitudes_ = jnp.asarray(amplitudes, dtype=lagrangian.dtype)
        wavenumbers = jnp.asarray(medium_wavenumbers, dtype=lagrangian.dtype)
        omega = jnp.asarray(angular_frequency)
        if amplitudes_.shape != leading or wavenumbers.shape != leading:
            raise ValueError(
                "amplitudes and medium_wavenumbers must match chief leading shape."
            )
        if omega.shape != () or not jnp.issubdtype(omega.dtype, jnp.floating):
            raise ValueError("angular_frequency must be a real scalar.")
        h = lagrangian[..., :2, :]
        determinant = _determinant_2x2(h)
        invariant = beamlet_lagrange_invariant(lagrangian)
        reference = (
            invariant
            if reference_invariant is None
            else jnp.asarray(reference_invariant, dtype=lagrangian.dtype)
        )
        initial = (
            determinant
            if initial_determinant is None
            else jnp.asarray(initial_determinant, dtype=lagrangian.dtype)
        )
        phase = (
            jnp.angle(determinant)
            if determinant_phase is None
            else jnp.asarray(determinant_phase, dtype=origins.dtype)
        )
        index = (
            jnp.zeros(leading, dtype=jnp.int32)
            if topology_index is None
            else jnp.asarray(topology_index, dtype=jnp.int32)
        )
        requested_valid = (
            jnp.ones(leading, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        requested_status = (
            jnp.zeros(leading, dtype=jnp.int32)
            if status is None
            else jnp.asarray(status, dtype=jnp.int32)
        )
        if reference.shape != leading + (2, 2):
            raise ValueError("reference_invariant must have shape chief_shape + (2, 2).")
        if initial.shape != leading or phase.shape != leading:
            raise ValueError("Determinant records must match chief leading shape.")
        if (
            index.shape != leading
            or requested_valid.shape != leading
            or requested_status.shape != leading
        ):
            raise ValueError("Beamlet status records must match chief leading shape.")
        chief_valid, chief_finite = _chief_ray_validity(chief_ray)
        finite_data = (
            chief_finite
            & jnp.all(jnp.isfinite(lagrangian), axis=(-2, -1))
            & jnp.all(jnp.isfinite(reference), axis=(-2, -1))
            & jnp.isfinite(initial)
            & jnp.isfinite(phase)
            & jnp.isfinite(amplitudes_)
            & jnp.isfinite(wavenumbers)
            & jnp.isfinite(omega)
        )
        well_defined = (jnp.abs(wavenumbers) > 0.0) & (omega > 0.0)
        base_valid = finite_data & chief_valid & well_defined
        base_status = jnp.where(
            ~finite_data,
            int(BeamletStatus.NONFINITE_INPUT),
            jnp.where(
                ~chief_valid,
                int(BeamletStatus.INVALID_CHIEF_RAY),
                int(BeamletStatus.SINGULAR_LAGRANGIAN_BLOCK),
            ),
        ).astype(jnp.int32)
        valid_ = requested_valid & base_valid
        status_ = jnp.where(base_valid, requested_status, base_status)
        identifiers = (str(topology_id), str(source_prepared_id))
        if any(not identifier for identifier in identifiers):
            raise ValueError("Beamlet topology and source IDs must be non-empty.")
        self.chief_ray = chief_ray
        self.frame = frame
        self.lagrangian_state = lagrangian
        self.amplitudes = amplitudes_
        self.medium_wavenumbers = wavenumbers
        self.angular_frequency = omega
        self.reference_invariant = reference
        self.initial_determinant = initial
        self.determinant_phase = phase
        self.topology_index = index
        self.valid = valid_
        self.status = status_
        self.topology_id = identifiers[0]
        self.source_prepared_id = identifiers[1]

    @property
    def frame_id(self) -> str:
        return self.frame.frame_id

    @property
    def transverse_basis(self) -> Array:
        return jnp.broadcast_to(
            self.frame.frame.rotation[:, :2],
            self.beamlet_shape + (3, 2),
        )

    @property
    def h(self) -> Array:
        return self.lagrangian_state[..., :2, :]

    @property
    def u(self) -> Array:
        return self.lagrangian_state[..., 2:, :]

    @property
    def beamlet_shape(self) -> tuple[int, ...]:
        return self.lagrangian_state.shape[:-2]


class BeamletCurvatureResult(StrictModule):
    curvature: Array
    determinant: Array
    condition_estimate: Array
    successful: Array
    status: Array


class BeamletTransportEvidence(StrictModule, NonTrainableState):
    invariant_error: Array
    symplectic_error: Array
    caustic_distance: Array
    determinant: Array
    phase_increment: Array
    branch_margin: Array
    finite: Array
    frame_consistent: Array
    coordinate_consistent: Array
    topology_consistent: Array
    valid: Array
    status: Array


class GaussianBeamletTransportResult(StrictModule):
    state: GaussianBeamletState
    evidence: BeamletTransportEvidence

    @property
    def successful(self) -> Array:
        return self.evidence.valid


class GaussianWaistSpecification(StrictModule):
    """Astigmatic 1/e field radii and deterministic transverse rotation."""

    radii: Array
    rotation_angle: Array

    def __init__(self, radii: ArrayLike, rotation_angle: ArrayLike = 0.0):
        radii_ = jnp.asarray(radii)
        if not jnp.issubdtype(radii_.dtype, jnp.floating):
            radii_ = radii_.astype(float)
        if radii_.shape[-1:] != (2,):
            raise ValueError("Gaussian waist radii must have shape (..., 2).")
        angle = jnp.asarray(rotation_angle, dtype=radii_.dtype)
        if angle.shape != radii_.shape[:-1]:
            angle = jnp.broadcast_to(angle, radii_.shape[:-1])
        radii_ = eqx.error_if(
            radii_,
            jnp.any(~jnp.isfinite(radii_)) | jnp.any(radii_ <= 0.0),
            "Gaussian waist radii must be finite and positive.",
        )
        angle = eqx.error_if(
            angle,
            jnp.any(~jnp.isfinite(angle)),
            "Gaussian waist rotation must be finite.",
        )
        self.radii = radii_
        self.rotation_angle = angle


def _determinant_2x2(matrix: Array, /) -> Array:
    return matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]


def _chief_ray_validity(chief_ray: OpticalRayState, /) -> tuple[Array, Array]:
    origins = jnp.asarray(chief_ray.origins)
    directions = jnp.asarray(chief_ray.directions)
    refractive_indices = jnp.asarray(chief_ray.refractive_indices)
    geometric_paths = jnp.asarray(chief_ray.geometric_path_lengths)
    optical_paths = jnp.asarray(chief_ray.optical_path_lengths)
    finite = (
        jnp.all(jnp.isfinite(origins), axis=-1)
        & jnp.all(jnp.isfinite(directions), axis=-1)
        & jnp.isfinite(refractive_indices)
        & jnp.isfinite(geometric_paths)
        & jnp.isfinite(optical_paths)
    )
    direction_norm = jnp.sqrt(jnp.sum(directions * directions, axis=-1))
    valid = (
        finite
        & (refractive_indices > 0.0)
        & (jnp.abs(direction_norm - 1.0) <= 64.0 * jnp.finfo(directions.dtype).eps)
    )
    return valid, finite


def deterministic_transverse_basis(directions: ArrayLike, /) -> Array:
    """Construct right-handed transverse bases with a deterministic tie break."""
    direction = jnp.asarray(directions)
    if direction.shape[-1:] != (3,):
        raise ValueError("directions must have shape (..., 3).")
    norm = jnp.sqrt(jnp.sum(direction * direction, axis=-1, keepdims=True))
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    normal = direction / safe_norm
    reference_index = jnp.argmin(jnp.abs(normal), axis=-1)
    reference = jax.nn.one_hot(reference_index, 3, dtype=normal.dtype)
    first = reference - jnp.sum(reference * normal, axis=-1, keepdims=True) * normal
    first_norm = jnp.sqrt(jnp.sum(first * first, axis=-1, keepdims=True))
    first = first / jnp.where(first_norm > 0.0, first_norm, 1.0)
    second = jnp.cross(normal, first)
    return jnp.stack((first, second), axis=-1)


def transport_transverse_basis(
    previous_basis: ArrayLike,
    directions: ArrayLike,
    /,
) -> Array:
    """Parallel-transport a moving frame, deterministically resolving degeneracy."""
    previous = jnp.asarray(previous_basis)
    direction = jnp.asarray(directions)
    if previous.shape != direction.shape[:-1] + (3, 2):
        raise ValueError("previous_basis and directions have incompatible shapes.")
    norm = jnp.sqrt(jnp.sum(direction * direction, axis=-1, keepdims=True))
    normal = direction / jnp.where(norm > 0.0, norm, 1.0)
    first = previous[..., :, 0]
    projected = first - jnp.sum(first * normal, axis=-1, keepdims=True) * normal
    projected_norm = jnp.sqrt(jnp.sum(projected * projected, axis=-1, keepdims=True))
    canonical = deterministic_transverse_basis(normal)[..., :, 0]
    usable = projected_norm > 32.0 * jnp.finfo(normal.dtype).eps
    first = jnp.where(
        usable,
        projected / jnp.where(projected_norm > 0.0, projected_norm, 1.0),
        canonical,
    )
    second = jnp.cross(normal, first)
    first = jnp.cross(second, normal)
    return jnp.stack((first, second), axis=-1)


def _host_moving_basis(
    direction: ArrayLike,
    /,
    *,
    previous_first: ArrayLike | None = None,
) -> np.ndarray:
    direction_host = np.asarray(direction, dtype=float)
    if direction_host.shape != (3,) or not np.all(np.isfinite(direction_host)):
        raise ValueError("Beamlet-frame direction must be a finite three-vector.")
    norm = np.linalg.norm(direction_host)
    if norm <= 0.0:
        raise ValueError("Beamlet-frame direction must have nonzero length.")
    normal = direction_host / norm
    if previous_first is None:
        reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
        first = reference - np.dot(reference, normal) * normal
    else:
        previous = np.asarray(previous_first, dtype=float)
        if previous.shape != (3,) or not np.all(np.isfinite(previous)):
            raise ValueError("Previous frame direction must be a finite three-vector.")
        first = previous - np.dot(previous, normal) * normal
        if np.linalg.norm(first) <= 32.0 * np.finfo(float).eps:
            reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
            first = reference - np.dot(reference, normal) * normal
    first = first / np.linalg.norm(first)
    second = np.cross(normal, first)
    first = np.cross(second, normal)
    return np.column_stack((first, second, normal))


def deterministic_beamlet_frame(
    origin: ArrayLike,
    direction: ArrayLike,
    /,
) -> BeamletFrame:
    """Create a deterministic right-handed ray frame at one chief-ray point."""
    origin_host = np.asarray(origin, dtype=float)
    if origin_host.shape != (3,) or not np.all(np.isfinite(origin_host)):
        raise ValueError("Beamlet-frame origin must be a finite three-vector.")
    rotation = _host_moving_basis(direction)
    return BeamletFrame(RigidFrame(rotation, origin_host))


def transport_beamlet_frame(
    previous: BeamletFrame,
    origin: ArrayLike,
    direction: ArrayLike,
    /,
) -> BeamletFrame:
    """Construct the deterministic rotation-minimizing next chief-ray frame."""
    if not isinstance(previous, BeamletFrame):
        raise TypeError("previous must be a BeamletFrame.")
    origin_host = np.asarray(origin, dtype=float)
    if origin_host.shape != (3,) or not np.all(np.isfinite(origin_host)):
        raise ValueError("Beamlet-frame origin must be a finite three-vector.")
    rotation = _host_moving_basis(
        direction,
        previous_first=np.asarray(previous.frame.rotation)[:, 0],
    )
    return BeamletFrame(RigidFrame(rotation, origin_host))


def beamlet_lagrange_invariant(lagrangian_state: ArrayLike, /) -> Array:
    """Return HᴴU - UᴴH for a complex transverse Lagrangian plane."""
    lagrangian = jnp.asarray(lagrangian_state)
    if lagrangian.shape[-2:] != (4, 2):
        raise ValueError("lagrangian_state must have shape (..., 4, 2).")
    h = lagrangian[..., :2, :]
    u = lagrangian[..., 2:, :]
    return contract("...ji,...jk->...ik", jnp.conj(h), u) - contract(
        "...ji,...jk->...ik", jnp.conj(u), h
    )


def beamlet_curvature(
    state: GaussianBeamletState,
    /,
    *,
    solve_plan: SmallLinearSolvePlan | None = None,
) -> BeamletCurvatureResult:
    """Compute U H⁻¹ using the native fixed-size linear solve."""
    if not isinstance(state, GaussianBeamletState):
        raise TypeError("state must be a GaussianBeamletState.")
    resolved_plan = SmallLinearSolvePlan(2) if solve_plan is None else solve_plan
    if (
        not isinstance(resolved_plan, SmallLinearSolvePlan)
        or resolved_plan.dimension != 2
    ):
        raise TypeError("solve_plan must be a two-dimensional SmallLinearSolvePlan.")
    solve = solve_small_linear(
        resolved_plan,
        jnp.swapaxes(state.h, -1, -2),
        jnp.swapaxes(state.u, -1, -2),
    )
    curvature = jnp.swapaxes(solve.value, -1, -2)
    status = jnp.where(
        solve.successful,
        int(BeamletStatus.SUCCESS),
        int(BeamletStatus.SINGULAR_LAGRANGIAN_BLOCK),
    ).astype(jnp.int32)
    return BeamletCurvatureResult(
        curvature,
        solve.determinant,
        solve.condition_estimate,
        solve.successful,
        status,
    )


def gaussian_beamlets_at_waist(
    chief_ray: OpticalRayState,
    waist: GaussianWaistSpecification,
    frame: BeamletFrame,
    medium_wavenumbers: ArrayLike,
    angular_frequency: ArrayLike,
    /,
    *,
    amplitudes: ArrayLike = 1.0,
    topology_id: str,
    source_prepared_id: str,
) -> GaussianBeamletState:
    """Create fundamental astigmatic Gaussian beamlets at their waist planes."""
    if (
        not isinstance(chief_ray, OpticalRayState)
        or not isinstance(waist, GaussianWaistSpecification)
        or not isinstance(frame, BeamletFrame)
    ):
        raise TypeError(
            "Expected an OpticalRayState, GaussianWaistSpecification, and BeamletFrame."
        )
    directions = jnp.asarray(chief_ray.directions)
    leading = directions.shape[:-1]
    radii = jnp.broadcast_to(waist.radii, leading + (2,))
    angle = jnp.broadcast_to(waist.rotation_angle, leading)
    wavenumbers = jnp.broadcast_to(jnp.asarray(medium_wavenumbers), leading)
    amplitudes_ = jnp.broadcast_to(jnp.asarray(amplitudes), leading)
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    rotation = jnp.stack((cosine, -sine, sine, cosine), axis=-1).reshape(leading + (2, 2))
    diagonal = jnp.zeros(leading + (2, 2), dtype=jnp.result_type(wavenumbers, 1j))
    inverse_radius_squared = 1.0 / (radii * radii)
    diagonal = diagonal.at[..., 0, 0].set(inverse_radius_squared[..., 0])
    diagonal = diagonal.at[..., 1, 1].set(inverse_radius_squared[..., 1])
    rotated = contract("...ik,...kl,...jl->...ij", rotation, diagonal, rotation)
    curvature = (2j / wavenumbers[..., None, None]) * rotated
    identity = jnp.broadcast_to(jnp.eye(2, dtype=curvature.dtype), leading + (2, 2))
    lagrangian = jnp.concatenate((identity, curvature), axis=-2)
    ray_valid, ray_finite = _chief_ray_validity(chief_ray)
    finite = (
        ray_finite
        & jnp.all(jnp.isfinite(lagrangian), axis=(-2, -1))
        & jnp.isfinite(amplitudes_)
        & jnp.isfinite(wavenumbers)
        & (jnp.abs(wavenumbers) > 0.0)
    )
    valid = ray_valid & finite
    status = jnp.where(
        valid,
        int(BeamletStatus.SUCCESS),
        jnp.where(
            finite,
            int(BeamletStatus.INVALID_CHIEF_RAY),
            int(BeamletStatus.NONFINITE_INPUT),
        ),
    ).astype(jnp.int32)
    return GaussianBeamletState(
        chief_ray,
        frame,
        lagrangian,
        amplitudes_,
        wavenumbers,
        angular_frequency,
        topology_id=topology_id,
        source_prepared_id=source_prepared_id,
        valid=valid,
        status=status,
    )


def _symplectic_error(jacobian: Array, /) -> Array:
    symplectic_form = jnp.asarray(
        (
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
        ),
        dtype=jacobian.dtype,
    )
    pulled_back = contract(
        "...ji,jk,...kl->...il",
        jacobian,
        symplectic_form,
        jacobian,
    )
    defect = pulled_back - symplectic_form
    return jnp.sqrt(jnp.sum(jnp.abs(defect) ** 2, axis=(-2, -1)))


def transport_gaussian_beamlets(
    state: GaussianBeamletState,
    differential_map: DifferentialRayMap,
    output_chief_ray: OpticalRayState,
    output_frame: BeamletFrame,
    /,
    *,
    invariant_tolerance: float = 1e-7,
    symplectic_tolerance: float = 1e-7,
    caustic_tolerance: float = 1e-10,
    output_topology_id: str | None = None,
) -> GaussianBeamletTransportResult:
    """Transport H/U through a real differential ray map with evidence."""
    if not isinstance(state, GaussianBeamletState):
        raise TypeError("state must be a GaussianBeamletState.")
    if (
        not isinstance(differential_map, DifferentialRayMap)
        or not isinstance(output_chief_ray, OpticalRayState)
        or not isinstance(output_frame, BeamletFrame)
    ):
        raise TypeError(
            "Expected a DifferentialRayMap, output OpticalRayState, and BeamletFrame."
        )
    jacobian = jnp.asarray(differential_map.jacobian)
    if jacobian.shape != state.beamlet_shape + (4, 4):
        raise ValueError(
            "Differential ray-map shape must match the beamlet leading shape."
        )
    lagrangian = contract("...ij,...jk->...ik", jacobian, state.lagrangian_state)
    invariant = beamlet_lagrange_invariant(lagrangian)
    invariant_scale = jnp.maximum(
        jnp.sqrt(jnp.sum(jnp.abs(state.reference_invariant) ** 2, axis=(-2, -1))),
        jnp.asarray(1.0, dtype=jacobian.dtype),
    )
    invariant_error = (
        jnp.sqrt(
            jnp.sum(
                jnp.abs(invariant - state.reference_invariant) ** 2,
                axis=(-2, -1),
            )
        )
        / invariant_scale
    )
    symplectic_error = _symplectic_error(jacobian)
    determinant = _determinant_2x2(lagrangian[..., :2, :])
    h_scale = jnp.maximum(
        jnp.sum(jnp.abs(lagrangian[..., :2, :]) ** 2, axis=(-2, -1)),
        jnp.finfo(jacobian.dtype).tiny,
    )
    caustic_distance = jnp.abs(determinant) / h_scale
    previous_determinant = _determinant_2x2(state.h)
    safe_previous = jnp.where(
        jnp.abs(previous_determinant) > 0.0,
        previous_determinant,
        1.0,
    )
    phase_increment = jnp.angle(determinant / safe_previous)
    determinant_phase = state.determinant_phase + phase_increment
    topology_index = jnp.floor((determinant_phase + jnp.pi) / (2.0 * jnp.pi)).astype(
        jnp.int32
    )
    target_topology = (
        state.topology_id if output_topology_id is None else str(output_topology_id)
    )
    frame_consistent_static = (
        state.frame_id == differential_map.input_frame_id
        and output_frame.frame_id == differential_map.output_frame_id
    )
    coordinate_consistent_static = (
        differential_map.coordinate_convention == _COORDINATE_CONVENTION
    )
    topology_consistent_static = (
        state.source_prepared_id == differential_map.source_prepared_id
        and target_topology == state.topology_id
    )
    frame_consistent = jnp.full(state.beamlet_shape, frame_consistent_static)
    coordinate_consistent = jnp.full(state.beamlet_shape, coordinate_consistent_static)
    topology_consistent = jnp.full(state.beamlet_shape, topology_consistent_static)
    chief_valid, chief_finite = _chief_ray_validity(output_chief_ray)
    finite = (
        chief_finite
        & jnp.all(jnp.isfinite(jacobian), axis=(-2, -1))
        & jnp.all(jnp.isfinite(lagrangian), axis=(-2, -1))
        & jnp.isfinite(invariant_error)
        & jnp.isfinite(symplectic_error)
        & jnp.isfinite(caustic_distance)
    )
    map_valid = jnp.asarray(differential_map.valid, dtype=bool)
    roundoff_tolerance = 64.0 * jnp.finfo(jacobian.dtype).eps
    invariant_valid = invariant_error <= jnp.maximum(
        invariant_tolerance, roundoff_tolerance
    )
    symplectic_valid = symplectic_error <= jnp.maximum(
        symplectic_tolerance, roundoff_tolerance
    )
    away_from_caustic = caustic_distance > jnp.maximum(
        caustic_tolerance, roundoff_tolerance
    )
    valid = (
        state.valid
        & map_valid
        & chief_valid
        & finite
        & frame_consistent
        & coordinate_consistent
        & topology_consistent
        & invariant_valid
        & symplectic_valid
        & away_from_caustic
    )
    status = jnp.full(state.beamlet_shape, int(BeamletStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(~away_from_caustic, int(BeamletStatus.CAUSTIC), status)
    status = jnp.where(
        ~(invariant_valid & symplectic_valid),
        int(BeamletStatus.INVARIANT_VIOLATION),
        status,
    )
    status = jnp.where(~topology_consistent, int(BeamletStatus.TOPOLOGY_MISMATCH), status)
    status = jnp.where(
        ~coordinate_consistent,
        int(BeamletStatus.COORDINATE_CONVENTION_MISMATCH),
        status,
    )
    status = jnp.where(~frame_consistent, int(BeamletStatus.FRAME_MISMATCH), status)
    status = jnp.where(~map_valid, int(BeamletStatus.DIFFERENTIAL_MAP_INVALID), status)
    status = jnp.where(~chief_valid, int(BeamletStatus.INVALID_CHIEF_RAY), status)
    status = jnp.where(~finite, int(BeamletStatus.NONFINITE_INPUT), status)
    next_state = GaussianBeamletState(
        output_chief_ray,
        output_frame,
        lagrangian,
        state.amplitudes,
        state.medium_wavenumbers,
        state.angular_frequency,
        topology_id=target_topology,
        source_prepared_id=state.source_prepared_id,
        reference_invariant=state.reference_invariant,
        initial_determinant=state.initial_determinant,
        determinant_phase=determinant_phase,
        topology_index=topology_index,
        valid=valid,
        status=status,
    )
    evidence = BeamletTransportEvidence(
        invariant_error,
        symplectic_error,
        caustic_distance,
        determinant,
        phase_increment,
        jnp.asarray(differential_map.branch_margin),
        finite,
        frame_consistent,
        coordinate_consistent,
        topology_consistent,
        valid,
        status,
    )
    return GaussianBeamletTransportResult(next_state, evidence)


__all__ = [
    "BeamletCurvatureResult",
    "BeamletFrame",
    "BeamletStatus",
    "BeamletTransportEvidence",
    "GaussianBeamletState",
    "GaussianBeamletTransportResult",
    "GaussianWaistSpecification",
    "beamlet_curvature",
    "beamlet_lagrange_invariant",
    "deterministic_beamlet_frame",
    "deterministic_transverse_basis",
    "gaussian_beamlets_at_waist",
    "transport_gaussian_beamlets",
    "transport_beamlet_frame",
    "transport_transverse_basis",
]
