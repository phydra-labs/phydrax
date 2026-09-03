#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import StateLayout
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    JacobianLinearOperator,
    prepare_linearization,
)
from ._rod_dynamics import (
    evaluate_rod,
    PreparedRod,
    rod_potential_energy,
    RodEvaluation,
    RodState,
)


def _real_array(name: str, value: ArrayLike, rank: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _planar_rotation_matrix(angle: Array, /) -> Array:
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    return jnp.stack((cosine, -sine, sine, cosine), axis=-1).reshape(
        angle.shape + (2, 2)
    )


class ReducedRodPlan(StrictModule, NonTrainableState):
    """Finite planar strain basis for one prepared native Cosserat rod.

    The final axis of each basis is the reduced-coordinate axis. Stretch/shear
    basis values have native shape ``(segments, 2, coordinates)`` and
    bend/twist basis values have shape ``(junctions, 1, coordinates)``.
    Reduced coefficients always measure strain from the native rod rest state;
    ``reference_coefficients`` declares the default prepared state without
    changing that zero-strain origin.
    """

    stretch_shear_basis: Array
    bend_twist_basis: Array
    reference_coefficients: Array
    fixed_base_position: Array | None
    fixed_base_orientation: Array | None
    coordinate_count: int = eqx.field(static=True)
    quadrature_tolerance: float = eqx.field(static=True)
    certification_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stretch_shear_basis: ArrayLike,
        bend_twist_basis: ArrayLike,
        /,
        *,
        reference_coefficients: ArrayLike | None = None,
        fixed_base_position: ArrayLike | None = None,
        fixed_base_orientation: ArrayLike | None = None,
        quadrature_tolerance: float = 1.0e-6,
        certification_tolerance: float = 1.0e-6,
        plan_id: str | None = None,
    ):
        stretch = _real_array("stretch_shear_basis", stretch_shear_basis, 3)
        bend = _real_array("bend_twist_basis", bend_twist_basis, 3)
        if stretch.shape[0] < 1 or stretch.shape[1] != 2 or stretch.shape[2] < 1:
            raise ValueError(
                "stretch_shear_basis must have shape (segments, 2, coordinates)."
            )
        coordinate_count = int(stretch.shape[2])
        if bend.shape != (stretch.shape[0] - 1, 1, coordinate_count):
            raise ValueError(
                "bend_twist_basis must have shape "
                "(segments - 1, 1, coordinates)."
            )
        reference = (
            np.zeros((coordinate_count,), dtype=stretch.dtype)
            if reference_coefficients is None
            else _real_array("reference_coefficients", reference_coefficients, 1)
        )
        if reference.shape != (coordinate_count,):
            raise ValueError(
                "reference_coefficients must contain one value per reduced coordinate."
            )
        dtype = stretch.dtype
        arrays = {
            "stretch_shear_basis": stretch,
            "bend_twist_basis": bend.astype(dtype, copy=False),
            "reference_coefficients": reference.astype(dtype, copy=False),
        }
        basis_matrix = np.concatenate(
            (
                arrays["stretch_shear_basis"].reshape((-1, coordinate_count)),
                arrays["bend_twist_basis"].reshape((-1, coordinate_count)),
            ),
            axis=0,
        )
        if np.linalg.matrix_rank(basis_matrix) != coordinate_count:
            raise ValueError(
                "The combined rod strain basis must have full column rank."
            )

        base_position = (
            None
            if fixed_base_position is None
            else _real_array("fixed_base_position", fixed_base_position, 1)
        )
        if base_position is not None and base_position.shape != (2,):
            raise ValueError(
                "fixed_base_position must have shape (2,) for a planar rod."
            )
        base_orientation = (
            None
            if fixed_base_orientation is None
            else _real_array("fixed_base_orientation", fixed_base_orientation, 0)
        )
        quadrature = float(quadrature_tolerance)
        certification = float(certification_tolerance)
        if (
            not isfinite(quadrature)
            or quadrature <= 0.0
            or not isfinite(certification)
            or certification <= 0.0
        ):
            raise ValueError("Rod reduction tolerances must be finite and positive.")

        generated = canonical_fingerprint(
            {
                "kind": "reduced-rod-strain-coordinate-plan",
                "coordinate_count": coordinate_count,
                "quadrature_tolerance": quadrature,
                "certification_tolerance": certification,
                "values": array_tree_fingerprint(arrays),
                "fixed_base_position": (
                    None
                    if base_position is None
                    else array_tree_fingerprint(base_position.astype(dtype, copy=False))
                ),
                "fixed_base_orientation": (
                    None
                    if base_orientation is None
                    else array_tree_fingerprint(
                        base_orientation.astype(dtype, copy=False)
                    )
                ),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")

        self.stretch_shear_basis = jnp.asarray(arrays["stretch_shear_basis"])
        self.bend_twist_basis = jnp.asarray(arrays["bend_twist_basis"])
        self.reference_coefficients = jnp.asarray(arrays["reference_coefficients"])
        self.fixed_base_position = (
            None
            if base_position is None
            else jnp.asarray(base_position.astype(dtype, copy=False))
        )
        self.fixed_base_orientation = (
            None
            if base_orientation is None
            else jnp.asarray(base_orientation.astype(dtype, copy=False))
        )
        self.coordinate_count = coordinate_count
        self.quadrature_tolerance = quadrature
        self.certification_tolerance = certification
        self.plan_id = identifier


class ReducedRodState(StrictModule):
    """Canonical packed reduced configuration and velocity state."""

    values: Array
    coordinate_count: int = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        coefficient_velocities: ArrayLike,
        /,
    ):
        coefficients_ = jnp.asarray(coefficients)
        velocities_ = jnp.asarray(coefficient_velocities)
        if coefficients_.ndim != 1 or coefficients_.shape[0] < 1:
            raise ValueError(
                "Reduced rod coefficients must be a nonempty rank-1 array."
            )
        if velocities_.shape != coefficients_.shape:
            raise ValueError(
                "Reduced rod coefficient velocities must match coefficients."
            )
        for name, value in (
            ("coefficients", coefficients_),
            ("coefficient_velocities", velocities_),
        ):
            if not jnp.issubdtype(value.dtype, jnp.inexact) or jnp.iscomplexobj(value):
                raise TypeError(f"Reduced rod {name} must be a real inexact array.")
        if velocities_.dtype != coefficients_.dtype:
            raise TypeError(
                "Reduced rod coefficients and velocities must share one dtype."
            )
        self.values = jnp.concatenate((coefficients_, velocities_))
        self.coordinate_count = int(coefficients_.shape[0])

    @property
    def coefficients(self) -> Array:
        return self.values[: self.coordinate_count]

    @property
    def coefficient_velocities(self) -> Array:
        return self.values[self.coordinate_count :]


class ReducedRodLiftEvidence(StrictModule):
    """Finite fixed-base evidence for one reduced-to-native lift."""

    base_position_error: Array
    base_orientation_error: Array
    finite: Array
    base_pose_valid: Array
    valid: Array


class ReducedRodPowerEvidence(StrictModule):
    """Virtual-power equality for a native load and its reduced pullback."""

    native_power: Array
    reduced_power: Array
    absolute_residual: Array
    relative_residual: Array
    finite: Array
    valid: Array


class ReducedRodStrainEvidence(StrictModule):
    """Residual between requested reduced strains and lifted native strains."""

    stretch_shear_residual: Array
    bend_twist_residual: Array
    maximum_stretch_shear_error: Array
    maximum_bend_twist_error: Array
    finite: Array
    within_tolerance: Array
    valid: Array


class ReducedRodEvaluation(StrictModule):
    """Native rod evaluation and reduction-specific finite-domain evidence."""

    native_state: RodState
    native_evaluation: RodEvaluation
    generalized_internal_load: Array
    potential_energy: Array
    kinetic_energy: Array
    total_energy: Array
    strain_quadrature_energy: Array
    quadrature_error: Array
    lift_evidence: ReducedRodLiftEvidence
    power_evidence: ReducedRodPowerEvidence
    strain_evidence: ReducedRodStrainEvidence
    finite: Array
    quadrature_valid: Array
    valid: Array
    reduction_id: str = eqx.field(static=True)


class PreparedReducedRod(StrictModule, NonTrainableState):
    """A planar strain-coordinate parameterization bound to one PreparedRod."""

    rod: PreparedRod
    plan: ReducedRodPlan
    stretch_shear_basis: Array
    bend_twist_basis: Array
    reference_coefficients: Array
    path_node_ids: Array
    reference_positions: Array
    reference_orientations: Array
    base_position: Array
    base_orientation: Array
    coefficient_space: ArraySpace
    native_velocity_space: ArraySpace
    state_layout: StateLayout
    native_position_size: int = eqx.field(static=True)
    configuration_slice: slice = eqx.field(static=True)
    velocity_slice: slice = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, rod: PreparedRod, plan: ReducedRodPlan, /):
        if not isinstance(rod, PreparedRod):
            raise TypeError("rod must be a PreparedRod.")
        if not isinstance(plan, ReducedRodPlan):
            raise TypeError("plan must be a ReducedRodPlan.")
        if rod.plan.dimension != 2:
            raise ValueError(
                "Reduced rod strain lifting currently supports planar "
                "PreparedRod values."
            )
        if rod.plan.inextensible:
            raise ValueError(
                "Inextensible rods are unsupported because native projection "
                "would leave the declared reduced strain manifold."
            )
        expected_stretch = (rod.plan.segment_count, 2, plan.coordinate_count)
        expected_bend = (rod.plan.segment_count - 1, 1, plan.coordinate_count)
        if plan.stretch_shear_basis.shape != expected_stretch:
            raise ValueError(
                "stretch_shear_basis is incompatible with the prepared rod "
                "segment count."
            )
        if plan.bend_twist_basis.shape != expected_bend:
            raise ValueError(
                "bend_twist_basis is incompatible with the prepared rod junction count."
            )

        dtype = np.dtype(rod.plan.rest_positions.dtype)
        stretch = jnp.asarray(plan.stretch_shear_basis, dtype=dtype)
        bend = jnp.asarray(plan.bend_twist_basis, dtype=dtype)
        reference_coefficients = jnp.asarray(plan.reference_coefficients, dtype=dtype)
        prepared_basis_matrix = np.concatenate(
            (
                np.asarray(stretch).reshape((-1, plan.coordinate_count)),
                np.asarray(bend).reshape((-1, plan.coordinate_count)),
            ),
            axis=0,
        )
        if np.linalg.matrix_rank(prepared_basis_matrix) != plan.coordinate_count:
            raise ValueError(
                "The rod strain basis must retain full column rank in the "
                "prepared rod dtype."
            )
        path_node_ids = jnp.concatenate(
            (rod.plan.segment_node_ids[:1, 0], rod.plan.segment_node_ids[:, 1])
        )
        start_node = int(np.asarray(rod.plan.segment_node_ids[0, 0]))
        native_base_position = rod.plan.rest_positions[start_node]
        native_base_orientation = rod.rest_orientations[0]
        base_position = (
            native_base_position
            if plan.fixed_base_position is None
            else jnp.asarray(plan.fixed_base_position, dtype=dtype)
        )
        base_orientation = (
            native_base_orientation
            if plan.fixed_base_orientation is None
            else jnp.asarray(plan.fixed_base_orientation, dtype=dtype)
        )
        angle_offset = base_orientation - native_base_orientation
        if plan.fixed_base_orientation is None:
            reference_positions = rod.plan.rest_positions + (
                base_position - native_base_position
            )
            reference_orientations = rod.rest_orientations
        else:
            rotation = _planar_rotation_matrix(angle_offset)
            offsets = rod.plan.rest_positions - native_base_position
            reference_positions = base_position + ein.contract(
                "ij,nj->ni", rotation, offsets
            )
            reference_orientations = rod.rest_orientations + angle_offset

        coefficient_space = ArraySpace(
            (plan.coordinate_count,),
            dtype=dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "reduced-rod-coefficient-space",
                    "plan": plan.plan_id,
                    "rod": rod.prepared_id,
                }
            ),
        )
        native_position_size = rod.plan.node_count * rod.plan.dimension
        native_velocity_space = ArraySpace(
            (native_position_size + rod.plan.segment_count,),
            dtype=dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "reduced-rod-native-velocity-space",
                    "rod": rod.prepared_id,
                }
            ),
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-reduced-rod-strain-parameterization",
                "plan": plan.plan_id,
                "rod": rod.prepared_id,
                "reference_pose": array_tree_fingerprint(
                    {
                        "positions": np.asarray(reference_positions),
                        "orientations": np.asarray(reference_orientations),
                    }
                ),
            }
        )
        state_layout = StateLayout(
            (2 * plan.coordinate_count,),
            axes=("reduced_rod_state",),
            component_names=tuple(
                [f"q:{index}" for index in range(plan.coordinate_count)]
                + [f"v:{index}" for index in range(plan.coordinate_count)]
            ),
            layout_id=f"state-layout:reduced-rod:{prepared_id}",
        )

        self.rod = rod
        self.plan = plan
        self.stretch_shear_basis = stretch
        self.bend_twist_basis = bend
        self.reference_coefficients = reference_coefficients
        self.path_node_ids = path_node_ids
        self.reference_positions = reference_positions
        self.reference_orientations = reference_orientations
        self.base_position = base_position
        self.base_orientation = base_orientation
        self.coefficient_space = coefficient_space
        self.native_velocity_space = native_velocity_space
        self.state_layout = state_layout
        self.native_position_size = native_position_size
        self.configuration_slice = slice(0, plan.coordinate_count)
        self.velocity_slice = slice(plan.coordinate_count, 2 * plan.coordinate_count)
        self.prepared_id = prepared_id

    @property
    def state_size(self) -> int:
        return self.state_layout.size

    def initialize_state(self) -> ReducedRodState:
        return ReducedRodState(
            self.reference_coefficients,
            jnp.zeros_like(self.reference_coefficients),
        )

    def rest_state(self) -> ReducedRodState:
        return ReducedRodState(
            jnp.zeros_like(self.reference_coefficients),
            jnp.zeros_like(self.reference_coefficients),
        )

    def lift(self, state: ReducedRodState, /) -> RodState:
        return lift_reduced_rod_state(self, state)

    def lift_operator(self, coefficients: ArrayLike, /) -> AbstractLinearOperator:
        return reduced_rod_lift_operator(self, coefficients)

    def pullback_loads(
        self,
        coefficients: ArrayLike,
        native_forces: ArrayLike,
        native_moments: ArrayLike,
        /,
    ) -> Array:
        return pullback_reduced_rod_loads(
            self, coefficients, native_forces, native_moments
        )

    def evaluate(self, state: ReducedRodState, /) -> ReducedRodEvaluation:
        return evaluate_reduced_rod(self, state)


def prepare_reduced_rod(
    rod: PreparedRod,
    plan: ReducedRodPlan,
    /,
) -> PreparedReducedRod:
    """Bind a finite planar strain basis to one prepared native rod."""
    return PreparedReducedRod(rod, plan)


def _validate_coefficients(
    prepared: PreparedReducedRod,
    coefficients: ArrayLike,
    /,
) -> Array:
    values = jnp.asarray(coefficients)
    return prepared.coefficient_space.validate(values)


def _validate_state(prepared: PreparedReducedRod, state: ReducedRodState, /) -> None:
    if not isinstance(state, ReducedRodState):
        raise TypeError("state must be a ReducedRodState.")
    if (
        state.coordinate_count != prepared.plan.coordinate_count
        or state.values.shape != prepared.state_layout.shape
    ):
        raise ValueError("Reduced rod state must match the prepared state layout.")
    if np.dtype(state.values.dtype) != prepared.coefficient_space.dtype:
        raise TypeError(
            "Reduced rod state dtype does not match the prepared rod dtype."
        )


def _target_strains(
    prepared: PreparedReducedRod,
    coefficients: Array,
    /,
) -> tuple[Array, Array]:
    stretch_shear = ein.contract(
        "sdk,k->sd", prepared.stretch_shear_basis, coefficients
    )
    bend_twist = ein.contract("sjk,k->sj", prepared.bend_twist_basis, coefficients)
    return stretch_shear, bend_twist


def _lift_configuration(
    prepared: PreparedReducedRod,
    coefficients: Array,
    /,
) -> tuple[Array, Array]:
    stretch_shear, bend_twist = _target_strains(prepared, coefficients)
    angle_increments = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=coefficients.dtype),
            jnp.cumsum(prepared.rod.dual_lengths * bend_twist[:, 0]),
        )
    )
    orientations = prepared.reference_orientations + angle_increments
    current_frames = _planar_rotation_matrix(orientations)
    reference_frames = _planar_rotation_matrix(prepared.reference_orientations)
    current_material_tangents = prepared.rod.rest_stretch_shear + stretch_shear
    tangent_correction = prepared.rod.plan.rest_lengths[:, None] * (
        ein.contract("sij,sj->si", current_frames, current_material_tangents)
        - ein.contract(
            "sij,sj->si", reference_frames, prepared.rod.rest_stretch_shear
        )
    )
    path_correction = jnp.concatenate(
        (
            jnp.zeros((1, 2), dtype=coefficients.dtype),
            jnp.cumsum(tangent_correction, axis=0),
        ),
        axis=0,
    )
    positions = prepared.reference_positions.at[prepared.path_node_ids].add(
        path_correction
    )
    return positions, orientations


def _pack_configuration(
    prepared: PreparedReducedRod,
    coefficients: Array,
    /,
) -> Array:
    positions, orientations = _lift_configuration(prepared, coefficients)
    return jnp.concatenate((positions.reshape((-1,)), orientations))


def _unpack_native_configuration(
    prepared: PreparedReducedRod,
    packed: Array,
    /,
) -> tuple[Array, Array]:
    positions = packed[: prepared.native_position_size].reshape(
        (prepared.rod.plan.node_count, prepared.rod.plan.dimension)
    )
    orientations = packed[prepared.native_position_size :]
    return positions, orientations


def _unpack_native_velocity(
    prepared: PreparedReducedRod,
    packed: Array,
    /,
) -> tuple[Array, Array]:
    velocities = packed[: prepared.native_position_size].reshape(
        (prepared.rod.plan.node_count, prepared.rod.plan.dimension)
    )
    angular_velocities = packed[prepared.native_position_size :]
    return velocities, angular_velocities


def _pack_native_loads(
    prepared: PreparedReducedRod,
    native_forces: ArrayLike,
    native_moments: ArrayLike,
    /,
) -> Array:
    forces = jnp.asarray(native_forces)
    moments = jnp.asarray(native_moments)
    if forces.shape != (prepared.rod.plan.node_count, 2):
        raise ValueError("native_forces must contain one planar force per rod node.")
    if moments.shape != (prepared.rod.plan.segment_count,):
        raise ValueError(
            "native_moments must contain one scalar moment per rod segment."
        )
    packed = jnp.concatenate((forces.reshape((-1,)), moments))
    return prepared.native_velocity_space.validate(packed)


def _prepare_lift_operator(
    prepared: PreparedReducedRod,
    point: Array,
    /,
) -> JacobianLinearOperator:
    def configuration(values):
        return _pack_configuration(prepared, values)

    linearization = prepare_linearization(
        configuration,
        point,
        source=prepared.coefficient_space,
        target=prepared.native_velocity_space,
        linearization_id=canonical_fingerprint(
            {
                "kind": "reduced-rod-lift-linearization",
                "reduction": prepared.prepared_id,
            }
        ),
    )
    return JacobianLinearOperator(
        linearization,
        operator_id=canonical_fingerprint(
            {
                "kind": "reduced-rod-lift-operator",
                "reduction": prepared.prepared_id,
            }
        ),
    )


def reduced_rod_lift_operator(
    prepared: PreparedReducedRod,
    coefficients: ArrayLike,
    /,
) -> AbstractLinearOperator:
    """Return the native configuration Jacobian at one reduced configuration."""
    if not isinstance(prepared, PreparedReducedRod):
        raise TypeError("prepared must be a PreparedReducedRod.")
    point = _validate_coefficients(prepared, coefficients)
    return _prepare_lift_operator(prepared, point)


def lift_reduced_rod_velocity(
    prepared: PreparedReducedRod,
    coefficients: ArrayLike,
    coefficient_velocities: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Push a reduced velocity through the exact configuration JVP."""
    operator = reduced_rod_lift_operator(prepared, coefficients)
    packed_velocity = operator.mv(
        _validate_coefficients(prepared, coefficient_velocities)
    )
    return _unpack_native_velocity(prepared, packed_velocity)


def _native_state_from_operator(
    prepared: PreparedReducedRod,
    state: ReducedRodState,
    operator: JacobianLinearOperator,
    /,
) -> RodState:
    positions, orientations = _unpack_native_configuration(
        prepared, operator.linearization.primal
    )
    packed_velocity = operator.mv(state.coefficient_velocities)
    velocities, angular_velocities = _unpack_native_velocity(
        prepared, packed_velocity
    )
    return RodState(positions, velocities, orientations, angular_velocities)


def lift_reduced_rod_state(
    prepared: PreparedReducedRod,
    state: ReducedRodState,
    /,
) -> RodState:
    """Lift one packed reduced phase state to the native RodState contract."""
    if not isinstance(prepared, PreparedReducedRod):
        raise TypeError("prepared must be a PreparedReducedRod.")
    _validate_state(prepared, state)
    operator = _prepare_lift_operator(prepared, state.coefficients)
    return _native_state_from_operator(prepared, state, operator)


def pullback_reduced_rod_loads(
    prepared: PreparedReducedRod,
    coefficients: ArrayLike,
    native_forces: ArrayLike,
    native_moments: ArrayLike,
    /,
) -> Array:
    """Pull native force/moment covectors back with the lift-operator VJP."""
    operator = reduced_rod_lift_operator(prepared, coefficients)
    native_loads = _pack_native_loads(prepared, native_forces, native_moments)
    return operator.transpose_mv(native_loads)


def _power_evidence_from_operator(
    prepared: PreparedReducedRod,
    operator: JacobianLinearOperator,
    rates: Array,
    native_loads: Array,
    /,
) -> ReducedRodPowerEvidence:
    native_velocity = operator.mv(rates)
    generalized_load = operator.transpose_mv(native_loads)
    native_power = jnp.vdot(native_loads, native_velocity).real
    reduced_power = jnp.vdot(generalized_load, rates).real
    absolute = jnp.abs(native_power - reduced_power)
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=absolute.dtype),
        jnp.maximum(jnp.abs(native_power), jnp.abs(reduced_power)),
    )
    relative = absolute / scale
    finite = (
        jnp.isfinite(native_power)
        & jnp.isfinite(reduced_power)
        & jnp.isfinite(absolute)
        & jnp.isfinite(relative)
    )
    valid = finite & (relative <= prepared.plan.certification_tolerance)
    return ReducedRodPowerEvidence(
        native_power,
        reduced_power,
        absolute,
        relative,
        finite,
        valid,
    )


def reduced_rod_power_evidence(
    prepared: PreparedReducedRod,
    coefficients: ArrayLike,
    coefficient_velocities: ArrayLike,
    native_forces: ArrayLike,
    native_moments: ArrayLike,
    /,
) -> ReducedRodPowerEvidence:
    """Certify virtual-power duality for one load and reduced velocity."""
    values = _validate_coefficients(prepared, coefficients)
    rates = _validate_coefficients(prepared, coefficient_velocities)
    operator = _prepare_lift_operator(prepared, values)
    native_loads = _pack_native_loads(prepared, native_forces, native_moments)
    return _power_evidence_from_operator(prepared, operator, rates, native_loads)


def reduced_rod_potential_energy(
    prepared: PreparedReducedRod,
    coefficients: ArrayLike,
    /,
) -> Array:
    """Evaluate reduced potential through the native rod energy function."""
    values = _validate_coefficients(prepared, coefficients)
    positions, orientations = _lift_configuration(prepared, values)
    return rod_potential_energy(prepared.rod, positions, orientations)


def reduced_rod_kinetic_energy(
    prepared: PreparedReducedRod,
    state: ReducedRodState,
    /,
) -> Array:
    """Evaluate reduced kinetic energy through the native rod evaluation."""
    native_state = lift_reduced_rod_state(prepared, state)
    return evaluate_rod(prepared.rod, native_state).kinetic_energy


def _lift_evidence(
    prepared: PreparedReducedRod,
    native_state: RodState,
    /,
) -> ReducedRodLiftEvidence:
    base_node = prepared.path_node_ids[0]
    base_position_error = jnp.sqrt(
        jnp.sum((native_state.positions[base_node] - prepared.base_position) ** 2)
    )
    orientation_delta = native_state.orientations[0] - prepared.base_orientation
    base_orientation_error = jnp.abs(
        jnp.arctan2(jnp.sin(orientation_delta), jnp.cos(orientation_delta))
    )
    finite = (
        jnp.all(jnp.isfinite(native_state.positions))
        & jnp.all(jnp.isfinite(native_state.velocities))
        & jnp.all(jnp.isfinite(native_state.orientations))
        & jnp.all(jnp.isfinite(native_state.angular_velocities))
        & jnp.isfinite(base_position_error)
        & jnp.isfinite(base_orientation_error)
    )
    tolerance = prepared.plan.certification_tolerance
    base_pose_valid = (base_position_error <= tolerance) & (
        base_orientation_error <= tolerance
    )
    return ReducedRodLiftEvidence(
        base_position_error,
        base_orientation_error,
        finite,
        base_pose_valid,
        finite & base_pose_valid,
    )


def _strain_evidence(
    prepared: PreparedReducedRod,
    coefficients: Array,
    native_evaluation: RodEvaluation,
    /,
) -> ReducedRodStrainEvidence:
    target_stretch_shear, target_bend_twist = _target_strains(
        prepared, coefficients
    )
    stretch_residual = (
        native_evaluation.stretch_shear_strain - target_stretch_shear
    )
    bend_residual = native_evaluation.bend_twist_strain - target_bend_twist
    maximum_stretch = jnp.max(jnp.abs(stretch_residual))
    maximum_bend = jnp.max(
        jnp.concatenate(
            (
                jnp.zeros((1,), dtype=maximum_stretch.dtype),
                jnp.abs(bend_residual).reshape((-1,)),
            )
        )
    )
    finite = (
        jnp.all(jnp.isfinite(stretch_residual))
        & jnp.all(jnp.isfinite(bend_residual))
        & jnp.isfinite(maximum_stretch)
        & jnp.isfinite(maximum_bend)
    )
    tolerance = prepared.plan.certification_tolerance
    within_tolerance = (maximum_stretch <= tolerance) & (maximum_bend <= tolerance)
    return ReducedRodStrainEvidence(
        stretch_residual,
        bend_residual,
        maximum_stretch,
        maximum_bend,
        finite,
        within_tolerance,
        finite & within_tolerance,
    )


def _strain_quadrature_energy(
    prepared: PreparedReducedRod,
    coefficients: Array,
    /,
) -> Array:
    stretch_shear, bend_twist = _target_strains(prepared, coefficients)
    stretch_density = 0.5 * ein.contract(
        "si,sij,sj->s",
        stretch_shear,
        prepared.rod.plan.stretch_shear_stiffness,
        stretch_shear,
    )
    bend_density = 0.5 * ein.contract(
        "si,sij,sj->s",
        bend_twist,
        prepared.rod.plan.bend_twist_stiffness,
        bend_twist,
    )
    return jnp.sum(prepared.rod.plan.rest_lengths * stretch_density) + jnp.sum(
        prepared.rod.dual_lengths * bend_density
    )


def evaluate_reduced_rod(
    prepared: PreparedReducedRod,
    state: ReducedRodState,
    /,
) -> ReducedRodEvaluation:
    """Lift and evaluate one reduced state with power and strain certificates."""
    if not isinstance(prepared, PreparedReducedRod):
        raise TypeError("prepared must be a PreparedReducedRod.")
    _validate_state(prepared, state)
    operator = _prepare_lift_operator(prepared, state.coefficients)
    native_state = _native_state_from_operator(prepared, state, operator)
    native_evaluation = evaluate_rod(prepared.rod, native_state)
    native_internal_loads = _pack_native_loads(
        prepared,
        native_evaluation.internal_forces,
        native_evaluation.internal_moments,
    )
    generalized_internal_load = operator.transpose_mv(native_internal_loads)
    lift_evidence = _lift_evidence(prepared, native_state)
    strain_evidence = _strain_evidence(
        prepared, state.coefficients, native_evaluation
    )
    power_evidence = _power_evidence_from_operator(
        prepared,
        operator,
        state.coefficient_velocities,
        native_internal_loads,
    )
    strain_quadrature_energy = _strain_quadrature_energy(
        prepared, state.coefficients
    )
    quadrature_error = jnp.abs(
        native_evaluation.potential_energy - strain_quadrature_energy
    )
    quadrature_scale = jnp.maximum(
        jnp.asarray(1.0, dtype=quadrature_error.dtype),
        jnp.maximum(
            jnp.abs(native_evaluation.potential_energy),
            jnp.abs(strain_quadrature_energy),
        ),
    )
    quadrature_valid = jnp.isfinite(quadrature_error) & (
        quadrature_error
        <= prepared.plan.quadrature_tolerance * quadrature_scale
    )
    finite = (
        native_evaluation.finite
        & lift_evidence.finite
        & strain_evidence.finite
        & power_evidence.finite
        & jnp.all(jnp.isfinite(generalized_internal_load))
        & jnp.isfinite(strain_quadrature_energy)
        & jnp.isfinite(quadrature_error)
    )
    valid = (
        finite
        & native_evaluation.valid
        & lift_evidence.valid
        & strain_evidence.valid
        & power_evidence.valid
        & quadrature_valid
    )
    return ReducedRodEvaluation(
        native_state,
        native_evaluation,
        generalized_internal_load,
        native_evaluation.potential_energy,
        native_evaluation.kinetic_energy,
        native_evaluation.total_energy,
        strain_quadrature_energy,
        quadrature_error,
        lift_evidence,
        power_evidence,
        strain_evidence,
        finite,
        quadrature_valid,
        valid,
        prepared.prepared_id,
    )


__all__ = [
    "PreparedReducedRod",
    "ReducedRodEvaluation",
    "ReducedRodLiftEvidence",
    "ReducedRodPlan",
    "ReducedRodPowerEvidence",
    "ReducedRodState",
    "ReducedRodStrainEvidence",
    "evaluate_reduced_rod",
    "lift_reduced_rod_state",
    "lift_reduced_rod_velocity",
    "prepare_reduced_rod",
    "pullback_reduced_rod_loads",
    "reduced_rod_kinetic_energy",
    "reduced_rod_lift_operator",
    "reduced_rod_potential_energy",
    "reduced_rod_power_evidence",
]
