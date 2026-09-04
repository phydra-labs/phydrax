#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import StateLayout
from ...linalg import AbstractLinearOperator, ArraySpace, BlockSpace, DualSpace
from ._rod_dynamics import (
    evaluate_rod,
    PreparedRod,
    rod_potential_energy,
    RodEvaluation,
    RodState,
)
from ._rod_reduced_basis import (
    prepare_rod_strain_basis,
    PreparedRodStrainBasis,
    RodStrainBasisEvidence,
    RodStrainBasisPlan,
)
from ._rod_reduced_kinematics import (
    lift_configuration,
    lift_effort_pullback_operator,
    lift_reduced_rod_state,
    lift_reduced_rod_velocity,
    lift_velocity_operator,
    pullback_reduced_rod_loads,
    target_native_strains,
)


ReducedRodBasePolicy: TypeAlias = Literal["reference", "fixed"]


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
    return jnp.stack((cosine, -sine, sine, cosine), axis=-1).reshape(angle.shape + (2, 2))


def _quaternion_conjugate(quaternion: Array, /) -> Array:
    return jnp.concatenate((quaternion[..., :1], -quaternion[..., 1:]), axis=-1)


def _quaternion_multiply(left: Array, right: Array, /) -> Array:
    left_scalar = left[..., :1]
    right_scalar = right[..., :1]
    left_vector = left[..., 1:]
    right_vector = right[..., 1:]
    scalar = left_scalar * right_scalar - jnp.sum(
        left_vector * right_vector, axis=-1, keepdims=True
    )
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + jnp.cross(left_vector, right_vector)
    )
    return jnp.concatenate((scalar, vector), axis=-1)


def _quaternion_rotation_matrix(quaternion: Array, /) -> Array:
    normalized = quaternion / jnp.sqrt(
        jnp.sum(quaternion * quaternion, axis=-1, keepdims=True)
    )
    scalar = normalized[..., 0]
    x = normalized[..., 1]
    y = normalized[..., 2]
    z = normalized[..., 3]
    return jnp.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - scalar * z),
            2.0 * (x * z + scalar * y),
            2.0 * (x * y + scalar * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - scalar * x),
            2.0 * (x * z - scalar * y),
            2.0 * (y * z + scalar * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(normalized.shape[:-1] + (3, 3))


def _canonical_unit_quaternion(value: np.ndarray, dtype: np.dtype, /) -> np.ndarray:
    quaternion = value.astype(dtype, copy=False)
    norm = float(np.linalg.norm(quaternion))
    tolerance = 500.0 * np.finfo(dtype).eps
    if not np.isclose(norm, 1.0, rtol=tolerance, atol=tolerance):
        raise ValueError("fixed_base_orientation must be a unit quaternion.")
    quaternion = quaternion / norm
    nonzero = np.flatnonzero(quaternion != 0.0)
    if nonzero.size and quaternion[int(nonzero[0])] < 0.0:
        quaternion = -quaternion
    quaternion = np.where(quaternion == 0.0, 0.0, quaternion)
    return quaternion.astype(dtype, copy=False)


def _quaternion_angle(quaternion: Array, /) -> Array:
    normalized = quaternion / jnp.sqrt(jnp.sum(quaternion * quaternion))
    canonical = jnp.where(normalized[0] < 0.0, -normalized, normalized)
    vector_norm = jnp.sqrt(jnp.sum(canonical[1:] * canonical[1:]))
    return 2.0 * jnp.arctan2(vector_norm, jnp.abs(canonical[0]))


class ReducedRodPlan(StrictModule, NonTrainableState):
    """Dimension-generic fixed-base reduction over a continuous strain basis.

    ``base_policy='reference'`` fixes the first native rod frame at its native
    reference pose. ``base_policy='fixed'`` fixes it at the explicitly supplied
    pose. Floating-base reduction is intentionally outside this contract.
    """

    basis: RodStrainBasisPlan
    reference_coefficients: Array
    fixed_base_position: Array | None
    fixed_base_orientation: Array | None
    coordinate_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    base_policy: ReducedRodBasePolicy = eqx.field(static=True)
    quadrature_tolerance: float = eqx.field(static=True)
    certification_tolerance: float = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: RodStrainBasisPlan,
        /,
        *,
        reference_coefficients: ArrayLike | None = None,
        base_policy: ReducedRodBasePolicy = "reference",
        fixed_base_position: ArrayLike | None = None,
        fixed_base_orientation: ArrayLike | None = None,
        quadrature_tolerance: float = 1.0e-6,
        certification_tolerance: float = 1.0e-6,
        label: str | None = None,
    ):
        if not isinstance(basis, RodStrainBasisPlan):
            raise TypeError("basis must be a RodStrainBasisPlan.")
        if base_policy not in ("reference", "fixed"):
            raise ValueError(
                "base_policy must be 'reference' or 'fixed'; floating rods are unsupported."
            )
        dtype = np.dtype(basis.polynomial_coefficients.dtype)
        coordinate_count = basis.coordinate_count
        reference = (
            np.zeros((coordinate_count,), dtype=dtype)
            if reference_coefficients is None
            else _real_array("reference_coefficients", reference_coefficients, 1).astype(
                dtype, copy=False
            )
        )
        if reference.shape != (coordinate_count,):
            raise ValueError(
                "reference_coefficients must contain one value per reduced coordinate."
            )

        if base_policy == "reference":
            if fixed_base_position is not None or fixed_base_orientation is not None:
                raise ValueError(
                    "Reference base policy derives the fixed pose from the native rod; "
                    "explicit base values are forbidden."
                )
            base_position = None
            base_orientation = None
        else:
            if fixed_base_position is None or fixed_base_orientation is None:
                raise ValueError(
                    "Fixed base policy requires both fixed_base_position and fixed_base_orientation."
                )
            base_position = _real_array(
                "fixed_base_position", fixed_base_position, 1
            ).astype(dtype, copy=False)
            if base_position.shape != (basis.dimension,):
                raise ValueError(
                    f"fixed_base_position must have shape {(basis.dimension,)}."
                )
            if basis.dimension == 2:
                base_orientation = _real_array(
                    "fixed_base_orientation", fixed_base_orientation, 0
                ).astype(dtype, copy=False)
            else:
                raw_orientation = _real_array(
                    "fixed_base_orientation", fixed_base_orientation, 1
                )
                if raw_orientation.shape != (4,):
                    raise ValueError(
                        "Spatial fixed_base_orientation must be a scalar-first quaternion of shape (4,)."
                    )
                base_orientation = _canonical_unit_quaternion(raw_orientation, dtype)

        quadrature = float(quadrature_tolerance)
        certification = float(certification_tolerance)
        if (
            not isfinite(quadrature)
            or quadrature <= 0.0
            or not isfinite(certification)
            or certification <= 0.0
        ):
            raise ValueError("Rod reduction tolerances must be finite and positive.")
        values = {"reference_coefficients": reference}
        if base_position is not None:
            values["fixed_base_position"] = base_position
            values["fixed_base_orientation"] = base_orientation
        generated = canonical_fingerprint(
            {
                "kind": "reduced-rod-strain-coordinate-plan",
                "dimension": basis.dimension,
                "coordinate_count": coordinate_count,
                "basis": basis.plan_id,
                "base_policy": base_policy,
                "quadrature_tolerance": quadrature,
                "certification_tolerance": certification,
                "values": array_tree_fingerprint(values),
            }
        )
        self.basis = basis
        self.reference_coefficients = jnp.asarray(reference)
        self.fixed_base_position = (
            None if base_position is None else jnp.asarray(base_position)
        )
        self.fixed_base_orientation = (
            None if base_orientation is None else jnp.asarray(base_orientation)
        )
        self.coordinate_count = coordinate_count
        self.dimension = basis.dimension
        self.base_policy = base_policy
        self.quadrature_tolerance = quadrature
        self.certification_tolerance = certification
        self.label = None if label is None else str(label)
        self.plan_id = generated


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
            raise ValueError("Reduced rod coefficients must be a nonempty rank-1 array.")
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
    base_linear_velocity_error: Array
    base_angular_velocity_error: Array
    finite: Array
    base_pose_valid: Array
    base_velocity_valid: Array
    valid: Array


class ReducedRodPowerEvidence(StrictModule):
    """Algebraic virtual-power equality for a native effort pullback."""

    native_power: Array
    reduced_power: Array
    absolute_residual: Array
    relative_residual: Array
    finite: Array
    valid: Array


class ReducedRodStrainEvidence(StrictModule):
    """Residual between requested and evaluated native discrete rod strains."""

    stretch_shear_residual: Array
    bend_twist_residual: Array
    maximum_stretch_shear_error: Array
    maximum_bend_twist_error: Array
    finite: Array
    within_tolerance: Array
    valid: Array


class ReducedRodEvaluation(StrictModule):
    """Native rod mechanics and finite-domain reduction evidence."""

    native_state: RodState
    native_evaluation: RodEvaluation
    generalized_internal_load: Array
    potential_energy: Array
    kinetic_energy: Array
    total_energy: Array
    native_discrete_strain_energy: Array
    native_discrete_energy_error: Array
    lift_evidence: ReducedRodLiftEvidence
    power_evidence: ReducedRodPowerEvidence
    strain_evidence: ReducedRodStrainEvidence
    finite: Array
    native_discrete_energy_valid: Array
    valid: Array
    reduction_id: str = eqx.field(static=True)

    @property
    def strain_quadrature_energy(self) -> Array:
        return self.native_discrete_strain_energy

    @property
    def quadrature_error(self) -> Array:
        return self.native_discrete_energy_error

    @property
    def quadrature_valid(self) -> Array:
        return self.native_discrete_energy_valid


class PreparedReducedRod(StrictModule, NonTrainableState):
    """Fixed-base native-discrete reduction of one prepared 2-D or 3-D rod."""

    rod: PreparedRod
    plan: ReducedRodPlan
    basis: PreparedRodStrainBasis
    reference_coefficients: Array
    path_node_ids: Array
    reference_positions: Array
    reference_orientations: Array
    base_position: Array
    base_orientation: Array
    coefficient_space: ArraySpace
    reduced_effort_space: DualSpace
    native_velocity_space: BlockSpace
    native_effort_space: DualSpace
    state_layout: StateLayout
    configuration_slice: slice = eqx.field(static=True)
    velocity_slice: slice = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, rod: PreparedRod, plan: ReducedRodPlan, /):
        if not isinstance(rod, PreparedRod):
            raise TypeError("rod must be a PreparedRod.")
        if not isinstance(plan, ReducedRodPlan):
            raise TypeError("plan must be a ReducedRodPlan.")
        if plan.dimension != rod.plan.dimension:
            raise ValueError(
                "Reduced rod plan dimension is incompatible with the prepared rod."
            )
        basis = prepare_rod_strain_basis(plan.basis, rod)
        dtype = np.dtype(rod.plan.rest_positions.dtype)
        reference_coefficients = jnp.asarray(plan.reference_coefficients)
        if np.dtype(reference_coefficients.dtype) != dtype:
            raise TypeError(
                "Reduced rod reference coefficients must retain the native rod dtype."
            )
        path_node_ids = jnp.concatenate(
            (rod.plan.segment_node_ids[:1, 0], rod.plan.segment_node_ids[:, 1])
        )
        start_node = rod.plan.segment_node_ids[0, 0]
        native_base_position = rod.plan.rest_positions[start_node]
        native_base_orientation = rod.rest_orientations[0]
        base_position = (
            native_base_position
            if plan.base_policy == "reference"
            else jnp.asarray(plan.fixed_base_position)
        )
        base_orientation = (
            native_base_orientation
            if plan.base_policy == "reference"
            else jnp.asarray(plan.fixed_base_orientation)
        )
        reference_positions, reference_orientations = _reference_pose(
            rod, base_position, base_orientation
        )

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
        native_velocity_space = rod.velocity_space
        reduced_effort_space = DualSpace(coefficient_space)
        native_effort_space = rod.effort_space
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-reduced-rod-native-discrete-parameterization",
                "plan": plan.plan_id,
                "basis": basis.prepared_id,
                "rod": rod.prepared_id,
                "base_policy": plan.base_policy,
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
        self.basis = basis
        self.reference_coefficients = reference_coefficients
        self.path_node_ids = path_node_ids
        self.reference_positions = reference_positions
        self.reference_orientations = reference_orientations
        self.base_position = base_position
        self.base_orientation = base_orientation
        self.coefficient_space = coefficient_space
        self.reduced_effort_space = reduced_effort_space
        self.native_velocity_space = native_velocity_space
        self.native_effort_space = native_effort_space
        self.state_layout = state_layout
        self.configuration_slice = slice(0, plan.coordinate_count)
        self.velocity_slice = slice(plan.coordinate_count, 2 * plan.coordinate_count)
        self.prepared_id = prepared_id

    @property
    def state_size(self) -> int:
        return self.state_layout.size

    @property
    def stretch_shear_basis(self) -> Array:
        return self.basis.stretch_shear_basis

    @property
    def bend_twist_basis(self) -> Array:
        return self.basis.bend_twist_basis

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

    def validate_state(self, state: ReducedRodState, /) -> None:
        if not isinstance(state, ReducedRodState):
            raise TypeError("state must be a ReducedRodState.")
        if (
            state.coordinate_count != self.plan.coordinate_count
            or state.values.shape != self.state_layout.shape
        ):
            raise ValueError("Reduced rod state must match the prepared state layout.")
        if np.dtype(state.values.dtype) != self.coefficient_space.dtype:
            raise TypeError(
                "Reduced rod state dtype does not match the prepared rod dtype."
            )

    def lift_configuration(self, coefficients: ArrayLike, /) -> tuple[Array, Array]:
        return lift_configuration(self, coefficients)

    def lift_velocity_operator(
        self, coefficients: ArrayLike, /
    ) -> AbstractLinearOperator:
        return lift_velocity_operator(self, coefficients)

    def lift_effort_pullback_operator(
        self, coefficients: ArrayLike, /
    ) -> AbstractLinearOperator:
        return lift_effort_pullback_operator(self, coefficients)

    def lift(self, state: ReducedRodState, /) -> RodState:
        return lift_reduced_rod_state(self, state)

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


def _reference_pose(
    rod: PreparedRod,
    base_position: Array,
    base_orientation: Array,
    /,
) -> tuple[Array, Array]:
    start = rod.plan.segment_node_ids[0, 0]
    source_position = rod.plan.rest_positions[start]
    source_orientation = rod.rest_orientations[0]
    if rod.plan.dimension == 2:
        offset = base_orientation - source_orientation
        rotation = _planar_rotation_matrix(offset)
        positions = base_position + ein.contract(
            "ij,nj->ni", rotation, rod.plan.rest_positions - source_position
        )
        orientations = rod.rest_orientations + offset
    else:
        offset = _quaternion_multiply(
            base_orientation, _quaternion_conjugate(source_orientation)
        )
        rotation = _quaternion_rotation_matrix(offset)
        positions = base_position + ein.contract(
            "ij,nj->ni", rotation, rod.plan.rest_positions - source_position
        )
        orientations = _quaternion_multiply(
            jnp.broadcast_to(offset, rod.rest_orientations.shape),
            rod.rest_orientations,
        )
    return positions, orientations


def prepare_reduced_rod(
    rod: PreparedRod,
    plan: ReducedRodPlan,
    /,
) -> PreparedReducedRod:
    """Bind a fixed-base strain-coordinate plan to native rod mechanics."""
    return PreparedReducedRod(rod, plan)


def _native_efforts(
    prepared: PreparedReducedRod,
    native_forces: ArrayLike,
    native_moments: ArrayLike,
    /,
) -> tuple[Array, Array]:
    return prepared.rod.effort_from_load(native_forces, native_moments)


def _power_evidence_from_operator(
    prepared: PreparedReducedRod,
    velocity_operator: AbstractLinearOperator,
    effort_pullback_operator: AbstractLinearOperator,
    rates: Array,
    native_efforts: tuple[Array, Array],
    /,
) -> ReducedRodPowerEvidence:
    native_velocity = velocity_operator.mv(rates)
    generalized_load = effort_pullback_operator.mv(native_efforts)
    native_power = prepared.native_effort_space.pair(native_efforts, native_velocity).real
    reduced_power = prepared.reduced_effort_space.pair(generalized_load, rates).real
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
    """Certify algebraic JVP/VJP virtual-power duality."""
    values = prepared.coefficient_space.validate(jnp.asarray(coefficients))
    rates = prepared.coefficient_space.validate(jnp.asarray(coefficient_velocities))
    velocity_operator = lift_velocity_operator(prepared, values)
    effort_pullback = lift_effort_pullback_operator(prepared, values)
    efforts = _native_efforts(prepared, native_forces, native_moments)
    return _power_evidence_from_operator(
        prepared, velocity_operator, effort_pullback, rates, efforts
    )


def reduced_rod_potential_energy(
    prepared: PreparedReducedRod,
    coefficients: ArrayLike,
    /,
) -> Array:
    """Evaluate native rod potential energy at the lifted configuration."""
    positions, orientations = lift_configuration(prepared, coefficients)
    return rod_potential_energy(prepared.rod, positions, orientations)


def reduced_rod_kinetic_energy(
    prepared: PreparedReducedRod,
    state: ReducedRodState,
    /,
) -> Array:
    """Evaluate native rod kinetic energy at the lifted physical velocity."""
    return evaluate_rod(
        prepared.rod, lift_reduced_rod_state(prepared, state)
    ).kinetic_energy


def _lift_evidence(
    prepared: PreparedReducedRod,
    native_state: RodState,
    /,
) -> ReducedRodLiftEvidence:
    base_node = prepared.path_node_ids[0]
    base_position_error = jnp.sqrt(
        jnp.sum((native_state.positions[base_node] - prepared.base_position) ** 2)
    )
    base_linear_velocity_error = jnp.sqrt(
        jnp.sum(native_state.velocities[base_node] ** 2)
    )
    if prepared.rod.plan.dimension == 2:
        orientation_delta = native_state.orientations[0] - prepared.base_orientation
        base_orientation_error = jnp.abs(
            jnp.arctan2(jnp.sin(orientation_delta), jnp.cos(orientation_delta))
        )
        base_angular_velocity_error = jnp.abs(native_state.angular_velocities[0])
    else:
        relative = _quaternion_multiply(
            _quaternion_conjugate(prepared.base_orientation),
            native_state.orientations[0],
        )
        base_orientation_error = _quaternion_angle(relative)
        base_angular_velocity_error = jnp.sqrt(
            jnp.sum(native_state.angular_velocities[0] ** 2)
        )
    finite = (
        jnp.all(jnp.isfinite(native_state.positions))
        & jnp.all(jnp.isfinite(native_state.velocities))
        & jnp.all(jnp.isfinite(native_state.orientations))
        & jnp.all(jnp.isfinite(native_state.angular_velocities))
        & jnp.isfinite(base_position_error)
        & jnp.isfinite(base_orientation_error)
        & jnp.isfinite(base_linear_velocity_error)
        & jnp.isfinite(base_angular_velocity_error)
    )
    tolerance = prepared.plan.certification_tolerance
    base_pose_valid = (base_position_error <= tolerance) & (
        base_orientation_error <= tolerance
    )
    base_velocity_valid = (base_linear_velocity_error <= tolerance) & (
        base_angular_velocity_error <= tolerance
    )
    return ReducedRodLiftEvidence(
        base_position_error,
        base_orientation_error,
        base_linear_velocity_error,
        base_angular_velocity_error,
        finite,
        base_pose_valid,
        base_velocity_valid,
        finite & base_pose_valid & base_velocity_valid,
    )


def _strain_evidence(
    prepared: PreparedReducedRod,
    coefficients: Array,
    native_evaluation: RodEvaluation,
    /,
) -> ReducedRodStrainEvidence:
    target_stretch_shear, target_bend_twist = target_native_strains(
        prepared, coefficients
    )
    stretch_residual = native_evaluation.stretch_shear_strain - target_stretch_shear
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


def evaluate_reduced_rod(
    prepared: PreparedReducedRod,
    state: ReducedRodState,
    /,
) -> ReducedRodEvaluation:
    """Evaluate the lifted state with native-discrete mechanics authority."""
    if not isinstance(prepared, PreparedReducedRod):
        raise TypeError("prepared must be a PreparedReducedRod.")
    prepared.validate_state(state)
    velocity_operator = lift_velocity_operator(prepared, state.coefficients)
    positions, orientations = lift_configuration(prepared, state.coefficients)
    velocities, angular_velocities = velocity_operator.mv(state.coefficient_velocities)
    native_state = RodState(positions, velocities, orientations, angular_velocities)
    native_evaluation = evaluate_rod(prepared.rod, native_state)
    native_internal_efforts = _native_efforts(
        prepared,
        native_evaluation.internal_forces,
        native_evaluation.internal_moments,
    )
    effort_pullback = lift_effort_pullback_operator(prepared, state.coefficients)
    generalized_internal_load = effort_pullback.mv(native_internal_efforts)
    lift_evidence = _lift_evidence(prepared, native_state)
    strain_evidence = _strain_evidence(prepared, state.coefficients, native_evaluation)
    power_evidence = _power_evidence_from_operator(
        prepared,
        velocity_operator,
        effort_pullback,
        state.coefficient_velocities,
        native_internal_efforts,
    )
    native_discrete_strain_energy = native_evaluation.potential_energy
    energy_error = jnp.abs(
        native_evaluation.potential_energy - native_discrete_strain_energy
    )
    energy_scale = jnp.maximum(
        jnp.asarray(1.0, dtype=energy_error.dtype),
        jnp.maximum(
            jnp.abs(native_evaluation.potential_energy),
            jnp.abs(native_discrete_strain_energy),
        ),
    )
    native_discrete_energy_valid = jnp.isfinite(energy_error) & (
        energy_error <= prepared.plan.quadrature_tolerance * energy_scale
    )
    finite = (
        native_evaluation.finite
        & lift_evidence.finite
        & strain_evidence.finite
        & power_evidence.finite
        & jnp.all(jnp.isfinite(generalized_internal_load))
        & jnp.isfinite(native_discrete_strain_energy)
        & jnp.isfinite(energy_error)
    )
    valid = (
        finite
        & native_evaluation.valid
        & lift_evidence.valid
        & strain_evidence.valid
        & power_evidence.valid
        & native_discrete_energy_valid
    )
    return ReducedRodEvaluation(
        native_state,
        native_evaluation,
        generalized_internal_load,
        native_evaluation.potential_energy,
        native_evaluation.kinetic_energy,
        native_evaluation.total_energy,
        native_discrete_strain_energy,
        energy_error,
        lift_evidence,
        power_evidence,
        strain_evidence,
        finite,
        native_discrete_energy_valid,
        valid,
        prepared.prepared_id,
    )


__all__ = [
    "PreparedReducedRod",
    "PreparedRodStrainBasis",
    "ReducedRodBasePolicy",
    "ReducedRodEvaluation",
    "ReducedRodLiftEvidence",
    "ReducedRodPlan",
    "ReducedRodPowerEvidence",
    "ReducedRodState",
    "ReducedRodStrainEvidence",
    "RodStrainBasisEvidence",
    "RodStrainBasisPlan",
    "evaluate_reduced_rod",
    "lift_configuration",
    "lift_effort_pullback_operator",
    "lift_reduced_rod_state",
    "lift_reduced_rod_velocity",
    "lift_velocity_operator",
    "prepare_reduced_rod",
    "pullback_reduced_rod_loads",
    "reduced_rod_kinetic_energy",
    "reduced_rod_potential_energy",
    "reduced_rod_power_evidence",
]
