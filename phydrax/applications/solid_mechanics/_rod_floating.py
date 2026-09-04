#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import (
    AbstractDiscretePlant,
    ArrayPyTreeSchema,
    ExecutableSignature,
    NumericRevision,
    PlantParameters,
    PlantProposal,
    PlantStepContext,
    SemanticProvenance,
    StateLayout,
)
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    dual_transpose,
    DualSpace,
    FunctionLinearOperator,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ...metrix import (
    EuclideanStateGeometry,
    ProductStateGeometry,
    ProductStateGeometryBlock,
    QuaternionPoseStateGeometry,
)
from ._rod_dynamics import PreparedRod, RodState
from ._rod_loads import RodLoadLedger
from ._rod_reduced_dynamics import (
    PreparedReducedRodDynamics,
    ReducedRodDenseCholeskyPlan,
    ReducedRodDynamicsEvaluation,
    ReducedRodDynamicsPlan,
    ReducedRodMaterial,
    ReducedRodMaterialControl,
    ReducedRodMaterialState,
)
from ._rod_reduction import (
    prepare_reduced_rod,
    PreparedReducedRod,
    ReducedRodPlan,
    ReducedRodState,
)


FloatingRodTwistConvention: TypeAlias = Literal["body", "spatial"]


def _positive_finite(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _quaternion_conjugate(quaternion: Array, /) -> Array:
    return jnp.concatenate((quaternion[..., :1], -quaternion[..., 1:]), axis=-1)


def _quaternion_multiply(left: Array, right: Array, /) -> Array:
    left_scalar = left[..., :1]
    left_vector = left[..., 1:]
    right_scalar = right[..., :1]
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


def _rotate(quaternion: Array, vector: Array, /) -> Array:
    imaginary = quaternion[..., 1:]
    doubled_cross = 2.0 * jnp.cross(imaginary, vector)
    return (
        vector + quaternion[..., :1] * doubled_cross + jnp.cross(imaginary, doubled_cross)
    )


def _body_angular_velocity(quaternion: Array, tangent: Array, /) -> Array:
    return 2.0 * _quaternion_multiply(_quaternion_conjugate(quaternion), tangent)[..., 1:]


def _tree_select(selector: Array, candidate, source):
    return jax.tree_util.tree_map(
        lambda candidate_leaf, source_leaf: jnp.where(
            selector, candidate_leaf, source_leaf
        ),
        candidate,
        source,
    )


class FloatingReducedRodPlan(StrictModule, NonTrainableState):
    """Quaternion free-root wrapper around one spatial reduced-strain plan.

    The wrapped reduction remains a native-discrete, fixed-root parameterization.
    Its root pose is used only as the material reference from which this profile
    applies a free SE(3) action; it is not a kinematic constraint here.
    """

    reduction: ReducedRodPlan
    convention: FloatingRodTwistConvention = eqx.field(static=True)
    pose_tolerance: float = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduction: ReducedRodPlan,
        /,
        *,
        convention: FloatingRodTwistConvention = "body",
        pose_tolerance: float = 1.0e-9,
        label: str | None = None,
    ):
        if not isinstance(reduction, ReducedRodPlan):
            raise TypeError("reduction must be a ReducedRodPlan.")
        if reduction.dimension != 3:
            raise ValueError("Floating reduced rods require a spatial 3-D reduction.")
        if convention not in ("body", "spatial"):
            raise ValueError("convention must be 'body' or 'spatial'.")
        tolerance = _positive_finite(pose_tolerance, "pose_tolerance")
        if tolerance >= np.pi:
            raise ValueError("pose_tolerance must be smaller than pi.")
        self.reduction = reduction
        self.convention = convention
        self.pose_tolerance = tolerance
        self.label = None if label is None else str(label)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "floating-reduced-rod-plan",
                "reduction": reduction.plan_id,
                "dimension": 3,
                "root": "se3-quaternion-position",
                "twist_convention": convention,
                "pose_tolerance": tolerance,
                "contact": "not-composed",
            }
        )


class FloatingReducedRodState(StrictModule):
    """Free-root point and physical tangent with reduced strain coordinates."""

    base_pose: Array
    coefficients: Array
    base_twist: Array
    coefficient_velocities: Array
    coordinate_count: int = eqx.field(static=True)

    def __init__(
        self,
        base_pose: ArrayLike,
        coefficients: ArrayLike,
        base_twist: ArrayLike,
        coefficient_velocities: ArrayLike,
        /,
    ):
        pose = jnp.asarray(base_pose)
        values = jnp.asarray(coefficients)
        twist = jnp.asarray(base_twist)
        rates = jnp.asarray(coefficient_velocities)
        if pose.shape != (7,):
            raise ValueError("base_pose must have quaternion-position shape (7,).")
        if twist.shape != (6,):
            raise ValueError("base_twist must have linear-angular shape (6,).")
        if values.ndim != 1 or values.shape[0] < 1:
            raise ValueError("coefficients must be a nonempty rank-one array.")
        if rates.shape != values.shape:
            raise ValueError("coefficient_velocities must match coefficients.")
        arrays = (pose, values, twist, rates)
        if any(
            not jnp.issubdtype(array.dtype, jnp.inexact) or jnp.iscomplexobj(array)
            for array in arrays
        ):
            raise TypeError("Floating reduced rod state arrays must be real and inexact.")
        if len({np.dtype(array.dtype) for array in arrays}) != 1:
            raise TypeError("Floating reduced rod state arrays must share one dtype.")
        self.base_pose = pose
        self.coefficients = values
        self.base_twist = twist
        self.coefficient_velocities = rates
        self.coordinate_count = int(values.shape[0])

    @property
    def configuration(self) -> Array:
        """Return the 7+q point representation."""
        return jnp.concatenate((self.base_pose, self.coefficients))

    @property
    def velocity(self) -> Array:
        """Return the 6+q physical tangent representation."""
        return jnp.concatenate((self.base_twist, self.coefficient_velocities))

    @property
    def values(self) -> Array:
        """Return packed point followed by physical tangent storage."""
        return jnp.concatenate((self.configuration, self.velocity))

    @property
    def reduced_state(self) -> ReducedRodState:
        return ReducedRodState(self.coefficients, self.coefficient_velocities)


class FloatingReducedRodDirectLoad(StrictModule):
    """One source-resolved effort in the full free-root tangent dual."""

    effort: Array
    source_id: str = eqx.field(static=True)
    power_channel: str = eqx.field(static=True)

    def __init__(self, effort: ArrayLike, /, *, source_id: str, power_channel: str):
        value = jnp.asarray(effort)
        if (
            value.ndim != 1
            or not jnp.issubdtype(value.dtype, jnp.inexact)
            or jnp.iscomplexobj(value)
        ):
            raise TypeError("A floating rod direct effort must be a real rank-one array.")
        source = str(source_id).strip()
        channel = str(power_channel).strip()
        if not source or not channel:
            raise ValueError("source_id and power_channel must be nonempty.")
        self.effort = value
        self.source_id = source
        self.power_channel = channel


class FloatingReducedRodMassEvidence(StrictModule):
    symmetry_error: Array
    minimum_eigenvalue: Array
    maximum_eigenvalue: Array
    minimum_cholesky_pivot: Array
    condition_estimate: Array
    finite: Array
    symmetric: Array
    positive_definite: Array
    conditioned: Array
    valid: Array


class FloatingReducedRodMassResult(StrictModule):
    operator: AbstractLinearOperator
    matrix: Array
    base_base: Array
    base_reduced: Array
    reduced_base: Array
    reduced_reduced: Array
    evidence: FloatingReducedRodMassEvidence
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodSolveEvidence(StrictModule):
    status: Array
    residual_norm: Array
    relative_residual: Array
    iterations: Array
    roundtrip_error: Array
    relative_roundtrip_error: Array
    finite: Array
    converged: Array
    roundtrip_valid: Array
    valid: Array
    solver: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodInverseMassResult(StrictModule):
    acceleration: Array
    inverse_mass_operator: AbstractLinearOperator
    mass: FloatingReducedRodMassResult
    solve_evidence: FloatingReducedRodSolveEvidence
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodBiasResult(StrictModule):
    effort: Array
    lift_acceleration: tuple[Array, Array]
    native_gyroscopic_effort: tuple[Array, Array]
    finite: Array
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodGravityResult(StrictModule):
    effort: Array
    native_effort: tuple[Array, Array]
    finite: Array
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodForceResult(StrictModule):
    elastic_effort: Array
    kelvin_voigt_effort: Array
    gravity_effort: Array
    native_external_effort: Array
    direct_effort: Array
    total_effort: Array
    source_power: Array
    channel_power: Array
    total_power: Array
    finite: Array
    source_ids: tuple[str, ...] = eqx.field(static=True)
    channel_names: tuple[str, ...] = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodEnergyResult(StrictModule):
    kinetic_energy: Array
    stored_energy: Array
    gravitational_potential: Array
    viscous_dissipation: Array
    total_mechanical_energy: Array
    finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodDynamicsEvaluation(StrictModule):
    native_state: RodState
    mass: FloatingReducedRodMassResult
    bias: FloatingReducedRodBiasResult
    forces: FloatingReducedRodForceResult
    energy: FloatingReducedRodEnergyResult
    reduced_evaluation: ReducedRodDynamicsEvaluation
    finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodForwardDynamicsResult(StrictModule):
    acceleration: Array
    rhs_effort: Array
    evaluation: FloatingReducedRodDynamicsEvaluation
    solve_evidence: FloatingReducedRodSolveEvidence
    finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodInverseDynamicsResult(StrictModule):
    required_effort: Array
    dynamic_effort: Array
    residual: Array
    evaluation: FloatingReducedRodDynamicsEvaluation
    finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodMomentum(StrictModule):
    linear: Array
    angular_about_origin: Array
    finite: Array
    dynamics_id: str = eqx.field(static=True)

    @property
    def values(self) -> Array:
        return jnp.concatenate((self.linear, self.angular_about_origin))


class PreparedFloatingReducedRod(StrictModule, NonTrainableState):
    """Contact-free free-root dynamics composed with native reduced rod mechanics."""

    plan: FloatingReducedRodPlan
    reduction: PreparedReducedRod
    reduced_dynamics: PreparedReducedRodDynamics
    fixed_base_dynamics: PreparedReducedRodDynamics
    gravity: Array | None
    pose_geometry: QuaternionPoseStateGeometry
    configuration_geometry: ProductStateGeometry
    configuration_layout: StateLayout
    state_layout: StateLayout
    tangent_space: ArraySpace
    effort_space: DualSpace
    configuration_slice: slice = eqx.field(static=True)
    velocity_slice: slice = eqx.field(static=True)
    base_point_slice: slice = eqx.field(static=True)
    coefficient_point_slice: slice = eqx.field(static=True)
    base_tangent_slice: slice = eqx.field(static=True)
    coefficient_tangent_slice: slice = eqx.field(static=True)
    supports_contact: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        rod: PreparedRod,
        plan: FloatingReducedRodPlan,
        dynamics_plan: ReducedRodDynamicsPlan | None = None,
        /,
        *,
        stretch_shear_material: ReducedRodMaterial | None = None,
        bend_twist_material: ReducedRodMaterial | None = None,
        gravity: ArrayLike | None = None,
    ):
        if not isinstance(rod, PreparedRod):
            raise TypeError("rod must be a PreparedRod.")
        if not isinstance(plan, FloatingReducedRodPlan):
            raise TypeError("plan must be a FloatingReducedRodPlan.")
        if rod.plan.dimension != 3:
            raise ValueError("Floating reduced rods require a spatial 3-D rod.")
        reduction = prepare_reduced_rod(rod, plan.reduction)
        reduced_dynamics = PreparedReducedRodDynamics(
            reduction,
            dynamics_plan,
            stretch_shear_material=stretch_shear_material,
            bend_twist_material=bend_twist_material,
            gravity=None,
        )
        acceleration = None
        if gravity is not None:
            gravity_ = np.asarray(gravity)
            dtype = np.dtype(rod.plan.rest_positions.dtype)
            if (
                gravity_.shape != (3,)
                or not np.issubdtype(gravity_.dtype, np.inexact)
                or np.iscomplexobj(gravity_)
                or not np.all(np.isfinite(gravity_))
            ):
                raise ValueError("gravity must be one finite real spatial vector.")
            acceleration = jnp.asarray(gravity_.astype(dtype, copy=False))
        fixed_base_dynamics = (
            reduced_dynamics
            if acceleration is None
            else PreparedReducedRodDynamics(
                reduction,
                dynamics_plan,
                stretch_shear_material=stretch_shear_material,
                bend_twist_material=bend_twist_material,
                gravity=acceleration,
            )
        )
        dtype = np.dtype(rod.plan.rest_positions.dtype)
        count = reduction.plan.coordinate_count
        pose_geometry = QuaternionPoseStateGeometry(
            convention=plan.convention, tolerance=plan.pose_tolerance
        )
        pose_tangent_space = ArraySpace(
            (6,),
            dtype=dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "floating-rod-base-tangent",
                    "plan": plan.plan_id,
                    "convention": plan.convention,
                }
            ),
        )
        configuration_geometry = ProductStateGeometry(
            (
                ProductStateGeometryBlock(
                    pose_geometry,
                    (7,),
                    block_id="free_root_pose",
                    local_space=pose_tangent_space,
                    tangent_space=pose_tangent_space,
                ),
                ProductStateGeometryBlock(
                    EuclideanStateGeometry(),
                    (count,),
                    block_id="reduced_strain_coordinates",
                    local_space=reduction.coefficient_space,
                    tangent_space=reduction.coefficient_space,
                ),
            ),
            geometry_id=canonical_fingerprint(
                {
                    "kind": "floating-reduced-rod-configuration-geometry",
                    "plan": plan.plan_id,
                    "reduction": reduction.prepared_id,
                }
            ),
        )
        tangent_space = ArraySpace(
            (6 + count,),
            dtype=dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "floating-reduced-rod-physical-tangent",
                    "plan": plan.plan_id,
                    "reduction": reduction.prepared_id,
                }
            ),
        )
        effort_space = DualSpace(
            tangent_space,
            space_id=canonical_fingerprint(
                {
                    "kind": "floating-reduced-rod-physical-effort-dual",
                    "tangent": tangent_space.space_id,
                }
            ),
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-floating-reduced-rod",
                "plan": plan.plan_id,
                "reduction": reduction.prepared_id,
                "reduced_dynamics": reduced_dynamics.dynamics_id,
                "gravity": None
                if acceleration is None
                else array_tree_fingerprint(np.asarray(acceleration)),
                "configuration_geometry": configuration_geometry.geometry_id,
                "tangent_space": tangent_space.space_id,
                "effort_space": effort_space.space_id,
                "contact": "not-composed",
            }
        )
        configuration_layout = StateLayout(
            (7 + count,),
            axes=("floating_reduced_configuration",),
            component_names=(
                "root_quaternion_w",
                "root_quaternion_x",
                "root_quaternion_y",
                "root_quaternion_z",
                "root_position_x",
                "root_position_y",
                "root_position_z",
                *(f"q:{index}" for index in range(count)),
            ),
            geometry=configuration_geometry,
            local_space=tangent_space,
            tangent_space=tangent_space,
            local_component_names=(
                "root_linear_x",
                "root_linear_y",
                "root_linear_z",
                "root_angular_x",
                "root_angular_y",
                "root_angular_z",
                *(f"dq:{index}" for index in range(count)),
            ),
            tangent_component_names=(
                "root_linear_velocity_x",
                "root_linear_velocity_y",
                "root_linear_velocity_z",
                "root_angular_velocity_x",
                "root_angular_velocity_y",
                "root_angular_velocity_z",
                *(f"v:{index}" for index in range(count)),
            ),
            layout_id=f"state-layout:floating-reduced-rod:{prepared_id}",
        )
        self.plan = plan
        self.reduction = reduction
        self.reduced_dynamics = reduced_dynamics
        self.fixed_base_dynamics = fixed_base_dynamics
        self.gravity = acceleration
        self.pose_geometry = pose_geometry
        self.configuration_geometry = configuration_geometry
        self.configuration_layout = configuration_layout
        self.state_layout = configuration_layout
        self.tangent_space = tangent_space
        self.effort_space = effort_space
        self.configuration_slice = slice(0, 7 + count)
        self.velocity_slice = slice(7 + count, 13 + 2 * count)
        self.base_point_slice = slice(0, 7)
        self.coefficient_point_slice = slice(7, 7 + count)
        self.base_tangent_slice = slice(0, 6)
        self.coefficient_tangent_slice = slice(6, 6 + count)
        self.supports_contact = False
        self.prepared_id = prepared_id
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "floating-reduced-rod-block-dynamics",
                "prepared": prepared_id,
                "solver": reduced_dynamics.plan.plan_id,
                "mechanics": "native-discrete-reduced-rod-composition",
            }
        )

    @property
    def coordinate_count(self) -> int:
        return self.reduction.plan.coordinate_count

    @property
    def point_size(self) -> int:
        return self.configuration_layout.size

    @property
    def tangent_size(self) -> int:
        return self.configuration_layout.tangent_size

    @property
    def state_size(self) -> int:
        return self.point_size + self.tangent_size

    def initialize_state(
        self, base_pose: ArrayLike | None = None, /
    ) -> FloatingReducedRodState:
        pose = (
            jnp.concatenate(
                (self.reduction.base_orientation, self.reduction.base_position)
            )
            if base_pose is None
            else jnp.asarray(base_pose)
        )
        return FloatingReducedRodState(
            pose,
            self.reduction.reference_coefficients,
            jnp.zeros((6,), dtype=self.reduction.reference_coefficients.dtype),
            jnp.zeros_like(self.reduction.reference_coefficients),
        )

    def initialize_material_state(self, /) -> ReducedRodMaterialState:
        return self.reduced_dynamics.initialize_material_state()

    def initialize_material_control(self, /) -> ReducedRodMaterialControl:
        return self.reduced_dynamics.initialize_material_control()

    def state_from_configuration_velocity(
        self, configuration: ArrayLike, velocity: ArrayLike, /
    ) -> FloatingReducedRodState:
        point = jnp.asarray(configuration)
        tangent = jnp.asarray(velocity)
        if point.shape != self.configuration_layout.shape:
            raise ValueError("configuration does not match the 7+q point layout.")
        if tangent.shape != (self.tangent_size,):
            raise ValueError("velocity does not match the 6+q tangent layout.")
        return FloatingReducedRodState(
            point[self.base_point_slice],
            point[self.coefficient_point_slice],
            tangent[self.base_tangent_slice],
            tangent[self.coefficient_tangent_slice],
        )

    def _validated_configuration(self, state: FloatingReducedRodState, /) -> Array:
        if not isinstance(state, FloatingReducedRodState):
            raise TypeError("state must be a FloatingReducedRodState.")
        if state.coordinate_count != self.coordinate_count:
            raise ValueError("Floating state coordinate count does not match the plan.")
        configuration = state.configuration
        velocity = state.velocity
        if configuration.shape != self.configuration_layout.shape:
            raise ValueError("Floating state point does not match its layout.")
        self.tangent_space.validate(velocity)
        dtype = self.reduction.coefficient_space.dtype
        if np.dtype(configuration.dtype) != dtype or np.dtype(velocity.dtype) != dtype:
            raise TypeError("Floating state dtype does not match the prepared rod.")
        return eqx.error_if(
            configuration,
            ~self.configuration_geometry.contains(configuration),
            "Floating rod configuration lies outside its quaternion-product geometry.",
        )

    def validate_state(self, state: FloatingReducedRodState, /) -> None:
        self._validated_configuration(state)

    def _lift_configuration_values(self, configuration: Array, /) -> tuple[Array, Array]:
        base_pose, coefficients = self.configuration_geometry.split_point(configuration)
        base_orientation = base_pose[:4]
        base_position = base_pose[4:]
        fixed_positions, fixed_orientations = self.reduction.lift_configuration(
            coefficients
        )
        rotation = _quaternion_multiply(
            base_orientation,
            _quaternion_conjugate(self.reduction.base_orientation),
        )
        positions = base_position + _rotate(
            rotation, fixed_positions - self.reduction.base_position
        )
        orientations = _quaternion_multiply(rotation, fixed_orientations)
        return positions, orientations

    def lift_configuration(
        self, state: FloatingReducedRodState, /
    ) -> tuple[Array, Array]:
        return self._lift_configuration_values(self._validated_configuration(state))

    def _native_velocity_values(
        self, configuration: Array, velocity: Array, /
    ) -> tuple[Array, Array]:
        tangent = self.tangent_space.validate(velocity)
        zero = jnp.zeros_like(tangent)

        def configuration_curve(local):
            point = self.configuration_geometry.retract(configuration, local)
            return self._lift_configuration_values(point)

        (positions, orientations), (linear, orientation_tangent) = jax.jvp(
            configuration_curve, (zero,), (tangent,)
        )
        del positions
        angular = _body_angular_velocity(orientations, orientation_tangent)
        return self.reduction.native_velocity_space.validate((linear, angular))

    def velocity_operator(
        self, state: FloatingReducedRodState, /
    ) -> AbstractLinearOperator:
        configuration = self._validated_configuration(state)
        return FunctionLinearOperator(
            lambda tangent: self._native_velocity_values(configuration, tangent),
            source=self.tangent_space,
            target=self.reduction.native_velocity_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "floating-reduced-rod-velocity-lift",
                    "prepared": self.prepared_id,
                }
            ),
        )

    def effort_pullback_operator(
        self, state: FloatingReducedRodState, /
    ) -> AbstractLinearOperator:
        """Return the exact algebraic native-effort to 6+q dual pullback."""
        return dual_transpose(self.velocity_operator(state))

    def lift(self, state: FloatingReducedRodState, /) -> RodState:
        configuration = self._validated_configuration(state)
        positions, orientations = self._lift_configuration_values(configuration)
        linear, angular = self._native_velocity_values(configuration, state.velocity)
        return RodState(positions, linear, orientations, angular)

    def _native_inertia_action(
        self, velocity: tuple[Array, Array], /
    ) -> tuple[Array, Array]:
        linear, angular = self.reduction.native_velocity_space.validate(velocity)
        forces = self.reduction.rod.node_masses[:, None] * linear
        moments = ein.contract("sij,sj->si", self.reduction.rod.segment_inertias, angular)
        return self.reduction.native_effort_space.validate((forces, moments))

    def _pullback(
        self, configuration: Array, native_effort: tuple[Array, Array], /
    ) -> Array:
        zero = jnp.zeros((self.tangent_size,), dtype=configuration.dtype)
        return self.effort_space.validate(
            jax.linear_transpose(
                lambda tangent: self._native_velocity_values(configuration, tangent),
                zero,
            )(self.reduction.native_effort_space.validate(native_effort))[0]
        )

    def _mass_action(self, configuration: Array, tangent: Array, /) -> Array:
        native_velocity = self._native_velocity_values(configuration, tangent)
        return self._pullback(configuration, self._native_inertia_action(native_velocity))

    def mass(self, state: FloatingReducedRodState, /) -> FloatingReducedRodMassResult:
        configuration = self._validated_configuration(state)
        size = self.tangent_size
        dtype = configuration.dtype
        columns = jax.vmap(lambda basis: self._mass_action(configuration, basis))(
            jnp.eye(size, dtype=dtype)
        )
        matrix = jnp.swapaxes(columns, -1, -2)
        operator = DenseLinearOperator(
            matrix,
            source=self.tangent_space,
            target=self.effort_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "floating-reduced-rod-dense-block-mass",
                    "dynamics": self.dynamics_id,
                }
            ),
        )
        symmetric_part = 0.5 * (matrix + matrix.T)
        eigenvalues = jnp.linalg.eigvalsh(symmetric_part)
        factor = jnp.linalg.cholesky(symmetric_part)
        symmetry_error = jnp.max(jnp.abs(matrix - matrix.T))
        minimum = jnp.min(eigenvalues)
        maximum = jnp.max(eigenvalues)
        pivot = jnp.min(jnp.diag(factor))
        condition = maximum / minimum
        plan = self.reduced_dynamics.plan
        positivity_tolerance = (
            plan.pivot_tolerance
            if isinstance(plan, ReducedRodDenseCholeskyPlan)
            else plan.positivity_tolerance
        )
        finite = (
            jnp.all(jnp.isfinite(matrix))
            & jnp.all(jnp.isfinite(eigenvalues))
            & jnp.isfinite(pivot)
            & jnp.isfinite(condition)
        )
        symmetric = symmetry_error <= plan.symmetry_tolerance * jnp.maximum(1.0, maximum)
        positive = (minimum > positivity_tolerance) & (pivot > positivity_tolerance)
        conditioned = jnp.isfinite(condition) & (condition <= plan.condition_limit)
        evidence = FloatingReducedRodMassEvidence(
            symmetry_error,
            minimum,
            maximum,
            pivot,
            condition,
            finite,
            symmetric,
            positive,
            conditioned,
            finite & symmetric & positive & conditioned,
        )
        split = 6
        return FloatingReducedRodMassResult(
            operator,
            matrix,
            matrix[:split, :split],
            matrix[:split, split:],
            matrix[split:, :split],
            matrix[split:, split:],
            evidence,
            self.dynamics_id,
        )

    def _inverse_mass_from_mass(
        self, mass: FloatingReducedRodMassResult, effort: ArrayLike, /
    ) -> FloatingReducedRodInverseMassResult:
        if not isinstance(mass, FloatingReducedRodMassResult):
            raise TypeError("mass must be a FloatingReducedRodMassResult.")
        if mass.dynamics_id != self.dynamics_id:
            raise ValueError("Floating rod mass result belongs to different dynamics.")
        rhs = self.effort_space.validate(jnp.asarray(effort))
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
        solver_operator = DenseLinearOperator(
            mass.matrix,
            source=self.tangent_space,
            target=self.tangent_space,
            properties=properties,
        )
        tangent_rhs = self.tangent_space.inverse_riesz(rhs)
        result = solve(
            LinearSystem(solver_operator),
            tangent_rhs,
            policy=self.reduced_dynamics.solve_policy,
        )
        acceleration = self.tangent_space.validate(result.value)
        reconstructed = mass.operator.mv(acceleration)
        error = jnp.linalg.norm(reconstructed - rhs)
        scale = jnp.maximum(1.0, jnp.linalg.norm(rhs))
        relative = error / scale
        diagnostics = result.diagnostics
        finite = (
            jnp.all(jnp.isfinite(acceleration))
            & jnp.isfinite(error)
            & diagnostics.finite
            & mass.evidence.finite
        )
        roundtrip_valid = relative <= self.reduced_dynamics.plan.roundtrip_tolerance
        evidence = FloatingReducedRodSolveEvidence(
            result.status,
            diagnostics.residual_norm,
            diagnostics.relative_residual,
            diagnostics.iterations,
            error,
            relative,
            finite,
            result.successful & diagnostics.converged,
            roundtrip_valid,
            finite
            & mass.evidence.valid
            & result.successful
            & diagnostics.converged
            & roundtrip_valid,
            self.reduced_dynamics.plan.solver,
            self.dynamics_id,
        )

        def inverse_action(value):
            tangent_value = self.tangent_space.inverse_riesz(value)
            return solve(
                LinearSystem(solver_operator),
                tangent_value,
                policy=self.reduced_dynamics.solve_policy,
            ).value

        inverse_operator = FunctionLinearOperator(
            inverse_action,
            source=self.effort_space,
            target=self.tangent_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "floating-reduced-rod-inverse-mass",
                    "dynamics": self.dynamics_id,
                }
            ),
        )
        return FloatingReducedRodInverseMassResult(
            acceleration, inverse_operator, mass, evidence, self.dynamics_id
        )

    def inverse_mass(
        self, state: FloatingReducedRodState, effort: ArrayLike, /
    ) -> FloatingReducedRodInverseMassResult:
        return self._inverse_mass_from_mass(self.mass(state), effort)

    def bias(self, state: FloatingReducedRodState, /) -> FloatingReducedRodBiasResult:
        configuration = self._validated_configuration(state)
        velocity = self.tangent_space.validate(state.velocity)
        native_velocity = self._native_velocity_values(configuration, velocity)
        zero = jnp.zeros_like(velocity)

        def lifted_at(local):
            point = self.configuration_geometry.retract(configuration, local)
            return self._native_velocity_values(point, velocity)

        _, lift_acceleration = jax.jvp(lifted_at, (zero,), (velocity,))
        angular_momentum = ein.contract(
            "sij,sj->si",
            self.reduction.rod.segment_inertias,
            native_velocity[1],
        )
        gyroscopic = (
            jnp.zeros_like(native_velocity[0]),
            jnp.cross(native_velocity[1], angular_momentum),
        )
        inertial = self._native_inertia_action(lift_acceleration)
        native_bias = (inertial[0], inertial[1] + gyroscopic[1])
        effort = self._pullback(configuration, native_bias)
        finite = (
            jnp.all(jnp.isfinite(effort))
            & jnp.all(jnp.isfinite(lift_acceleration[0]))
            & jnp.all(jnp.isfinite(lift_acceleration[1]))
            & jnp.all(jnp.isfinite(gyroscopic[1]))
        )
        return FloatingReducedRodBiasResult(
            effort, lift_acceleration, gyroscopic, finite, self.dynamics_id
        )

    def gravity_effort(
        self, state: FloatingReducedRodState, /
    ) -> FloatingReducedRodGravityResult:
        configuration = self._validated_configuration(state)
        native_effort = (
            jnp.zeros(
                self.reduction.rod.plan.rest_positions.shape,
                dtype=configuration.dtype,
            ),
            jnp.zeros(
                (self.reduction.rod.plan.segment_count, 3),
                dtype=configuration.dtype,
            ),
        )
        if self.gravity is not None:
            native_effort = (
                self.reduction.rod.node_masses[:, None] * self.gravity[None, :],
                native_effort[1],
            )
        effort = self._pullback(configuration, native_effort)
        finite = (
            jnp.all(jnp.isfinite(native_effort[0]))
            & jnp.all(jnp.isfinite(native_effort[1]))
            & jnp.all(jnp.isfinite(effort))
        )
        return FloatingReducedRodGravityResult(
            effort, native_effort, finite, self.dynamics_id
        )

    def _forces(
        self,
        state: FloatingReducedRodState,
        reduced_evaluation: ReducedRodDynamicsEvaluation,
        native_loads: RodLoadLedger | None,
        direct_loads: Sequence[FloatingReducedRodDirectLoad],
        /,
    ) -> FloatingReducedRodForceResult:
        configuration = self._validated_configuration(state)
        velocity = state.velocity
        zeros = jnp.zeros((6,), dtype=velocity.dtype)
        elastic = jnp.concatenate((zeros, reduced_evaluation.forces.elastic_effort))
        viscous = jnp.concatenate((zeros, reduced_evaluation.forces.kelvin_voigt_effort))
        total_zeros = jnp.zeros_like(velocity)
        gravity = total_zeros
        external = total_zeros
        direct = total_zeros
        source_ids: list[str] = ["elastic", "kelvin_voigt"]
        source_channels: list[str] = ["elastic", "kelvin_voigt"]
        source_efforts: list[Array] = [elastic, viscous]
        gravity_result = self.gravity_effort(state)
        gravity = gravity_result.effort
        if self.gravity is not None:
            source_ids.append("gravity")
            source_channels.append("gravity")
            source_efforts.append(gravity)
        if native_loads is not None:
            for load, native_effort in zip(
                native_loads.loads,
                native_loads.source_efforts(self.reduction.rod),
                strict=True,
            ):
                if load.source_id in source_ids:
                    raise ValueError("Every floating rod load source_id must be unique.")
                effort = self._pullback(configuration, native_effort)
                external = external + effort
                source_ids.append(load.source_id)
                source_channels.append(load.power_channel)
                source_efforts.append(effort)
        for load in tuple(direct_loads):
            if not isinstance(load, FloatingReducedRodDirectLoad):
                raise TypeError(
                    "direct_loads must contain FloatingReducedRodDirectLoad values."
                )
            if load.source_id in source_ids:
                raise ValueError("Every floating rod load source_id must be unique.")
            effort = self.effort_space.validate(load.effort)
            direct = direct + effort
            source_ids.append(load.source_id)
            source_channels.append(load.power_channel)
            source_efforts.append(effort)
        efforts = jnp.stack(tuple(source_efforts))
        source_power = jax.vmap(
            lambda effort: self.effort_space.pair(effort, velocity).real
        )(efforts)
        channel_names = tuple(dict.fromkeys(source_channels))
        channel_power = jnp.stack(
            tuple(
                jnp.sum(
                    source_power[
                        jnp.asarray(
                            [
                                index
                                for index, name in enumerate(source_channels)
                                if name == channel
                            ],
                            dtype=jnp.int32,
                        )
                    ]
                )
                for channel in channel_names
            )
        )
        total = elastic + viscous + gravity + external + direct
        total_power = jnp.sum(source_power)
        finite = (
            jnp.all(jnp.isfinite(efforts))
            & jnp.all(jnp.isfinite(source_power))
            & jnp.all(jnp.isfinite(channel_power))
            & jnp.all(jnp.isfinite(total))
            & jnp.isfinite(total_power)
        )
        return FloatingReducedRodForceResult(
            elastic,
            viscous,
            gravity,
            external,
            direct,
            total,
            source_power,
            channel_power,
            total_power,
            finite,
            tuple(source_ids),
            channel_names,
            self.dynamics_id,
        )

    def _energy(
        self,
        state: FloatingReducedRodState,
        native_state: RodState,
        reduced_evaluation: ReducedRodDynamicsEvaluation,
        /,
    ) -> FloatingReducedRodEnergyResult:
        native_velocity = (native_state.velocities, native_state.angular_velocities)
        native_momentum = self._native_inertia_action(native_velocity)
        kinetic = (
            0.5
            * self.reduction.native_effort_space.pair(
                native_momentum, native_velocity
            ).real
        )
        stored = reduced_evaluation.energy.stored_energy
        dissipation = reduced_evaluation.energy.viscous_dissipation
        gravitational = (
            jnp.asarray(0.0, dtype=kinetic.dtype)
            if self.gravity is None
            else -jnp.sum(
                self.reduction.rod.node_masses[:, None]
                * native_state.positions
                * self.gravity[None, :]
            )
        )
        total = kinetic + stored + gravitational
        finite = (
            jnp.isfinite(kinetic)
            & jnp.isfinite(stored)
            & jnp.isfinite(dissipation)
            & jnp.isfinite(gravitational)
            & jnp.isfinite(total)
        )
        return FloatingReducedRodEnergyResult(
            kinetic,
            stored,
            gravitational,
            dissipation,
            total,
            finite,
            finite & reduced_evaluation.energy.valid,
            self.dynamics_id,
        )

    def evaluate(
        self,
        state: FloatingReducedRodState,
        /,
        *,
        source_state: FloatingReducedRodState | None = None,
        material_state: ReducedRodMaterialState | None = None,
        material_control: ReducedRodMaterialControl | None = None,
        time: ArrayLike = 0.0,
        step_size: ArrayLike = 1.0,
        native_loads: RodLoadLedger | None = None,
        direct_loads: Sequence[FloatingReducedRodDirectLoad] = (),
    ) -> FloatingReducedRodDynamicsEvaluation:
        self._validated_configuration(state)
        reduced_source = None
        if source_state is not None:
            self._validated_configuration(source_state)
            reduced_source = source_state.reduced_state
        reduced_evaluation = self.reduced_dynamics.evaluate(
            state.reduced_state,
            source_state=reduced_source,
            material_state=material_state,
            material_control=material_control,
            time=time,
            step_size=step_size,
        )
        native_state = self.lift(state)
        mass = self.mass(state)
        bias = self.bias(state)
        forces = self._forces(state, reduced_evaluation, native_loads, direct_loads)
        energy = self._energy(state, native_state, reduced_evaluation)
        finite = (
            reduced_evaluation.finite
            & mass.evidence.finite
            & bias.finite
            & forces.finite
            & energy.finite
            & jnp.all(jnp.isfinite(native_state.positions))
            & jnp.all(jnp.isfinite(native_state.velocities))
            & jnp.all(jnp.isfinite(native_state.orientations))
            & jnp.all(jnp.isfinite(native_state.angular_velocities))
        )
        valid = (
            finite
            & reduced_evaluation.valid
            & mass.evidence.valid
            & energy.valid
            & self.configuration_geometry.contains(state.configuration)
        )
        return FloatingReducedRodDynamicsEvaluation(
            native_state,
            mass,
            bias,
            forces,
            energy,
            reduced_evaluation,
            finite,
            valid,
            self.dynamics_id,
        )

    def forward_dynamics(
        self, state: FloatingReducedRodState, /, **kwargs
    ) -> FloatingReducedRodForwardDynamicsResult:
        evaluation = self.evaluate(state, **kwargs)
        rhs = evaluation.forces.total_effort - evaluation.bias.effort
        inverse = self._inverse_mass_from_mass(evaluation.mass, rhs)
        finite = evaluation.finite & inverse.solve_evidence.finite
        valid = evaluation.valid & inverse.solve_evidence.valid
        return FloatingReducedRodForwardDynamicsResult(
            inverse.acceleration,
            rhs,
            evaluation,
            inverse.solve_evidence,
            finite,
            valid,
            self.dynamics_id,
        )

    def inverse_dynamics(
        self,
        state: FloatingReducedRodState,
        acceleration: ArrayLike,
        /,
        **kwargs,
    ) -> FloatingReducedRodInverseDynamicsResult:
        evaluation = self.evaluate(state, **kwargs)
        acceleration_ = self.tangent_space.validate(jnp.asarray(acceleration))
        dynamic = evaluation.mass.operator.mv(acceleration_) + evaluation.bias.effort
        required = dynamic - evaluation.forces.total_effort
        residual = dynamic - (evaluation.forces.total_effort + required)
        finite = (
            evaluation.finite
            & jnp.all(jnp.isfinite(required))
            & jnp.all(jnp.isfinite(residual))
        )
        return FloatingReducedRodInverseDynamicsResult(
            required,
            dynamic,
            residual,
            evaluation,
            finite,
            evaluation.valid & finite,
            self.dynamics_id,
        )

    def fixed_base_evaluation(
        self, state: FloatingReducedRodState, /, **kwargs
    ) -> ReducedRodDynamicsEvaluation:
        """Delegate the constrained-root limit to the authoritative fixed profile."""
        self._validated_configuration(state)
        source_state = kwargs.pop("source_state", None)
        if source_state is not None:
            self._validated_configuration(source_state)
            kwargs["source_state"] = source_state.reduced_state
        return self.fixed_base_dynamics.evaluate(state.reduced_state, **kwargs)

    def fixed_base_forward_dynamics(self, state: FloatingReducedRodState, /, **kwargs):
        """Return the exact fixed-profile evolution after imposing zero root motion."""
        self._validated_configuration(state)
        source_state = kwargs.pop("source_state", None)
        if source_state is not None:
            self._validated_configuration(source_state)
            kwargs["source_state"] = source_state.reduced_state
        return self.fixed_base_dynamics.forward_dynamics(state.reduced_state, **kwargs)

    def spatial_momentum(
        self, state: FloatingReducedRodState, /
    ) -> FloatingReducedRodMomentum:
        native = self.lift(state)
        nodal_momentum = self.reduction.rod.node_masses[:, None] * native.velocities
        body_angular_momentum = ein.contract(
            "sij,sj->si",
            self.reduction.rod.segment_inertias,
            native.angular_velocities,
        )
        linear = jnp.sum(nodal_momentum, axis=0)
        angular = jnp.sum(jnp.cross(native.positions, nodal_momentum), axis=0) + jnp.sum(
            _rotate(native.orientations, body_angular_momentum), axis=0
        )
        finite = jnp.all(jnp.isfinite(linear)) & jnp.all(jnp.isfinite(angular))
        return FloatingReducedRodMomentum(linear, angular, finite, self.dynamics_id)

    def spatial_momentum_rate(
        self, state: FloatingReducedRodState, acceleration: ArrayLike, /
    ) -> FloatingReducedRodMomentum:
        configuration = self._validated_configuration(state)
        velocity = self.tangent_space.validate(state.velocity)
        acceleration_ = self.tangent_space.validate(jnp.asarray(acceleration))
        zero = jnp.zeros_like(velocity)

        def momentum_at(local, tangent):
            point = self.configuration_geometry.retract(configuration, local)
            candidate = self.state_from_configuration_velocity(point, tangent)
            return self.spatial_momentum(candidate).values

        _, rate = jax.jvp(
            momentum_at,
            (zero, velocity),
            (velocity, acceleration_),
        )
        finite = jnp.all(jnp.isfinite(rate))
        return FloatingReducedRodMomentum(rate[:3], rate[3:], finite, self.dynamics_id)


class FloatingReducedRodPlantState(StrictModule):
    """Complete domain payload committed by the floating rod plant."""

    rod_state: FloatingReducedRodState
    material_state: ReducedRodMaterialState


class FloatingReducedRodPlantControl(StrictModule):
    """Applied effort in the exact full floating tangent dual."""

    effort: Array

    def __init__(self, effort: ArrayLike, /):
        value = jnp.asarray(effort)
        if (
            value.ndim != 1
            or not jnp.issubdtype(value.dtype, jnp.inexact)
            or jnp.iscomplexobj(value)
        ):
            raise TypeError("Floating rod plant effort must be a real rank-one array.")
        self.effort = value


class FloatingReducedRodPlantParameterValues(StrictModule):
    """Explicit zero-width dynamic parameter payload for a prepared plant."""

    values: Array

    def __init__(self, values: ArrayLike, /):
        array = jnp.asarray(values)
        if array.shape != (0,):
            raise ValueError(
                "Prepared floating rod parameter values must have shape (0,)."
            )
        if not jnp.issubdtype(array.dtype, jnp.inexact) or jnp.iscomplexobj(array):
            raise TypeError(
                "Prepared floating rod parameter values must be real and inexact."
            )
        self.values = array


class FloatingReducedRodPlantEvidence(StrictModule):
    dynamics: FloatingReducedRodForwardDynamicsResult
    requested_increment: Array
    chart_valid: Array
    candidate_finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodResetEvidence(StrictModule):
    configuration_valid: Array
    finite: Array
    valid: Array
    dynamics_id: str = eqx.field(static=True)


class FloatingReducedRodPlant(AbstractDiscretePlant):
    """Single-case, contact-free transactional plant for a prepared floating rod."""

    prepared: PreparedFloatingReducedRod
    initial_payload: FloatingReducedRodPlantState
    parameter_values: FloatingReducedRodPlantParameterValues
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: FloatingReducedRodPlantState
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    supports_contact: bool = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedFloatingReducedRod,
        /,
        *,
        initial_state: FloatingReducedRodState | None = None,
    ):
        if not isinstance(prepared, PreparedFloatingReducedRod):
            raise TypeError("prepared must be a PreparedFloatingReducedRod.")
        state = prepared.initialize_state() if initial_state is None else initial_state
        prepared.validate_state(state)
        payload = FloatingReducedRodPlantState(
            state, prepared.initialize_material_state()
        )
        control = FloatingReducedRodPlantControl(
            jnp.zeros((prepared.tangent_size,), dtype=state.velocity.dtype)
        )
        parameter_values = FloatingReducedRodPlantParameterValues(
            jnp.zeros((0,), dtype=state.velocity.dtype)
        )
        state_schema = ArrayPyTreeSchema.from_tree(
            payload,
            case_ndim=0,
            schema_id=f"floating-reduced-rod-state:{prepared.prepared_id}",
        )
        control_schema = ArrayPyTreeSchema.from_tree(
            control,
            case_ndim=0,
            schema_id=f"floating-reduced-rod-control:{prepared.prepared_id}",
        )
        parameter_schema = ArrayPyTreeSchema.from_tree(
            parameter_values,
            case_ndim=0,
            schema_id=f"floating-reduced-rod-parameters:{prepared.prepared_id}",
        )
        semantic = SemanticProvenance(
            {
                "kind": "floating-reduced-rod-contact-free-plant",
                "prepared": prepared.prepared_id,
                "dynamics": prepared.dynamics_id,
                "state_schema": state_schema.content_id,
                "control_schema": control_schema.content_id,
                "parameter_schema": parameter_schema.content_id,
                "step": "semi-implicit-euler-with-se3-retraction",
                "contact": "not-composed",
            }
        )
        numeric = NumericRevision(
            semantic,
            {
                "rest_positions": prepared.reduction.rod.plan.rest_positions,
                "node_masses": prepared.reduction.rod.node_masses,
                "segment_inertias": prepared.reduction.rod.segment_inertias,
                "stretch_shear_basis": prepared.reduction.stretch_shear_basis,
                "bend_twist_basis": prepared.reduction.bend_twist_basis,
                "reference_coefficients": prepared.reduction.reference_coefficients,
                "gravity": jnp.zeros((0,), dtype=state.velocity.dtype)
                if prepared.gravity is None
                else prepared.gravity,
            },
        )
        signature = ExecutableSignature(
            shapes={
                "configuration": (prepared.point_size,),
                "velocity": (prepared.tangent_size,),
                "effort": (prepared.tangent_size,),
            },
            dtypes={"state": state.velocity.dtype},
            space_ids={
                "configuration": prepared.configuration_layout.layout_id,
                "tangent": prepared.tangent_space.space_id,
                "effort": prepared.effort_space.space_id,
            },
            topology_ids={"prepared": prepared.prepared_id},
            capacities={"contact_constraints": 0},
            algorithm_facts={
                "integrator": "semi-implicit-euler",
                "configuration_update": "product-se3-retraction",
                "root_convention": prepared.plan.convention,
                "contact": "not-composed",
            },
        )
        self.prepared = prepared
        self.initial_payload = payload
        self.parameter_values = parameter_values
        self.state_schema = state_schema
        self.control_schema = control_schema
        self.parameter_schema = parameter_schema
        self.reset_fallback = payload
        self.semantic_provenance = semantic
        self.numeric_revision = numeric
        self.execution_signature = signature
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True
        self.supports_contact = False

    def bind_parameters(self, /) -> PlantParameters:
        """Bind the explicit zero-width values to this exact numeric revision."""
        return PlantParameters(
            self.parameter_values,
            self.parameter_schema.schema_id,
            self.numeric_revision,
        )

    def zero_control(self, /) -> FloatingReducedRodPlantControl:
        return FloatingReducedRodPlantControl(
            jnp.zeros(
                (self.prepared.tangent_size,),
                dtype=self.initial_payload.rod_state.velocity.dtype,
            )
        )

    def propose_reset(
        self,
        keys: Array,
        parameters: FloatingReducedRodPlantParameterValues,
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        del keys, parameters, initial_time
        if case_shape:
            raise ValueError("FloatingReducedRodPlant is a single-case plant.")
        state = self.initial_payload.rod_state
        configuration_valid = self.prepared.configuration_geometry.contains(
            state.configuration
        )
        finite = self.state_schema.finite_mask(self.initial_payload)
        valid = configuration_valid & finite
        evidence = FloatingReducedRodResetEvidence(
            configuration_valid, finite, valid, self.prepared.dynamics_id
        )
        status = jnp.where(valid, 0, 1).astype(jnp.int32)
        return PlantProposal(
            self.initial_payload,
            self.initial_payload,
            jnp.asarray(True),
            valid,
            status,
            jnp.asarray(0, dtype=jnp.int32),
            evidence,
        )

    def propose_step(
        self,
        context: PlantStepContext,
        source: FloatingReducedRodPlantState,
        commands: FloatingReducedRodPlantControl | None,
        parameters: FloatingReducedRodPlantParameterValues,
        keys: Array,
        /,
    ) -> PlantProposal:
        del parameters, keys
        if commands is None:
            raise TypeError(
                "FloatingReducedRodPlant requires a full dual effort control."
            )
        effort = self.prepared.effort_space.validate(commands.effort)
        direct = FloatingReducedRodDirectLoad(
            effort, source_id="plant_control", power_channel="control"
        )
        duration = jnp.asarray(context.duration, dtype=source.rod_state.velocity.dtype)
        dynamics = self.prepared.forward_dynamics(
            source.rod_state,
            material_state=source.material_state,
            time=context.target_time,
            step_size=duration,
            direct_loads=(direct,),
        )
        candidate_velocity = source.rod_state.velocity + duration * dynamics.acceleration
        requested_increment = duration * candidate_velocity
        angular_increment = requested_increment[3:6]
        chart_valid = jnp.all(jnp.isfinite(requested_increment)) & (
            jnp.linalg.norm(angular_increment)
            < jnp.pi - self.prepared.plan.pose_tolerance
        )
        safe_increment = jnp.where(
            chart_valid, requested_increment, jnp.zeros_like(requested_increment)
        )
        candidate_configuration = self.prepared.configuration_geometry.retract(
            source.rod_state.configuration, safe_increment
        )
        candidate_state = self.prepared.state_from_configuration_velocity(
            candidate_configuration, candidate_velocity
        )
        candidate = FloatingReducedRodPlantState(
            candidate_state,
            dynamics.evaluation.reduced_evaluation.candidate_material_state,
        )
        candidate_finite = self.state_schema.finite_mask(candidate)
        configuration_valid = self.prepared.configuration_geometry.contains(
            candidate_configuration
        )
        successful = dynamics.valid & chart_valid & candidate_finite & configuration_valid
        accepted = _tree_select(successful, candidate, source)
        status = jnp.where(
            successful,
            0,
            jnp.where(~chart_valid, 2, jnp.where(~dynamics.valid, 3, 4)),
        ).astype(jnp.int32)
        evidence = FloatingReducedRodPlantEvidence(
            dynamics,
            requested_increment,
            chart_valid,
            candidate_finite,
            successful,
            self.prepared.dynamics_id,
        )
        return PlantProposal(
            candidate,
            accepted,
            jnp.asarray(True),
            successful,
            status,
            dynamics.solve_evidence.status,
            evidence,
        )


def prepare_floating_reduced_rod(
    rod: PreparedRod,
    plan: FloatingReducedRodPlan,
    dynamics_plan: ReducedRodDynamicsPlan | None = None,
    /,
    **kwargs,
) -> PreparedFloatingReducedRod:
    return PreparedFloatingReducedRod(rod, plan, dynamics_plan, **kwargs)


def floating_reduced_rod_mass(
    prepared: PreparedFloatingReducedRod, state: FloatingReducedRodState, /
) -> FloatingReducedRodMassResult:
    return prepared.mass(state)


def floating_reduced_rod_inverse_mass(
    prepared: PreparedFloatingReducedRod,
    state: FloatingReducedRodState,
    effort: ArrayLike,
    /,
) -> FloatingReducedRodInverseMassResult:
    return prepared.inverse_mass(state, effort)


def floating_reduced_rod_bias(
    prepared: PreparedFloatingReducedRod, state: FloatingReducedRodState, /
) -> FloatingReducedRodBiasResult:
    return prepared.bias(state)


def floating_reduced_rod_gravity(
    prepared: PreparedFloatingReducedRod, state: FloatingReducedRodState, /
) -> FloatingReducedRodGravityResult:
    return prepared.gravity_effort(state)


def evaluate_floating_reduced_rod(
    prepared: PreparedFloatingReducedRod, state: FloatingReducedRodState, /, **kwargs
) -> FloatingReducedRodDynamicsEvaluation:
    return prepared.evaluate(state, **kwargs)


def floating_reduced_rod_forward_dynamics(
    prepared: PreparedFloatingReducedRod, state: FloatingReducedRodState, /, **kwargs
) -> FloatingReducedRodForwardDynamicsResult:
    return prepared.forward_dynamics(state, **kwargs)


def floating_reduced_rod_inverse_dynamics(
    prepared: PreparedFloatingReducedRod,
    state: FloatingReducedRodState,
    acceleration: ArrayLike,
    /,
    **kwargs,
) -> FloatingReducedRodInverseDynamicsResult:
    return prepared.inverse_dynamics(state, acceleration, **kwargs)


__all__ = [
    "evaluate_floating_reduced_rod",
    "floating_reduced_rod_bias",
    "floating_reduced_rod_gravity",
    "floating_reduced_rod_forward_dynamics",
    "floating_reduced_rod_inverse_dynamics",
    "floating_reduced_rod_inverse_mass",
    "floating_reduced_rod_mass",
    "FloatingReducedRodBiasResult",
    "FloatingReducedRodDirectLoad",
    "FloatingReducedRodDynamicsEvaluation",
    "FloatingReducedRodEnergyResult",
    "FloatingReducedRodForceResult",
    "FloatingReducedRodGravityResult",
    "FloatingReducedRodForwardDynamicsResult",
    "FloatingReducedRodInverseDynamicsResult",
    "FloatingReducedRodInverseMassResult",
    "FloatingReducedRodMassEvidence",
    "FloatingReducedRodMassResult",
    "FloatingReducedRodMomentum",
    "FloatingReducedRodPlan",
    "FloatingReducedRodPlant",
    "FloatingReducedRodPlantControl",
    "FloatingReducedRodPlantEvidence",
    "FloatingReducedRodPlantParameterValues",
    "FloatingReducedRodPlantState",
    "FloatingReducedRodResetEvidence",
    "FloatingReducedRodSolveEvidence",
    "FloatingReducedRodState",
    "FloatingRodTwistConvention",
    "PreparedFloatingReducedRod",
    "prepare_floating_reduced_rod",
]
