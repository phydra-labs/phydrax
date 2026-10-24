#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Configuration-bound articulated inertia and its true-dual mass actions.

The articulated-body algorithm (ABA) is split into one configuration/inertia
preparation and reusable right-hand-side substitutions. Preparation at ``q``
stores parent-to-child twist transforms ``X_e``, motion subspaces ``S_e``,
articulated spatial inertias ``I^A``, projected columns ``U_e = I^A S_e``,
scalar pivots ``D_e = S_eᵀ U_e`` and reciprocal pivots as dynamic leaves. The
reverse sweep visits edges in reverse-topological order and substitution
visits them in topological order, so the factor is the tree-ordered
``LDLᵀ``-type factorization of ``M(q)`` and never forms a dense matrix.

The inverse mass maps the generalized effort dual space into the generalized
velocity space. The mass action ``M(q)x = Σ_b J_bᵀ G_b J_b x`` is evaluated by
an independent composite recursion that uses the same edge geometry but not
the articulated pivots; it certifies inverse actions through their relative
energy-norm residual without materializing ``M``.

The factor recurrence, the reverse/forward substitutions and the reverse
force reduction of the mass action are owned by the coarse sweep primitives
of ``_articulated_sweeps``; the forward twist propagation of the mass action
stays an ordinary JAX loop.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DualSpace,
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from ._articulated_sweeps import (
    factor_articulated_inertia,
    reduce_articulated_forces,
    solve_articulated_factor,
)
from ._reduced_articulation import ArticulationKinematics, PreparedReducedArticulation
from ._rigid_joints import RigidJointKind


class ArticulatedMassStatus(IntEnum):
    """Preparation and action status of one configuration-bound articulated mass."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    CAPABILITY_REJECTED = 2
    PIVOT_NOT_POSITIVE = 3
    NONFINITE_FACTOR = 4
    NONFINITE_OUTPUT = 5
    RESIDUAL_TOO_LARGE = 6
    STALE_BINDING = 7
    PREPARATION_FAILED = 8


def _require_articulation(
    articulation: PreparedReducedArticulation, /
) -> PreparedReducedArticulation:
    """Require the fixed-base three-dimensional scalar-joint tree profile."""

    if not isinstance(articulation, PreparedReducedArticulation):
        raise TypeError("articulation must be PreparedReducedArticulation.")
    if articulation.graph.bodies.ambient_dimension != 3:
        raise ValueError("Reduced rigid dynamics supports three-dimensional bodies only.")
    if articulation.nq != articulation.nv:
        raise ValueError("Reduced rigid dynamics requires fixed-base nq == nv state.")
    return articulation


def _capability_supported(articulation: PreparedReducedArticulation, /) -> Array:
    """Return whether the tree has a fixed root and only fixed/hinge/prismatic joints."""

    kinds = articulation.joint_kinds
    supported_kind = (
        (kinds == int(RigidJointKind.FIXED))
        | (kinds == int(RigidJointKind.HINGE))
        | (kinds == int(RigidJointKind.PRISMATIC))
    )
    root_fixed = articulation.graph.bodies.fixed_mask[articulation.root_index]
    return root_fixed & jnp.all(supported_kind)


def _skew(value: Array, /) -> Array:
    x, y, z = value[..., 0], value[..., 1], value[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        axis=-1,
    ).reshape(value.shape[:-1] + (3, 3))


def _world_body_inertia(
    articulation: PreparedReducedArticulation,
    rotation: Array,
    /,
) -> tuple[Array, Array]:
    """Return body masses and world-frame COM inertias ``R I_com Rᵀ``."""

    properties = articulation.graph.bodies.mass_properties
    inertia = rotation @ properties.inertia_com @ jnp.swapaxes(rotation, -1, -2)
    mass = properties.masses.astype(rotation.dtype)
    return mass, inertia


def _edge_motion_geometry(
    articulation: PreparedReducedArticulation,
    position: Array,
    rotation: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Return ``X_e``, ``S_e`` and the moving-edge mask at given body poses.

    Twists are ``(COM linear velocity, angular velocity)`` in world axes.
    ``X_e`` maps the parent twist to the child COM, and ``S_e`` is the twist
    of the child per unit joint rate. Fixed edges have a zero subspace.
    """

    parent = articulation.parent_indices
    child = articulation.child_indices
    parent_rotation = rotation[parent]
    axes = contract("eij,ej->ei", parent_rotation, articulation.parent_axes)
    anchors = position[parent] + contract(
        "eij,ej->ei", parent_rotation, articulation.parent_anchors
    )
    displacement = position[child] - position[parent]
    edge_count = articulation.edge_count
    identity = jnp.broadcast_to(jnp.eye(6, dtype=position.dtype), (edge_count, 6, 6))
    motion_transform = identity.at[:, :3, 3:].set(-_skew(displacement))
    hinge = articulation.joint_kinds == int(RigidJointKind.HINGE)
    prismatic = articulation.joint_kinds == int(RigidJointKind.PRISMATIC)
    linear = jnp.where(
        hinge[:, None],
        jnp.cross(axes, position[child] - anchors),
        jnp.where(prismatic[:, None], axes, 0.0),
    )
    angular = jnp.where(hinge[:, None], axes, 0.0)
    motion_subspace = jnp.concatenate((linear, angular), axis=-1)
    return motion_transform, motion_subspace, hinge | prismatic


@final
class ArticulatedMassPolicy(StrictModule):
    """Immutable pivot, certification and right-hand-side resource policy.

    ``relative_pivot_tolerance`` requires every moving pivot to satisfy
    ``D_e > tol * |S_e|ᵀ|I^A||S_e|``; ``None`` selects ``16 eps``.
    ``residual_tolerance`` bounds the relative energy-norm inverse residual
    ``sqrt(rᵀM⁻¹r / fᵀM⁻¹f)`` with ``r = M x - f``; ``None`` selects
    ``sqrt(eps)``. ``maximum_rhs`` bounds the columns of one multiple
    right-hand-side action.
    """

    relative_pivot_tolerance: float | None = eqx.field(static=True)
    residual_tolerance: float | None = eqx.field(static=True)
    maximum_rhs: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_pivot_tolerance: float | None = None,
        residual_tolerance: float | None = None,
        maximum_rhs: int = 64,
    ) -> None:
        pivot = (
            None if relative_pivot_tolerance is None else float(relative_pivot_tolerance)
        )
        residual = None if residual_tolerance is None else float(residual_tolerance)
        if pivot is not None and (not math.isfinite(pivot) or pivot < 0.0):
            raise ValueError("relative_pivot_tolerance must be finite and non-negative.")
        if residual is not None and (not math.isfinite(residual) or residual < 0.0):
            raise ValueError("residual_tolerance must be finite and non-negative.")
        if isinstance(maximum_rhs, bool) or not isinstance(maximum_rhs, int):
            raise TypeError("maximum_rhs must be an integer.")
        if maximum_rhs < 1:
            raise ValueError("maximum_rhs must be positive.")
        self.relative_pivot_tolerance = pivot
        self.residual_tolerance = residual
        self.maximum_rhs = maximum_rhs

    def resolved_pivot_tolerance(self, dtype: np.dtype, /) -> float:
        """Return the relative pivot tolerance for one floating dtype."""

        if self.relative_pivot_tolerance is not None:
            return self.relative_pivot_tolerance
        return 16.0 * float(jnp.finfo(dtype).eps)

    def resolved_residual_tolerance(self, dtype: np.dtype, /) -> float:
        """Return the relative energy-norm residual tolerance for one dtype."""

        if self.residual_tolerance is not None:
            return self.residual_tolerance
        return math.sqrt(float(jnp.finfo(dtype).eps))


@final
class ArticulatedMassBinding(StrictModule, NonTrainableState):
    """Exact numerical provenance of one articulated-mass preparation.

    The static identities name the prepared topology and mass-property schema;
    they do not prove equal numbers. Every numerical leaf that enters the
    factor is retained so a consumer can reject stale factors after a state,
    frame or material change without host fingerprints.
    """

    configuration: Array
    masses: Array
    inertia_com: Array
    reference_position: Array
    reference_orientation: Array
    parent_reference_translation: Array
    parent_reference_orientation: Array
    parent_axes: Array
    parent_anchors: Array
    articulation_id: str = eqx.field(static=True)
    mass_properties_id: str = eqx.field(static=True)

    def _leaves(self, /) -> tuple[Array, ...]:
        return (
            self.configuration,
            self.masses,
            self.inertia_com,
            self.reference_position,
            self.reference_orientation,
            self.parent_reference_translation,
            self.parent_reference_orientation,
            self.parent_axes,
            self.parent_anchors,
        )

    def matches(self, other: ArticulatedMassBinding, /) -> Array:
        """Return exact equality of every bound numerical leaf.

        Foreign topology or mass-property identities raise. A binding that
        contains any NaN or infinite configuration, mass, inertia or frame
        value never matches, even against an identical copy of itself.
        """

        if not isinstance(other, ArticulatedMassBinding):
            raise TypeError("other must be ArticulatedMassBinding.")
        if (
            other.articulation_id != self.articulation_id
            or other.mass_properties_id != self.mass_properties_id
        ):
            raise ValueError("Articulated-mass bindings use different identities.")
        result = jnp.asarray(True)
        for left, right in zip(self._leaves(), other._leaves(), strict=True):
            if left.shape != right.shape:
                raise ValueError("Articulated-mass bindings have different shapes.")
            result = (
                result
                & jnp.all(left == right)
                & jnp.all(jnp.isfinite(left))
                & jnp.all(jnp.isfinite(right))
            )
        return result


def _bind(
    articulation: PreparedReducedArticulation,
    configuration: ArrayLike,
    /,
) -> ArticulatedMassBinding:
    """Capture the exact numerical state that defines ``M(q)``."""

    articulation = _require_articulation(articulation)
    dtype = articulation.reference_position.dtype
    q = jnp.asarray(configuration, dtype=dtype)
    if q.shape != (articulation.nq,):
        raise ValueError("configuration must have articulation configuration shape.")
    properties = articulation.graph.bodies.mass_properties
    return ArticulatedMassBinding(
        q,
        properties.masses,
        properties.inertia_com,
        articulation.reference_position,
        articulation.reference_orientation,
        articulation.parent_reference_translation,
        articulation.parent_reference_orientation,
        articulation.parent_axes,
        articulation.parent_anchors,
        articulation.prepared_id,
        properties.properties_id,
    )


@final
class ArticulatedInertiaFactor(StrictModule, NonTrainableState):
    """Dynamic ABA factors of ``M(q)`` in the articulation's topological edge order.

    Fixed edges keep a zero subspace, projected column and reciprocal pivot, so
    they transmit the complete child articulated inertia. ``articulated_inertia``
    holds the final subtree inertia of each body at its COM in world axes.
    """

    motion_transforms: Array
    motion_subspaces: Array
    moving: Array
    body_masses: Array
    world_inertia: Array
    articulated_inertia: Array
    projected_columns: Array
    pivots: Array
    pivot_scales: Array
    reciprocal_pivots: Array
    parent_indices: Array
    child_indices: Array
    dof_edge_indices: Array

    @property
    def edge_count(self) -> int:
        return self.pivots.shape[0]

    @property
    def body_capacity(self) -> int:
        return self.articulated_inertia.shape[0]


def _spatial_inertia(mass: Array, world_inertia: Array, /) -> Array:
    capacity = mass.shape[0]
    dtype = world_inertia.dtype
    spatial = jnp.zeros((capacity, 6, 6), dtype=dtype)
    spatial = spatial.at[:, :3, :3].set(
        mass[:, None, None] * jnp.eye(3, dtype=dtype)[None, :, :]
    )
    return spatial.at[:, 3:, 3:].set(world_inertia)


def _prepare_factor(
    articulation: PreparedReducedArticulation,
    position: Array,
    rotation: Array,
    /,
) -> ArticulatedInertiaFactor:
    motion_transform, motion_subspace, moving = _edge_motion_geometry(
        articulation, position, rotation
    )
    mass, world_inertia = _world_body_inertia(articulation, rotation)
    parent = articulation.parent_indices
    child = articulation.child_indices
    articulated, projected, pivots, scales, reciprocal = factor_articulated_inertia(
        _spatial_inertia(mass, world_inertia),
        motion_transform,
        motion_subspace,
        moving,
        parent,
        child,
    )
    return ArticulatedInertiaFactor(
        motion_transform,
        motion_subspace,
        moving,
        mass,
        world_inertia,
        articulated,
        projected,
        pivots,
        scales,
        reciprocal,
        parent,
        child,
        articulation.dof_joint_indices,
    )


def _articulated_solve(
    factor: ArticulatedInertiaFactor,
    generalized_effort: Array,
    /,
    *,
    body_bias: Array | None = None,
    edge_convective: Array | None = None,
) -> Array:
    """Substitute ``(nv, K)`` effort columns through one prepared ABA factor.

    ``body_bias`` ``(C, 6, K)`` and ``edge_convective`` ``(E, 6, K)`` are the
    optional velocity/force terms of forward dynamics; without them this is
    the pure inverse-mass action ``M(q)⁻¹ f``.
    """

    edge_count = factor.edge_count
    columns = generalized_effort.shape[-1]
    dtype = factor.articulated_inertia.dtype
    edge_effort = (
        jnp.zeros((edge_count, columns), dtype=dtype)
        .at[factor.dof_edge_indices]
        .set(generalized_effort)
    )
    edge_acceleration = solve_articulated_factor(
        factor.articulated_inertia,
        factor.motion_transforms,
        factor.motion_subspaces,
        factor.projected_columns,
        factor.reciprocal_pivots,
        factor.parent_indices,
        factor.child_indices,
        edge_effort,
        body_bias,
        edge_convective,
    )
    return edge_acceleration[factor.dof_edge_indices]


def _mass_columns(factor: ArticulatedInertiaFactor, velocity: Array, /) -> Array:
    """Evaluate ``M(q)x = Jᵀ G J x`` for ``(nv, K)`` columns without pivots."""

    edge_count = factor.edge_count
    columns = velocity.shape[-1]
    dtype = factor.articulated_inertia.dtype
    parent = factor.parent_indices
    child = factor.child_indices
    transform = factor.motion_transforms
    subspace = factor.motion_subspaces
    edge_rate = (
        jnp.zeros((edge_count, columns), dtype=dtype)
        .at[factor.dof_edge_indices]
        .set(velocity)
    )

    def forward_edge(edge: Array, twist: Array) -> Array:
        return twist.at[child[edge]].set(
            transform[edge] @ twist[parent[edge]]
            + subspace[edge][:, None] * edge_rate[edge][None, :]
        )

    twist = jax.lax.fori_loop(
        0,
        edge_count,
        forward_edge,
        jnp.zeros((factor.body_capacity, 6, columns), dtype=dtype),
    )
    momentum = jnp.concatenate(
        (
            factor.body_masses[:, None, None] * twist[:, :3],
            contract("bij,bjk->bik", factor.world_inertia, twist[:, 3:]),
        ),
        axis=1,
    )
    edge_effort = reduce_articulated_forces(transform, subspace, parent, child, momentum)
    # Efforts are stored once into the articulated-dtype effort carry.
    return edge_effort[factor.dof_edge_indices].astype(dtype)


@final
class ArticulatedInverseMassOperator(AbstractLinearOperator):
    """``M(q)⁻¹``: generalized effort dual space to generalized velocity space.

    The map is symmetric by construction, so its algebraic transpose is the
    same substitution. Validity is the preparing ``PreparedArticulatedMass``
    evidence; this operator never materializes a dense inverse.
    """

    factor: ArticulatedInertiaFactor

    def __init__(
        self,
        factor: ArticulatedInertiaFactor,
        velocity_space: ArraySpace,
        effort_space: DualSpace,
        /,
        *,
        operator_id: str,
    ) -> None:
        if not isinstance(factor, ArticulatedInertiaFactor):
            raise TypeError("factor must be ArticulatedInertiaFactor.")
        if not isinstance(velocity_space, ArraySpace) or not isinstance(
            effort_space, DualSpace
        ):
            raise TypeError(
                "Inverse mass requires ArraySpace velocity and DualSpace effort."
            )
        if not effort_space.primal.compatible(velocity_space):
            raise ValueError("effort_space must be the dual of velocity_space.")
        self.factor = factor
        self.source = effort_space
        self.target = velocity_space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False
        )
        self.batch_shape = ()
        self.operator_id = str(operator_id)

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        return self.target.validate(_articulated_solve(self.factor, value[:, None])[:, 0])

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(_articulated_solve(self.factor, value[:, None])[:, 0])

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        covector = jnp.conj(self.target.riesz(vector))
        return self.source.inverse_riesz(jnp.conj(self.transpose_mv(covector)))

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "Articulated inverse mass does not materialize a dense matrix."
        )


@final
class ArticulatedMassOperator(AbstractLinearOperator):
    """``M(q)``: generalized velocity space to generalized effort dual space.

    Evaluated as the composite ``Jᵀ G J`` recursion over the prepared edge
    geometry, independently of the articulated pivots.
    """

    factor: ArticulatedInertiaFactor

    def __init__(
        self,
        factor: ArticulatedInertiaFactor,
        velocity_space: ArraySpace,
        effort_space: DualSpace,
        /,
        *,
        operator_id: str,
    ) -> None:
        if not isinstance(factor, ArticulatedInertiaFactor):
            raise TypeError("factor must be ArticulatedInertiaFactor.")
        if not isinstance(velocity_space, ArraySpace) or not isinstance(
            effort_space, DualSpace
        ):
            raise TypeError("Mass requires ArraySpace velocity and DualSpace effort.")
        if not effort_space.primal.compatible(velocity_space):
            raise ValueError("effort_space must be the dual of velocity_space.")
        self.factor = factor
        self.source = velocity_space
        self.target = effort_space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False
        )
        self.batch_shape = ()
        self.operator_id = str(operator_id)

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        return self.target.validate(_mass_columns(self.factor, value[:, None])[:, 0])

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(_mass_columns(self.factor, value[:, None])[:, 0])

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        covector = jnp.conj(self.target.riesz(vector))
        return self.source.inverse_riesz(jnp.conj(self.transpose_mv(covector)))

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "Articulated mass does not materialize a dense matrix."
        )


@final
class ArticulatedMassEvidence(StrictModule):
    """Finite, pivot and capability evidence of one articulated-mass preparation.

    ``minimum_pivot`` and ``minimum_relative_pivot`` are taken over moving
    edges and are infinite when no edge moves. A positive pivot certificate
    means every moving pivot is finite and exceeds ``relative_pivot_tolerance``
    times its magnitude scale; it is not an eigenvalue bound.
    """

    pivots: Array
    pivot_scales: Array
    moving: Array
    minimum_pivot: Array
    minimum_relative_pivot: Array
    relative_pivot_tolerance: Array
    input_finite: Array
    factor_finite: Array
    pivots_positive: Array
    capability_supported: Array
    successful: Array
    status: Array


@final
class ArticulatedMassActionResult(StrictModule):
    """Certified inverse-mass action for one vector or explicit RHS columns.

    ``relative_residual`` is ``sqrt(rᵀM⁻¹r / fᵀM⁻¹f)`` per column with the
    independent residual ``r = M x - f``; it is scalar for a vector action.
    It is zero only for an exactly zero residual and infinite when the
    residual is not certifiable (nonpositive effort energy, or a nonzero
    residual with vanishing measured energy). Acceptance also requires a
    successful preparation. ``value`` is ``candidate`` on success and zero
    otherwise.
    """

    candidate: Array
    value: Array
    relative_residual: Array
    finite: Array
    current: Array
    successful: Array
    status: Array


@final
class ArticulatedMassProductResult(StrictModule):
    """Independent mass action ``M(q)x`` with binding and finiteness status."""

    candidate: Array
    value: Array
    finite: Array
    current: Array
    successful: Array
    status: Array


@final
class ArticulatedMassAdjointEvidence(StrictModule):
    """Measured symmetry ``⟨f₁, M⁻¹f₂⟩ = ⟨f₂, M⁻¹f₁⟩`` of two effort covectors.

    ``scale`` is the Cauchy-Schwarz bound ``sqrt(⟨f₁,M⁻¹f₁⟩⟨f₂,M⁻¹f₂⟩)``.
    """

    first_pairing: Array
    second_pairing: Array
    residual: Array
    scale: Array
    finite: Array
    valid: Array


def _action_status(
    *,
    current: Array,
    prepared: Array,
    input_finite: Array,
    output_finite: Array,
    residual_accepted: Array,
) -> Array:
    status = jnp.asarray(int(ArticulatedMassStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~residual_accepted, int(ArticulatedMassStatus.RESIDUAL_TOO_LARGE), status
    )
    status = jnp.where(
        ~output_finite, int(ArticulatedMassStatus.NONFINITE_OUTPUT), status
    )
    status = jnp.where(~input_finite, int(ArticulatedMassStatus.NONFINITE_INPUT), status)
    status = jnp.where(~prepared, int(ArticulatedMassStatus.PREPARATION_FAILED), status)
    return jnp.where(~current, int(ArticulatedMassStatus.STALE_BINDING), status).astype(
        jnp.int32
    )


def _positive_sqrt(value: Array, /) -> Array:
    positive = value > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, value, 1.0)), 0.0)


@final
class PreparedArticulatedMass(StrictModule, NonTrainableState):
    """Configuration-bound articulated inertia with true-dual mass actions.

    Construct with `prepare_articulated_mass`. ``mass_id`` is the static
    topology/mass-schema/policy identity; ``binding`` holds the exact dynamic
    numbers. Operators do not gate on ``successful``; consumers must combine
    it with their own result status.
    """

    binding: ArticulatedMassBinding
    factor: ArticulatedInertiaFactor
    evidence: ArticulatedMassEvidence
    policy: ArticulatedMassPolicy
    velocity_space: ArraySpace
    effort_space: DualSpace
    mass_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def inverse_mass_operator(self) -> ArticulatedInverseMassOperator:
        return ArticulatedInverseMassOperator(
            self.factor,
            self.velocity_space,
            self.effort_space,
            operator_id=f"{self.mass_id}:inverse-mass",
        )

    @property
    def mass_operator(self) -> ArticulatedMassOperator:
        return ArticulatedMassOperator(
            self.factor,
            self.velocity_space,
            self.effort_space,
            operator_id=f"{self.mass_id}:mass",
        )

    def is_current(
        self,
        articulation: PreparedReducedArticulation,
        configuration: ArrayLike,
        /,
    ) -> Array:
        """Return whether ``(articulation, configuration)`` is the bound state."""

        return self.binding.matches(_bind(articulation, configuration))

    def _columns(self, value: ArrayLike, rhs_axis: int | None, name: str, /) -> Array:
        array = jnp.asarray(value, dtype=self.velocity_space.dtype)
        size = self.velocity_space.shape[0]
        if rhs_axis is None:
            if array.shape != (size,):
                raise ValueError(f"{name} must have generalized shape {(size,)}.")
            return array[:, None]
        if isinstance(rhs_axis, bool) or not isinstance(rhs_axis, int):
            raise TypeError("rhs_axis must be an integer or None.")
        if array.ndim != 2 or rhs_axis not in (-2, -1, 0, 1):
            raise ValueError(
                f"{name} with rhs_axis must be rank 2 with axis in {{0, 1}}."
            )
        columns = array if rhs_axis % 2 == 1 else array.T
        if columns.shape[0] != size:
            raise ValueError(f"{name} must have generalized extent {size}.")
        if columns.shape[1] > self.policy.maximum_rhs:
            raise ValueError(
                f"{name} has {columns.shape[1]} right-hand sides; the declared "
                f"maximum_rhs resource bound is {self.policy.maximum_rhs}."
            )
        return columns

    @staticmethod
    def _restore(columns: Array, rhs_axis: int | None, /) -> Array:
        if rhs_axis is None:
            return columns[:, 0]
        return columns if rhs_axis % 2 == 1 else columns.T

    def apply_inverse(
        self,
        articulation: PreparedReducedArticulation,
        configuration: ArrayLike,
        effort: ArrayLike,
        /,
        *,
        rhs_axis: int | None = None,
    ) -> ArticulatedMassActionResult:
        """Apply ``M(q)⁻¹`` to effort covectors and certify the residual.

        ``rhs_axis`` selects the column axis of a rank-2 effort; the result
        keeps the same layout. The supplied state must equal the bound state.
        """

        current = self.is_current(articulation, configuration)
        columns = self._columns(effort, rhs_axis, "effort")
        input_finite = jnp.all(jnp.isfinite(columns))
        candidate = _articulated_solve(self.factor, columns)
        residual = _mass_columns(self.factor, candidate) - columns
        correction = _articulated_solve(self.factor, residual)
        residual_energy = jnp.abs(jnp.sum(residual * correction, axis=0))
        effort_energy = jnp.sum(columns * candidate, axis=0)
        exact = jnp.all(residual == 0.0, axis=0)
        positive = effort_energy > 0.0
        ratio = residual_energy / jnp.where(positive, effort_energy, 1.0)
        # A zero measured ratio is reported only for an exactly zero residual;
        # a nonzero residual with vanishing energy (e.g. an indefinite factor)
        # or a nonpositive effort energy is not certifiable.
        relative = jnp.where(
            exact,
            0.0,
            jnp.where(positive & (ratio > 0.0), _positive_sqrt(ratio), jnp.inf),
        )
        finite = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(correction))
        )
        tolerance = self.policy.resolved_residual_tolerance(self.velocity_space.dtype)
        residual_accepted = self.successful & jnp.all(relative <= tolerance)
        successful = current & self.successful & input_finite & finite & residual_accepted
        value = jnp.where(successful, candidate, jnp.zeros_like(candidate))
        return ArticulatedMassActionResult(
            self._restore(candidate, rhs_axis),
            self._restore(value, rhs_axis),
            relative[0] if rhs_axis is None else relative,
            finite,
            current,
            successful,
            _action_status(
                current=current,
                prepared=self.successful,
                input_finite=input_finite,
                output_finite=finite,
                residual_accepted=residual_accepted,
            ),
        )

    def apply_mass(
        self,
        articulation: PreparedReducedArticulation,
        configuration: ArrayLike,
        velocity: ArrayLike,
        /,
        *,
        rhs_axis: int | None = None,
    ) -> ArticulatedMassProductResult:
        """Apply the independent ``M(q)`` action to generalized velocities."""

        current = self.is_current(articulation, configuration)
        columns = self._columns(velocity, rhs_axis, "velocity")
        input_finite = jnp.all(jnp.isfinite(columns))
        candidate = _mass_columns(self.factor, columns)
        finite = jnp.all(jnp.isfinite(candidate))
        successful = current & self.successful & input_finite & finite
        value = jnp.where(successful, candidate, jnp.zeros_like(candidate))
        return ArticulatedMassProductResult(
            self._restore(candidate, rhs_axis),
            self._restore(value, rhs_axis),
            finite,
            current,
            successful,
            _action_status(
                current=current,
                prepared=self.successful,
                input_finite=input_finite,
                output_finite=finite,
                residual_accepted=jnp.asarray(True),
            ),
        )

    def adjoint_evidence(
        self, first_effort: ArrayLike, second_effort: ArrayLike, /
    ) -> ArticulatedMassAdjointEvidence:
        """Measure the virtual-work symmetry of the inverse mass on two covectors."""

        first = self.effort_space.validate(
            jnp.asarray(first_effort, dtype=self.velocity_space.dtype)
        )
        second = self.effort_space.validate(
            jnp.asarray(second_effort, dtype=self.velocity_space.dtype)
        )
        responses = _articulated_solve(self.factor, jnp.stack((first, second), axis=1))
        first_response = self.velocity_space.validate(responses[:, 0])
        second_response = self.velocity_space.validate(responses[:, 1])
        first_pairing = self.effort_space.pair(first, second_response)
        second_pairing = self.effort_space.pair(second, first_response)
        residual = first_pairing - second_pairing
        scale = _positive_sqrt(
            jnp.abs(self.effort_space.pair(first, first_response))
            * jnp.abs(self.effort_space.pair(second, second_response))
        )
        dtype = self.velocity_space.dtype
        tolerance = float(jnp.finfo(dtype).eps) * 64.0 * max(1, self.factor.edge_count)
        finite = jnp.all(
            jnp.isfinite(jnp.stack((first_pairing, second_pairing, residual, scale)))
        )
        return ArticulatedMassAdjointEvidence(
            first_pairing,
            second_pairing,
            residual,
            scale,
            finite,
            finite & (jnp.abs(residual) <= tolerance * scale),
        )


def prepare_articulated_mass(
    articulation: PreparedReducedArticulation,
    configuration: ArrayLike,
    /,
    *,
    policy: ArticulatedMassPolicy | None = None,
) -> PreparedArticulatedMass:
    """Prepare the ABA inertia factors of ``M(q)`` once at one configuration.

    Velocity, bias, force and right-hand-side work are excluded. Failure is
    reported by status; pivots are never jittered or sign-repaired.
    """

    articulation = _require_articulation(articulation)
    policy_ = ArticulatedMassPolicy() if policy is None else policy
    if not isinstance(policy_, ArticulatedMassPolicy):
        raise TypeError("policy must be ArticulatedMassPolicy or None.")
    binding = _bind(articulation, configuration)
    q = binding.configuration
    kinematics = articulation.forward_kinematics(q)
    return _prepare_mass_from_kinematics(articulation, policy_, binding, kinematics)


def _prepare_mass_from_kinematics(
    articulation: PreparedReducedArticulation,
    policy_: ArticulatedMassPolicy,
    binding: ArticulatedMassBinding,
    kinematics: ArticulationKinematics,
    /,
) -> PreparedArticulatedMass:
    """Certify one factor using the owner's already evaluated zero-rate geometry."""

    q = binding.configuration
    rotation = kinematics.body_transforms[:, :3, :3]
    factor = _prepare_factor(articulation, kinematics.bodies.position, rotation)
    dtype = np.dtype(q.dtype)
    pivot_tolerance = policy_.resolved_pivot_tolerance(dtype)
    moving = factor.moving
    pivots = factor.pivots
    scales = factor.pivot_scales
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    minimum_pivot = jnp.min(jnp.where(moving, pivots, infinity), initial=jnp.inf)
    has_scale = scales > 0.0
    relative = jnp.where(has_scale, pivots / jnp.where(has_scale, scales, 1.0), 0.0)
    minimum_relative = jnp.min(jnp.where(moving, relative, infinity), initial=jnp.inf)
    pivots_positive = jnp.all(
        ~moving
        | (
            jnp.isfinite(pivots)
            & jnp.isfinite(scales)
            & (pivots > 0.0)
            & (pivots > pivot_tolerance * scales)
        )
    )
    input_finite = (
        jnp.all(jnp.isfinite(q))
        & jnp.all(jnp.isfinite(binding.masses))
        & jnp.all(jnp.isfinite(binding.inertia_com))
    )
    factor_finite = (
        kinematics.finite
        & jnp.all(jnp.isfinite(factor.motion_transforms))
        & jnp.all(jnp.isfinite(factor.motion_subspaces))
        & jnp.all(jnp.isfinite(factor.world_inertia))
        & jnp.all(jnp.isfinite(factor.articulated_inertia))
        & jnp.all(jnp.isfinite(factor.projected_columns))
        & jnp.all(jnp.isfinite(factor.reciprocal_pivots))
    )
    capability = _capability_supported(articulation)
    successful = input_finite & capability & pivots_positive & factor_finite
    status = jnp.asarray(int(ArticulatedMassStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~factor_finite, int(ArticulatedMassStatus.NONFINITE_FACTOR), status
    )
    status = jnp.where(
        ~pivots_positive, int(ArticulatedMassStatus.PIVOT_NOT_POSITIVE), status
    )
    status = jnp.where(
        ~capability, int(ArticulatedMassStatus.CAPABILITY_REJECTED), status
    )
    status = jnp.where(
        ~input_finite, int(ArticulatedMassStatus.NONFINITE_INPUT), status
    ).astype(jnp.int32)
    evidence = ArticulatedMassEvidence(
        pivots,
        scales,
        moving,
        minimum_pivot,
        minimum_relative,
        jnp.asarray(pivot_tolerance, dtype=dtype),
        input_finite,
        factor_finite,
        pivots_positive,
        capability,
        successful,
        status,
    )
    mass_id = canonical_fingerprint(
        {
            "kind": "prepared-articulated-mass",
            "articulation": articulation.prepared_id,
            "mass_properties": binding.mass_properties_id,
            "dtype": dtype.str,
            "policy": {
                "relative_pivot_tolerance": pivot_tolerance,
                "residual_tolerance": policy_.resolved_residual_tolerance(dtype),
                "maximum_rhs": policy_.maximum_rhs,
            },
        }
    )
    return PreparedArticulatedMass(
        binding,
        factor,
        evidence,
        policy_,
        articulation.generalized_velocity_space,
        articulation.generalized_effort_space,
        mass_id,
    )


__all__ = [
    "ArticulatedInertiaFactor",
    "ArticulatedInverseMassOperator",
    "ArticulatedMassActionResult",
    "ArticulatedMassAdjointEvidence",
    "ArticulatedMassBinding",
    "ArticulatedMassEvidence",
    "ArticulatedMassOperator",
    "ArticulatedMassPolicy",
    "ArticulatedMassProductResult",
    "ArticulatedMassStatus",
    "PreparedArticulatedMass",
    "prepare_articulated_mass",
]
