#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ...equations._finite_element_variational import CompiledFiniteElementProblem
    from ..particle._rigid_body import (
        PreparedRigidBodySet,
        RigidBodyKinematics,
        RigidBodyLoad,
    )
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    FunctionLinearOperator,
    saddle_point_operator,
    ScaledLinearOperator,
)
from ._generic import FiniteElementDiscretization, FiniteElementRuntimeData


class InterpolationTransposeEvidence(StrictModule):
    primal_pairing: Array
    transpose_pairing: Array
    residual: Array
    scale: Array
    finite: Array
    valid: Array


class PreparedFiniteElementPointInterpolation(StrictModule, NonTrainableState):
    """Fixed FE point routes with an exact algebraic transpose scatter."""

    discretization: FiniteElementDiscretization
    field_name: str = eqx.field(static=True)
    dof_routes: Array
    weights: Array
    reference_positions: Array
    dof_reference_positions: Array
    tolerance: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        field_name: str,
        dof_routes: ArrayLike,
        weights: ArrayLike,
        reference_positions: ArrayLike,
        dof_reference_positions: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
        prepared_id: str | None = None,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        name = str(field_name)
        field_index = discretization._field_index(name)
        field_space = discretization.field_spaces[field_index].vector_space
        if not isinstance(field_space, ArraySpace):
            raise TypeError("Point interpolation requires an array-valued FE field.")
        dimension = discretization.mesh.ambient_dimension
        if field_space.shape != (
            discretization.dof_maps[field_index].global_dof_count,
            dimension,
        ):
            raise ValueError(
                "Rigid coupling requires a nodal vector field matching mesh dimension."
            )
        routes = np.asarray(dof_routes, dtype=np.int32)
        weights_ = np.asarray(weights)
        positions = np.asarray(reference_positions)
        dof_positions = np.asarray(dof_reference_positions)
        limit = float(tolerance)
        if routes.ndim != 2 or routes.shape[0] == 0:
            raise ValueError("Interpolation routes must be a nonempty rank-2 array.")
        if weights_.shape != routes.shape:
            raise ValueError("Interpolation weights must match fixed route shape.")
        if positions.shape != (routes.shape[0], dimension):
            raise ValueError("Reference points must match interpolation count/dimension.")
        if dof_positions.shape != (field_space.shape[0], dimension):
            raise ValueError("DOF reference positions must match the FE vector field.")
        if (
            np.any(routes < 0)
            or np.any(routes >= field_space.shape[0])
            or np.any(~np.isfinite(weights_))
            or np.any(~np.isfinite(positions))
            or np.any(~np.isfinite(dof_positions))
        ):
            raise ValueError(
                "Interpolation routes and geometric data must be finite/valid."
            )
        partition_defect = np.max(np.abs(np.sum(weights_, axis=1) - 1.0))
        if not isfinite(limit) or limit < 0.0 or partition_defect > limit:
            raise ValueError(
                "FE interpolation must reproduce constants within tolerance."
            )
        generated = canonical_fingerprint(
            {
                "kind": "prepared-finite-element-point-interpolation",
                "discretization": discretization.prepared_id,
                "field": name,
                "routes": array_tree_fingerprint(routes),
                "weights": array_tree_fingerprint(weights_),
                "reference_positions": array_tree_fingerprint(positions),
                "tolerance": limit.hex(),
            }
        )
        identifier = generated if prepared_id is None else str(prepared_id)
        if not identifier:
            raise ValueError("prepared_id must be non-empty or None.")
        dtype = field_space.dtype
        self.discretization = discretization
        self.field_name = name
        self.dof_routes = jnp.asarray(routes)
        self.weights = jnp.asarray(weights_, dtype=dtype)
        self.reference_positions = jnp.asarray(positions, dtype=dtype)
        self.dof_reference_positions = jnp.asarray(dof_positions, dtype=dtype)
        self.tolerance = limit
        self.prepared_id = identifier

    @property
    def attachment_count(self) -> int:
        return int(self.dof_routes.shape[0])

    @property
    def ambient_dimension(self) -> int:
        return int(self.reference_positions.shape[1])

    @property
    def field_space(self) -> ArraySpace:
        index = self.discretization._field_index(self.field_name)
        space = self.discretization.field_spaces[index].vector_space
        if not isinstance(space, ArraySpace):
            raise TypeError("Prepared point interpolation lost its array field space.")
        return space

    def interpolate(self, coefficients: ArrayLike, /) -> Array:
        values = self.field_space.validate(coefficients)
        local = values[self.dof_routes]
        return contract("ai,aid->ad", self.weights, local)

    def transpose_scatter(self, point_dual: ArrayLike, /) -> Array:
        dual = jnp.asarray(point_dual, dtype=self.weights.dtype)
        if dual.shape != (self.attachment_count, self.ambient_dimension):
            raise ValueError("Point dual must match attachment count and dimension.")
        payload = self.weights[..., None] * dual[:, None, :]
        return (
            jnp.zeros(self.field_space.shape, dtype=payload.dtype)
            .at[self.dof_routes]
            .add(payload)
        )

    def duality_evidence(
        self,
        coefficients: ArrayLike,
        point_dual: ArrayLike,
        /,
    ) -> InterpolationTransposeEvidence:
        values = self.field_space.validate(coefficients)
        dual = jnp.asarray(point_dual, dtype=values.dtype)
        interpolated = self.interpolate(values)
        scattered = self.transpose_scatter(dual)
        primal = jnp.sum(interpolated * dual)
        transpose = jnp.sum(values * scattered)
        residual = primal - transpose
        scale = jnp.maximum(
            jnp.asarray(1.0, dtype=residual.real.dtype),
            jnp.maximum(jnp.abs(primal), jnp.abs(transpose)),
        )
        finite = jnp.all(jnp.isfinite(jnp.stack((primal, transpose, residual, scale))))
        valid = finite & (jnp.abs(residual) <= self.tolerance * scale)
        return InterpolationTransposeEvidence(
            primal,
            transpose,
            residual,
            scale,
            finite,
            valid,
        )

    def deformed_dof_positions(self, displacement: ArrayLike, /) -> Array:
        value = self.field_space.validate(displacement)
        return self.dof_reference_positions + value


def prepare_finite_element_point_interpolation(
    discretization: FiniteElementDiscretization,
    field_name: str,
    block_name: str,
    cell_indices: ArrayLike,
    reference_points: ArrayLike,
    /,
    *,
    runtime: FiniteElementRuntimeData | None = None,
    tolerance: float = 1.0e-10,
) -> PreparedFiniteElementPointInterpolation:
    """Prepare fixed block-local cell/point routes for a nodal H1 vector field."""
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    dof_map = discretization.dof_maps[field_index]
    block = str(block_name)
    if block not in dof_map.block_names:
        raise KeyError(f"Unknown FE block {block!r} for field {field_name!r}.")
    block_index = dof_map.block_names.index(block)
    element = discretization.elements[field_index][block_index]
    if element.conformity != "H1" or element.mapping != "identity" or element.value_shape:
        raise ValueError(
            "Point attachments require a scalar-basis identity-mapped H1 field."
        )
    cells = np.asarray(cell_indices, dtype=np.int32)
    points = np.asarray(reference_points)
    cell_count = discretization.mesh.blocks[block_index].cell_count
    if cells.ndim != 1 or cells.size == 0:
        raise ValueError("cell_indices must be a nonempty rank-1 array.")
    if points.shape != (cells.size, element.topological_dimension):
        raise ValueError("reference_points must provide one point per selected cell.")
    if np.any(cells < 0) or np.any(cells >= cell_count) or np.any(~np.isfinite(points)):
        raise ValueError("Interpolation cell routes/reference points are invalid.")
    basis, _ = element.tabulate(jnp.asarray(points))
    if basis.ndim != 2 or basis.shape != (cells.size, element.local_dof_count):
        raise ValueError(
            "Attachment interpolation requires scalar reference basis values."
        )
    routes = np.asarray(dof_map.cell_dofs[block_index])[cells]
    orientation = np.asarray(dof_map.orientations[block_index])[cells]
    weights = np.asarray(basis) * orientation
    realized = discretization.default_runtime if runtime is None else runtime
    if not isinstance(realized, FiniteElementRuntimeData):
        raise TypeError("runtime must be FiniteElementRuntimeData or None.")
    if (
        realized.topology_id != discretization.mesh.topology_id
        or realized.geometry_layout_id
        != discretization.default_runtime.geometry_layout_id
    ):
        raise ValueError("Interpolation runtime does not match the FE discretization.")
    geometry = discretization.evaluate_block_geometry(
        field_name,
        block_index,
        realized.coordinates,
        jnp.asarray(points),
        jnp.ones((cells.size,), dtype=jnp.asarray(points).dtype),
    )
    reference_positions = np.asarray(geometry.physical_points)[
        cells, np.arange(cells.size)
    ]
    dof_positions = np.asarray(
        dof_map.evaluate_coordinates(discretization.mesh, realized.coordinates)
    )
    return PreparedFiniteElementPointInterpolation(
        discretization,
        field_name,
        routes,
        weights,
        reference_positions,
        dof_positions,
        tolerance=tolerance,
    )


class AttachmentRankEvidence(StrictModule, NonTrainableState):
    row_count: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    numerical_rank: int = eqx.field(static=True)
    smallest_singular_value: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    full_row_rank: bool = eqx.field(static=True)


class RigidDeformableAttachmentEvaluation(StrictModule):
    deformable_points: Array
    rigid_points: Array
    body_levers: Array
    residual: Array
    maximum_residual: Array
    finite: Array


class RigidDeformableKKTLinearization(StrictModule, NonTrainableState):
    interpolation: PreparedFiniteElementPointInterpolation
    bodies: PreparedRigidBodySet
    body_indices: Array
    body_levers: Array
    operator_id: str = eqx.field(static=True)

    def constraint_action(
        self,
        deformable_increment: ArrayLike,
        body_translation_increment: ArrayLike,
        body_rotation_increment: ArrayLike,
        /,
    ) -> Array:
        translation = jnp.asarray(body_translation_increment)
        rotation = jnp.asarray(body_rotation_increment)
        expected = (self.bodies.capacity, 3)
        if translation.shape != expected or rotation.shape != expected:
            raise ValueError("Rigid KKT increments must have body-capacity 3-D shape.")
        deformable = self.interpolation.interpolate(deformable_increment)
        rotational = jnp.cross(self.body_levers, rotation[self.body_indices])
        return deformable - translation[self.body_indices] + rotational

    def transpose_action(
        self,
        multiplier: ArrayLike,
        /,
    ) -> tuple[Array, RigidBodyLoad]:
        from ..particle._rigid_body import RigidBodyLoad

        value = jnp.asarray(multiplier, dtype=self.body_levers.dtype)
        if value.shape != self.body_levers.shape:
            raise ValueError("Attachment multiplier shape is invalid.")
        deformable = self.interpolation.transpose_scatter(value)
        force = (
            jnp.zeros((self.bodies.capacity, 3), dtype=value.dtype)
            .at[self.body_indices]
            .add(-value)
        )
        torque = (
            jnp.zeros_like(force)
            .at[self.body_indices]
            .add(jnp.cross(self.body_levers, -value))
        )
        return deformable, RigidBodyLoad(force, torque)

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        dtype = self.interpolation.field_space.dtype
        rigid_space = ArraySpace((self.bodies.capacity, 3), dtype=dtype)
        source = BlockSpace(
            (self.interpolation.field_space, rigid_space, rigid_space),
            names=("deformable", "rigid_translation", "rigid_rotation"),
        )
        target = ArraySpace(self.body_levers.shape, dtype=dtype)

        def transpose(multiplier):
            deformable, rigid = self.transpose_action(multiplier)
            return deformable, rigid.force, rigid.torque

        return FunctionLinearOperator(
            lambda increment: self.constraint_action(*increment),
            source=source,
            target=target,
            transpose_action=transpose,
            operator_id=self.operator_id,
        )

    def duality_residual(
        self,
        deformable_increment: ArrayLike,
        body_translation_increment: ArrayLike,
        body_rotation_increment: ArrayLike,
        multiplier: ArrayLike,
        /,
    ) -> Array:
        constraint = self.constraint_action(
            deformable_increment,
            body_translation_increment,
            body_rotation_increment,
        )
        deformable_load, rigid_load = self.transpose_action(multiplier)
        left = jnp.sum(constraint * jnp.asarray(multiplier))
        right = (
            jnp.sum(jnp.asarray(deformable_increment) * deformable_load)
            + jnp.sum(jnp.asarray(body_translation_increment) * rigid_load.force)
            + jnp.sum(jnp.asarray(body_rotation_increment) * rigid_load.torque)
        )
        return left - right


class AttachmentActionReactionCertificate(StrictModule):
    deformable_resultant: Array
    rigid_resultant: Array
    force_balance: Array
    deformable_moment: Array
    rigid_moment: Array
    moment_balance: Array
    interpolation_duality_residual: Array
    maximum_force_defect: Array
    maximum_moment_defect: Array
    finite: Array
    valid: Array


class RigidDeformableKKTPayload(StrictModule):
    constraint_residual: Array
    multiplier: Array
    deformable_load: Array
    rigid_load: RigidBodyLoad
    linearization: RigidDeformableKKTLinearization
    operator: FunctionLinearOperator
    certificate: AttachmentActionReactionCertificate
    finite: Array
    valid: Array


class RigidDeformableAttachmentPlan(StrictModule, NonTrainableState):
    """Fixed FE-to-rigid translational constraints with exact KKT actions."""

    interpolation: PreparedFiniteElementPointInterpolation
    bodies: PreparedRigidBodySet
    body_indices: Array
    local_anchors: Array
    rank: AttachmentRankEvidence
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interpolation: PreparedFiniteElementPointInterpolation,
        bodies: PreparedRigidBodySet,
        body_ids: ArrayLike,
        local_anchors: ArrayLike,
        /,
        *,
        rank_tolerance: float = 1.0e-10,
        tolerance: float = 1.0e-9,
        plan_id: str | None = None,
    ):
        if not isinstance(interpolation, PreparedFiniteElementPointInterpolation):
            raise TypeError("interpolation must be prepared FE point interpolation.")
        from ..particle._rigid_body import PreparedRigidBodySet

        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be PreparedRigidBodySet.")
        if interpolation.ambient_dimension != 3 or bodies.ambient_dimension != 3:
            raise ValueError(
                "Rigid-deformable attachments currently require 3-D supports."
            )
        identifiers = np.asarray(body_ids)
        anchors = np.asarray(local_anchors)
        count = interpolation.attachment_count
        if (
            identifiers.shape != (count,)
            or not np.issubdtype(identifiers.dtype, np.integer)
            or anchors.shape != (count, 3)
            or np.any(~np.isfinite(anchors))
        ):
            raise ValueError("Attachment body IDs/local anchors have invalid layout.")
        particle_ids = np.asarray(bodies.particles.particle_ids)
        order = np.argsort(particle_ids, kind="stable")
        sorted_ids = particle_ids[order]
        ranks = np.searchsorted(sorted_ids, identifiers)
        safe = np.minimum(ranks, max(sorted_ids.size - 1, 0))
        present = (ranks < sorted_ids.size) & (sorted_ids[safe] == identifiers)
        if not np.all(present):
            raise ValueError("An attachment body ID is absent from rigid-body support.")
        indices = order[ranks].astype(np.int32)
        active = np.asarray(bodies.particles.active_mask, dtype=bool)
        if not np.all(active[indices]):
            raise ValueError("Attachments require active rigid bodies.")
        if np.any(np.asarray(bodies.fixed_mask, dtype=bool)[indices]):
            raise ValueError("KKT attachments require mobile rigid bodies.")
        rank_limit = float(rank_tolerance)
        physical_limit = float(tolerance)
        if (
            not isfinite(rank_limit)
            or rank_limit <= 0.0
            or not isfinite(physical_limit)
            or physical_limit < 0.0
        ):
            raise ValueError("Attachment rank/physical tolerances are invalid.")
        rows = 3 * count
        field_coordinates = interpolation.field_space.size
        rigid_coordinates = 6 * bodies.capacity
        matrix = np.zeros((rows, field_coordinates + rigid_coordinates))
        routes = np.asarray(interpolation.dof_routes)
        weights = np.asarray(interpolation.weights)
        for attachment in range(count):
            row = 3 * attachment
            for local, dof in enumerate(routes[attachment]):
                for component in range(3):
                    matrix[row + component, 3 * int(dof) + component] += weights[
                        attachment, local
                    ]
            body = int(indices[attachment])
            rigid_start = field_coordinates + 6 * body
            matrix[row : row + 3, rigid_start : rigid_start + 3] -= np.eye(3)
            x, y, z = anchors[attachment]
            cross_matrix = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
            matrix[row : row + 3, rigid_start + 3 : rigid_start + 6] += cross_matrix
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        largest = max(float(singular_values[0]), 1.0)
        numerical_rank = int(np.count_nonzero(singular_values > rank_limit * largest))
        smallest = float(singular_values[-1])
        evidence = AttachmentRankEvidence(
            rows,
            matrix.shape[1],
            numerical_rank,
            smallest,
            rank_limit,
            numerical_rank == rows,
        )
        if not evidence.full_row_rank:
            raise ValueError(
                "Rigid-deformable attachment rows are duplicate or rank deficient."
            )
        generated = canonical_fingerprint(
            {
                "kind": "rigid-deformable-attachment-plan",
                "interpolation": interpolation.prepared_id,
                "bodies": bodies.prepared_id,
                "body_indices": array_tree_fingerprint(indices),
                "local_anchors": array_tree_fingerprint(anchors),
                "rank_tolerance": rank_limit.hex(),
                "tolerance": physical_limit.hex(),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty or None.")
        self.interpolation = interpolation
        self.bodies = bodies
        self.body_indices = jnp.asarray(indices)
        self.local_anchors = jnp.asarray(anchors, dtype=interpolation.weights.dtype)
        self.rank = evidence
        self.tolerance = physical_limit
        self.plan_id = identifier

    def evaluate(
        self,
        displacement: ArrayLike,
        kinematics: RigidBodyKinematics,
        /,
    ) -> RigidDeformableAttachmentEvaluation:
        from ..particle._rigid_body import (
            quaternion_rotation_matrix,
            RigidBodyKinematics,
        )

        if not isinstance(kinematics, RigidBodyKinematics):
            raise TypeError("kinematics must be RigidBodyKinematics.")
        expected = (self.bodies.capacity, 3)
        if (
            kinematics.position.shape != expected
            or kinematics.velocity.shape != expected
            or kinematics.orientation.shape != (self.bodies.capacity, 4)
            or kinematics.angular_velocity.shape != expected
        ):
            raise ValueError("Rigid kinematics do not match the attachment body set.")
        rotation = quaternion_rotation_matrix(kinematics.orientation)
        levers = contract("aij,aj->ai", rotation[self.body_indices], self.local_anchors)
        rigid_points = kinematics.position[self.body_indices] + levers
        deformable_points = (
            self.interpolation.reference_positions
            + self.interpolation.interpolate(displacement)
        )
        residual = deformable_points - rigid_points
        maximum = jnp.max(jnp.abs(residual), initial=0.0)
        finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (deformable_points, rigid_points, levers, residual), axis=0
                )
            )
        )
        return RigidDeformableAttachmentEvaluation(
            deformable_points,
            rigid_points,
            levers,
            residual,
            maximum,
            finite,
        )

    def linearization(
        self,
        evaluation: RigidDeformableAttachmentEvaluation,
        /,
    ) -> RigidDeformableKKTLinearization:
        if not isinstance(evaluation, RigidDeformableAttachmentEvaluation):
            raise TypeError("evaluation must be RigidDeformableAttachmentEvaluation.")
        return RigidDeformableKKTLinearization(
            self.interpolation,
            self.bodies,
            self.body_indices,
            evaluation.body_levers,
            canonical_fingerprint(
                {
                    "kind": "rigid-deformable-kkt-linearization",
                    "plan": self.plan_id,
                }
            ),
        )

    def kkt_payload(
        self,
        displacement: ArrayLike,
        kinematics: RigidBodyKinematics,
        multiplier: ArrayLike,
        /,
    ) -> RigidDeformableKKTPayload:
        evaluation = self.evaluate(displacement, kinematics)
        linearization = self.linearization(evaluation)
        operator = linearization.as_linear_operator()
        value = jnp.asarray(multiplier, dtype=evaluation.residual.dtype)
        if value.shape != evaluation.residual.shape:
            raise ValueError(
                "Attachment multiplier must match constraint residual shape."
            )
        deformable_load, rigid_load = linearization.transpose_action(value)
        deformable_resultant = jnp.sum(deformable_load, axis=0)
        rigid_resultant = jnp.sum(rigid_load.force, axis=0)
        force_balance = deformable_resultant + rigid_resultant
        deformed_dofs = self.interpolation.deformed_dof_positions(displacement)
        deformable_moment = jnp.sum(jnp.cross(deformed_dofs, deformable_load), axis=0)
        rigid_moment = jnp.sum(
            jnp.cross(kinematics.position, rigid_load.force) + rigid_load.torque,
            axis=0,
        )
        moment_balance = deformable_moment + rigid_moment
        duality = self.interpolation.duality_evidence(displacement, value)
        maximum_force = jnp.max(jnp.abs(force_balance), initial=0.0)
        maximum_moment = jnp.max(jnp.abs(moment_balance), initial=0.0)
        finite = evaluation.finite & jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        deformable_load,
                        rigid_load.force,
                        rigid_load.torque,
                        force_balance[None, :],
                        moment_balance[None, :],
                    ),
                    axis=0,
                )
            )
        )
        certificate_valid = (
            finite
            & duality.valid
            & (maximum_force <= self.tolerance)
            & (maximum_moment <= self.tolerance)
        )
        valid = certificate_valid & (evaluation.maximum_residual <= self.tolerance)
        certificate = AttachmentActionReactionCertificate(
            deformable_resultant,
            rigid_resultant,
            force_balance,
            deformable_moment,
            rigid_moment,
            moment_balance,
            duality.residual,
            maximum_force,
            maximum_moment,
            finite,
            certificate_valid,
        )
        return RigidDeformableKKTPayload(
            evaluation.residual,
            value,
            deformable_load,
            rigid_load,
            linearization,
            operator,
            certificate,
            finite,
            valid,
        )


class PressureGaugeEvidence(StrictModule):
    residual: Array
    scale: Array
    finite: Array
    valid: Array
    mode: Literal["mean-zero", "pinned"] = eqx.field(static=True)


class PressureGaugePlan(StrictModule, NonTrainableState):
    """Explicit constant-pressure gauge with pure projection and evidence."""

    space: ArraySpace
    weights: Array
    constant_mode: Array
    pinned_index: int | None = eqx.field(static=True)
    mode: Literal["mean-zero", "pinned"] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: ArraySpace,
        /,
        *,
        mode: Literal["mean-zero", "pinned"] = "mean-zero",
        weights: ArrayLike | None = None,
        constant_mode: ArrayLike | None = None,
        pinned_index: int | None = None,
        tolerance: float = 1.0e-10,
    ):
        if not isinstance(space, ArraySpace) or len(space.shape) != 1:
            raise TypeError("Pressure gauge requires a rank-1 ArraySpace.")
        mode_ = str(mode)
        if mode_ not in ("mean-zero", "pinned"):
            raise ValueError("Pressure gauge mode must be 'mean-zero' or 'pinned'.")
        limit = float(tolerance)
        if not isfinite(limit) or limit < 0.0:
            raise ValueError("Pressure gauge tolerance must be finite and nonnegative.")
        if mode_ == "mean-zero":
            if pinned_index is not None:
                raise ValueError("mean-zero pressure gauges do not accept pinned_index.")
            weights_ = (
                np.ones(space.shape, dtype=space.dtype)
                if weights is None
                else np.asarray(weights, dtype=space.dtype)
            )
            if weights_.shape != space.shape or np.any(~np.isfinite(weights_)):
                raise ValueError(
                    "Mean-zero gauge weights must be finite pressure vectors."
                )
            pin = None
        else:
            if weights is not None:
                raise ValueError("Pinned pressure gauges do not accept weights.")
            pin = 0 if pinned_index is None else int(pinned_index)
            if pin < 0 or pin >= space.size:
                raise ValueError("pinned_index lies outside the pressure space.")
            weights_ = np.zeros(space.shape, dtype=space.dtype)
            weights_[pin] = 1.0
        constant = (
            np.ones(space.shape, dtype=space.dtype)
            if constant_mode is None
            else np.asarray(constant_mode, dtype=space.dtype)
        )
        denominator = (
            float(np.sum(weights_ * constant)) if constant.shape == space.shape else 0.0
        )
        if (
            constant.shape != space.shape
            or np.any(~np.isfinite(constant))
            or abs(denominator) <= limit
        ):
            raise ValueError(
                "Pressure constant mode must be finite and visible to the gauge."
            )
        self.space = space
        self.weights = jnp.asarray(weights_)
        self.constant_mode = jnp.asarray(constant)
        self.pinned_index = pin
        self.mode = mode_  # type: ignore[assignment]
        self.tolerance = limit
        self.gauge_id = canonical_fingerprint(
            {
                "kind": "pressure-gauge",
                "space": space.space_id,
                "mode": mode_,
                "weights": array_tree_fingerprint(weights_),
                "constant_mode": array_tree_fingerprint(constant),
                "pinned_index": pin,
                "tolerance": limit.hex(),
            }
        )

    def residual(self, pressure: ArrayLike, /) -> Array:
        value = self.space.validate(pressure)
        return jnp.sum(self.weights * value)

    def project(self, pressure: ArrayLike, /) -> Array:
        value = self.space.validate(pressure)
        shift = self.residual(value) / jnp.sum(self.weights * self.constant_mode)
        return value - shift * self.constant_mode

    def evidence(self, pressure: ArrayLike, /) -> PressureGaugeEvidence:
        value = self.space.validate(pressure)
        residual = self.residual(value)
        scale = jnp.maximum(
            jnp.asarray(1.0, dtype=value.real.dtype), jnp.max(jnp.abs(value))
        )
        finite = jnp.all(jnp.isfinite(value)) & jnp.isfinite(residual)
        valid = finite & (jnp.abs(residual) <= self.tolerance * scale)
        return PressureGaugeEvidence(residual, scale, finite, valid, self.mode)


class VolumetricConstraintRankEvidence(StrictModule, NonTrainableState):
    displacement_dimension: int = eqx.field(static=True)
    pressure_dimension: int = eqx.field(static=True)
    numerical_rank: int = eqx.field(static=True)
    nullity: int = eqx.field(static=True)
    expected_rank: int = eqx.field(static=True)
    smallest_retained_singular_value: float = eqx.field(static=True)
    largest_discarded_singular_value: float = eqx.field(static=True)
    gauge_nullspace_coupling: float = eqx.field(static=True)
    adjoint_sign: int = eqx.field(static=True)
    adjoint_defect: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    valid: bool = eqx.field(static=True)


class MixedVolumetricConstraintPayload(StrictModule):
    residual: tuple[Array, Array]
    displacement_residual: Array
    incompressibility_residual: Array
    gauged_pressure: Array
    gauge: PressureGaugeEvidence
    rank: VolumetricConstraintRankEvidence
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class MixedVolumetricConstraintPlan(StrictModule, NonTrainableState):
    """Prepared displacement-pressure saddle operator and pressure-gauge proof."""

    problem: CompiledFiniteElementProblem
    operator: BlockLinearOperator
    primal_operator: AbstractLinearOperator
    constraint_operator: AbstractLinearOperator
    stabilization: AbstractLinearOperator | None
    gauge: PressureGaugePlan
    rank: VolumetricConstraintRankEvidence
    displacement_index: int = eqx.field(static=True)
    pressure_index: int = eqx.field(static=True)
    constraint_sign: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def evaluate(
        self,
        state: tuple[ArrayLike, ArrayLike],
        args: object = None,
        /,
    ) -> MixedVolumetricConstraintPayload:
        values = self.problem.state_space.validate(state)
        raw = self.problem.residual(values, args)
        displacement = self.primal_operator.target.inverse_riesz(
            raw[self.displacement_index]
        )
        incompressibility = (
            self.constraint_sign
            * self.constraint_operator.target.inverse_riesz(raw[self.pressure_index])
        )
        pressure = values[self.pressure_index]
        gauged = self.gauge.project(pressure)
        gauge_evidence = self.gauge.evidence(gauged)
        finite = (
            jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(incompressibility))
            & jnp.all(jnp.isfinite(gauged))
        )
        valid = finite & gauge_evidence.valid & jnp.asarray(self.rank.valid)
        return MixedVolumetricConstraintPayload(
            (displacement, incompressibility),
            displacement,
            incompressibility,
            gauged,
            gauge_evidence,
            self.rank,
            finite,
            valid,
            self.plan_id,
        )


def _primalized_block(
    operator: AbstractLinearOperator,
    target,
    /,
) -> FunctionLinearOperator:
    return FunctionLinearOperator(
        lambda value: target.inverse_riesz(operator.mv(value)),
        source=operator.source,
        target=target,
        operator_id=canonical_fingerprint(
            {
                "kind": "primalized-finite-element-block",
                "operator": operator.operator_id,
                "target": target.space_id,
            }
        ),
    )


def _dense_operator(operator: AbstractLinearOperator, /) -> np.ndarray:
    columns = []
    for index in range(operator.source.size):
        coordinate = np.zeros(
            (operator.source.size,),
            dtype=operator.source.flatten(operator.source.zeros()).dtype,
        )
        coordinate[index] = 1.0
        vector = operator.source.unflatten(jnp.asarray(coordinate))
        columns.append(np.asarray(operator.target.flatten(operator.mv(vector))))
    if not columns:
        return np.empty((operator.target.size, 0))
    return np.stack(columns, axis=1)


def prepare_mixed_volumetric_constraint(
    problem: CompiledFiniteElementProblem,
    state: tuple[ArrayLike, ArrayLike],
    displacement_field: str,
    pressure_field: str,
    /,
    *,
    args: object = None,
    gauge_mode: Literal["mean-zero", "pinned"] = "mean-zero",
    pressure_weights: ArrayLike | None = None,
    pinned_pressure_dof: int | None = None,
    rank_tolerance: float = 1.0e-10,
    gauge_tolerance: float = 1.0e-10,
    plan_id: str | None = None,
) -> MixedVolumetricConstraintPlan:
    """Extract and certify a two-field incompressibility saddle structure."""
    from ...equations._finite_element_variational import CompiledFiniteElementProblem

    if not isinstance(problem, CompiledFiniteElementProblem):
        raise TypeError("problem must be CompiledFiniteElementProblem.")
    fields = problem.form.field_names
    if len(fields) != 2 or set(fields) != {str(displacement_field), str(pressure_field)}:
        raise ValueError(
            "Mixed volumetric constraints require exactly the named displacement/pressure fields."
        )
    displacement_index = fields.index(str(displacement_field))
    pressure_index = fields.index(str(pressure_field))
    values = problem.state_space.validate(state)
    if not isinstance(problem.state_space, BlockSpace):
        raise ValueError("Mixed volumetric constraints require a block state space.")
    block = problem.block_linearization_operator(values, args)
    primal_raw = block.blocks[displacement_index][displacement_index]
    constraint_raw = block.blocks[pressure_index][displacement_index]
    coupling_raw = block.blocks[displacement_index][pressure_index]
    pressure_diagonal_raw = block.blocks[pressure_index][pressure_index]
    if primal_raw is None or constraint_raw is None or coupling_raw is None:
        raise ValueError("Mixed volumetric form is missing a required saddle block.")
    displacement_space = problem.state_space.spaces[displacement_index]
    pressure_space = problem.state_space.spaces[pressure_index]
    if not isinstance(pressure_space, ArraySpace) or len(pressure_space.shape) != 1:
        raise ValueError("Incompressibility pressure must use one rank-1 array field.")
    primal = _primalized_block(primal_raw, displacement_space)
    raw_constraint = _primalized_block(constraint_raw, pressure_space)
    pressure_coupling = _primalized_block(coupling_raw, displacement_space)
    pressure_diagonal = (
        None
        if pressure_diagonal_raw is None
        else _primalized_block(pressure_diagonal_raw, pressure_space)
    )
    tolerance = float(rank_tolerance)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive and finite.")
    constraint_matrix = _dense_operator(raw_constraint)
    coupling_matrix = _dense_operator(pressure_coupling)
    left_vectors, singular_values, _ = np.linalg.svd(
        constraint_matrix, full_matrices=True
    )
    largest = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    threshold = tolerance * largest
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    pressure_dimension = pressure_space.size
    expected_rank = pressure_dimension - 1
    nullity = pressure_dimension - numerical_rank
    smallest_retained = (
        float(singular_values[numerical_rank - 1]) if numerical_rank else 0.0
    )
    largest_discarded = (
        float(singular_values[numerical_rank])
        if numerical_rank < singular_values.size
        else 0.0
    )
    null_basis = left_vectors[:, numerical_rank:]
    constant_mode = (
        np.ones((pressure_dimension,), dtype=constraint_matrix.dtype)
        if null_basis.shape[1] == 0
        else null_basis[:, 0]
    )
    gauge = PressureGaugePlan(
        pressure_space,
        mode=gauge_mode,
        weights=pressure_weights,
        constant_mode=constant_mode,
        pinned_index=pinned_pressure_dof,
        tolerance=gauge_tolerance,
    )
    gauge_vector = np.asarray(gauge.weights).reshape((1, pressure_dimension))
    gauge_coupling = (
        0.0
        if null_basis.shape[1] == 0
        else float(np.min(np.linalg.svd(gauge_vector @ null_basis, compute_uv=False)))
    )
    plus_defect = float(
        np.max(np.abs(coupling_matrix - constraint_matrix.T), initial=0.0)
    )
    minus_defect = float(
        np.max(np.abs(coupling_matrix + constraint_matrix.T), initial=0.0)
    )
    sign = 1 if plus_defect <= minus_defect else -1
    adjoint_defect = min(plus_defect, minus_defect)
    adjoint_scale = max(
        float(np.max(np.abs(coupling_matrix), initial=0.0)),
        float(np.max(np.abs(constraint_matrix), initial=0.0)),
        1.0,
    )
    valid = (
        numerical_rank == expected_rank
        and nullity == 1
        and gauge_coupling > tolerance
        and adjoint_defect <= tolerance * adjoint_scale
    )
    rank = VolumetricConstraintRankEvidence(
        raw_constraint.source.size,
        pressure_dimension,
        numerical_rank,
        nullity,
        expected_rank,
        smallest_retained,
        largest_discarded,
        gauge_coupling,
        sign,
        adjoint_defect,
        tolerance,
        valid,
    )
    if numerical_rank < expected_rank:
        raise ValueError(
            "Volumetric constraint rows are duplicate or rank deficient beyond the pressure gauge."
        )
    if numerical_rank > expected_rank:
        raise ValueError(
            "Pressure block has no one-dimensional constant nullspace for the declared gauge."
        )
    if gauge_coupling <= tolerance:
        raise ValueError(
            "Pressure gauge does not remove the detected constant nullspace."
        )
    if adjoint_defect > tolerance * adjoint_scale:
        raise ValueError(
            "Displacement-pressure coupling blocks are not adjoints up to sign."
        )
    constraint = (
        raw_constraint if sign == 1 else ScaledLinearOperator(raw_constraint, -1.0)
    )
    stabilization = (
        None
        if pressure_diagonal is None
        else ScaledLinearOperator(pressure_diagonal, -float(sign))
    )
    generated = canonical_fingerprint(
        {
            "kind": "mixed-volumetric-constraint",
            "compilation": problem.compilation_id,
            "linearization": block.operator_id,
            "displacement_field": displacement_field,
            "pressure_field": pressure_field,
            "gauge": gauge.gauge_id,
            "rank_tolerance": tolerance.hex(),
            "constraint_sign": sign,
        }
    )
    identifier = generated if plan_id is None else str(plan_id)
    if not identifier:
        raise ValueError("plan_id must be non-empty or None.")
    operator = saddle_point_operator(
        primal,
        constraint,
        stabilization,
        operator_id=f"{identifier}:saddle",
    )
    return MixedVolumetricConstraintPlan(
        problem,
        operator,
        primal,
        constraint,
        stabilization,
        gauge,
        rank,
        displacement_index,
        pressure_index,
        sign,
        identifier,
    )


__all__ = [
    "AttachmentActionReactionCertificate",
    "AttachmentRankEvidence",
    "InterpolationTransposeEvidence",
    "MixedVolumetricConstraintPayload",
    "MixedVolumetricConstraintPlan",
    "PreparedFiniteElementPointInterpolation",
    "PressureGaugeEvidence",
    "PressureGaugePlan",
    "RigidDeformableAttachmentEvaluation",
    "RigidDeformableAttachmentPlan",
    "RigidDeformableKKTLinearization",
    "RigidDeformableKKTPayload",
    "VolumetricConstraintRankEvidence",
    "prepare_finite_element_point_interpolation",
    "prepare_mixed_volumetric_constraint",
]
