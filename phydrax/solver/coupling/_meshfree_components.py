#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scalar native meshfree equations published without a facet-trace claim."""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...discretization import BoundaryImposition, PreparedFieldReconstruction
from ...discretization._point_cloud import PreparedPointCloudDiscretization
from ...discretization._point_cloud_view import PointCloudFieldReconstructionKernel
from ...discretization.meshfree._exterior import PreparedMeshfreeExteriorCalculus
from ...discretization.meshfree._surface import (
    PreparedSurfacePointCloud,
    SurfaceFieldReconstructionKernel,
)
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BlockLinearOperator,
    ConstraintMap,
    DualSpace,
    FunctionLinearOperator,
)
from ...sparse import SparseCoordinateOperator
from ...typing import Dim, Float
from ._components import (
    AbstractCapacityComponent,
    AbstractPreparedCapacity,
    AbstractReconstructionComponent,
    ComponentBlock,
    ComponentField,
    ComponentSpace,
)


type MeshfreeOwner = (
    PreparedPointCloudDiscretization
    | PreparedSurfacePointCloud
    | PreparedMeshfreeExteriorCalculus
)


class _MeshfreePointDim(Dim):
    """Full point-value coordinates."""


class _MeshfreeStateDim(Dim):
    """Native, possibly constrained state coordinates."""


class _MeshfreeNullityDim(Dim):
    """Declared homogeneous kernel columns."""


@final
class MeshfreeCapacity(AbstractPreparedCapacity, NonTrainableState):
    __strict_contract__ = True
    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    diagonal: Float[_MeshfreePointDim]
    full_space: ArraySpace

    def operator(self, args: object, /) -> AbstractLinearOperator:
        del args
        return FunctionLinearOperator(
            lambda value: self.diagonal * value,
            source=self.full_space,
            target=DualSpace(self.full_space),
            transpose_action=lambda value: self.diagonal * value,
        )


@final
class MeshfreeComponent(AbstractCapacityComponent, AbstractReconstructionComponent):
    """Publish a prepared point-cloud, intrinsic, or conservative exterior equation.

    ``operator`` is the native full-row operator, not a reconstructed solver;
    ``mass_diagonal`` is the owner's physical quadrature measure.  Reconstruction
    and equation coordinates must agree.  A supplied constraint parameterizes
    ``u=Pz+lift`` and pulls residual rows back exactly once.  No facet capability
    is advertised.  Nullspace columns are explicit declarations, verified against
    the homogeneous published operator at host preparation.
    """

    __strict_contract__ = True
    name: str = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    space: ComponentSpace = eqx.field(static=True)
    owner: MeshfreeOwner
    native_operator: AbstractLinearOperator
    row_operator: AbstractLinearOperator
    reconstruction: PreparedFieldReconstruction
    mass_diagonal: Float[_MeshfreePointDim]
    offset: Float[_MeshfreePointDim]
    load: Float[_MeshfreePointDim]
    kernel: Float[_MeshfreeStateDim, _MeshfreeNullityDim] | None
    impositions: tuple[BoundaryImposition, ...]
    state_blocks: tuple[ComponentBlock, ...]
    row_blocks: tuple[ComponentBlock, ...]
    fields: tuple[ComponentField, ...]

    def __init__(
        self,
        owner: MeshfreeOwner,
        operator: AbstractLinearOperator,
        reconstruction: PreparedFieldReconstruction,
        mass_diagonal: ArrayLike,
        /,
        *,
        name: str,
        owner_id: str,
        field: str = "concentration",
        constraint: ConstraintMap | None = None,
        lift: ArrayLike | None = None,
        load: ArrayLike | None = None,
        free_rows: Array | None = None,
        nullspace: ArrayLike | None = None,
        boundary_impositions: tuple[BoundaryImposition, ...] = (),
    ) -> None:
        if not isinstance(
            owner,
            (
                PreparedPointCloudDiscretization,
                PreparedSurfacePointCloud,
                PreparedMeshfreeExteriorCalculus,
            ),
        ):
            raise TypeError(
                "owner must be a native prepared point cloud, surface, or exterior owner."
            )
        if not isinstance(operator, AbstractLinearOperator) or operator.batch_shape:
            raise TypeError("operator must be an unbatched native linear operator.")
        if not operator.capabilities.transpose:
            raise ValueError(
                "The native equation must publish its exact coordinate transpose."
            )
        if not isinstance(reconstruction, PreparedFieldReconstruction):
            raise TypeError("reconstruction must be a PreparedFieldReconstruction.")
        if not isinstance(operator.source, ArraySpace):
            raise TypeError("Meshfree scalar equations require an ArraySpace.")
        full = operator.source
        if len(full.shape) != 1 or reconstruction.coefficient_shape != full.shape:
            raise ValueError(
                "Equation and reconstruction coefficient coordinates differ."
            )
        if reconstruction.value_shape or not reconstruction.coefficient_linear:
            raise ValueError("MeshfreeComponent requires a scalar linear reconstruction.")
        if not (
            operator.target.compatible(full)
            or operator.target.compatible(DualSpace(full))
        ):
            raise ValueError(
                "Native residual rows must use the field's nominal coordinates or coordinate dual."
            )
        diagonal = np.asarray(mass_diagonal, dtype=np.float64)
        if diagonal.shape != full.shape or not np.all(
            np.isfinite(diagonal) & (diagonal > 0)
        ):
            raise ValueError("Native mass measures must be finite and strictly positive.")
        if isinstance(owner, PreparedPointCloudDiscretization):
            native_measures = owner.quadrature_weights
            native_space = owner.field_spaces[0].vector_space
            if (
                not isinstance(reconstruction.kernel, PointCloudFieldReconstructionKernel)
                or reconstruction.kernel.source_owner_id != owner.prepared_id
            ):
                raise ValueError(
                    "Reconstruction belongs to another native point-cloud owner revision."
                )
        elif isinstance(owner, PreparedSurfacePointCloud):
            native_measures = owner.measures
            native_space = owner.laplace_beltrami.source
            if (
                not isinstance(reconstruction.kernel, SurfaceFieldReconstructionKernel)
                or reconstruction.kernel.source_owner_id != owner.prepared_id
            ):
                raise ValueError(
                    "Reconstruction belongs to another native intrinsic-surface owner revision."
                )
        else:
            native_measures = owner.node_volumes
            native_space = owner.incidence.source
            kernel_ = reconstruction.kernel
            if isinstance(kernel_, PointCloudFieldReconstructionKernel):
                reconstruction_points = kernel_.points
            elif isinstance(kernel_, SurfaceFieldReconstructionKernel):
                reconstruction_points = kernel_.surface.points
            else:
                raise TypeError(
                    "Exterior fields require a canonical point-cloud or intrinsic reconstruction."
                )
            if not np.array_equal(
                np.asarray(reconstruction_points), np.asarray(owner.points)
            ):
                raise ValueError(
                    "Exterior reconstruction must explicitly share the compact native point enumeration."
                )
        if not full.compatible(native_space):
            raise ValueError(
                "The operator source must be the native owner's nominal field space."
            )
        if not np.array_equal(diagonal, np.asarray(native_measures)):
            raise ValueError(
                "Mass diagonal must be the declared native owner's quadrature measures."
            )
        offset = np.zeros(full.shape) if lift is None else np.asarray(lift)
        rhs = np.zeros(full.shape) if load is None else np.asarray(load)
        if offset.shape != full.shape or rhs.shape != full.shape:
            raise ValueError("Lift and load must match full field coordinates.")
        if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(rhs)):
            raise ValueError("Lift and load must be finite.")
        if not all(
            isinstance(value, BoundaryImposition) for value in boundary_impositions
        ):
            raise TypeError(
                "Boundary declarations must be native BoundaryImposition values."
            )
        name_ = canonical_identifier(name, "name")
        field_ = canonical_identifier(field, "field")
        record = ComponentField(
            field_,
            state_block=field_,
            row_block=field_,
            full_space=full,
            constraint=constraint,
            free_rows=free_rows,
        )
        state = full if constraint is None else constraint.reduced_space
        if not isinstance(state, ArraySpace) or len(state.shape) != 1:
            raise ValueError("Meshfree constraints must retain scalar array coordinates.")
        owner_id_ = canonical_identifier(owner_id, "owner_id")
        kernel = None if nullspace is None else np.asarray(nullspace)
        if kernel is not None:
            if (
                kernel.ndim != 2
                or kernel.shape[0] != state.size
                or not np.all(np.isfinite(kernel))
            ):
                raise ValueError(
                    "Nullspace columns must use the declared state coordinates."
                )
            for column in kernel.T:
                value = jnp.asarray(column, dtype=full.dtype)
                expanded = (
                    value
                    if constraint is None
                    else constraint.homogeneous_correction(value)
                )
                residual = record.pull_back(operator.mv(expanded))
                if np.linalg.norm(np.asarray(residual)) > 1e-9 * max(
                    1.0, np.linalg.norm(column)
                ):
                    raise ValueError(
                        "Declared nullspace is not a kernel of the native operator."
                    )
        if operator.target.compatible(DualSpace(full)):
            row_operator = operator
        elif isinstance(operator, SparseCoordinateOperator):
            row_operator = SparseCoordinateOperator(
                operator.relation,
                operator.coefficients,
                source=full,
                target=DualSpace(full),
                accumulation_dtype=operator.accumulation_dtype,
                block_shape=operator.block_shape,
                storage_plan=operator._storage_plan,
                operator_id=f"{operator.operator_id}:field-rows",
            )
        else:
            row_operator = FunctionLinearOperator(
                operator.mv,
                source=full,
                target=DualSpace(full),
                transpose_action=operator.transpose_mv,
                operator_id=f"{operator.operator_id}:field-rows",
            )
        self.name = name_
        self.owner_id = owner_id_
        self.space = "full" if constraint is None else "reduced"
        self.owner = owner
        self.native_operator = operator
        self.row_operator = row_operator
        self.reconstruction = reconstruction
        self.mass_diagonal = jnp.asarray(diagonal, dtype=full.dtype)
        self.offset = jnp.asarray(offset, dtype=full.dtype)
        self.load = jnp.asarray(rhs, dtype=full.dtype)
        self.kernel = None if kernel is None else jnp.asarray(kernel, dtype=full.dtype)
        self.impositions = boundary_impositions
        self.fields = (record,)
        self.state_blocks = (ComponentBlock(field_, state),)
        self.row_blocks = (ComponentBlock(field_, DualSpace(state)),)

    def residual(self, state: tuple[Array, ...], args: object, /) -> tuple[Array, ...]:
        record = self.fields[0]
        full = self.expand(record.name, state, args)
        return (record.pull_back(self.native_operator.mv(full) - self.load),)

    def linear_operator(self, args: object, /) -> BlockLinearOperator:
        del args
        record = self.fields[0]
        native = self.row_operator
        if record.constraint is None:
            return BlockLinearOperator(
                ((native,),), source=self.state_space, target=self.row_space
            )

        def apply(value: Array) -> Array:
            full = (
                value
                if record.constraint is None
                else record.constraint.homogeneous_correction(value)
            )
            return record.pull_back(native.mv(full))

        def transpose(value: Array) -> Array:
            full = (
                value
                if record.constraint is None
                else record.constraint.prolongation.mv(value)
            )
            result = native.transpose_mv(full)
            return (
                result
                if record.constraint is None
                else record.constraint.prolongation.transpose_mv(result)
            )

        block = FunctionLinearOperator(
            apply,
            source=self.state_blocks[0].space,
            target=self.row_blocks[0].space,
            transpose_action=transpose,
        )
        return BlockLinearOperator(
            ((block,),), source=self.state_space, target=self.row_space
        )

    def lift(self, field: str, args: object, /) -> Array:
        del args
        self.field(field)
        return self.offset

    def nullspace(self, args: object, /) -> tuple[Array, ...] | None:
        del args
        return None if self.kernel is None else (self.kernel,)

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]:
        return self.impositions

    def field_space_id(self, field: str, /) -> str:
        self.field(field)
        return self.reconstruction.field_space_id

    def prepare_field_reconstruction(self, field: str, /) -> PreparedFieldReconstruction:
        self.field(field)
        return self.reconstruction

    def prepare_capacity(
        self, field: str, /, *, coefficient: ArrayLike = 1.0
    ) -> MeshfreeCapacity:
        record = self.field(field)
        value = np.asarray(coefficient, dtype=np.float64)
        if value.shape not in ((), self.mass_diagonal.shape) or not np.all(
            np.isfinite(value) & (value > 0)
        ):
            raise ValueError(
                "Capacity coefficient must be finite and positive, scalar or per point."
            )
        return MeshfreeCapacity(
            self.name, field, self.mass_diagonal * jnp.asarray(value), record.full_space
        )


__all__ = ["MeshfreeCapacity", "MeshfreeComponent"]
