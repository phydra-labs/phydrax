#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    ArraySpace,
    FunctionLinearOperator,
    LocalEliminationPlan,
    LocalEliminationResult,
    OperatorProperties,
)
from ._execution import RelationExecutionPlan, RelationExecutionState
from ._linear import SparseCoordinateOperator
from ._relation import EdgeRelation


def scatter_local(
    residual: Array,
    dofs: Array,
    local: Array,
    accumulation: str = "fast",
    /,
) -> Array:
    """Scatter local rows with an explicit reduction-order policy."""
    if residual.ndim < 1:
        raise ValueError("residual must contain a target axis.")
    if dofs.shape != local.shape[: dofs.ndim]:
        raise ValueError("Local values must begin with the local DOF shape.")
    flat_dofs = jnp.asarray(dofs, dtype=jnp.int32).reshape((-1,))
    trailing = local.shape[dofs.ndim :]
    flat_local = jnp.asarray(local).reshape((flat_dofs.size,) + trailing)
    relation = EdgeRelation(
        jnp.arange(flat_dofs.size, dtype=jnp.int32),
        flat_dofs,
        source_size=flat_dofs.size,
        target_size=residual.shape[0],
    )
    execution = RelationExecutionPlan().prepare(relation)
    grouped, _ = execution.reduce(
        flat_local,
        accumulation=accumulation,
        output="dense",
    )
    return residual + grouped


class ElementTensorOperator(StrictModule, NonTrainableState):
    """Rectangular local tensors with independent input/output scatter routes."""

    local_matrices: Array
    input_gathers: Array
    output_gathers: Array
    valid: Array
    input_execution: RelationExecutionState
    output_execution: RelationExecutionState
    source_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    accumulation: str = eqx.field(static=True)
    properties: OperatorProperties
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_matrices: ArrayLike,
        input_gathers: ArrayLike,
        output_gathers: ArrayLike,
        source_size: int,
        target_size: int,
        /,
        *,
        valid: ArrayLike | None = None,
        accumulation: str = "fast",
        properties: OperatorProperties | None = None,
    ):
        matrices = jnp.asarray(local_matrices)
        inputs = jnp.asarray(input_gathers, dtype=jnp.int32)
        outputs = jnp.asarray(output_gathers, dtype=jnp.int32)
        source = int(source_size)
        target = int(target_size)
        accumulation_ = str(accumulation)
        if matrices.ndim != 3:
            raise ValueError("Local element matrices must have shape (entity, out, in).")
        if inputs.shape != (matrices.shape[0], matrices.shape[2]):
            raise ValueError("Input gathers must match local matrix columns.")
        if outputs.shape != (matrices.shape[0], matrices.shape[1]):
            raise ValueError("Output gathers must match local matrix rows.")
        if source <= 0 or target <= 0:
            raise ValueError("Element tensor source/target sizes must be positive.")
        if bool(jnp.any((inputs < 0) | (inputs >= source))):
            raise ValueError("Element tensor input gathers are out of bounds.")
        if bool(jnp.any((outputs < 0) | (outputs >= target))):
            raise ValueError("Element tensor output gathers are out of bounds.")
        if accumulation_ not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown local accumulation policy.")
        valid_ = (
            jnp.ones((matrices.shape[0],), dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if valid_.shape != (matrices.shape[0],):
            raise ValueError("Element tensor validity must have one entry per entity.")
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties or None.")
        output_routes = outputs.reshape((-1,))
        output_valid = jnp.broadcast_to(valid_[:, None], outputs.shape).reshape((-1,))
        output_relation = EdgeRelation(
            jnp.arange(output_routes.size, dtype=jnp.int32),
            output_routes,
            source_size=output_routes.size,
            target_size=target,
            valid=output_valid,
        )
        input_routes = inputs.reshape((-1,))
        input_valid = jnp.broadcast_to(valid_[:, None], inputs.shape).reshape((-1,))
        input_relation = EdgeRelation(
            jnp.arange(input_routes.size, dtype=jnp.int32),
            input_routes,
            source_size=input_routes.size,
            target_size=source,
            valid=input_valid,
        )
        output_execution = RelationExecutionPlan().prepare(output_relation)
        input_execution = RelationExecutionPlan().prepare(input_relation)
        self.local_matrices = matrices
        self.input_gathers = inputs
        self.output_gathers = outputs
        self.valid = valid_
        self.input_execution = input_execution
        self.output_execution = output_execution
        self.source_size = source
        self.target_size = target
        self.accumulation = accumulation_
        self.properties = properties_
        self.operator_id = canonical_fingerprint(
            {
                "kind": "element-tensor-operator",
                "matrix_shape": list(matrices.shape),
                "input_shape": list(inputs.shape),
                "output_shape": list(outputs.shape),
                "source_size": source,
                "target_size": target,
                "accumulation": accumulation_,
            }
        )

    def mv(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if value_.shape != (self.source_size,):
            raise ValueError("Element tensor input shape is incompatible.")
        local_input = value_[self.input_gathers]
        contribution = ein.contract("eoi,ei->eo", self.local_matrices, local_input)
        contribution = jnp.where(self.valid[:, None], contribution, 0.0)
        reduced, _ = self.output_execution.reduce(
            contribution.reshape((-1,)),
            accumulation=self.accumulation,
            output="dense",
        )
        return reduced

    def transpose_mv(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if value_.shape != (self.target_size,):
            raise ValueError("Element tensor transpose input shape is incompatible.")
        local_input = value_[self.output_gathers]
        contribution = ein.contract("eoi,eo->ei", self.local_matrices, local_input)
        contribution = jnp.where(self.valid[:, None], contribution, 0.0)
        reduced, _ = self.input_execution.reduce(
            contribution.reshape((-1,)),
            accumulation=self.accumulation,
            output="dense",
        )
        return reduced

    def diagonal(self, /) -> Array:
        if self.source_size != self.target_size or not bool(
            jnp.array_equal(self.input_gathers, self.output_gathers)
        ):
            raise ValueError("Diagonal requires square operators with identical routes.")
        local = jnp.diagonal(self.local_matrices, axis1=-2, axis2=-1)
        local = jnp.where(self.valid[:, None], local, 0.0)
        reduced, _ = self.input_execution.reduce(
            local.reshape((-1,)),
            accumulation=self.accumulation,
            output="dense",
        )
        return reduced

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        source = ArraySpace((self.source_size,), dtype=self.local_matrices.dtype)
        target = ArraySpace((self.target_size,), dtype=self.local_matrices.dtype)
        return FunctionLinearOperator(
            self.mv,
            source=source,
            target=target,
            transpose_action=self.transpose_mv,
            properties=self.properties,
            operator_id=self.operator_id,
            closure_convert=False,
        )

    def as_sparse_coordinate(self, /) -> SparseCoordinateOperator:
        entity_count, output_width, input_width = self.local_matrices.shape
        source_routes = jnp.broadcast_to(
            self.input_gathers[:, None, :],
            (entity_count, output_width, input_width),
        ).reshape((-1,))
        target_routes = jnp.broadcast_to(
            self.output_gathers[:, :, None],
            (entity_count, output_width, input_width),
        ).reshape((-1,))
        valid = jnp.broadcast_to(
            self.valid[:, None, None],
            (entity_count, output_width, input_width),
        ).reshape((-1,))
        relation = EdgeRelation(
            source_routes,
            target_routes,
            source_size=self.source_size,
            target_size=self.target_size,
            valid=valid,
        )
        return SparseCoordinateOperator(
            relation,
            self.local_matrices.reshape((-1,)),
            source=ArraySpace((self.source_size,), dtype=self.local_matrices.dtype),
            target=ArraySpace((self.target_size,), dtype=self.local_matrices.dtype),
            properties=self.properties,
            operator_id=canonical_fingerprint(
                {"kind": "element-tensor-sparse", "operator": self.operator_id}
            ),
        )

    def condense(
        self,
        plan: LocalEliminationPlan,
        local_right_hand_side: ArrayLike,
        retained_gathers: ArrayLike,
        retained_global_size: int,
        /,
    ) -> tuple["ElementTensorOperator", LocalEliminationResult]:
        if not isinstance(plan, LocalEliminationPlan):
            raise TypeError("plan must be LocalEliminationPlan.")
        if (
            self.local_matrices.shape[1] != self.local_matrices.shape[2]
            or self.local_matrices.shape[1] != plan.local_size
        ):
            raise ValueError("Static condensation requires square compatible tensors.")
        result = plan.condense(self.local_matrices, local_right_hand_side)
        retained = jnp.asarray(retained_gathers, dtype=jnp.int32)
        condensed = ElementTensorOperator(
            result.schur,
            retained,
            retained,
            retained_global_size,
            retained_global_size,
            valid=self.valid & ~result.failed,
            accumulation=self.accumulation,
        )
        return condensed, result


__all__ = ["ElementTensorOperator", "scatter_local"]
