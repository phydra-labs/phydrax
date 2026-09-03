#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.fem import SumFactorizationPlan
from ...linalg import ArraySpace, FunctionLinearOperator, OperatorProperties
from ...sparse import ElementTensorOperator, SparseCoordinateOperator


class PartialAssemblyOperator(StrictModule, NonTrainableState):
    """Scalar quadrature action with reusable geometry/coefficient data."""

    basis_values: Array
    quadrature_weights: Array
    quadrature_coefficient: Array
    gathers: Array
    valid: Array
    global_size: int = eqx.field(static=True)
    properties: OperatorProperties
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis_values: ArrayLike,
        quadrature_weights: ArrayLike,
        quadrature_coefficient: ArrayLike,
        gathers: ArrayLike,
        global_size: int,
        /,
        *,
        valid: ArrayLike | None = None,
        properties: OperatorProperties | None = None,
    ):
        basis = jnp.asarray(basis_values)
        weights = jnp.asarray(quadrature_weights)
        coefficient = jnp.asarray(quadrature_coefficient)
        routes = jnp.asarray(gathers, dtype=jnp.int32)
        size = int(global_size)
        if basis.ndim != 2 or routes.ndim != 2 or basis.shape[1] != routes.shape[1]:
            raise ValueError("Partial basis and gather local widths must match.")
        if (
            weights.shape != (routes.shape[0], basis.shape[0])
            or coefficient.shape != weights.shape
        ):
            raise ValueError("Partial quadrature weights/coefficient shapes are invalid.")
        if size <= 0 or bool(jnp.any((routes < 0) | (routes >= size))):
            raise ValueError("Partial assembly routes or global size are invalid.")
        valid_ = (
            jnp.ones((routes.shape[0],), dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if valid_.shape != (routes.shape[0],):
            raise ValueError("Partial assembly validity must match entities.")
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties or None.")
        self.basis_values = basis
        self.quadrature_weights = weights
        self.quadrature_coefficient = coefficient
        self.gathers = routes
        self.valid = valid_
        self.global_size = size
        self.properties = properties_
        self.operator_id = canonical_fingerprint(
            {
                "kind": "partial-assembly-operator",
                "cells": int(routes.shape[0]),
                "quadrature": int(basis.shape[0]),
                "local_width": int(basis.shape[1]),
                "global_size": size,
            }
        )

    def mv(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if value_.shape != (self.global_size,):
            raise ValueError("Partial assembly input shape is incompatible.")
        local = value_[self.gathers]
        quadrature = ein.contract("qi,ci->cq", self.basis_values, local)
        weighted = self.quadrature_weights * self.quadrature_coefficient * quadrature
        contribution = ein.contract("qi,cq->ci", self.basis_values, weighted)
        contribution = jnp.where(self.valid[:, None], contribution, 0.0)
        return (
            jnp.zeros((self.global_size,), dtype=contribution.dtype)
            .at[self.gathers]
            .add(contribution)
        )

    def transpose_mv(self, value: ArrayLike, /) -> Array:
        return self.mv(value)

    def as_element_tensor(self, /) -> ElementTensorOperator:
        local = ein.contract(
            "cq,cq,qi,qj->cij",
            self.quadrature_weights,
            self.quadrature_coefficient,
            self.basis_values,
            self.basis_values,
        )
        return ElementTensorOperator(
            local,
            self.gathers,
            self.gathers,
            self.global_size,
            self.global_size,
            valid=self.valid,
            properties=self.properties,
        )

    def as_sparse_coordinate(self, /) -> SparseCoordinateOperator:
        return self.as_element_tensor().as_sparse_coordinate()

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        space = ArraySpace((self.global_size,), dtype=self.quadrature_weights.dtype)
        return FunctionLinearOperator(
            self.mv,
            source=space,
            target=space,
            transpose_action=self.transpose_mv,
            properties=self.properties,
            operator_id=self.operator_id,
            closure_convert=False,
        )


TensorProductAction = Literal["mass", "diffusion"]


class FiniteElementMassPolicy(StrictModule, NonTrainableState):
    """Exact, nodally collocated, or row-sum-lumped finite-element mass."""

    kind: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, kind: str = "exact", /):
        kind_ = str(kind)
        if kind_ not in ("exact", "collocated_diagonal", "lumped"):
            raise ValueError("Unknown finite-element mass policy.")
        self.kind = kind_
        self.policy_id = canonical_fingerprint(
            {"kind": "finite-element-mass-policy", "mass_kind": kind_}
        )


class FiniteElementDiagonalData(StrictModule, NonTrainableState):
    """Exact coordinate diagonal with explicit construction provenance."""

    diagonal: object
    zero_mask: object
    negative_mask: object
    method: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    numeric_version: Array
    diagonal_id: str = eqx.field(static=True)

    def __init__(
        self,
        diagonal: object,
        method: str,
        operator_id: str,
        /,
        *,
        numeric_version: ArrayLike = 0,
    ):
        method_ = str(method)
        operator = str(operator_id)
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if method_ not in ("sparse-coordinate", "workset", "coordinate-linearization"):
            raise ValueError("Unknown finite-element diagonal construction method.")
        if not operator or version.shape != () or version < 0:
            raise ValueError("Diagonal operator identity/version are invalid.")
        leaves = jax.tree.leaves(diagonal)
        if not leaves or any(
            not jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact) for value in leaves
        ):
            raise TypeError("Finite-element diagonal must contain inexact arrays.")
        self.diagonal = diagonal
        self.zero_mask = jax.tree.map(lambda value: jnp.asarray(value) == 0, diagonal)
        self.negative_mask = jax.tree.map(
            lambda value: jnp.real(jnp.asarray(value)) < 0, diagonal
        )
        self.method = method_
        self.operator_id = operator
        self.numeric_version = version
        self.diagonal_id = canonical_fingerprint(
            {
                "kind": "finite-element-diagonal-data",
                "method": method_,
                "operator": operator,
                "shape": [list(jnp.asarray(value).shape) for value in leaves],
            }
        )


class FiniteElementPreconditionerData(StrictModule, NonTrainableState):
    """Prepared diagonal/block/workset evidence for generic preconditioner builders."""

    diagonal: object
    block_graph: tuple[tuple[bool, ...], ...] = eqx.field(static=True)
    workset_ids: tuple[str, ...] = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        diagonal: object,
        block_graph: tuple[tuple[bool, ...], ...],
        workset_ids: tuple[str, ...],
        /,
    ):
        graph = tuple(tuple(bool(value) for value in row) for row in block_graph)
        identifiers = tuple(str(value) for value in workset_ids)
        if not graph or any(len(row) != len(graph) for row in graph):
            raise ValueError("Preconditioner block graph must be non-empty and square.")
        if not identifiers or any(not value for value in identifiers):
            raise ValueError("Preconditioner workset identities must be non-empty.")
        self.diagonal = diagonal
        self.block_graph = graph
        self.workset_ids = identifiers
        self.data_id = canonical_fingerprint(
            {
                "kind": "finite-element-preconditioner-data",
                "diagonal": array_tree_fingerprint(diagonal),
                "block_graph": [list(row) for row in graph],
                "worksets": list(identifiers),
            }
        )


class TensorProductPartialAssemblyOperator(StrictModule, NonTrainableState):
    """Physical tensor-product mass or diffusion action without dense tabulation."""

    plan: SumFactorizationPlan

    quadrature_data: Array
    gathers: Array
    valid: Array
    action_kind: TensorProductAction = eqx.field(static=True)
    global_size: int = eqx.field(static=True)
    properties: OperatorProperties
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SumFactorizationPlan,
        quadrature_data: ArrayLike,
        gathers: ArrayLike,
        global_size: int,
        /,
        *,
        action_kind: TensorProductAction,
        valid: ArrayLike | None = None,
        properties: OperatorProperties | None = None,
    ):
        if not isinstance(plan, SumFactorizationPlan):
            raise TypeError("plan must be SumFactorizationPlan.")
        kind = str(action_kind)
        if kind not in ("mass", "diffusion"):
            raise ValueError("Unknown tensor-product action kind.")
        data = jnp.asarray(quadrature_data)
        routes = jnp.asarray(gathers, dtype=jnp.int32)
        size = int(global_size)
        nodal_shape = plan.tabulation.nodal_shape
        quadrature_shape = plan.tabulation.evaluation_shape
        dimension = plan.tabulation.dimension
        if routes.ndim != 2 or routes.shape[1] != int(jnp.prod(jnp.asarray(nodal_shape))):
            raise ValueError("Tensor-product gathers do not match nodal axes.")
        expected = (
            (routes.shape[0],) + quadrature_shape
            if kind == "mass"
            else (routes.shape[0],) + quadrature_shape + (dimension, dimension)
        )
        if data.shape != expected:
            raise ValueError("Tensor-product quadrature data have incompatible shape.")
        if size <= 0 or bool(jnp.any((routes < 0) | (routes >= size))):
            raise ValueError("Tensor-product routes or global size are invalid.")
        valid_ = (
            jnp.ones((routes.shape[0],), dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if valid_.shape != (routes.shape[0],):
            raise ValueError("Tensor-product validity must match entities.")
        self.plan = plan
        self.quadrature_data = data
        self.gathers = routes
        self.valid = valid_
        self.action_kind = kind
        self.global_size = size
        properties_ = (
            OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_semidefinite": "construction",
                },
            )
            if properties is None
            else properties
        )
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties or None.")
        self.properties = properties_
        self.operator_id = canonical_fingerprint(
            {
                "kind": "tensor-product-partial-assembly",
                "plan": plan.plan_id,
                "action": kind,
                "data_shape": list(data.shape),
                "gather_shape": list(routes.shape),
                "global_size": size,
            }
        )

    def mv(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if value_.shape != (self.global_size,):
            raise ValueError("Tensor-product input shape is incompatible.")
        nodal_shape = self.plan.tabulation.nodal_shape
        local = value_[self.gathers].reshape((self.gathers.shape[0],) + nodal_shape)
        if self.action_kind == "mass":
            quadrature = self.plan.interpolate(local)
            local_output = self.plan.interpolate_transpose(
                self.quadrature_data * quadrature
            )
        else:
            gradient = self.plan.gradient(local)
            flux = ein.contract("...ab,...b->...a", self.quadrature_data, gradient)
            local_output = self.plan.gradient_transpose(flux)
        contribution = local_output.reshape(self.gathers.shape)
        contribution = jnp.where(self.valid[:, None], contribution, 0.0)
        return (
            jnp.zeros((self.global_size,), dtype=contribution.dtype)
            .at[self.gathers]
            .add(contribution)
        )

    def transpose_mv(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if value_.shape != (self.global_size,):
            raise ValueError("Tensor-product input shape is incompatible.")
        nodal_shape = self.plan.tabulation.nodal_shape
        local = value_[self.gathers].reshape((self.gathers.shape[0],) + nodal_shape)
        if self.action_kind == "mass":
            quadrature = self.plan.interpolate(local)
            local_output = self.plan.interpolate_transpose(
                self.quadrature_data * quadrature
            )
        else:
            gradient = self.plan.gradient(local)
            flux = ein.contract("...ba,...b->...a", self.quadrature_data, gradient)
            local_output = self.plan.gradient_transpose(flux)
        contribution = local_output.reshape(self.gathers.shape)
        contribution = jnp.where(self.valid[:, None], contribution, 0.0)
        return (
            jnp.zeros((self.global_size,), dtype=contribution.dtype)
            .at[self.gathers]
            .add(contribution)
        )

    def as_element_tensor(self, /) -> ElementTensorOperator:
        local_width = self.gathers.shape[1]
        identity = jnp.eye(local_width, dtype=self.quadrature_data.dtype).reshape(
            (local_width,) + self.plan.tabulation.nodal_shape
        )
        if self.action_kind == "mass":
            basis = self.plan.interpolate(identity).reshape((local_width, -1))
            data = self.quadrature_data.reshape((self.gathers.shape[0], -1))
            local = ein.contract("cq,iq,jq->cij", data, basis, basis)
        else:
            gradients = self.plan.gradient(identity).reshape(
                (local_width, -1, self.plan.tabulation.dimension)
            )
            data = self.quadrature_data.reshape(
                (
                    self.gathers.shape[0],
                    -1,
                    self.plan.tabulation.dimension,
                    self.plan.tabulation.dimension,
                )
            )
            local = ein.contract("cqab,iqa,jqb->cij", data, gradients, gradients)
        return ElementTensorOperator(
            local,
            self.gathers,
            self.gathers,
            self.global_size,
            self.global_size,
            valid=self.valid,
            properties=self.properties,
        )

    def as_sparse_coordinate(self, /) -> SparseCoordinateOperator:
        return self.as_element_tensor().as_sparse_coordinate()

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        space = ArraySpace((self.global_size,), dtype=self.quadrature_data.dtype)
        return FunctionLinearOperator(
            self.mv,
            source=space,
            target=space,
            transpose_action=self.transpose_mv,
            properties=self.properties,
            operator_id=self.operator_id,
            closure_convert=False,
        )


class CollocatedTensorProductOperator(StrictModule, NonTrainableState):
    """Collocated quad/hex mass-diffusion action with packed metric entries."""

    derivatives: tuple[Array, ...]
    weighted_metric: Array
    weighted_mass: Array
    gathers: Array
    valid: Array
    dimension: int = eqx.field(static=True)
    global_size: int = eqx.field(static=True)
    properties: OperatorProperties
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        derivative: ArrayLike | Sequence[ArrayLike],
        weighted_metric: ArrayLike,
        weighted_mass: ArrayLike,
        gathers: ArrayLike,
        global_size: int,
        /,
        *,
        valid: ArrayLike | None = None,
    ):
        metric = jnp.asarray(weighted_metric)
        mass = jnp.asarray(weighted_mass)
        routes = jnp.asarray(gathers, dtype=jnp.int32)
        size = int(global_size)
        if metric.ndim == 4 and metric.shape[-1] == 3:
            dimension = 2
        elif metric.ndim == 5 and metric.shape[-1] == 6:
            dimension = 3
        else:
            raise ValueError("Collocated metric must pack 2-D or 3-D symmetric entries.")
        expected_grid = tuple(int(value) for value in metric.shape[1:-1])
        if isinstance(derivative, Sequence):
            derivatives = tuple(jnp.asarray(value) for value in derivative)
        else:
            derivative_ = jnp.asarray(derivative)
            derivatives = (derivative_,) * dimension
        if len(derivatives) != dimension or any(
            value.ndim != 2
            or value.shape[0] != value.shape[1]
            or value.shape[0] != expected_grid[axis]
            for axis, value in enumerate(derivatives)
        ):
            raise ValueError(
                "Collocated derivatives must be square and match each tensor axis."
            )
        if mass.shape != metric.shape[:-1]:
            raise ValueError("Collocated metric/mass grids are incompatible.")
        if routes.shape != (metric.shape[0], int(jnp.prod(jnp.asarray(expected_grid)))):
            raise ValueError("Collocated gathers do not match tensor grid size.")
        valid_ = (
            jnp.ones((routes.shape[0],), dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if valid_.shape != (routes.shape[0],):
            raise ValueError("Collocated validity must match element count.")
        self.derivatives = derivatives
        self.weighted_metric = metric
        self.weighted_mass = mass
        self.gathers = routes
        self.valid = valid_
        self.dimension = dimension
        self.global_size = size
        self.properties = OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        )
        self.operator_id = canonical_fingerprint(
            {
                "kind": "collocated-tensor-product-operator",
                "dimension": dimension,
                "points": expected_grid,
                "elements": int(routes.shape[0]),
                "global_size": size,
            }
        )

    def mv(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if value_.shape != (self.global_size,):
            raise ValueError("Collocated tensor input shape is incompatible.")
        local = value_[self.gathers]
        if self.dimension == 2:
            nx, ny = self.weighted_mass.shape[1:]
            dx, dy = self.derivatives
            q = local.reshape((-1, nx, ny))
            qx = ein.contract("ia,eaj->eij", dx, q)
            qy = ein.contract("ja,eia->eij", dy, q)
            g00, g01, g11 = (
                self.weighted_metric[..., 0],
                self.weighted_metric[..., 1],
                self.weighted_metric[..., 2],
            )
            flux_x = g00 * qx + g01 * qy
            flux_y = g01 * qx + g11 * qy
            output = (
                ein.contract("ia,eij->eaj", dx, flux_x)
                + ein.contract("ja,eij->eia", dy, flux_y)
                + self.weighted_mass * q
            )
        else:
            nx, ny, nz = self.weighted_mass.shape[1:]
            dx, dy, dz = self.derivatives
            q = local.reshape((-1, nx, ny, nz))
            qx = ein.contract("ia,eajk->eijk", dx, q)
            qy = ein.contract("ja,eiak->eijk", dy, q)
            qz = ein.contract("ka,eija->eijk", dz, q)
            g00, g01, g02, g11, g12, g22 = tuple(
                self.weighted_metric[..., index] for index in range(6)
            )
            flux_x = g00 * qx + g01 * qy + g02 * qz
            flux_y = g01 * qx + g11 * qy + g12 * qz
            flux_z = g02 * qx + g12 * qy + g22 * qz
            output = (
                ein.contract("ia,eijk->eajk", dx, flux_x)
                + ein.contract("ja,eijk->eiak", dy, flux_y)
                + ein.contract("ka,eijk->eija", dz, flux_z)
                + self.weighted_mass * q
            )
        contribution = output.reshape(self.gathers.shape)
        contribution = jnp.where(self.valid[:, None], contribution, 0.0)
        return (
            jnp.zeros((self.global_size,), dtype=contribution.dtype)
            .at[self.gathers]
            .add(contribution)
        )

    def transpose_mv(self, value: ArrayLike, /) -> Array:
        return self.mv(value)

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        space = ArraySpace((self.global_size,), dtype=self.weighted_metric.dtype)
        return FunctionLinearOperator(
            self.mv,
            source=space,
            target=space,
            transpose_action=self.transpose_mv,
            properties=self.properties,
            operator_id=self.operator_id,
            closure_convert=False,
        )


__all__ = [
    "CollocatedTensorProductOperator",
    "FiniteElementDiagonalData",
    "FiniteElementMassPolicy",
    "FiniteElementPreconditionerData",
    "PartialAssemblyOperator",
    "TensorProductAction",
    "TensorProductPartialAssemblyOperator",
]
