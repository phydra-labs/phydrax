#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, FunctionLinearOperator


class ElementTensorOperator(StrictModule, NonTrainableState):
    local_matrices: Array
    gathers: Array
    global_size: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_matrices: ArrayLike,
        gathers: ArrayLike,
        global_size: int,
        /,
    ):
        matrices = jnp.asarray(local_matrices)
        routes = jnp.asarray(gathers, dtype=jnp.int32)
        size = int(global_size)
        if matrices.ndim != 3 or matrices.shape[0] != routes.shape[0]:
            raise ValueError("Element matrices and gather routes must share cells.")
        if matrices.shape[1:] != (routes.shape[1], routes.shape[1]):
            raise ValueError("Element matrix local width must match gathers.")
        if size <= 0 or jnp.any(routes < 0) or jnp.any(routes >= size):
            raise ValueError("Element gather routes or global size are invalid.")
        self.local_matrices = matrices
        self.gathers = routes
        self.global_size = size
        self.operator_id = canonical_fingerprint(
            {
                "kind": "element-tensor-operator",
                "cell_count": int(routes.shape[0]),
                "local_width": int(routes.shape[1]),
                "global_size": size,
            }
        )

    def mv(self, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if value_.shape != (self.global_size,):
            raise ValueError("Element tensor input has invalid shape.")
        local = value_[self.gathers]
        contribution = oe.contract("cij,cj->ci", self.local_matrices, local)
        return jnp.zeros_like(value_).at[self.gathers].add(contribution)

    def diagonal(self, /) -> Array:
        local = jnp.diagonal(self.local_matrices, axis1=-2, axis2=-1)
        return (
            jnp.zeros((self.global_size,), dtype=local.dtype).at[self.gathers].add(local)
        )

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        space = ArraySpace((self.global_size,), dtype=self.local_matrices.dtype)
        return FunctionLinearOperator(
            self.mv,
            source=space,
            target=space,
            operator_id=self.operator_id,
        )


class PartialAssemblyOperator(StrictModule, NonTrainableState):
    basis_values: Array
    quadrature_weights: Array
    quadrature_coefficient: Array
    gathers: Array
    global_size: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis_values: ArrayLike,
        quadrature_weights: ArrayLike,
        quadrature_coefficient: ArrayLike,
        gathers: ArrayLike,
        global_size: int,
        /,
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
        self.basis_values = basis
        self.quadrature_weights = weights
        self.quadrature_coefficient = coefficient
        self.gathers = routes
        self.global_size = size
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
        local = value_[self.gathers]
        quadrature = oe.contract("qi,ci->cq", self.basis_values, local)
        weighted = self.quadrature_weights * self.quadrature_coefficient * quadrature
        contribution = oe.contract("qi,cq->ci", self.basis_values, weighted)
        return (
            jnp.zeros((self.global_size,), dtype=value_.dtype)
            .at[self.gathers]
            .add(contribution)
        )

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        space = ArraySpace((self.global_size,), dtype=self.quadrature_weights.dtype)
        return FunctionLinearOperator(
            self.mv,
            source=space,
            target=space,
            operator_id=self.operator_id,
        )


__all__ = ["ElementTensorOperator", "PartialAssemblyOperator"]
