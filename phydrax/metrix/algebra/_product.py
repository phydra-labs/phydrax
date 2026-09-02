#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import AbstractFiniteRealAlgebraSpec
from ._layout import AlgebraElementLayout
from ._resources import AlgebraResourceEvidence


AlgebraProductBackend: TypeAlias = Literal["sparse", "dense"]


class AlgebraProductEvidence(StrictModule, NonTrainableState):
    algebra_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    backend: AlgebraProductBackend = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    resource_evidence: AlgebraResourceEvidence
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        algebra_id: str,
        layout_id: str,
        backend: AlgebraProductBackend,
        term_count: int,
        resource_evidence: AlgebraResourceEvidence,
    ):
        if backend not in ("sparse", "dense"):
            raise ValueError("Unknown algebra product backend.")
        if not algebra_id or not layout_id:
            raise ValueError("Algebra product evidence IDs must be non-empty.")
        self.algebra_id = algebra_id
        self.layout_id = layout_id
        self.backend = backend
        self.term_count = int(term_count)
        self.resource_evidence = resource_evidence
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "algebra-product-evidence-v1",
                "algebra": algebra_id,
                "layout": layout_id,
                "backend": backend,
                "terms": int(term_count),
                "resources": resource_evidence.evidence_id,
            }
        )


class AlgebraProductPlan(StrictModule, NonTrainableState):
    """Prepared bilinear product over a complete finite-algebra coordinate layout."""

    algebra: AbstractFiniteRealAlgebraSpec
    layout: AlgebraElementLayout
    left_indices: Array
    right_indices: Array
    output_indices: Array
    coefficient_numerators: Array
    coefficient_denominators: Array
    dense_kernel: Array | None
    backend: AlgebraProductBackend = eqx.field(static=True)
    fractional_coefficients: bool = eqx.field(static=True)
    evidence: AlgebraProductEvidence
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: AbstractFiniteRealAlgebraSpec,
        /,
        *,
        layout: AlgebraElementLayout | None = None,
        backend: Literal["auto", "sparse", "dense"] = "auto",
    ):
        if not isinstance(algebra, AbstractFiniteRealAlgebraSpec):
            raise TypeError("algebra must implement AbstractFiniteRealAlgebraSpec.")
        layout_ = AlgebraElementLayout(algebra) if layout is None else layout
        if not isinstance(layout_, AlgebraElementLayout):
            raise TypeError("layout must be AlgebraElementLayout or None.")
        algebra.require_compatible(layout_.algebra)
        if backend not in ("auto", "sparse", "dense"):
            raise ValueError("backend must be 'auto', 'sparse', or 'dense'.")
        terms = algebra.structure.terms
        term_count = len(terms)
        fractional = any(denominator != 1 for *_, denominator in terms)
        sparse_bytes = term_count * 5 * 4
        dimension = algebra.coordinate_dimension
        dense_entries = dimension**3
        dense_bytes = dense_entries * 8
        dense_admitted = (
            dense_bytes <= algebra.budget.maximum_dense_kernel_bytes
            and dense_bytes + sparse_bytes <= algebra.budget.maximum_plan_bytes
        )
        if backend == "dense" and not dense_admitted:
            raise ValueError("Dense algebra product exceeds its resource budget.")
        resolved: AlgebraProductBackend = (
            "dense"
            if backend == "dense" or (backend == "auto" and dense_admitted)
            else "sparse"
        )
        dense_kernel = None
        if resolved == "dense":
            kernel = np.zeros((dimension, dimension, dimension), dtype=np.float64)
            for left, right, output, numerator, denominator in terms:
                kernel[output, left, right] += numerator / denominator
            dense_kernel = jnp.asarray(kernel)
        realized_bytes = sparse_bytes + (dense_bytes if resolved == "dense" else 0)
        algebra.budget.admit_product(term_count, realized_bytes)
        resources = AlgebraResourceEvidence(
            coordinate_count=dimension,
            product_pairs=dimension**2,
            product_terms=term_count,
            audit_terms=algebra.resources.audit_terms,
            plan_bytes=realized_bytes,
            dense_kernel_bytes=dense_bytes if resolved == "dense" else 0,
            budget=algebra.budget,
        )
        evidence = AlgebraProductEvidence(
            algebra_id=algebra.algebra_id,
            layout_id=layout_.layout_id,
            backend=resolved,
            term_count=term_count,
            resource_evidence=resources,
        )
        self.algebra = algebra
        self.layout = layout_
        self.left_indices = jnp.asarray([term[0] for term in terms], dtype=jnp.int32)
        self.right_indices = jnp.asarray([term[1] for term in terms], dtype=jnp.int32)
        self.output_indices = jnp.asarray([term[2] for term in terms], dtype=jnp.int32)
        self.coefficient_numerators = jnp.asarray(
            [term[3] for term in terms], dtype=jnp.int32
        )
        self.coefficient_denominators = jnp.asarray(
            [term[4] for term in terms], dtype=jnp.int32
        )
        self.dense_kernel = dense_kernel
        self.backend = resolved
        self.evidence = evidence
        self.plan_id = canonical_fingerprint(
            {
                "kind": "algebra-product-plan-v1",
                "evidence": evidence.evidence_id,
                "structure": algebra.structure.table_id,
            }
        )
        self.fractional_coefficients = fractional

    def _last_axis(self, value: ArrayLike, owner: str, /) -> Array:
        array = jnp.asarray(value)
        if array.ndim < 1:
            raise ValueError(f"{owner} must expose one algebra coordinate axis.")
        axis = self.layout.algebra_axis
        if axis < 0:
            axis += array.ndim
        if (
            axis < 0
            or axis >= array.ndim
            or array.shape[axis] != self.algebra.coordinate_dimension
        ):
            raise ValueError(f"{owner} algebra axis has the wrong coordinate dimension.")
        return jnp.moveaxis(array, axis, -1)

    def __call__(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = self._last_axis(left, "Left value")
        right_ = self._last_axis(right, "Right value")
        leading = jnp.broadcast_shapes(left_.shape[:-1], right_.shape[:-1])
        dtype = jnp.result_type(left_, right_)
        if self.fractional_coefficients:
            dtype = jnp.result_type(dtype, jnp.float64)
        if not (
            jnp.issubdtype(dtype, jnp.signedinteger)
            or jnp.issubdtype(dtype, jnp.floating)
            or jnp.issubdtype(dtype, jnp.complexfloating)
        ):
            raise TypeError(
                "Algebra products require signed-integer, floating, or complex values."
            )
        dimension = self.algebra.coordinate_dimension
        left_ = jnp.broadcast_to(left_, leading + (dimension,)).astype(dtype)
        right_ = jnp.broadcast_to(right_, leading + (dimension,)).astype(dtype)
        if self.backend == "dense":
            if self.dense_kernel is None:
                raise RuntimeError("Dense algebra product lost its kernel.")
            output = ein.contract(
                "...l,olr,...r->...o",
                left_,
                self.dense_kernel.astype(dtype),
                right_,
                backend="jax",
            )
        else:
            output = jnp.zeros(leading + (dimension,), dtype=dtype)
            coefficients = self.coefficient_numerators.astype(dtype)
            if self.fractional_coefficients:
                coefficients = coefficients / self.coefficient_denominators.astype(dtype)
            terms = (
                left_[..., self.left_indices]
                * right_[..., self.right_indices]
                * coefficients
            )
            output = output.at[..., self.output_indices].add(terms)
        target_axis = self.layout.algebra_axis
        if target_axis < 0:
            target_axis += output.ndim
        return jnp.moveaxis(output, -1, target_axis)

    def commutator(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Return ``left * right - right * left`` under this product plan."""
        return self(left, right) - self(right, left)

    def jordan_product(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Return the symmetrized product without integer truncation."""
        left_right = self(left, right)
        right_left = self(right, left)
        dtype = jnp.result_type(left_right, right_left, 0.5)
        return (left_right.astype(dtype) + right_left.astype(dtype)) * jnp.asarray(
            0.5, dtype=dtype
        )

    def associator(
        self,
        left: ArrayLike,
        middle: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        """Return the explicit bracket defect ``(left * middle) * right - left * (middle * right)``."""
        return self(self(left, middle), right) - self(
            left,
            self(middle, right),
        )

    def lower(self, leading_shape: Sequence[int], dtype: Any, /):
        from ...discretization._lowered_operator import (
            LoweredBufferSpec,
            LoweredKernel,
            LoweredOperatorProgram,
        )

        leading = tuple(int(size) for size in leading_shape)
        if any(size <= 0 for size in leading):
            raise ValueError("Lowered algebra leading shape must be positive.")
        value_shape = leading + (self.algebra.coordinate_dimension,)
        dtype_ = np.dtype(dtype)
        if self.fractional_coefficients and not np.issubdtype(dtype_, np.inexact):
            raise TypeError(
                "Fractional algebra products require floating or complex lowered dtype."
            )

        def jax_action(state):
            return {"output": self(state["left"], state["right"])}

        terms = self.algebra.structure.terms

        def numpy_action(state):
            left = np.asarray(state["left"])
            right = np.asarray(state["right"])
            output_dtype = (
                np.result_type(left, right, np.float64)
                if self.fractional_coefficients
                else np.result_type(left, right)
            )
            output = np.zeros(value_shape, dtype=output_dtype)
            for left_index, right_index, output_index, numerator, denominator in terms:
                output[..., output_index] += (
                    left[..., left_index]
                    * right[..., right_index]
                    * numerator
                    / denominator
                )
            return {"output": output}

        buffers = (
            LoweredBufferSpec("left", value_shape, dtype_),
            LoweredBufferSpec("right", value_shape, dtype_),
            LoweredBufferSpec("output", value_shape, dtype_),
        )
        kernel = LoweredKernel(
            "algebra-product",
            ("left", "right"),
            ("output",),
            jax_action,
            numpy_action,
            implementation_id=f"algebra-product:{self.plan_id}",
        )
        return LoweredOperatorProgram(buffers, (kernel,))


__all__ = ["AlgebraProductBackend", "AlgebraProductEvidence", "AlgebraProductPlan"]
