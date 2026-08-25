#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._blades import CliffordBladeLayout
from ._reports import CliffordProductEvidence
from ._resources import CliffordResourceEvidence
from ._spec import CliffordAlgebraSpec


CliffordProductKind: TypeAlias = Literal[
    "geometric",
    "exterior",
    "left_contraction",
    "right_contraction",
]


def basis_blade_product(
    algebra: CliffordAlgebraSpec,
    left_bitmap: int,
    right_bitmap: int,
    /,
) -> tuple[int, int]:
    """Return the exact coefficient and bitmap of two canonical basis blades."""
    left_axes = tuple(
        axis for axis in range(algebra.dimension) if left_bitmap & (1 << axis)
    )
    right_axes = tuple(
        axis for axis in range(algebra.dimension) if right_bitmap & (1 << axis)
    )
    inversions = sum(left > right for left in left_axes for right in right_axes)
    coefficient = -1 if inversions % 2 else 1
    repeated = left_bitmap & right_bitmap
    for axis in range(algebra.dimension):
        if repeated & (1 << axis):
            coefficient *= algebra.diagonal[axis]
            if coefficient == 0:
                break
    return coefficient, left_bitmap ^ right_bitmap


def _retain_product(
    kind: CliffordProductKind,
    left_grade: int,
    right_grade: int,
    output_grade: int,
    overlap: bool,
    /,
) -> bool:
    if kind == "geometric":
        return True
    if kind == "exterior":
        return not overlap and output_grade == left_grade + right_grade
    if kind == "left_contraction":
        return left_grade <= right_grade and output_grade == right_grade - left_grade
    if kind == "right_contraction":
        return right_grade <= left_grade and output_grade == left_grade - right_grade
    raise ValueError(f"Unknown Clifford product kind {kind!r}.")


class CliffordProductPlan(StrictModule, NonTrainableState):
    """Prepared exact product over fixed source and output blade supports."""

    __hash__ = object.__hash__

    algebra: CliffordAlgebraSpec
    left_layout: CliffordBladeLayout
    right_layout: CliffordBladeLayout
    output_layout: CliffordBladeLayout
    left_indices: Array
    right_indices: Array
    output_indices: Array
    coefficients: Array
    dense_kernel: Array | None
    kind: CliffordProductKind = eqx.field(static=True)
    backend: Literal["dense", "sparse"] = eqx.field(static=True)
    evidence: CliffordProductEvidence
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: CliffordAlgebraSpec,
        left_layout: CliffordBladeLayout,
        right_layout: CliffordBladeLayout,
        /,
        *,
        kind: CliffordProductKind = "geometric",
        output_layout: CliffordBladeLayout | None = None,
        backend: Literal["auto", "dense", "sparse"] = "auto",
    ):
        if not isinstance(algebra, CliffordAlgebraSpec):
            raise TypeError("algebra must be a CliffordAlgebraSpec.")
        for layout in (left_layout, right_layout):
            if not isinstance(layout, CliffordBladeLayout):
                raise TypeError("Clifford product layouts must be CliffordBladeLayout.")
            algebra.require_compatible(layout.algebra)
        if kind not in (
            "geometric",
            "exterior",
            "left_contraction",
            "right_contraction",
        ):
            raise ValueError("Unsupported Clifford product kind.")
        if backend not in ("auto", "dense", "sparse"):
            raise ValueError("backend must be 'auto', 'dense', or 'sparse'.")

        terms: list[tuple[int, int, int, int]] = []
        structural_zeros = 0
        output_bitmaps: set[int] = set()
        for left_position, (left_bitmap, left_grade) in enumerate(
            zip(left_layout.bitmaps, left_layout.grades)
        ):
            for right_position, (right_bitmap, right_grade) in enumerate(
                zip(right_layout.bitmaps, right_layout.grades)
            ):
                coefficient, output_bitmap = basis_blade_product(
                    algebra, left_bitmap, right_bitmap
                )
                output_grade = output_bitmap.bit_count()
                retained = coefficient != 0 and _retain_product(
                    kind,
                    left_grade,
                    right_grade,
                    output_grade,
                    bool(left_bitmap & right_bitmap),
                )
                if not retained:
                    structural_zeros += 1
                    continue
                output_bitmaps.add(output_bitmap)
                terms.append((left_position, right_position, output_bitmap, coefficient))

        closure_layout = CliffordBladeLayout.blades(algebra, tuple(output_bitmaps))
        if output_layout is None:
            output = closure_layout
        else:
            if not isinstance(output_layout, CliffordBladeLayout):
                raise TypeError("output_layout must be CliffordBladeLayout or None.")
            algebra.require_compatible(output_layout.algebra)
            missing = output_bitmaps.difference(output_layout.bitmaps)
            if missing:
                raise ValueError(
                    "Clifford output layout drops nonzero product blades: "
                    f"{tuple(sorted(missing))}."
                )
            output = output_layout
        output_lookup = {
            bitmap: position for position, bitmap in enumerate(output.bitmaps)
        }
        left_indices = tuple(term[0] for term in terms)
        right_indices = tuple(term[1] for term in terms)
        output_indices = tuple(output_lookup[term[2]] for term in terms)
        coefficients = tuple(term[3] for term in terms)
        term_count = len(terms)
        sparse_bytes = term_count * (3 * 4 + 1)
        dense_entries = (
            output.blade_count * left_layout.blade_count * right_layout.blade_count
        )
        dense_bytes = dense_entries
        dense_plan_bytes = dense_bytes + sparse_bytes
        algebra.budget.admit_product(term_count, sparse_bytes)
        dense_admitted = (
            term_count > 0
            and dense_bytes <= algebra.budget.maximum_dense_kernel_bytes
            and dense_plan_bytes <= algebra.budget.maximum_plan_bytes
        )
        if backend == "dense" and not dense_admitted:
            raise ValueError(
                f"Dense Clifford plan requires {dense_plan_bytes} bytes, exceeding "
                "the configured dense-kernel or total-plan budget."
            )
        resolved_backend: Literal["dense", "sparse"] = (
            "dense"
            if backend == "dense" or (backend == "auto" and dense_admitted)
            else "sparse"
        )
        dense_kernel = None
        if resolved_backend == "dense":
            dense_kernel = jnp.zeros(
                (
                    output.blade_count,
                    left_layout.blade_count,
                    right_layout.blade_count,
                ),
                dtype=jnp.int8,
            )
            if term_count:
                dense_kernel = dense_kernel.at[
                    jnp.asarray(output_indices, dtype=jnp.int32),
                    jnp.asarray(left_indices, dtype=jnp.int32),
                    jnp.asarray(right_indices, dtype=jnp.int32),
                ].add(jnp.asarray(coefficients, dtype=jnp.int8))
        realized_bytes = dense_plan_bytes if resolved_backend == "dense" else sparse_bytes
        resources = CliffordResourceEvidence(
            blade_count=max(
                left_layout.blade_count,
                right_layout.blade_count,
                output.blade_count,
            ),
            product_terms=term_count,
            plan_bytes=realized_bytes,
            dense_kernel_bytes=dense_bytes if resolved_backend == "dense" else 0,
            budget=algebra.budget,
        )
        evidence = CliffordProductEvidence(
            algebra_id=algebra.algebra_id,
            left_layout_id=left_layout.layout_id,
            right_layout_id=right_layout.layout_id,
            output_layout_id=output.layout_id,
            product_kind=kind,
            backend=resolved_backend,
            term_count=term_count,
            structural_zero_count=structural_zeros,
            exact_closure=output.layout_id == closure_layout.layout_id,
            resource_evidence=resources,
        )
        plan_id = canonical_fingerprint(
            {
                "kind": "clifford-product-plan-v1",
                "evidence": evidence.evidence_id,
                "left_indices": list(left_indices),
                "right_indices": list(right_indices),
                "output_indices": list(output_indices),
                "coefficients": list(coefficients),
            }
        )
        self.algebra = algebra
        self.left_layout = left_layout
        self.right_layout = right_layout
        self.output_layout = output
        self.left_indices = jnp.asarray(left_indices, dtype=jnp.int32)
        self.right_indices = jnp.asarray(right_indices, dtype=jnp.int32)
        self.output_indices = jnp.asarray(output_indices, dtype=jnp.int32)
        self.coefficients = jnp.asarray(coefficients, dtype=jnp.int8)
        self.dense_kernel = dense_kernel
        self.kind = kind
        self.backend = resolved_backend
        self.evidence = evidence
        self.plan_id = plan_id

    def __call__(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.ndim < 1 or left_.shape[-1] != self.left_layout.blade_count:
            raise ValueError(
                "Left Clifford values must end in the prepared left blade count."
            )
        if right_.ndim < 1 or right_.shape[-1] != self.right_layout.blade_count:
            raise ValueError(
                "Right Clifford values must end in the prepared right blade count."
            )
        leading = jnp.broadcast_shapes(left_.shape[:-1], right_.shape[:-1])
        dtype = jnp.result_type(left_, right_)
        left_ = jnp.broadcast_to(left_, leading + (self.left_layout.blade_count,)).astype(
            dtype
        )
        right_ = jnp.broadcast_to(
            right_, leading + (self.right_layout.blade_count,)
        ).astype(dtype)
        if self.backend == "dense":
            if self.dense_kernel is None:
                raise RuntimeError("Dense Clifford plan lost its kernel.")
            return jnp.einsum(
                "...l,olr,...r->...o",
                left_,
                self.dense_kernel.astype(dtype),
                right_,
            )
        output = jnp.zeros(leading + (self.output_layout.blade_count,), dtype=dtype)
        if self.left_indices.size == 0:
            return output
        terms = (
            left_[..., self.left_indices]
            * right_[..., self.right_indices]
            * self.coefficients.astype(dtype)
        )
        return output.at[..., self.output_indices].add(terms)


def prepare_product(
    algebra: CliffordAlgebraSpec,
    left_layout: CliffordBladeLayout,
    right_layout: CliffordBladeLayout,
    /,
    **kwargs,
) -> CliffordProductPlan:
    """Prepare one exact fixed-layout Clifford product."""
    return CliffordProductPlan(algebra, left_layout, right_layout, **kwargs)


__all__ = [
    "basis_blade_product",
    "CliffordProductKind",
    "CliffordProductPlan",
    "prepare_product",
]
