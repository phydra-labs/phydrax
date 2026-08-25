#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._blades import CliffordBladeLayout


def _values(value: ArrayLike, layout: CliffordBladeLayout, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[-1] != layout.blade_count:
        raise ValueError(
            f"{name} must end in {layout.blade_count} Clifford blade coefficients."
        )
    return array


def embed_layout(
    values: ArrayLike,
    source: CliffordBladeLayout,
    target: CliffordBladeLayout,
    /,
) -> Array:
    """Embed one blade support into a compatible superset layout."""
    source.require_compatible(target)
    array = _values(values, source, "Clifford values")
    missing = set(source.bitmaps).difference(target.bitmaps)
    if missing:
        raise ValueError(
            f"Target Clifford layout omits source blades {tuple(sorted(missing))}."
        )
    positions = jnp.asarray(
        tuple(target.position(bitmap) for bitmap in source.bitmaps), dtype=jnp.int32
    )
    output = jnp.zeros(array.shape[:-1] + (target.blade_count,), dtype=array.dtype)
    return output.at[..., positions].set(array)


def extract_layout(
    values: ArrayLike,
    source: CliffordBladeLayout,
    target: CliffordBladeLayout,
    /,
) -> Array:
    """Extract a compatible subset of blade coefficients in canonical order."""
    source.require_compatible(target)
    array = _values(values, source, "Clifford values")
    missing = set(target.bitmaps).difference(source.bitmaps)
    if missing:
        raise ValueError(
            f"Source Clifford layout omits requested blades {tuple(sorted(missing))}."
        )
    positions = jnp.asarray(
        tuple(source.position(bitmap) for bitmap in target.bitmaps), dtype=jnp.int32
    )
    return array[..., positions]


def grade_layout(
    layout: CliffordBladeLayout,
    grades: Sequence[int],
    /,
) -> CliffordBladeLayout:
    """Return the selected complete grades supported by one layout."""
    target = CliffordBladeLayout.grades_layout(layout.algebra, grades)
    missing = set(target.bitmaps).difference(layout.bitmaps)
    if missing:
        raise ValueError("Source Clifford layout does not contain the requested grades.")
    return target


def project_grades(
    values: ArrayLike,
    layout: CliffordBladeLayout,
    grades: Sequence[int],
    /,
) -> tuple[Array, CliffordBladeLayout]:
    """Extract complete grades and return their explicit output layout."""
    target = grade_layout(layout, grades)
    return extract_layout(values, layout, target), target


def _grade_signs(
    layout: CliffordBladeLayout,
    kind: Literal["grade", "reverse", "conjugate"],
    /,
) -> Array:
    if kind == "grade":
        values = tuple(-1 if grade % 2 else 1 for grade in layout.grades)
    elif kind == "reverse":
        values = tuple(
            -1 if (grade * (grade - 1) // 2) % 2 else 1 for grade in layout.grades
        )
    elif kind == "conjugate":
        values = tuple(
            -1 if (grade * (grade + 1) // 2) % 2 else 1 for grade in layout.grades
        )
    else:
        raise ValueError("Unknown Clifford involution.")
    return jnp.asarray(values, dtype=jnp.int8)


def grade_involution(values: ArrayLike, layout: CliffordBladeLayout, /) -> Array:
    array = _values(values, layout, "Clifford values")
    return array * _grade_signs(layout, "grade").astype(array.dtype)


def reverse(values: ArrayLike, layout: CliffordBladeLayout, /) -> Array:
    array = _values(values, layout, "Clifford values")
    return array * _grade_signs(layout, "reverse").astype(array.dtype)


def clifford_conjugate(values: ArrayLike, layout: CliffordBladeLayout, /) -> Array:
    array = _values(values, layout, "Clifford values")
    return array * _grade_signs(layout, "conjugate").astype(array.dtype)


def scalar_part(values: ArrayLike, layout: CliffordBladeLayout, /) -> Array:
    array = _values(values, layout, "Clifford values")
    if not layout.contains(0):
        return jnp.zeros(array.shape[:-1], dtype=array.dtype)
    return array[..., layout.position(0)]


def basis_blade(
    layout: CliffordBladeLayout,
    bitmap: int,
    /,
    *,
    dtype=float,
) -> Array:
    position = layout.position(bitmap)
    return jnp.zeros((layout.blade_count,), dtype=dtype).at[position].set(1)


__all__ = [
    "basis_blade",
    "clifford_conjugate",
    "embed_layout",
    "extract_layout",
    "grade_involution",
    "grade_layout",
    "project_grades",
    "reverse",
    "scalar_part",
]
