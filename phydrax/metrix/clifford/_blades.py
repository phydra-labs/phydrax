#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._exterior_basis import axes_bitmap, bitmap_axes, exterior_indices
from ._spec import CliffordAlgebraSpec


def _canonical_bitmaps(
    algebra: CliffordAlgebraSpec,
    bitmaps: Iterable[int],
    /,
) -> tuple[int, ...]:
    values = tuple(int(value) for value in bitmaps)
    if any(value < 0 or value >> algebra.dimension for value in values):
        raise ValueError("Clifford blade bitmap exceeds the algebra dimension.")
    if len(set(values)) != len(values):
        raise ValueError("Clifford blade layouts cannot contain duplicate blades.")
    return tuple(
        sorted(
            values,
            key=lambda value: (value.bit_count(), bitmap_axes(value, algebra.dimension)),
        )
    )


class CliffordBladeLayout(StrictModule, NonTrainableState):
    """Canonical fixed support over basis blades of one Clifford algebra."""

    algebra: CliffordAlgebraSpec
    bitmaps: tuple[int, ...] = eqx.field(static=True)
    grades: tuple[int, ...] = eqx.field(static=True)
    axes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: CliffordAlgebraSpec,
        bitmaps: Sequence[int],
        /,
    ):
        if not isinstance(algebra, CliffordAlgebraSpec):
            raise TypeError("algebra must be a CliffordAlgebraSpec.")
        resolved = _canonical_bitmaps(algebra, bitmaps)
        algebra.budget.admit_blades(len(resolved))
        axes = tuple(bitmap_axes(value, algebra.dimension) for value in resolved)
        grades = tuple(len(value) for value in axes)
        self.algebra = algebra
        self.bitmaps = resolved
        self.grades = grades
        self.axes = axes
        self.layout_id = canonical_fingerprint(
            {
                "kind": "clifford-blade-layout-v1",
                "algebra": algebra.algebra_id,
                "bitmaps": list(resolved),
            }
        )

    @classmethod
    def full(cls, algebra: CliffordAlgebraSpec, /) -> "CliffordBladeLayout":
        return cls(algebra, tuple(range(algebra.blade_count)))

    @classmethod
    def grades_layout(
        cls,
        algebra: CliffordAlgebraSpec,
        grades: Sequence[int],
        /,
    ) -> "CliffordBladeLayout":
        selected = tuple(int(value) for value in grades)
        if any(value < 0 or value > algebra.dimension for value in selected):
            raise ValueError("Clifford grades must lie in the algebra dimension.")
        if len(set(selected)) != len(selected):
            raise ValueError("Clifford grade selection must be unique.")
        bitmaps = tuple(
            axes_bitmap(index, algebra.dimension)
            for grade in sorted(selected)
            for index in exterior_indices(algebra.dimension, grade)
        )
        return cls(algebra, bitmaps)

    @classmethod
    def blades(
        cls,
        algebra: CliffordAlgebraSpec,
        bitmaps: Sequence[int],
        /,
    ) -> "CliffordBladeLayout":
        return cls(algebra, bitmaps)

    @property
    def blade_count(self) -> int:
        return len(self.bitmaps)

    @property
    def grade_set(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.grades)))

    @property
    def complete_grades(self) -> bool:
        return all(
            set(
                axes_bitmap(index, self.algebra.dimension)
                for index in exterior_indices(self.algebra.dimension, grade)
            ).issubset(self.bitmaps)
            for grade in self.grade_set
        )

    def grade_positions(self, grade: int, /) -> tuple[int, ...]:
        grade_ = int(grade)
        return tuple(index for index, value in enumerate(self.grades) if value == grade_)

    def position(self, bitmap: int, /) -> int:
        try:
            return self.bitmaps.index(int(bitmap))
        except ValueError as error:
            raise ValueError(
                "Blade bitmap is absent from the Clifford layout."
            ) from error

    def contains(self, bitmap: int, /) -> bool:
        return int(bitmap) in self.bitmaps

    def require_compatible(self, other: "CliffordBladeLayout", /) -> None:
        if not isinstance(other, CliffordBladeLayout):
            raise TypeError("Expected a CliffordBladeLayout.")
        self.algebra.require_compatible(other.algebra)

    def require_same(self, other: "CliffordBladeLayout", /) -> None:
        self.require_compatible(other)
        if self.layout_id != other.layout_id:
            raise ValueError("Clifford blade layouts do not match.")


__all__ = ["CliffordBladeLayout"]
