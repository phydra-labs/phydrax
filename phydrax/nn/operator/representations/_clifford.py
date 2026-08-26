#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.metrix.clifford import (
    CliffordAlgebraSpec,
    CliffordBladeLayout,
    CliffordOutermorphismPlan,
    MetricIsometryAction,
)

from ._o3 import O3Features, O3Representation


class CliffordGradeFeatures(eqx.Module):
    """One multiplicity-bearing array for every Clifford grade."""

    grades: tuple[Array, ...]


class CliffordGradeRepresentation(StrictModule, NonTrainableState):
    """Complete-grade channel layout for a real Clifford field."""

    algebra: CliffordAlgebraSpec
    multiplicities: tuple[int, ...] = eqx.field(static=True)
    grade_layouts: tuple[CliffordBladeLayout, ...]
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: CliffordAlgebraSpec,
        multiplicities: Sequence[int],
        /,
    ):
        if not isinstance(algebra, CliffordAlgebraSpec):
            raise TypeError("algebra must be a CliffordAlgebraSpec.")
        resolved = tuple(int(value) for value in multiplicities)
        if len(resolved) != algebra.dimension + 1:
            raise ValueError(
                "Clifford grade multiplicities must contain one count per grade."
            )
        if any(value < 0 for value in resolved):
            raise ValueError("Clifford grade multiplicities must be nonnegative.")
        if sum(resolved) == 0:
            raise ValueError("Clifford grade representation cannot be empty.")
        layouts = tuple(
            CliffordBladeLayout.grades_layout(algebra, (grade,))
            for grade in range(algebra.dimension + 1)
        )
        self.algebra = algebra
        self.multiplicities = resolved
        self.grade_layouts = layouts
        self.representation_id = canonical_fingerprint(
            {
                "kind": "clifford-grade-representation-v1",
                "algebra": algebra.algebra_id,
                "orientation": algebra.orientation,
                "multiplicities": list(resolved),
                "layouts": [layout.layout_id for layout in layouts],
            }
        )

    @property
    def packed_size(self) -> int:
        return sum(
            multiplicity * comb(self.algebra.dimension, grade)
            for grade, multiplicity in enumerate(self.multiplicities)
        )

    @property
    def uniform_multiplicity(self) -> int | None:
        nonzero = {value for value in self.multiplicities if value > 0}
        return next(iter(nonzero)) if len(nonzero) == 1 else None

    def split(self, values: Array, /) -> CliffordGradeFeatures:
        array = jnp.asarray(values)
        if array.ndim < 1 or array.shape[-1] != self.packed_size:
            raise ValueError(
                f"Packed Clifford field must end in {self.packed_size} components."
            )
        offset = 0
        grades = []
        for grade, (multiplicity, layout) in enumerate(
            zip(self.multiplicities, self.grade_layouts)
        ):
            count = multiplicity * layout.blade_count
            grades.append(
                array[..., offset : offset + count].reshape(
                    array.shape[:-1] + (multiplicity, comb(self.algebra.dimension, grade))
                )
            )
            offset += count
        return CliffordGradeFeatures(tuple(grades))

    def join(self, features: CliffordGradeFeatures, /) -> Array:
        if not isinstance(features, CliffordGradeFeatures):
            raise TypeError("features must be CliffordGradeFeatures.")
        if len(features.grades) != self.algebra.dimension + 1:
            raise ValueError(
                "Clifford feature grade count does not match representation."
            )
        leading = None
        flattened = []
        dtype = None
        for grade, (values, multiplicity, layout) in enumerate(
            zip(features.grades, self.multiplicities, self.grade_layouts)
        ):
            array = jnp.asarray(values)
            expected = (multiplicity, layout.blade_count)
            if array.shape[-2:] != expected:
                raise ValueError(
                    f"Grade {grade} Clifford features must end in {expected}; "
                    f"got {array.shape}."
                )
            if leading is None:
                leading = array.shape[:-2]
                dtype = array.dtype
            elif array.shape[:-2] != leading:
                raise ValueError("Clifford grade features must share leading axes.")
            elif array.dtype != dtype:
                raise TypeError("Clifford grade features must share one dtype.")
            flattened.append(array.reshape(array.shape[:-2] + (-1,)))
        if dtype is None or leading is None:
            raise RuntimeError("Clifford grade representation lost all feature grades.")
        return jnp.concatenate(flattened, axis=-1)

    def validate_affine_normalization(
        self,
        scales: Sequence[float],
        offsets: Sequence[float],
        /,
    ) -> None:
        scale = np.asarray(scales, dtype=float)
        offset = np.asarray(offsets, dtype=float)
        expected = (self.packed_size,)
        if scale.shape != expected or offset.shape != expected:
            raise ValueError(
                "Clifford affine normalization must match the packed representation."
            )
        cursor = 0
        for grade, (multiplicity, layout) in enumerate(
            zip(self.multiplicities, self.grade_layouts)
        ):
            count = multiplicity * layout.blade_count
            scale_block = scale[cursor : cursor + count].reshape(
                (multiplicity, layout.blade_count)
            )
            offset_block = offset[cursor : cursor + count]
            if multiplicity and np.any(scale_block != scale_block[:, :1]):
                raise ValueError(
                    "Clifford scales must be constant over the blades of each channel."
                )
            if grade > 0 and np.any(offset_block != 0.0):
                raise ValueError(
                    "Non-scalar Clifford grades require zero affine offsets."
                )
            cursor += count

    def to_dict(self) -> dict[str, object]:
        return {
            "algebra": self.algebra.to_dict(),
            "multiplicities": list(self.multiplicities),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        /,
    ) -> "CliffordGradeRepresentation":
        algebra_value = value["algebra"]
        multiplicities = value["multiplicities"]
        if not isinstance(algebra_value, Mapping):
            raise TypeError("Serialized Clifford algebra must be a mapping.")
        if not isinstance(multiplicities, Sequence) or isinstance(
            multiplicities, (str, bytes)
        ):
            raise TypeError("Serialized Clifford multiplicities must be a sequence.")
        return cls(
            CliffordAlgebraSpec.from_dict(algebra_value),
            tuple(int(item) for item in multiplicities),
        )

    def transform(self, values: Array, action: MetricIsometryAction, /) -> Array:
        self.algebra.require_compatible(action.algebra)
        features = self.split(values)
        transformed = []
        for grade_values, multiplicity, layout in zip(
            features.grades, self.multiplicities, self.grade_layouts
        ):
            if multiplicity == 0:
                transformed.append(grade_values)
                continue
            transformed.append(CliffordOutermorphismPlan(action, layout)(grade_values))
        return self.join(CliffordGradeFeatures(tuple(transformed)))

    def o3_representation(self) -> O3Representation:
        """Return the equivalent O(3) scalar/vector/pseudovector layout."""
        if self.algebra.dimension != 3 or not self.algebra.positive_definite:
            raise ValueError("O3 adapter requires a Euclidean three-dimensional algebra.")
        return O3Representation(
            scalars=self.multiplicities[0],
            vectors=self.multiplicities[1],
            pseudovectors=self.multiplicities[2],
            pseudoscalars=self.multiplicities[3],
        )

    def to_o3(self, values: Array, /) -> Array:
        features = self.split(values)
        o3 = self.o3_representation()
        bivector = features.grades[2]
        pseudovector = self.algebra.orientation * jnp.stack(
            (bivector[..., 2], -bivector[..., 1], bivector[..., 0]),
            axis=-1,
        )
        leading = jnp.asarray(values).shape[:-1]
        return o3.join(
            O3Features(
                scalars=features.grades[0][..., 0],
                pseudoscalars=features.grades[3][..., 0],
                vectors=features.grades[1],
                pseudovectors=pseudovector,
                tensors=jnp.zeros(leading + (0, 3, 3), dtype=jnp.asarray(values).dtype),
                pseudotensors=jnp.zeros(
                    leading + (0, 3, 3), dtype=jnp.asarray(values).dtype
                ),
            )
        )

    def from_o3(self, values: Array, /) -> Array:
        o3 = self.o3_representation()
        features = o3.split(values)
        pseudovector = features.pseudovectors
        bivector = self.algebra.orientation * jnp.stack(
            (pseudovector[..., 2], -pseudovector[..., 1], pseudovector[..., 0]),
            axis=-1,
        )
        return self.join(
            CliffordGradeFeatures(
                (
                    features.scalars[..., None],
                    features.vectors,
                    bivector,
                    features.pseudoscalars[..., None],
                )
            )
        )


__all__ = ["CliffordGradeFeatures", "CliffordGradeRepresentation"]
