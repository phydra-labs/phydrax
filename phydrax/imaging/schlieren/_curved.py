#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Camera-space lowering of continuously refracted ray fans."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...measurement import PreparedQuantityField, QuantitySpec, ValueKind, ValueLayout
from ...optics.geometric._graded_index import (
    GradedIndexRayResult,
    PreparedGradedIndexRay,
    RayFanPlan,
    RayFanResult,
)
from ...units import RADIAN
from .._plane import ImagePlaneSupport


class CurvedSchlierenResult(StrictModule, NonTrainableState):
    prediction: PreparedQuantityField
    rays: GradedIndexRayResult
    fan: RayFanResult
    successful: Array


@dataclass(frozen=True, slots=True)
class CurvedSchlierenPlan:
    rays: PreparedGradedIndexRay
    image_support: ImagePlaneSupport
    transverse_basis: np.ndarray
    deflection_quantity: QuantitySpec
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        basis = np.array(self.transverse_basis, dtype=float, copy=True)
        if basis.shape != (2, 3):
            raise ValueError("transverse_basis must have shape (2, 3).")
        if self.deflection_quantity.unit != RADIAN:
            raise ValueError("deflection_quantity must use radians.")
        if not isinstance(self.image_support, ImagePlaneSupport):
            raise TypeError("image_support must be ImagePlaneSupport.")
        basis.setflags(write=False)
        object.__setattr__(self, "transverse_basis", basis)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "curved-schlieren-plan",
                    "rays": self.rays.plan_id,
                    "image": self.image_support.support_id,
                    "basis": array_tree_fingerprint(basis),
                    "quantity": self.deflection_quantity.quantity_id,
                }
            ),
        )

    def evaluate(
        self, positions: ArrayLike, directions: ArrayLike, /
    ) -> CurvedSchlierenResult:
        result = self.rays.integrate(positions, directions)
        if (
            result.state.positions.shape[0]
            != self.image_support.image_shape[0] * self.image_support.image_shape[1]
        ):
            raise ValueError("Ray count must equal flattened image samples.")
        basis = jnp.asarray(self.transverse_basis, dtype=result.final_directions.dtype)
        deflection = contract(
            "ij,rj->ri", basis, result.final_directions - result.initial_directions
        )
        shape = self.image_support.image_shape
        values = deflection.reshape(shape + (2,))
        valid = result.state.valid.reshape(shape)
        layout = ValueLayout(
            ValueKind.VECTOR,
            (2,),
            ("row", "column"),
            self.image_support.detector_frame_id or "image-row-column",
        )
        prediction = PreparedQuantityField(
            values,
            valid,
            standard_uncertainty=None,
            quantity_id=self.deflection_quantity.quantity_id,
            compatibility_id=self.deflection_quantity.compatibility_id,
            layout_id=layout.layout_id,
            support_id=self.image_support.support_id,
            sampling_id=canonical_fingerprint(
                {"kind": "curved-schlieren-sampling", "plan": self.plan_id}
            ),
            unit_id=self.deflection_quantity.unit.unit_id,
            field_id=f"{self.plan_id}:deflection",
        )
        fan = RayFanPlan(self.transverse_basis).evaluate(result)
        return CurvedSchlierenResult(
            prediction, result, fan, result.evidence.successful & fan.successful
        )


__all__ = ["CurvedSchlierenPlan", "CurvedSchlierenResult"]
