#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Path-integrated Schlieren and background-oriented image formation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..measurement import (
    PreparedQuantityField,
    QuantitySpec,
    RaySampleSupport,
    ValueKind,
    ValueLayout,
)
from ..units import conversion_factor, derived_unit, ONE, RADIAN, UnitDefinition
from ._asset import ImageAsset
from ._plane import ImagePlaneSupport
from ._sampling import backward_warp


class SchlierenMethod(StrEnum):
    KNIFE_EDGE = "knife_edge"
    BACKGROUND_ORIENTED = "background_oriented"


@dataclass(frozen=True, slots=True)
class SchlierenImagePair:
    reference: ImageAsset
    observed: ImageAsset
    method: SchlierenMethod
    calibration_id: str
    pair_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ImageAsset) or not isinstance(
            self.observed, ImageAsset
        ):
            raise TypeError("reference and observed must be ImageAsset values.")
        if self.reference.support.support_id != self.observed.support.support_id:
            raise ValueError("Schlieren image pairs must share one image support.")
        if (
            self.reference.field.quantity.compatibility_id
            != self.observed.field.quantity.compatibility_id
            or self.reference.field.layout.layout_id
            != self.observed.field.layout.layout_id
        ):
            raise ValueError("Schlieren image quantities and layouts must be compatible.")
        if not isinstance(self.method, SchlierenMethod):
            raise TypeError("method must be SchlierenMethod.")
        calibration = str(self.calibration_id).strip()
        if not calibration or calibration != self.calibration_id:
            raise ValueError("calibration_id must be canonical non-empty text.")
        object.__setattr__(
            self,
            "pair_id",
            canonical_fingerprint(
                {
                    "kind": "schlieren-image-pair",
                    "reference": self.reference.image_id,
                    "observed": self.observed.image_id,
                    "method": self.method.value,
                    "calibration": calibration,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class GladstoneDaleRelation:
    coefficient: float
    coefficient_unit: UnitDefinition
    medium_id: str
    reference_refractive_index: float = 1.0
    relation_id: str = field(init=False)

    def __post_init__(self) -> None:
        coefficient = float(self.coefficient)
        reference = float(self.reference_refractive_index)
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("coefficient must be finite and positive.")
        if not isinstance(self.coefficient_unit, UnitDefinition):
            raise TypeError("coefficient_unit must be UnitDefinition.")
        medium = str(self.medium_id).strip()
        if not medium or medium != self.medium_id:
            raise ValueError("medium_id must be canonical non-empty text.")
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference_refractive_index must be finite and positive.")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "medium_id", medium)
        object.__setattr__(self, "reference_refractive_index", reference)
        object.__setattr__(
            self,
            "relation_id",
            canonical_fingerprint(
                {
                    "kind": "gladstone-dale-relation",
                    "coefficient": coefficient,
                    "unit": self.coefficient_unit.unit_id,
                    "medium": medium,
                    "reference_index": reference,
                }
            ),
        )


class SchlierenDeflectionResult(StrictModule, NonTrainableState):
    prediction: PreparedQuantityField
    deflection_rc: Array
    ray_valid: Array
    coverage_complete: Array
    small_angle_valid: Array
    finite: Array
    successful: Array


class SchlierenDeflectionPlan(StrictModule, NonTrainableState):
    """Fixed straight-ray quadrature for transverse optical deflection."""

    ray_active: Array
    image_support: ImagePlaneSupport
    segment_lengths: Array
    transverse_basis: Array
    gradient_quantity: QuantitySpec = eqx.field(static=True)
    deflection_quantity: QuantitySpec = eqx.field(static=True)
    relation: GladstoneDaleRelation | None = eqx.field(static=True)
    ray_support_id: str = eqx.field(static=True)
    deflection_layout_id: str = eqx.field(static=True)
    sampling_id: str = eqx.field(static=True)
    gradient_frame_id: str = eqx.field(static=True)
    dimensional_factor: float = eqx.field(static=True)
    small_angle_limit: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rays: RaySampleSupport,
        image_support: ImagePlaneSupport,
        segment_lengths: ArrayLike,
        transverse_basis: ArrayLike,
        gradient_quantity: QuantitySpec,
        deflection_quantity: QuantitySpec,
        /,
        *,
        relation: GladstoneDaleRelation | None = None,
        gradient_frame_id: str,
        small_angle_limit: float = 0.05,
    ):
        if not isinstance(rays, RaySampleSupport):
            raise TypeError("rays must be RaySampleSupport.")
        if not isinstance(image_support, ImagePlaneSupport):
            raise TypeError("image_support must be ImagePlaneSupport.")
        if (
            rays.sample_shape[0]
            != image_support.image_shape[0] * image_support.image_shape[1]
        ):
            raise ValueError("Ray count must equal the flattened image sample count.")
        if not isinstance(gradient_quantity, QuantitySpec) or not isinstance(
            deflection_quantity, QuantitySpec
        ):
            raise TypeError("gradient and deflection quantities must be QuantitySpec.")
        if deflection_quantity.unit != RADIAN:
            raise ValueError("deflection_quantity must use radians.")
        gradient_frame = str(gradient_frame_id).strip()
        if (
            not gradient_frame
            or gradient_frame != gradient_frame_id
            or gradient_frame != rays.coordinate_contract.reference_frame
        ):
            raise ValueError(
                "gradient_frame_id must equal the ray coordinate reference frame."
            )
        if relation is not None and not isinstance(relation, GladstoneDaleRelation):
            raise TypeError("relation must be GladstoneDaleRelation or None.")
        components = [
            (gradient_quantity.unit, 1),
            (rays.coordinate_contract.length_unit, 1),
        ]
        if relation is not None:
            components.append((relation.coefficient_unit, 1))
        closure_unit = derived_unit("schlieren-dimensional-closure", components)
        if closure_unit.dimension != ONE.dimension:
            raise ValueError(
                "Gradient, path length, and optional Gladstone-Dale coefficient must close dimensionlessly."
            )
        dimensional_factor = float(conversion_factor(closure_unit, ONE))
        lengths = np.asarray(segment_lengths, dtype=float)
        if lengths.ndim != 2 or lengths.shape[0] != rays.sample_shape[0]:
            raise ValueError(
                "segment_lengths must have shape (ray_count, segment_capacity)."
            )
        if not np.all(np.isfinite(lengths)) or np.any(lengths < 0.0):
            raise ValueError("segment_lengths must be finite and non-negative.")
        basis = np.asarray(transverse_basis, dtype=float)
        if basis.shape == (2, 3):
            basis = np.broadcast_to(basis, (rays.sample_shape[0], 2, 3)).copy()
        if basis.shape != (rays.sample_shape[0], 2, 3) or not np.all(np.isfinite(basis)):
            raise ValueError(
                "transverse_basis must have shape (ray_count, 2, 3) or (2, 3)."
            )
        gram = np.einsum("rij,rkj->rik", basis, basis)
        tolerance = 128.0 * np.finfo(basis.dtype).eps
        if not np.allclose(gram, np.eye(2)[None], atol=tolerance, rtol=0.0):
            raise ValueError("Every transverse basis must be orthonormal.")
        limit = float(small_angle_limit)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("small_angle_limit must be finite and positive.")
        layout = ValueLayout(
            ValueKind.VECTOR,
            (2,),
            ("row", "column"),
            image_support.detector_frame_id or "image-row-column",
        )
        self.ray_active = jnp.asarray(rays.active_mask)
        self.image_support = image_support
        self.segment_lengths = jnp.asarray(lengths)
        self.transverse_basis = jnp.asarray(basis)
        self.gradient_quantity = gradient_quantity
        self.deflection_quantity = deflection_quantity
        self.relation = relation
        self.ray_support_id = rays.support_id
        self.deflection_layout_id = layout.layout_id
        self.sampling_id = canonical_fingerprint(
            {
                "kind": "schlieren-path-integral-sampling",
                "rays": rays.support_id,
                "lengths": array_tree_fingerprint(lengths),
            }
        )
        self.gradient_frame_id = gradient_frame
        self.dimensional_factor = dimensional_factor
        self.small_angle_limit = limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "schlieren-deflection-plan",
                "rays": rays.support_id,
                "image": image_support.support_id,
                "lengths": array_tree_fingerprint(lengths),
                "basis": array_tree_fingerprint(basis),
                "gradient": gradient_quantity.quantity_id,
                "deflection": deflection_quantity.quantity_id,
                "relation": None if relation is None else relation.relation_id,
                "gradient_frame": gradient_frame,
                "dimensional_factor": dimensional_factor,
                "small_angle_limit": limit,
            }
        )

    def evaluate(self, gradient_samples: ArrayLike, /) -> SchlierenDeflectionResult:
        gradients = jnp.asarray(gradient_samples, dtype=self.segment_lengths.dtype)
        expected = self.segment_lengths.shape + (3,)
        if gradients.shape != expected:
            raise ValueError(f"gradient_samples must have shape {expected}.")
        active_segments = self.segment_lengths > 0.0
        finite_by_ray = jnp.all(
            jnp.where(active_segments[..., None], jnp.isfinite(gradients), True),
            axis=(1, 2),
        )
        safe_gradients = jnp.where(active_segments[..., None], gradients, 0.0)
        integrated = contract("rs,rsj->rj", self.segment_lengths, safe_gradients)
        deflection = contract("rij,rj->ri", self.transverse_basis, integrated)
        if self.relation is not None:
            deflection = (
                deflection
                * self.relation.coefficient
                / self.relation.reference_refractive_index
            )
        deflection = deflection * self.dimensional_factor
        ray_has_support = jnp.any(active_segments, axis=1)
        ray_valid = self.ray_active & ray_has_support & finite_by_ray
        shape = self.image_support.image_shape
        deflection_image = deflection.reshape(shape + (2,))
        valid_image = ray_valid.reshape(shape)
        deflection_image = jnp.where(valid_image[..., None], deflection_image, 0.0)

        prediction = PreparedQuantityField(
            deflection_image,
            valid_image,
            standard_uncertainty=None,
            quantity_id=self.deflection_quantity.quantity_id,
            compatibility_id=self.deflection_quantity.compatibility_id,
            layout_id=self.deflection_layout_id,
            support_id=self.image_support.support_id,
            sampling_id=self.sampling_id,
            unit_id=self.deflection_quantity.unit.unit_id,
            field_id=f"{self.plan_id}:deflection",
        )
        magnitude = jnp.sqrt(jnp.sum(deflection_image * deflection_image, axis=-1))
        small_angle = jnp.all(
            jnp.where(valid_image, magnitude <= self.small_angle_limit, True)
        )
        coverage = jnp.all(~self.ray_active | ray_has_support)
        finite = jnp.all(finite_by_ray) & jnp.all(jnp.isfinite(deflection_image))
        successful = coverage & finite & jnp.any(valid_image)
        return SchlierenDeflectionResult(
            prediction,
            deflection_image,
            valid_image,
            coverage,
            small_angle,
            finite,
            successful,
        )


class SchlierenImageFormationResult(StrictModule, NonTrainableState):
    prediction: PreparedQuantityField
    displacement_rc: Array
    valid: Array
    finite: Array
    approximation_valid: Array
    successful: Array


class KnifeEdgeSchlierenPlan(StrictModule, NonTrainableState):
    reference_values: Array
    reference_valid: Array
    cutoff_direction_rc: Array
    support_id: str = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True)
    compatibility_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    sampling_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    contrast_gain_per_radian: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: ImageAsset,
        cutoff_direction_rc: ArrayLike,
        /,
        *,
        contrast_gain_per_radian: float,
    ):
        if not isinstance(reference, ImageAsset):
            raise TypeError("reference must be ImageAsset.")
        direction = np.asarray(cutoff_direction_rc, dtype=float)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("cutoff_direction_rc must be a finite two-vector.")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("cutoff_direction_rc must be nonzero.")
        gain = float(contrast_gain_per_radian)
        if not np.isfinite(gain):
            raise ValueError("contrast_gain_per_radian must be finite.")
        prepared = reference.field.prepare()
        self.reference_values = prepared.values
        self.reference_valid = prepared.valid_mask
        self.cutoff_direction_rc = jnp.asarray(direction / norm)
        self.support_id = prepared.support_id
        self.quantity_id = prepared.quantity_id
        self.compatibility_id = prepared.compatibility_id
        self.layout_id = prepared.layout_id
        self.sampling_id = prepared.sampling_id
        self.unit_id = prepared.unit_id
        self.contrast_gain_per_radian = gain
        self.plan_id = canonical_fingerprint(
            {
                "kind": "knife-edge-schlieren-plan",
                "reference": reference.image_id,
                "cutoff": (direction / norm).tolist(),
                "gain": gain,
            }
        )

    def evaluate(
        self, deflection: SchlierenDeflectionResult, /
    ) -> SchlierenImageFormationResult:
        if not isinstance(deflection, SchlierenDeflectionResult):
            raise TypeError("deflection must be SchlierenDeflectionResult.")
        if deflection.prediction.support_id != self.support_id:
            raise ValueError("Deflection and reference image supports differ.")
        reference = self.reference_values
        if reference.shape != deflection.ray_valid.shape:
            raise ValueError("Knife-edge reference image must be scalar.")
        contrast = self.contrast_gain_per_radian * contract(
            "...i,i->...", deflection.deflection_rc, self.cutoff_direction_rc
        )
        signal = reference * (1.0 + contrast)
        valid = deflection.ray_valid & self.reference_valid
        signal = jnp.where(valid, signal, 0.0)
        prediction = PreparedQuantityField(
            signal,
            valid,
            standard_uncertainty=None,
            quantity_id=self.quantity_id,
            compatibility_id=self.compatibility_id,
            layout_id=self.layout_id,
            support_id=self.support_id,
            sampling_id=self.sampling_id,
            unit_id=self.unit_id,
            field_id=f"{self.plan_id}:signal",
        )
        finite = jnp.all(jnp.isfinite(signal))
        nonnegative = jnp.all(jnp.where(valid, signal >= 0.0, True))
        approximation_valid = deflection.small_angle_valid & nonnegative
        successful = finite & approximation_valid & deflection.successful & jnp.any(valid)
        return SchlierenImageFormationResult(
            prediction,
            jnp.zeros(deflection.deflection_rc.shape, dtype=signal.dtype),
            valid,
            finite,
            approximation_valid,
            successful,
        )


class BackgroundOrientedSchlierenPlan(StrictModule, NonTrainableState):
    background_values: Array
    background_valid: Array
    displacement_pixels_per_radian: Array
    support_id: str = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True)
    compatibility_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    sampling_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        background: ImageAsset,
        displacement_pixels_per_radian: ArrayLike,
        /,
    ):
        if not isinstance(background, ImageAsset):
            raise TypeError("background must be ImageAsset.")
        scale = np.asarray(displacement_pixels_per_radian, dtype=float)
        if scale.shape not in {(), (2,)} or not np.all(np.isfinite(scale)):
            raise ValueError(
                "displacement_pixels_per_radian must be finite scalar or two-vector."
            )
        if np.all(scale == 0.0):
            raise ValueError(
                "displacement_pixels_per_radian must not be identically zero."
            )
        scale = np.broadcast_to(scale, (2,)).copy()
        prepared = background.field.prepare()
        self.background_values = prepared.values
        self.background_valid = prepared.valid_mask
        self.displacement_pixels_per_radian = jnp.asarray(scale)
        self.support_id = prepared.support_id
        self.quantity_id = prepared.quantity_id
        self.compatibility_id = prepared.compatibility_id
        self.layout_id = prepared.layout_id
        self.sampling_id = prepared.sampling_id
        self.unit_id = prepared.unit_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "background-oriented-schlieren-plan",
                "background": background.image_id,
                "scale": scale.tolist(),
            }
        )

    def evaluate(
        self, deflection: SchlierenDeflectionResult, /
    ) -> SchlierenImageFormationResult:
        if not isinstance(deflection, SchlierenDeflectionResult):
            raise TypeError("deflection must be SchlierenDeflectionResult.")
        if deflection.prediction.support_id != self.support_id:
            raise ValueError("Deflection and background image supports differ.")
        background = self.background_values
        if background.shape != deflection.ray_valid.shape:
            raise ValueError("BOS background image must be scalar.")
        displacement = deflection.deflection_rc * self.displacement_pixels_per_radian
        warped = backward_warp(
            background,
            displacement,
            valid_mask=self.background_valid,
        )
        valid = warped.valid & deflection.ray_valid
        values = jnp.where(valid, warped.values, 0.0)

        prediction = PreparedQuantityField(
            values,
            valid,
            standard_uncertainty=None,
            quantity_id=self.quantity_id,
            compatibility_id=self.compatibility_id,
            layout_id=self.layout_id,
            support_id=self.support_id,
            sampling_id=self.sampling_id,
            unit_id=self.unit_id,
            field_id=f"{self.plan_id}:image",
        )
        finite = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(displacement))
        successful = finite & deflection.successful & jnp.any(valid)
        return SchlierenImageFormationResult(
            prediction,
            displacement,
            valid,
            finite,
            deflection.small_angle_valid,
            successful,
        )


__all__ = [
    "BackgroundOrientedSchlierenPlan",
    "GladstoneDaleRelation",
    "KnifeEdgeSchlierenPlan",
    "SchlierenDeflectionPlan",
    "SchlierenDeflectionResult",
    "SchlierenImageFormationResult",
    "SchlierenImagePair",
    "SchlierenMethod",
]
