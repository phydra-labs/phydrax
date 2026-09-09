#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generic acquired or synthetic two-dimensional image assets."""

from __future__ import annotations

from dataclasses import dataclass, field

from .._fingerprint import canonical_fingerprint
from ..measurement import (
    MeasurementAsset,
    QuantityField,
    QuantitySpec,
    SamplingSemantics,
    SpatialSamplingKind,
    ValueKind,
    ValueLayout,
)
from ..units import UnitDefinition
from ._plane import ImagePlaneSupport


@dataclass(frozen=True, slots=True)
class ImageFieldSpec:
    """Physical quantity, component layout, and sampling semantics for an image."""

    quantity: QuantitySpec
    layout: ValueLayout
    sampling: SamplingSemantics
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, QuantitySpec):
            raise TypeError("quantity must be QuantitySpec.")
        if not isinstance(self.layout, ValueLayout):
            raise TypeError("layout must be ValueLayout.")
        if not isinstance(self.sampling, SamplingSemantics):
            raise TypeError("sampling must be SamplingSemantics.")
        object.__setattr__(
            self,
            "spec_id",
            canonical_fingerprint(
                {
                    "kind": "image-field-spec",
                    "quantity": self.quantity.quantity_id,
                    "layout": self.layout.layout_id,
                    "sampling": self.sampling.sampling_id,
                }
            ),
        )

    @classmethod
    def named(
        cls,
        name: str,
        unit: UnitDefinition,
        kind: ValueKind,
        component_shape: tuple[int, ...] = (),
        component_frame_id: str | None = None,
        /,
        *,
        namespace: str = "imaging",
        quantity_kind: str | None = None,
        compatibility_key: str | None = None,
        spatial_sampling: SpatialSamplingKind = SpatialSamplingKind.POINT,
        component_labels: tuple[str, ...] = (),
    ) -> ImageFieldSpec:
        resolved_kind = name if quantity_kind is None else quantity_kind
        return cls(
            QuantitySpec(
                namespace,
                name,
                resolved_kind,
                unit,
                (
                    f"{namespace}.{resolved_kind}"
                    if compatibility_key is None
                    else compatibility_key
                ),
            ),
            ValueLayout(
                kind,
                component_shape,
                component_labels,
                component_frame_id,
            ),
            SamplingSemantics(spatial_sampling),
        )


@dataclass(frozen=True, slots=True)
class ImageAsset:
    """One typed image measurement on an explicit image-plane support."""

    measurement: MeasurementAsset
    image_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, MeasurementAsset):
            raise TypeError("measurement must be MeasurementAsset.")
        if not isinstance(self.measurement.field.support, ImagePlaneSupport):
            raise TypeError("ImageAsset requires an ImagePlaneSupport quantity field.")
        object.__setattr__(
            self,
            "image_id",
            canonical_fingerprint(
                {
                    "kind": "image-asset",
                    "measurement": self.measurement.content_id,
                }
            ),
        )

    @property
    def asset_id(self) -> str:
        return self.measurement.asset_id

    @property
    def field(self) -> QuantityField:
        return self.measurement.field

    @property
    def support(self) -> ImagePlaneSupport:
        support = self.measurement.field.support
        if not isinstance(support, ImagePlaneSupport):
            raise TypeError("ImageAsset support is not ImagePlaneSupport.")
        return support


__all__ = ["ImageAsset", "ImageFieldSpec"]
