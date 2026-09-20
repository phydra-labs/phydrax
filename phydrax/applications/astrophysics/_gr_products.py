#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import (
    conversion_factor,
    derived_unit,
    HERTZ,
    RADIAN,
    UnitDefinition,
)
from ._photometry import ObservationDataProvenance


_STERADIAN = derived_unit("sr", ((RADIAN, 2),))
_STOKES_COMPONENTS = ("I", "Q", "U", "V")


def _boolean_mask(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype(np.bool_) or result.shape != shape:
        raise ValueError(f"{name} must be a boolean array with shape {shape}.")
    return result


def _required_labels(values: Sequence[str], count: int, name: str, /) -> tuple[str, ...]:
    labels = tuple(str(value).strip() for value in values)
    if (
        len(labels) != count
        or len(set(labels)) != count
        or any(not value for value in labels)
    ):
        raise ValueError(f"{name} must contain {count} unique non-empty labels.")
    return labels


class GRImageScreen(StrictModule, NonTrainableState):
    """A physical observer screen with per-pixel solid-angle quadrature.

    Coordinates are angular offsets from the optical axis. ``valid_mask`` marks
    pixels owned by the physical screen; invalid pixels never contribute to an
    integral. No display coordinates, color transforms, or rendering state are
    part of this object.
    """

    screen_coordinates: Array
    solid_angle: Array
    valid_mask: Array
    angular_unit: UnitDefinition
    solid_angle_unit: UnitDefinition
    pixel_shape: tuple[int, int] = eqx.field(static=True)
    screen_id: str = eqx.field(static=True)

    def __init__(
        self,
        screen_coordinates: ArrayLike,
        solid_angle: ArrayLike,
        valid_mask: ArrayLike,
        /,
        *,
        angular_unit: UnitDefinition = RADIAN,
        solid_angle_unit: UnitDefinition = _STERADIAN,
    ):
        if not isinstance(angular_unit, UnitDefinition) or not isinstance(
            solid_angle_unit, UnitDefinition
        ):
            raise TypeError("Screen units must be UnitDefinition values.")
        conversion_factor(angular_unit, RADIAN)
        conversion_factor(solid_angle_unit, _STERADIAN)
        coordinates = np.asarray(screen_coordinates)
        weights = np.asarray(solid_angle)
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError("Screen coordinates must have shape (ny, nx, 2).")
        shape = coordinates.shape[:-1]
        if shape[0] == 0 or shape[1] == 0 or weights.shape != shape:
            raise ValueError("Screen solid angles must match a nonempty pixel grid.")
        mask = _boolean_mask(valid_mask, shape, "Screen valid_mask")
        if (
            np.any(~np.isfinite(coordinates))
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or np.any(mask & (weights <= 0.0))
        ):
            raise ValueError(
                "Screen coordinates must be finite and owned pixels need positive finite solid angle."
            )
        self.screen_coordinates = jnp.asarray(coordinates)
        self.solid_angle = jnp.asarray(weights)
        self.valid_mask = jnp.asarray(mask)
        self.angular_unit = angular_unit
        self.solid_angle_unit = solid_angle_unit
        self.pixel_shape = shape
        self.screen_id = canonical_fingerprint(
            {
                "kind": "gr-image-screen",
                "arrays": array_tree_fingerprint(
                    (self.screen_coordinates, self.solid_angle, self.valid_mask)
                ),
                "angular_unit": angular_unit.unit_id,
                "solid_angle_unit": solid_angle_unit.unit_id,
            }
        )

    @property
    def coordinates_in_radians(self) -> Array:
        factor = float(conversion_factor(self.angular_unit, RADIAN))
        return self.screen_coordinates * factor


class StokesImage(StrictModule, NonTrainableState):
    """A physical ``(I,Q,U,V)`` image on one immutable observer screen.

    ``redshift`` is the positive frequency ratio ν_observer/ν_emitter wherever
    ``redshift_valid`` is true. Lensing masks are named, disjoint physical ray
    classes. The contributing mask is the intersection of screen ownership,
    redshift validity, and membership in one lensing class.
    """

    stokes: Array
    screen: GRImageScreen
    redshift: Array
    redshift_valid: Array
    lensing_masks: Array
    valid_mask: Array
    frequency: Array
    intensity_unit: UnitDefinition
    flux_density_unit: UnitDefinition
    frequency_unit: UnitDefinition
    provenance: ObservationDataProvenance
    lensing_labels: tuple[str, ...] = eqx.field(static=True)
    stokes_components: tuple[str, str, str, str] = eqx.field(static=True)
    integration_unit_factor: float = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    def __init__(
        self,
        stokes: ArrayLike,
        screen: GRImageScreen,
        provenance: ObservationDataProvenance,
        /,
        *,
        frequency: ArrayLike,
        redshift: ArrayLike,
        redshift_valid: ArrayLike,
        lensing_masks: ArrayLike,
        lensing_labels: Sequence[str],
        intensity_unit: UnitDefinition,
        flux_density_unit: UnitDefinition,
        frequency_unit: UnitDefinition = HERTZ,
    ):
        if not isinstance(screen, GRImageScreen):
            raise TypeError("screen must be a GRImageScreen.")
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        if (
            not isinstance(intensity_unit, UnitDefinition)
            or not isinstance(flux_density_unit, UnitDefinition)
            or not isinstance(frequency_unit, UnitDefinition)
        ):
            raise TypeError("Image units must be UnitDefinition values.")
        conversion_factor(frequency_unit, HERTZ)
        integrated_unit = derived_unit(
            f"({intensity_unit.symbol})({screen.solid_angle_unit.symbol})",
            ((intensity_unit, 1), (screen.solid_angle_unit, 1)),
        )
        integration_factor = float(conversion_factor(integrated_unit, flux_density_unit))

        values = np.asarray(stokes)
        expected = (4, *screen.pixel_shape)
        if values.shape != expected or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"Stokes values must be a real array with shape {expected}.")
        shifts = np.asarray(redshift)
        if shifts.shape != screen.pixel_shape:
            raise ValueError("Image redshift must match the screen pixel grid.")
        redshift_mask = _boolean_mask(
            redshift_valid, screen.pixel_shape, "Image redshift_valid"
        )
        lensing = np.asarray(lensing_masks)
        if (
            lensing.dtype != np.dtype(np.bool_)
            or lensing.ndim != 3
            or lensing.shape[1:] != screen.pixel_shape
            or lensing.shape[0] == 0
        ):
            raise ValueError(
                "lensing_masks must be a nonempty boolean (class, ny, nx) array."
            )
        labels = _required_labels(
            lensing_labels, lensing.shape[0], "Image lensing_labels"
        )
        if np.any(np.sum(lensing, axis=0) > 1):
            raise ValueError("Image lensing masks must be disjoint.")
        frequency_host = np.asarray(frequency)
        if frequency_host.shape != ():
            raise ValueError("Image frequency must be scalar.")
        if (
            np.any(~np.isfinite(values))
            or np.any(~np.isfinite(shifts))
            or not np.isfinite(frequency_host)
            or float(frequency_host) <= 0.0
            or np.any(redshift_mask & (shifts <= 0.0))
            or np.any(shifts < 0.0)
        ):
            raise ValueError(
                "Image values must be finite; frequency and valid redshifts must be positive."
            )
        valid = np.asarray(screen.valid_mask) & redshift_mask & np.any(lensing, axis=0)
        intensity = values[0]
        polarized = np.sqrt(np.sum(values[1:] ** 2, axis=0))
        tolerance = 64.0 * np.finfo(values.dtype).eps * np.maximum(1.0, np.abs(intensity))
        if np.any(valid & ((intensity < 0.0) | (polarized > intensity + tolerance))):
            raise ValueError("Valid Stokes pixels must satisfy I ≥ 0 and I² ≥ Q²+U²+V².")

        self.stokes = jnp.asarray(values)
        self.screen = screen
        self.redshift = jnp.asarray(shifts)
        self.redshift_valid = jnp.asarray(redshift_mask)
        self.lensing_masks = jnp.asarray(lensing)
        self.valid_mask = jnp.asarray(valid)
        self.frequency = jnp.asarray(frequency_host, dtype=self.stokes.dtype)
        self.intensity_unit = intensity_unit
        self.flux_density_unit = flux_density_unit
        self.frequency_unit = frequency_unit
        self.provenance = provenance
        self.lensing_labels = labels
        self.stokes_components = _STOKES_COMPONENTS
        self.integration_unit_factor = integration_factor
        self.content_id = canonical_fingerprint(
            {
                "kind": "physical-stokes-image",
                "screen": screen.screen_id,
                "arrays": array_tree_fingerprint(
                    (
                        self.stokes,
                        self.redshift,
                        self.redshift_valid,
                        self.lensing_masks,
                        self.valid_mask,
                        self.frequency,
                    )
                ),
                "lensing_labels": list(labels),
                "intensity_unit": intensity_unit.unit_id,
                "flux_density_unit": flux_density_unit.unit_id,
                "frequency_unit": frequency_unit.unit_id,
                "provenance": provenance.provenance_id,
            }
        )

    @property
    def linear_polarization(self) -> Array:
        return self.stokes[1] + 1j * self.stokes[2]

    @property
    def polarized_intensity(self) -> Array:
        return jnp.sqrt(jnp.sum(self.stokes[1:] ** 2, axis=0))


__all__ = ["GRImageScreen", "StokesImage"]
