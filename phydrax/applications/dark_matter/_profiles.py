#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import CompiledGeometry, Sphere
from ...qualification import ReferenceArtifactManifest


_GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _target_ids(values: Sequence[str], /) -> tuple[str, ...]:
    identifiers = tuple(_identifier(value, "target ID") for value in values)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("target_ids must be non-empty and unique.")
    return identifiers


def _requested_use(
    commercial_use: bool,
    redistribution: bool,
    training_use: bool,
    export: bool,
    /,
) -> tuple[tuple[str, bool], ...]:
    values = (commercial_use, redistribution, training_use, export)
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("Requested-use flags must be booleans.")
    return (
        ("commercial_use", commercial_use),
        ("redistribution", redistribution),
        ("training_use", training_use),
        ("export", export),
    )


def _manifest(
    value: ReferenceArtifactManifest,
    requested_use: tuple[tuple[str, bool], ...],
    /,
) -> ReferenceArtifactManifest:
    if not isinstance(value, ReferenceArtifactManifest):
        raise TypeError("Radial profile data require a ReferenceArtifactManifest.")
    value.require_rights(**dict(requested_use))
    return value


def _external_array(value: ArrayLike, /) -> Array:
    return jax.lax.stop_gradient(jnp.asarray(value, dtype=jnp.float64))


class RadialProfileEvaluation(StrictModule):
    """Body-frame SI fields at one or more Cartesian positions."""

    radius_m: Array
    mass_density_kg_m3: Array
    target_number_densities_m3: Array
    temperature_K: Array
    enclosed_mass_kg: Array
    gravitational_acceleration_m_s2: Array
    layer_index: Array
    inside: Array
    finite: Array
    successful: Array
    profile_id: str = eqx.field(static=True)


class LayeredTerrestrialProfile(StrictModule, NonTrainableState):
    """Manifest-qualified concentric, piecewise-homogeneous body profile.

    ``outer_radii_m`` gives each layer's outer sphere. Densities and temperatures
    are constant within a layer; no named Earth model or detector composition is
    implied. The supplied arrays are stopped from gradient propagation because
    they are external products rather than inferred model parameters.
    """

    outer_radii_m: Array
    mass_densities_kg_m3: Array
    target_number_densities_m3: Array
    temperatures_K: Array
    target_ids: tuple[str, ...] = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    source: ReferenceArtifactManifest
    requested_use: tuple[tuple[str, bool], ...] = eqx.field(static=True)
    boundaries: tuple[CompiledGeometry, ...]
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        outer_radii_m: ArrayLike,
        mass_densities_kg_m3: ArrayLike,
        target_number_densities_m3: ArrayLike,
        temperatures_K: ArrayLike,
        target_ids: Sequence[str],
        source: ReferenceArtifactManifest,
        /,
        *,
        frame_id: str,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        radii = np.asarray(outer_radii_m, dtype=np.float64)
        densities = np.asarray(mass_densities_kg_m3, dtype=np.float64)
        number = np.asarray(target_number_densities_m3, dtype=np.float64)
        temperatures = np.asarray(temperatures_K, dtype=np.float64)
        targets = _target_ids(target_ids)
        frame = _identifier(frame_id, "frame_id")
        requested = _requested_use(commercial_use, redistribution, training_use, export)
        source_ = _manifest(source, requested)
        if radii.ndim != 1 or radii.size == 0:
            raise ValueError("outer_radii_m must be a non-empty vector.")
        layer_count = radii.size
        if (
            densities.shape != (layer_count,)
            or number.shape != (layer_count, len(targets))
            or temperatures.shape != (layer_count,)
        ):
            raise ValueError("Layer fields must match the layer and target axes exactly.")
        if (
            np.any(~np.isfinite(radii))
            or np.any(~np.isfinite(densities))
            or np.any(~np.isfinite(number))
            or np.any(~np.isfinite(temperatures))
            or np.any(radii <= 0.0)
            or np.any(np.diff(radii) <= 0.0)
            or np.any(densities < 0.0)
            or np.any(number < 0.0)
            or np.any(temperatures <= 0.0)
        ):
            raise ValueError("Layer radii and physical fields are outside their support.")
        self.outer_radii_m = _external_array(radii)
        self.mass_densities_kg_m3 = _external_array(densities)
        self.target_number_densities_m3 = _external_array(number)
        self.temperatures_K = _external_array(temperatures)
        self.target_ids = targets
        self.frame_id = frame
        self.source = source_
        self.requested_use = requested
        self.boundaries = tuple(
            Sphere(
                (0.0, 0.0, 0.0),
                float(radius),
                feature_id=f"{frame}:layer-boundary:{index}",
            ).compile()
            for index, radius in enumerate(radii)
        )
        self.profile_id = canonical_fingerprint(
            {
                "kind": "layered-terrestrial-profile",
                "frame": frame,
                "target_ids": targets,
                "source": source_.manifest_id,
                "requested_use": requested,
                "fields": array_tree_fingerprint(
                    {
                        "outer_radii_m": self.outer_radii_m,
                        "mass_densities_kg_m3": self.mass_densities_kg_m3,
                        "target_number_densities_m3": self.target_number_densities_m3,
                        "temperatures_K": self.temperatures_K,
                    }
                ),
            }
        )

    @property
    def radius_m(self) -> Array:
        return self.outer_radii_m[-1]

    @property
    def num_targets(self) -> int:
        return len(self.target_ids)

    @property
    def num_layers(self) -> int:
        return self.outer_radii_m.shape[0]

    def evaluate(self, positions_m: ArrayLike, /) -> RadialProfileEvaluation:
        positions = jnp.asarray(positions_m, dtype=self.outer_radii_m.dtype)
        if positions.shape[-1:] != (3,):
            raise ValueError(
                "Body-frame positions must have a trailing Cartesian axis of 3."
            )
        radius = jnp.sqrt(jnp.sum(positions * positions, axis=-1))
        inside = (
            self.boundaries[-1]
            .signed_distance(positions.reshape((-1, 3)))
            .reshape(radius.shape)
            <= 0.0
        )
        index = jnp.clip(
            jnp.searchsorted(self.outer_radii_m, radius, side="left"),
            0,
            self.num_layers - 1,
        )
        mass_density = jnp.where(inside, self.mass_densities_kg_m3[index], 0.0)
        number_density = jnp.where(
            inside[..., None], self.target_number_densities_m3[index], 0.0
        )
        temperature = self.temperatures_K[index]
        inner_radii = jnp.concatenate(
            (jnp.zeros((1,), dtype=radius.dtype), self.outer_radii_m[:-1])
        )
        shell_masses = (
            (4.0 * jnp.pi / 3.0)
            * self.mass_densities_kg_m3
            * (self.outer_radii_m**3 - inner_radii**3)
        )
        cumulative = jnp.cumsum(shell_masses)
        previous = jnp.where(index > 0, cumulative[jnp.maximum(index - 1, 0)], 0.0)
        local_radius = jnp.minimum(radius, self.radius_m)
        enclosed = previous + (
            (4.0 * jnp.pi / 3.0)
            * self.mass_densities_kg_m3[index]
            * (local_radius**3 - inner_radii[index] ** 3)
        )
        enclosed = jnp.where(inside, enclosed, cumulative[-1])
        safe_radius = jnp.maximum(radius, jnp.finfo(radius.dtype).tiny)
        gravity = _GRAVITATIONAL_CONSTANT_M3_KG_S2 * enclosed / safe_radius**2
        gravity = jnp.where(radius > 0.0, gravity, 0.0)
        finite = (
            jnp.isfinite(radius)
            & jnp.isfinite(mass_density)
            & jnp.all(jnp.isfinite(number_density), axis=-1)
            & jnp.isfinite(temperature)
            & jnp.isfinite(enclosed)
            & jnp.isfinite(gravity)
        )
        return RadialProfileEvaluation(
            radius,
            mass_density,
            number_density,
            temperature,
            enclosed,
            gravity,
            jnp.where(inside, index, -1).astype(jnp.int32),
            inside,
            finite,
            finite,
            self.profile_id,
        )


class SmoothStellarRadialProfile(StrictModule, NonTrainableState):
    """Manifest-qualified C1 radial stellar table in SI coordinates.

    Values use a smoothstep interpolant between explicit knots. The enclosed-mass
    column is supplied rather than reconstructed from density, so the source's
    hydrostatic convention remains auditable and gravity is not silently altered.
    """

    radii_m: Array
    mass_densities_kg_m3: Array
    target_number_densities_m3: Array
    temperatures_K: Array
    enclosed_masses_kg: Array
    target_ids: tuple[str, ...] = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    source: ReferenceArtifactManifest
    requested_use: tuple[tuple[str, bool], ...] = eqx.field(static=True)
    boundary: CompiledGeometry
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        radii_m: ArrayLike,
        mass_densities_kg_m3: ArrayLike,
        target_number_densities_m3: ArrayLike,
        temperatures_K: ArrayLike,
        enclosed_masses_kg: ArrayLike,
        target_ids: Sequence[str],
        source: ReferenceArtifactManifest,
        /,
        *,
        frame_id: str,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        radii = np.asarray(radii_m, dtype=np.float64)
        densities = np.asarray(mass_densities_kg_m3, dtype=np.float64)
        number = np.asarray(target_number_densities_m3, dtype=np.float64)
        temperatures = np.asarray(temperatures_K, dtype=np.float64)
        masses = np.asarray(enclosed_masses_kg, dtype=np.float64)
        targets = _target_ids(target_ids)
        frame = _identifier(frame_id, "frame_id")
        requested = _requested_use(commercial_use, redistribution, training_use, export)
        source_ = _manifest(source, requested)
        if radii.ndim != 1 or radii.size < 2:
            raise ValueError(
                "radii_m must contain at least the center and surface knots."
            )
        knot_count = radii.size
        if (
            densities.shape != (knot_count,)
            or number.shape != (knot_count, len(targets))
            or temperatures.shape != (knot_count,)
            or masses.shape != (knot_count,)
        ):
            raise ValueError(
                "Stellar fields must match the radial and target axes exactly."
            )
        if (
            radii[0] != 0.0
            or np.any(~np.isfinite(radii))
            or np.any(np.diff(radii) <= 0.0)
            or np.any(~np.isfinite(densities))
            or np.any(densities < 0.0)
            or np.any(~np.isfinite(number))
            or np.any(number < 0.0)
            or np.any(~np.isfinite(temperatures))
            or np.any(temperatures <= 0.0)
            or np.any(~np.isfinite(masses))
            or np.any(masses < 0.0)
            or np.any(np.diff(masses) < 0.0)
            or masses[0] != 0.0
            or masses[-1] <= 0.0
        ):
            raise ValueError("Stellar radial coordinates or physical fields are invalid.")
        self.radii_m = _external_array(radii)
        self.mass_densities_kg_m3 = _external_array(densities)
        self.target_number_densities_m3 = _external_array(number)
        self.temperatures_K = _external_array(temperatures)
        self.enclosed_masses_kg = _external_array(masses)
        self.target_ids = targets
        self.frame_id = frame
        self.source = source_
        self.requested_use = requested
        self.boundary = Sphere(
            (0.0, 0.0, 0.0), float(radii[-1]), feature_id=f"{frame}:surface"
        ).compile()
        self.profile_id = canonical_fingerprint(
            {
                "kind": "smooth-stellar-radial-profile",
                "frame": frame,
                "target_ids": targets,
                "source": source_.manifest_id,
                "requested_use": requested,
                "fields": array_tree_fingerprint(
                    {
                        "radii_m": self.radii_m,
                        "mass_densities_kg_m3": self.mass_densities_kg_m3,
                        "target_number_densities_m3": self.target_number_densities_m3,
                        "temperatures_K": self.temperatures_K,
                        "enclosed_masses_kg": self.enclosed_masses_kg,
                    }
                ),
            }
        )

    @property
    def radius_m(self) -> Array:
        return self.radii_m[-1]

    @property
    def total_mass_kg(self) -> Array:
        return self.enclosed_masses_kg[-1]

    @property
    def num_targets(self) -> int:
        return len(self.target_ids)

    def _interpolate(self, values: Array, radius: Array, /) -> Array:
        index = jnp.clip(
            jnp.searchsorted(self.radii_m, radius, side="right") - 1,
            0,
            self.radii_m.shape[0] - 2,
        )
        left = self.radii_m[index]
        right = self.radii_m[index + 1]
        coordinate = jnp.clip((radius - left) / (right - left), 0.0, 1.0)
        blend = coordinate * coordinate * (3.0 - 2.0 * coordinate)
        difference = values[index + 1] - values[index]
        if values.ndim == 2:
            return values[index] + blend[..., None] * difference
        return values[index] + blend * difference

    def _interpolate_enclosed_mass(self, radius: Array, /) -> Array:
        index = jnp.clip(
            jnp.searchsorted(self.radii_m, radius, side="right") - 1,
            0,
            self.radii_m.shape[0] - 2,
        )
        left = self.radii_m[index]
        right = self.radii_m[index + 1]
        coordinate = jnp.clip((radius - left) / (right - left), 0.0, 1.0)
        regular_center_blend = coordinate**3 * (4.0 - 3.0 * coordinate)
        ordinary_blend = coordinate**2 * (3.0 - 2.0 * coordinate)
        blend = jnp.where(index == 0, regular_center_blend, ordinary_blend)
        return self.enclosed_masses_kg[index] + blend * (
            self.enclosed_masses_kg[index + 1] - self.enclosed_masses_kg[index]
        )

    def evaluate(self, positions_m: ArrayLike, /) -> RadialProfileEvaluation:
        positions = jnp.asarray(positions_m, dtype=self.radii_m.dtype)
        if positions.shape[-1:] != (3,):
            raise ValueError(
                "Body-frame positions must have a trailing Cartesian axis of 3."
            )
        radius = jnp.sqrt(jnp.sum(positions * positions, axis=-1))
        inside = (
            self.boundary.signed_distance(positions.reshape((-1, 3))).reshape(
                radius.shape
            )
            <= 0.0
        )
        clipped = jnp.minimum(radius, self.radius_m)
        density = self._interpolate(self.mass_densities_kg_m3, clipped)
        number = self._interpolate(self.target_number_densities_m3, clipped)
        temperature = self._interpolate(self.temperatures_K, clipped)
        enclosed = self._interpolate_enclosed_mass(clipped)
        density = jnp.where(inside, density, 0.0)
        number = jnp.where(inside[..., None], number, 0.0)
        enclosed = jnp.where(inside, enclosed, self.total_mass_kg)
        safe_radius = jnp.maximum(radius, jnp.finfo(radius.dtype).tiny)
        gravity = _GRAVITATIONAL_CONSTANT_M3_KG_S2 * enclosed / safe_radius**2
        gravity = jnp.where(radius > 0.0, gravity, 0.0)
        finite = (
            jnp.isfinite(radius)
            & jnp.isfinite(density)
            & jnp.all(jnp.isfinite(number), axis=-1)
            & jnp.isfinite(temperature)
            & jnp.isfinite(enclosed)
            & jnp.isfinite(gravity)
        )
        return RadialProfileEvaluation(
            radius,
            density,
            number,
            temperature,
            enclosed,
            gravity,
            jnp.full(radius.shape, -1, dtype=jnp.int32),
            inside,
            finite,
            finite,
            self.profile_id,
        )


RadialBodyProfile = LayeredTerrestrialProfile | SmoothStellarRadialProfile


__all__ = [
    "LayeredTerrestrialProfile",
    "RadialBodyProfile",
    "RadialProfileEvaluation",
    "SmoothStellarRadialProfile",
]
