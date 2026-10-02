#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Polarized Fresnel facets and the UNIFIED rough-surface model.

Every reflection or refraction happens on a facet with normal ``m``: the
incident Jones vector is re-expressed on the facet frame ``(s', d × s')`` with
``s' = d × m`` normalized, multiplied by the complex ``(s, p)`` Fresnel
amplitudes of the refractive-interface owner (so total internal reflection
phases stay in the Jones vector), and renormalized; the outgoing candidate
carries ``s'`` as its transverse axis. Power fractions are
``R = |r_s J_s|² + |r_p J_p|²`` and the flux-weighted ``T``. Mirrors and paints
use the perfect-conductor amplitudes ``(r_s, r_p) = (-1, 1)``.

The UNIFIED model (Levin and Moisan 1996, after Nayar, Ikeuchi and Kanade
1991) follows the Geant4 convention: a ground surface samples a micro-facet
tilt ``alpha`` with density proportional to ``exp(-alpha² / 2 sigma_alpha²)
sin(alpha)`` on ``[0, pi/2)`` and a uniform azimuth; the facet Fresnel
powers decide reflection versus refraction, and a reflection is redirected as
a specular spike about the mean normal, a specular lobe about the facet, a
backscatter, or a Lambertian reflection with the declared probabilities.
Spike, backscatter, and Lambertian redirections apply the perfect-mirror
polarization transform about the normal that specularly produces the
outgoing direction. Sampled outcomes that leave on the wrong side of the mean
surface, and facets that face away from the photon, are resampled within a
fixed number of attempts; a lane without an accepted attempt reports an
invalid interaction. ``reflectivity`` is the probability that the surface
does not absorb the photon before the interaction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import get_args, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked, ConvertibleToArray, Dim, Float64, Int32, parse, Size
from ..geometric._interface import evaluate_refractive_interface
from ..geometric._nonsequential import (
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
)
from ._optical_monte_carlo import (
    _jones_to_incidence_frame,
    _unit_rows,
    OpticalSurfaceHit,
    OpticalSurfaceInteraction,
)


OpticalSurfaceFinish: TypeAlias = Literal[
    "polished", "ground", "polished-front-painted", "ground-front-painted"
]

_FINISHES = get_args(OpticalSurfaceFinish)
_POLISHED = _FINISHES.index("polished")
_GROUND = _FINISHES.index("ground")
_POLISHED_PAINTED = _FINISHES.index("polished-front-painted")
_GROUND_PAINTED = _FINISHES.index("ground-front-painted")
# Uniforms per facet attempt: tilt, tilt acceptance, azimuth, reflect or
# refract, reflection branch, and two for the Lambertian direction.
_ATTEMPT_UNIFORMS = 7
_PROBABILITY_TOLERANCE = 1e-12


class _SurfaceDim(Dim, minimum=1):
    """Physical surfaces addressed by surface identity."""


class _FacetResponse(StrictModule):
    """Fresnel candidates of lanes on one facet each (leading batch axes)."""

    reflected_directions: Array
    reflected_jones: Array
    transmitted_directions: Array
    transmitted_jones: Array
    axes: Array
    reflectance: Array
    transmittance: Array
    reflection_valid: Array
    transmission_valid: Array


type _Responder = Callable[[Array, Array], _FacetResponse]


def _facet_response(
    directions: Array,
    facet_normals: Array,
    incident_indices: Array,
    transmitted_indices: Array,
    jones_vectors: Array,
    frame_axes: Array,
    conducting: Array,
) -> _FacetResponse:
    """Polarized Fresnel (or perfect-conductor) response on facets ``m``."""

    interface = evaluate_refractive_interface(
        directions, facet_normals, incident_indices, transmitted_indices
    )
    axes, oblique = _unit_rows(jnp.cross(directions, facet_normals))
    axes = jnp.where(oblique[..., None], axes, frame_axes)
    local = _jones_to_incidence_frame(jones_vectors, frame_axes, directions, axes)
    mirror_amplitudes = jnp.asarray((-1.0 + 0.0j, 1.0 + 0.0j), dtype=jnp.complex128)
    reflection_amplitudes = jnp.where(
        conducting[..., None], mirror_amplitudes, interface.reflection_amplitudes
    )
    reflected = reflection_amplitudes * local
    transmitted = interface.transmission_amplitudes * local
    local_power = jnp.real(local * jnp.conj(local))
    reflectance = jnp.sum(jnp.real(reflected * jnp.conj(reflected)), axis=-1)
    transmittance = jnp.where(
        conducting, 0.0, jnp.sum(interface.transmittance * local_power, axis=-1)
    )
    transmitted_norm = jnp.sqrt(
        jnp.sum(jnp.real(transmitted * jnp.conj(transmitted)), axis=-1)
    )
    reflected_norm = jnp.sqrt(reflectance)
    return _FacetResponse(
        interface.reflected_directions,
        reflected / jnp.where(reflected_norm > 0.0, reflected_norm, 1.0)[..., None],
        interface.transmitted_directions,
        transmitted / jnp.where(transmitted_norm > 0.0, transmitted_norm, 1.0)[..., None],
        axes,
        reflectance,
        transmittance,
        interface.reflection_valid,
        interface.transmission_valid & ~conducting,
    )


def _per_surface(value: ConvertibleToArray, name: str, count: int, /) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real-valued.")
    if array.ndim > 1 or (array.ndim == 1 and array.shape != (count,)):
        raise ValueError(f"{name} must be a scalar or have one value per surface.")
    result = np.broadcast_to(array.astype(np.float64), (count,)).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


def _select(condition: Array, first: Array, second: Array) -> Array:
    """Lane-wise choice with ``condition`` broadcast over trailing components."""

    trailing = max(first.ndim, second.ndim) - condition.ndim
    return jnp.where(condition.reshape(condition.shape + (1,) * trailing), first, second)


def _first(values: Array, attempt: Array) -> Array:
    """Entry of the chosen attempt per lane from ``(attempts, lanes, ...)``."""

    return values[attempt, jnp.arange(attempt.shape[0])]


class UnifiedSurfaceModel(StrictModule, NonTrainableState):
    """UNIFIED polarized surfaces: one finish and parameter set per surface id.

    ``"polished"`` is the exact polarized Fresnel boundary (dielectrics) or a
    specular perfect mirror (mirror surfaces). ``"ground"`` samples
    micro-facets with Gaussian tilt ``sigma_alpha`` (radians) and redirects
    reflections by ``specular_spike``, ``specular_lobe``, ``backscatter``, and
    the Lambertian remainder. ``"polished-front-painted"`` and
    ``"ground-front-painted"`` are opaque specular and Lambertian paints.
    ``sigma_alpha`` and the branch probabilities apply only to ``"ground"``
    and are refused elsewhere; ``reflectivity`` in ``[0, 1]`` is the
    non-absorption probability of every finish. Surfaces of absorber or
    detector triangles must keep the default lossless polished declaration.
    """

    __strict_contract__ = True

    finish_codes: Int32[_SurfaceDim]
    sigma_alpha: Float64[_SurfaceDim]
    specular_spike: Float64[_SurfaceDim]
    specular_lobe: Float64[_SurfaceDim]
    backscatter: Float64[_SurfaceDim]
    reflectivity: Float64[_SurfaceDim]
    finishes: tuple[OpticalSurfaceFinish, ...] = eqx.field(static=True)
    surface_count: Size[_SurfaceDim] = eqx.field(static=True)
    facet_attempts: int = eqx.field(static=True)
    surface_model_id: str = eqx.field(static=True)

    def __init__(
        self,
        finishes: Sequence[OpticalSurfaceFinish],
        /,
        *,
        sigma_alpha: ConvertibleToArray = 0.0,
        specular_spike: ConvertibleToArray = 0.0,
        specular_lobe: ConvertibleToArray = 0.0,
        backscatter: ConvertibleToArray = 0.0,
        reflectivity: ConvertibleToArray = 1.0,
        facet_attempts: int = 32,
    ) -> None:
        if isinstance(finishes, str):
            raise TypeError("finishes must be a sequence with one finish per surface.")
        parsed = tuple(
            parse(value, OpticalSurfaceFinish, "finishes") for value in finishes
        )
        count = len(parsed)
        if count < 1:
            raise ValueError("At least one surface finish is required.")
        if type(facet_attempts) is not int:
            raise TypeError("facet_attempts must be an int.")
        if facet_attempts < 1:
            raise ValueError("facet_attempts must be positive.")
        sigma = _per_surface(sigma_alpha, "sigma_alpha", count)
        spike = _per_surface(specular_spike, "specular_spike", count)
        lobe = _per_surface(specular_lobe, "specular_lobe", count)
        back = _per_surface(backscatter, "backscatter", count)
        survival = _per_surface(reflectivity, "reflectivity", count)
        codes = np.asarray([_FINISHES.index(value) for value in parsed], dtype=np.int32)
        ground = codes == _GROUND
        if np.any(sigma < 0.0):
            raise ValueError("sigma_alpha must be non-negative.")
        if np.any(~ground & (sigma > 0.0)):
            raise ValueError("sigma_alpha applies only to the ground finish.")
        branches = np.stack((spike, lobe, back))
        if np.any((branches < 0.0) | (branches > 1.0)):
            raise ValueError("Reflection-branch probabilities must lie in [0, 1].")
        total = np.sum(branches, axis=0)
        if np.any(total > 1.0 + _PROBABILITY_TOLERANCE):
            raise ValueError(
                "specular_spike + specular_lobe + backscatter must not exceed one."
            )
        if np.any(~ground & (total > 0.0)):
            raise ValueError(
                "Reflection-branch probabilities apply only to the ground finish."
            )
        if np.any((survival < 0.0) | (survival > 1.0)):
            raise ValueError("reflectivity must lie in [0, 1].")
        self.finish_codes = jnp.asarray(codes)
        self.sigma_alpha = jnp.asarray(sigma)
        self.specular_spike = jnp.asarray(spike)
        self.specular_lobe = jnp.asarray(lobe)
        self.backscatter = jnp.asarray(back)
        self.reflectivity = jnp.asarray(survival)
        self.finishes = parsed
        self.surface_count = count
        self.facet_attempts = facet_attempts
        self.surface_model_id = canonical_fingerprint(
            {
                "kind": "unified-optical-surface",
                "finishes": list(parsed),
                "facet_attempts": facet_attempts,
                "content": array_tree_fingerprint(
                    {
                        "sigma_alpha": sigma,
                        "specular_spike": spike,
                        "specular_lobe": lobe,
                        "backscatter": back,
                        "reflectivity": survival,
                    }
                ),
            }
        )

    @checked
    def validate_surfaces(self, surfaces: NonSequentialSurfaceTable, /) -> None:
        if surfaces.surface_count != self.surface_count:
            raise ValueError(
                "UnifiedSurfaceModel needs one finish per surface id of the table."
            )
        kinds = np.full((self.surface_count,), -1, dtype=np.int32)
        kinds[np.asarray(surfaces.surface_ids)] = np.asarray(surfaces.surface_kinds)
        responsive = (kinds == int(NonSequentialSurfaceKind.DIELECTRIC)) | (
            kinds == int(NonSequentialSurfaceKind.MIRROR)
        )
        declared = (np.asarray(self.finish_codes) != _POLISHED) | (
            np.asarray(self.reflectivity) != 1.0
        )
        if np.any(~responsive & declared):
            raise ValueError(
                "Absorber, detector, and unused surface ids take no UNIFIED finish."
            )

    def interact(
        self, hit: OpticalSurfaceHit, keys: Array, /
    ) -> OpticalSurfaceInteraction:
        surface_ids = hit.surface_ids
        known = (surface_ids >= 0) & (surface_ids < self.surface_count)
        index = jnp.clip(surface_ids, 0, self.surface_count - 1)
        code = self.finish_codes[index]
        ground = code == _GROUND
        ground_painted = code == _GROUND_PAINTED
        painted = (code == _POLISHED_PAINTED) | ground_painted
        conducting = (hit.surface_kinds == int(NonSequentialSurfaceKind.MIRROR)) | painted
        directions = hit.directions
        normals = hit.normals
        frame_axes = hit.frame_axes

        def respond(facets: Array, conductor: Array) -> _FacetResponse:
            return _facet_response(
                directions,
                facets,
                hit.incident_indices,
                hit.transmitted_indices,
                hit.jones_vectors,
                frame_axes,
                jnp.broadcast_to(conductor, facets.shape[:-1]),
            )

        polished = respond(normals, conducting)
        reflected_directions = polished.reflected_directions
        reflected_jones = polished.reflected_jones
        reflected_axes = polished.axes
        transmitted_directions = polished.transmitted_directions
        transmitted_jones = polished.transmitted_jones
        transmitted_axes = polished.axes
        reflectance = polished.reflectance
        transmittance = polished.transmittance
        reflection_valid = polished.reflection_valid
        transmission_valid = polished.transmission_valid

        if "ground" in self.finishes or "ground-front-painted" in self.finishes:
            uniforms = jnp.moveaxis(
                jax.vmap(
                    lambda key: jr.uniform(
                        key,
                        (self.facet_attempts, _ATTEMPT_UNIFORMS),
                        dtype=jnp.float64,
                    )
                )(keys),
                1,
                0,
            )
            tangents = jnp.cross(normals, frame_axes)
            cosine = jnp.sqrt(uniforms[..., 5])
            sine = jnp.sqrt(1.0 - uniforms[..., 5])
            azimuth = 2.0 * jnp.pi * uniforms[..., 6]
            lambertian_directions = -cosine[..., None] * normals + sine[..., None] * (
                jnp.cos(azimuth)[..., None] * frame_axes
                + jnp.sin(azimuth)[..., None] * tangents
            )
            bisectors, _ = _unit_rows(directions - lambertian_directions)
            lambertian = respond(bisectors, jnp.ones_like(conducting))
            reflected_directions = _select(
                ground_painted, lambertian.reflected_directions[0], reflected_directions
            )
            reflected_jones = _select(
                ground_painted, lambertian.reflected_jones[0], reflected_jones
            )
            reflected_axes = _select(ground_painted, lambertian.axes[0], reflected_axes)
            reflection_valid = jnp.where(
                ground_painted, lambertian.reflection_valid[0], reflection_valid
            )
            if "ground" in self.finishes:
                outgoing, jones, axes, reflects, accepted = self._ground_attempts(
                    index,
                    uniforms,
                    directions,
                    normals,
                    frame_axes,
                    tangents,
                    conducting,
                    lambertian,
                    respond,
                )
                ground_reflects = ground & reflects
                ground_transmits = ground & ~reflects
                reflected_directions = _select(
                    ground_reflects, outgoing, reflected_directions
                )
                reflected_jones = _select(ground_reflects, jones, reflected_jones)
                reflected_axes = _select(ground_reflects, axes, reflected_axes)
                transmitted_directions = _select(
                    ground_transmits, outgoing, transmitted_directions
                )
                transmitted_jones = _select(ground_transmits, jones, transmitted_jones)
                transmitted_axes = _select(ground_transmits, axes, transmitted_axes)
                reflectance = jnp.where(
                    ground, jnp.where(reflects, 1.0, 0.0), reflectance
                )
                transmittance = jnp.where(
                    ground, jnp.where(reflects, 0.0, 1.0), transmittance
                )
                reflection_valid = jnp.where(ground, accepted, reflection_valid)
                transmission_valid = jnp.where(
                    ground, accepted & ~reflects, transmission_valid
                )

        survival = self.reflectivity[index]
        return OpticalSurfaceInteraction(
            reflected_directions,
            transmitted_directions,
            reflected_jones,
            transmitted_jones,
            survival * reflectance,
            survival * transmittance,
            known & reflection_valid,
            known & ~conducting & transmission_valid,
            reflected_axes,
            transmitted_axes,
        )

    def _ground_attempts(
        self,
        index: Array,
        uniforms: Array,
        directions: Array,
        normals: Array,
        frame_axes: Array,
        tangents: Array,
        conducting: Array,
        lambertian: _FacetResponse,
        respond: _Responder,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """First accepted ``(facet, outcome, branch)`` attempt per lane.

        Returns the accepted outcome's direction, Jones vector, and axis,
        whether it is a reflection, and whether any attempt was accepted.
        """

        sigma = self.sigma_alpha[index]
        rough = sigma > 0.0
        safe_sigma = jnp.where(rough, sigma, 1.0)
        # Truncated-Rayleigh proposal on [0, pi/2) by inversion, accepted with
        # probability sin(alpha) / alpha >= 2 / pi: the accepted tilt density
        # is proportional to exp(-alpha² / 2 sigma²) sin(alpha).
        tail = jnp.exp(-((0.5 * jnp.pi) ** 2) / (2.0 * safe_sigma * safe_sigma))
        tilt = jnp.where(
            rough,
            safe_sigma * jnp.sqrt(-2.0 * jnp.log1p(-uniforms[..., 0] * (1.0 - tail))),
            0.0,
        )
        tilt_accepted = uniforms[..., 1] * tilt <= jnp.sin(tilt)
        azimuth = 2.0 * jnp.pi * uniforms[..., 2]
        facets = jnp.cos(tilt)[..., None] * normals + jnp.sin(tilt)[..., None] * (
            jnp.cos(azimuth)[..., None] * frame_axes
            + jnp.sin(azimuth)[..., None] * tangents
        )
        facet = respond(facets, conducting)
        reflects = (
            uniforms[..., 3] * (facet.reflectance + facet.transmittance)
            < facet.reflectance
        )
        spike_bound = self.specular_spike[index]
        lobe_bound = spike_bound + self.specular_lobe[index]
        back_bound = lobe_bound + self.backscatter[index]
        branch = uniforms[..., 4]
        spike = branch < spike_bound
        lobe = ~spike & (branch < lobe_bound)
        back = ~spike & ~lobe & (branch < back_bound)
        lobe_leaves = jnp.sum(facet.reflected_directions * normals, axis=-1) < 0.0
        transmission_enters = facet.transmission_valid & (
            jnp.sum(facet.transmitted_directions * normals, axis=-1) > 0.0
        )
        valid = (
            tilt_accepted
            & facet.reflection_valid
            & jnp.where(reflects, ~lobe | lobe_leaves, transmission_enters)
        )
        attempt = jnp.argmax(valid, axis=0)
        accepted = jnp.any(valid, axis=0)
        mirrors = jnp.ones_like(conducting)
        specular = respond(normals, mirrors)
        backward = respond(directions, mirrors)

        def reflected(
            specular_value: Array,
            facet_value: Array,
            backward_value: Array,
            lambertian_value: Array,
        ) -> Array:
            return _select(
                spike,
                specular_value,
                _select(
                    lobe,
                    facet_value,
                    _select(back, backward_value, lambertian_value),
                ),
            )

        outgoing = _select(
            reflects,
            reflected(
                specular.reflected_directions,
                facet.reflected_directions,
                backward.reflected_directions,
                lambertian.reflected_directions,
            ),
            facet.transmitted_directions,
        )
        jones = _select(
            reflects,
            reflected(
                specular.reflected_jones,
                facet.reflected_jones,
                backward.reflected_jones,
                lambertian.reflected_jones,
            ),
            facet.transmitted_jones,
        )
        axes = _select(
            reflects,
            reflected(specular.axes, facet.axes, backward.axes, lambertian.axes),
            facet.axes,
        )
        return (
            _first(outgoing, attempt),
            _first(jones, attempt),
            _first(axes, attempt),
            _first(reflects, attempt),
            accepted,
        )


__all__ = ["OpticalSurfaceFinish", "UnifiedSurfaceModel"]
