#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Interference colors of thin-film samples from thickness and orientation."""

from __future__ import annotations

from enum import IntFlag
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..optics.wave import ThinFilmInterferencePlan
from ..typing import Bool, checked, Float, Identifier, Int32, VariadicDim
from ._colorimetry import SpectralColorimetryPlan, SpectralColorimetryResult


class _SampleDims(VariadicDim):
    """Film-sample axes of one appearance evaluation."""


class ThinFilmAppearanceStatus(IntFlag):
    """Fail-closed per-sample status bits of one appearance evaluation."""

    SUCCESS = 0
    INVALID_GEOMETRY = 1
    BACK_FACING = 2
    INTERFERENCE_REJECTED = 4
    COLORIMETRY_REJECTED = 8


class ThinFilmAppearanceEvidence(StrictModule):
    """Geometry, interference, and colorimetry evidence per film sample.

    ``facet_alignment`` is ``n . h``: one for the mirror configuration, which a
    smooth film requires in order to reflect a directional source towards the
    viewer. ``interference_status`` preserves the owning upstream cause;
    ``colorimetry_status`` records rejection of its acceptance-masked spectrum.
    """

    __strict_contract__ = True

    incidence_cosine: Float[_SampleDims]
    facet_alignment: Float[_SampleDims]
    illumination_directions: Float[_SampleDims, Literal[3]]
    minimum_samples_per_fringe: Float[_SampleDims]
    maximum_energy_residual: Float[_SampleDims]
    out_of_gamut: Bool[_SampleDims]
    interference_status: Int32[_SampleDims]
    colorimetry_status: Int32[_SampleDims]
    accepted: Bool[_SampleDims]
    status: Int32[_SampleDims]
    plan_id: Identifier = eqx.field(static=True)


class ThinFilmAppearanceResult(StrictModule):
    """Reflected interference colors and their evidence.

    ``colors`` holds XYZ, linear sRGB, and encoded sRGB of the unpolarized
    specular reflectance. Every nonzero interference status is rejected before
    colorimetry, and rejected samples carry NaN colors.
    """

    colors: SpectralColorimetryResult
    evidence: ThinFilmAppearanceEvidence

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted


def _directions(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be real floating data.")
    if array.shape[-1:] != (3,):
        raise ValueError(f"{name} must have a trailing axis of length 3.")
    return array


def _unit(vectors: Array, /) -> tuple[Array, Array]:
    norm = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    valid = jnp.all(jnp.isfinite(vectors), axis=-1) & jnp.isfinite(norm) & (norm > 0.0)
    safe = jnp.where(valid, norm, 1.0)
    return vectors / safe[..., None], valid


class ThinFilmAppearancePlan(StrictModule):
    """Specular interference color of film samples under one spectral illuminant.

    The interference and colorimetry plans must share the same wavelengths. Each
    sample has a film thickness (m), a unit normal pointing into the ambient
    medium, and a direction towards the viewer. The reflected light path is
    evaluated at the half-vector incidence ``cos theta = h . v`` with
    ``h = normalize(l + v)``; the returned color is the unpolarized Airy
    reflectance of that path integrated against the illuminant and observer.

    Without ``illumination_directions`` the film reflects a uniform spectral
    environment: the path arriving from the mirror direction
    ``l = 2 (n . v) n - v`` is observed, so ``h = n`` and ``cos theta = n . v``.
    With directional light, ``facet_alignment = n . h`` reports how far the
    sample is from the mirror configuration; shading lobes remain the caller's.
    ``two_sided=True`` flips normals towards the viewer and requires identical
    real ambient and substrate indices (a free-standing film). Plain arrays are
    accepted, so any film solver's vertex or face fields can be rendered;
    rendering never alters or certifies the film state.
    """

    __strict_contract__ = True

    interference: ThinFilmInterferencePlan
    colorimetry: SpectralColorimetryPlan
    two_sided: bool = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        interference: ThinFilmInterferencePlan,
        colorimetry: SpectralColorimetryPlan,
        /,
        *,
        two_sided: bool = False,
    ) -> None:
        if not isinstance(two_sided, bool):
            raise TypeError("two_sided must be a bool.")
        if not np.array_equal(
            np.asarray(interference.wavelengths), np.asarray(colorimetry.wavelengths)
        ):
            raise ValueError(
                "Interference and colorimetry plans must share identical wavelengths."
            )
        if two_sided and not np.array_equal(
            np.asarray(interference.ambient_index).astype(np.complex128),
            np.asarray(interference.substrate_index).astype(np.complex128),
        ):
            raise ValueError(
                "two_sided requires identical real ambient and substrate indices."
            )
        self.interference = interference
        self.colorimetry = colorimetry
        self.two_sided = two_sided
        self.plan_id = canonical_fingerprint(
            {
                "kind": "thin-film-appearance-plan",
                "interference": interference.plan_id,
                "colorimetry": colorimetry.plan_id,
                "two_sided": two_sided,
            }
        )

    def evaluate(
        self,
        thickness: ArrayLike,
        normals: ArrayLike,
        view_directions: ArrayLike,
        illumination_directions: ArrayLike | None = None,
        /,
    ) -> ThinFilmAppearanceResult:
        """Render samples ``B`` from thickness ``B`` and directions ``B + (3,)``.

        Directions are unit-normalized here and broadcast against the sample
        shape; zero or non-finite vectors are invalid geometry.
        """
        thickness_ = jnp.asarray(thickness)
        normals_, normal_valid = _unit(_directions(normals, "normals"))
        view, view_valid = _unit(_directions(view_directions, "view_directions"))
        supplied = (
            None
            if illumination_directions is None
            else _unit(_directions(illumination_directions, "illumination_directions"))
        )
        batch_shape = jnp.broadcast_shapes(
            thickness_.shape,
            normals_.shape[:-1],
            view.shape[:-1],
            () if supplied is None else supplied[0].shape[:-1],
        )
        normals_ = jnp.broadcast_to(normals_, batch_shape + (3,))
        view = jnp.broadcast_to(view, batch_shape + (3,))
        geometry_valid = jnp.broadcast_to(normal_valid & view_valid, batch_shape)
        view_cosine = jnp.sum(normals_ * view, axis=-1)
        if self.two_sided:
            normals_ = jnp.where((view_cosine < 0.0)[..., None], -normals_, normals_)
            view_cosine = jnp.abs(view_cosine)
        if supplied is None:
            light = 2.0 * view_cosine[..., None] * normals_ - view
            incidence_cosine = view_cosine
            alignment = jnp.ones(batch_shape, dtype=view_cosine.dtype)
            light_cosine = view_cosine
        else:
            light = jnp.broadcast_to(supplied[0], batch_shape + (3,))
            half, half_valid = _unit(light + view)
            geometry_valid = (
                geometry_valid & jnp.broadcast_to(supplied[1], batch_shape) & half_valid
            )
            incidence_cosine = jnp.sum(half * view, axis=-1)
            alignment = jnp.sum(normals_ * half, axis=-1)
            light_cosine = jnp.sum(normals_ * light, axis=-1)
        facing = (view_cosine > 0.0) & (light_cosine > 0.0)
        usable = geometry_valid & facing
        # Unusable samples enter interference as NaN. Interference masks every
        # nonzero-status spectrum while retaining finite quality diagnostics in
        # its evidence, so colorimetry can never certify a rejected spectrum.
        interference = self.interference.evaluate(
            thickness_, jnp.where(usable, incidence_cosine, jnp.nan)
        )
        colors = self.colorimetry.evaluate(interference.unpolarized_reflectance)
        status = _status(usable, geometry_valid, interference.status, colors.status)
        accepted = status == int(ThinFilmAppearanceStatus.SUCCESS)
        evidence = ThinFilmAppearanceEvidence(
            incidence_cosine=jnp.where(usable, incidence_cosine, jnp.nan),
            facet_alignment=jnp.where(usable, alignment, jnp.nan),
            illumination_directions=jnp.where(usable[..., None], light, jnp.nan),
            minimum_samples_per_fringe=interference.evidence.minimum_samples_per_fringe,
            maximum_energy_residual=interference.evidence.maximum_energy_residual,
            out_of_gamut=colors.evidence.out_of_gamut & usable,
            interference_status=interference.status,
            colorimetry_status=colors.status,
            accepted=accepted,
            status=status,
            plan_id=self.plan_id,
        )
        return ThinFilmAppearanceResult(colors=colors, evidence=evidence)


def _status(
    usable: Array,
    geometry_valid: Array,
    interference_status: Array,
    colorimetry_status: Array,
    /,
) -> Array:
    interference_accepted = usable & (interference_status == 0)
    bits = (
        (~geometry_valid, ThinFilmAppearanceStatus.INVALID_GEOMETRY),
        (geometry_valid & ~usable, ThinFilmAppearanceStatus.BACK_FACING),
        (usable & ~interference_accepted, ThinFilmAppearanceStatus.INTERFERENCE_REJECTED),
        (
            interference_accepted & (colorimetry_status != 0),
            ThinFilmAppearanceStatus.COLORIMETRY_REJECTED,
        ),
    )
    status = jnp.zeros(geometry_valid.shape, dtype=jnp.int32)
    for condition, flag in bits:
        status = status | jnp.where(condition, int(flag), 0).astype(jnp.int32)
    return status


__all__ = [
    "ThinFilmAppearanceEvidence",
    "ThinFilmAppearancePlan",
    "ThinFilmAppearanceResult",
    "ThinFilmAppearanceStatus",
]
