#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Support-aware thin-film colors for physical surface fields."""

from __future__ import annotations

from enum import IntFlag
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..typing import Bool, Float, Identifier, Int32, VariadicDim
from ._thin_film_appearance import ThinFilmAppearanceEvidence, ThinFilmAppearancePlan


class _SurfaceSampleDims(VariadicDim):
    """Vertex or face sample axes of one surface-color evaluation."""


class ThinFilmSurfaceColorStatus(IntFlag):
    """Fail-closed status bits for one rendered surface sample."""

    SUCCESS = 0
    UNSUPPORTED = 1
    APPEARANCE_REJECTED = 2


@final
class ThinFilmSurfaceColorResult(StrictModule):
    """Linear and encoded sRGB surface fields with explicit optical support.

    Both color fields have a trailing RGB axis and are directly consumable as
    vertex or face fields by :class:`SurfaceImagePlan`. Unsupported samples and
    samples rejected by the owning :class:`ThinFilmAppearancePlan` remain NaN.
    ``appearance_evidence`` preserves the owner's geometry, interference,
    colorimetry, and detailed rejection evidence.
    """

    __strict_contract__ = True

    linear_srgb: Float[_SurfaceSampleDims, Literal[3]]
    encoded_srgb: Float[_SurfaceSampleDims, Literal[3]]
    support: Bool[_SurfaceSampleDims]
    accepted: Bool[_SurfaceSampleDims]
    appearance_status: Int32[_SurfaceSampleDims]
    status: Int32[_SurfaceSampleDims]
    appearance_evidence: ThinFilmAppearanceEvidence
    appearance_plan_id: Identifier = eqx.field(static=True)


def thin_film_surface_colors(
    appearance: ThinFilmAppearancePlan,
    thickness_m: ArrayLike,
    normals: ArrayLike,
    view_directions: ArrayLike,
    optical_support: ArrayLike,
    illumination_directions: ArrayLike | None = None,
    /,
) -> ThinFilmSurfaceColorResult:
    """Evaluate support-aware thin-film colors from plain physical arrays.

    ``optical_support`` is supplied by the physics/geometry owner. This
    function neither infers thickness on unsupported Plateau borders or
    junctions nor certifies any input. It masks unsupported thickness before
    invoking ``appearance`` so no plausible optical sample can be created
    there. Rejected and unsupported outputs are NaN by contract.
    """
    if not isinstance(appearance, ThinFilmAppearancePlan):
        raise TypeError("appearance must be a ThinFilmAppearancePlan.")
    thickness = jnp.asarray(thickness_m)
    if not jnp.issubdtype(thickness.dtype, jnp.floating):
        raise TypeError("thickness_m must be real floating data.")
    support = jnp.asarray(optical_support)
    if support.dtype != jnp.bool_:
        raise TypeError("optical_support must be boolean data.")
    sample_shape = jnp.broadcast_shapes(thickness.shape, support.shape)
    support = jnp.broadcast_to(support, sample_shape)
    thickness = jnp.broadcast_to(thickness, sample_shape)
    masked_thickness = jnp.where(support, thickness, jnp.nan)
    result = appearance.evaluate(
        masked_thickness,
        normals,
        view_directions,
        illumination_directions,
    )
    support = jnp.broadcast_to(support, result.status.shape)
    accepted = support & result.accepted
    status = (
        jnp.where(
            ~support,
            int(ThinFilmSurfaceColorStatus.UNSUPPORTED),
            int(ThinFilmSurfaceColorStatus.SUCCESS),
        )
        | jnp.where(
            support & ~result.accepted,
            int(ThinFilmSurfaceColorStatus.APPEARANCE_REJECTED),
            int(ThinFilmSurfaceColorStatus.SUCCESS),
        )
    ).astype(jnp.int32)
    return ThinFilmSurfaceColorResult(
        linear_srgb=result.colors.linear_srgb,
        encoded_srgb=result.colors.encoded_srgb,
        support=support,
        accepted=accepted,
        appearance_status=result.status,
        status=status,
        appearance_evidence=result.evidence,
        appearance_plan_id=appearance.plan_id,
    )


__all__ = [
    "ThinFilmSurfaceColorResult",
    "ThinFilmSurfaceColorStatus",
    "thin_film_surface_colors",
]
