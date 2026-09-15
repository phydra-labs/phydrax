#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..geometric._interface import OpticalRayState
from ..geometric._paraxial import _COORDINATE_CONVENTION
from ..geometric._resonator import ParaxialResonatorMode
from ._core import BeamletFrame, BeamletStatus, GaussianBeamletState


def gaussian_beamlet_from_resonator_mode(
    mode: ParaxialResonatorMode,
    chief_ray: OpticalRayState,
    frame: BeamletFrame,
    /,
    *,
    amplitude: ArrayLike,
    angular_frequency: ArrayLike,
    medium_wavenumber: ArrayLike,
) -> GaussianBeamletState:
    """Create one beamlet from a certified positive resonator eigenspace."""
    if not isinstance(mode, ParaxialResonatorMode):
        raise TypeError("mode must be a ParaxialResonatorMode.")
    if not isinstance(chief_ray, OpticalRayState) or not isinstance(frame, BeamletFrame):
        raise TypeError(
            "chief_ray and frame must be optical ray and beamlet frame values."
        )
    if jnp.asarray(chief_ray.origins).shape != (3,):
        raise ValueError("A resonator mode requires one unbatched chief ray.")
    if mode.frame_id != frame.frame_id:
        raise ValueError("The beamlet frame does not match the resonator mode frame.")
    if mode.coordinate_convention != _COORDINATE_CONVENTION:
        raise ValueError("The resonator mode uses an incompatible coordinate convention.")
    if not mode.source_prepared_id or not mode.resonator_id:
        raise ValueError("The resonator mode is missing exact provenance identifiers.")
    amplitude_ = jnp.asarray(amplitude)
    frequency = jnp.asarray(angular_frequency)
    wavenumber = jnp.asarray(medium_wavenumber)
    if amplitude_.shape != ():
        raise ValueError("amplitude must be a scalar.")
    if frequency.shape != () or not jnp.issubdtype(frequency.dtype, jnp.floating):
        raise ValueError("angular_frequency must be a real scalar.")
    if wavenumber.shape != () or not jnp.issubdtype(wavenumber.dtype, jnp.floating):
        raise ValueError("medium_wavenumber must be a real scalar.")
    lagrangian = eqx.error_if(
        mode.lagrangian_state,
        ~jnp.asarray(mode.valid, dtype=bool),
        "The resonator mode is not positively certified.",
    )
    frequency = eqx.error_if(
        frequency,
        ~jnp.isfinite(frequency) | (frequency <= 0.0),
        "angular_frequency must be finite and positive.",
    )
    wavenumber = eqx.error_if(
        wavenumber,
        ~jnp.isfinite(wavenumber) | (wavenumber <= 0.0),
        "medium_wavenumber must be finite and positive.",
    )
    amplitude_ = eqx.error_if(
        amplitude_,
        ~jnp.isfinite(amplitude_),
        "amplitude must be finite.",
    )
    return GaussianBeamletState(
        chief_ray,
        frame,
        lagrangian,
        amplitude_,
        wavenumber,
        frequency,
        topology_id=mode.resonator_id,
        source_prepared_id=mode.source_prepared_id,
        valid=jnp.asarray(True),
        status=jnp.asarray(int(BeamletStatus.SUCCESS), dtype=jnp.int32),
    )


__all__ = ["gaussian_beamlet_from_resonator_mode"]
