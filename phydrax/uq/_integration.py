#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import coordax as cx

from ..integration import WeightedSampleTarget
from ._particle import ParticleFilterResult


def particle_posterior_measure(
    result: ParticleFilterResult,
    /,
    *,
    particle_dim: str = "particle",
    time_dim: str = "time",
) -> WeightedSampleTarget:
    """Expose filtering posteriors as dependent weighted empirical measures.

    Physical case and filtering-time axes are retained. The particle axis is reduced,
    failed particles and inactive filtering steps are masked, and resampling ancestry is
    preserved for diagnostics. IID uncertainty is intentionally disabled because
    filtering particles are dependent.
    """
    if not isinstance(result, ParticleFilterResult):
        raise TypeError("result must be a ParticleFilterResult.")
    particle_dim = str(particle_dim)
    time_dim = str(time_dim)
    if not particle_dim or not time_dim or particle_dim == time_dim:
        raise ValueError("particle_dim and time_dim must be distinct non-empty names.")
    if particle_dim in result.case_axes or time_dim in result.case_axes:
        raise ValueError("Particle and time dimensions must not collide with case axes.")
    weight_dims = result.case_axes + (time_dim, particle_dim)
    sample_dims = weight_dims + (None,) * len(result.state_shape)
    active = result.step_valid & result.valid
    mask = active[..., None] & result.transition_valid
    provenance = f"particle-filter-posterior:{result.model_id}:{result.sequence_id}"
    return WeightedSampleTarget(
        cx.Field(result.predicted_particles, dims=sample_dims),
        cx.Field(result.posterior_log_weights, dims=weight_dims),
        normalized=True,
        independent=False,
        mask=cx.Field(mask, dims=weight_dims),
        ancestry=cx.Field(result.ancestor_indices, dims=weight_dims),
        sample_axes=particle_dim,
        provenance=provenance,
    )


__all__ = ["particle_posterior_measure"]
