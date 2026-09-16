#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule


class ReptationObservablePlan(StrictModule):
    maximum_frames: int = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    maximum_chains: int = eqx.field(static=True)
    maximum_beads_per_chain: int = eqx.field(static=True)
    maximum_modes: int = eqx.field(static=True)
    minimum_origins: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_frames: int,
        maximum_particles: int,
        maximum_chains: int,
        maximum_beads_per_chain: int,
        /,
        *,
        maximum_modes: int = 4,
        minimum_origins: int = 8,
    ):
        values = (
            maximum_frames,
            maximum_particles,
            maximum_chains,
            maximum_beads_per_chain,
            maximum_modes,
            minimum_origins,
        )
        if any(int(value) <= 0 for value in values):
            raise ValueError("All reptation observable capacities must be positive.")
        self.maximum_frames = int(maximum_frames)
        self.maximum_particles = int(maximum_particles)
        self.maximum_chains = int(maximum_chains)
        self.maximum_beads_per_chain = int(maximum_beads_per_chain)
        self.maximum_modes = int(maximum_modes)
        self.minimum_origins = int(minimum_origins)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reptation-observables",
                "frames": self.maximum_frames,
                "particles": self.maximum_particles,
                "chains": self.maximum_chains,
                "beads": self.maximum_beads_per_chain,
                "modes": self.maximum_modes,
                "origins": self.minimum_origins,
            }
        )


class ReptationObservableResult(StrictModule):
    lag_steps: Array
    lag_times: Array
    monomer_msd: Array
    center_of_mass_msd: Array
    internal_msd: Array
    end_to_end_correlation: Array
    rouse_mode_msd: Array
    origin_counts: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def reptation_observables(
    plan: ReptationObservablePlan,
    unwrapped_positions: ArrayLike,
    chain_indices: ArrayLike,
    chain_mask: ArrayLike,
    lag_steps: ArrayLike,
    time_step: float,
    /,
    *,
    coordinate_representation: Literal["unwrapped"],
) -> ReptationObservableResult:
    if not isinstance(plan, ReptationObservablePlan):
        raise TypeError("plan must be ReptationObservablePlan.")
    if coordinate_representation != "unwrapped":
        raise ValueError("Reptation observables require unwrapped coordinates.")
    positions = jnp.asarray(unwrapped_positions)
    indices = jnp.asarray(chain_indices, dtype=jnp.int32)
    mask = jnp.asarray(chain_mask, dtype=bool)
    lags_host = np.asarray(lag_steps, dtype=np.int32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("unwrapped_positions must have shape (frames, particles, 3).")
    if indices.ndim != 2 or mask.shape != indices.shape:
        raise ValueError("chain_indices and chain_mask must have matching rank-2 shapes.")
    frames, particles, _ = positions.shape
    chains, beads = indices.shape
    if (
        frames > plan.maximum_frames
        or particles > plan.maximum_particles
        or chains > plan.maximum_chains
        or beads > plan.maximum_beads_per_chain
    ):
        raise ValueError("Reptation observable input exceeds the prepared capacities.")
    if lags_host.ndim != 1 or lags_host.size == 0:
        raise ValueError("lag_steps must be a nonempty rank-1 sequence.")
    if np.any(lags_host < 0) or np.any(lags_host >= frames):
        raise ValueError("Every lag must satisfy 0 <= lag < frame count.")
    if not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be finite and positive.")
    if np.any(np.asarray(indices)[np.asarray(mask)] < 0) or np.any(
        np.asarray(indices)[np.asarray(mask)] >= particles
    ):
        raise ValueError("Active chain indices must address trajectory particles.")

    chain_positions = positions[:, indices, :]
    weights = mask.astype(positions.dtype)
    chain_counts = jnp.sum(weights, axis=1)
    if bool(jnp.any(chain_counts <= 0.0)):
        raise ValueError("Every chain must contain at least one active bead.")
    center_of_mass = (
        jnp.sum(chain_positions * weights[None, ..., None], axis=2)
        / chain_counts[None, :, None]
    )
    relative = chain_positions - center_of_mass[:, :, None, :]
    last = jnp.sum(mask, axis=1).astype(jnp.int32) - 1
    end = jnp.take_along_axis(chain_positions, last[None, :, None, None], axis=2)[
        :, :, 0, :
    ]
    end_to_end = end - chain_positions[:, :, 0, :]
    end_to_end_norm = jnp.mean(jnp.sum(end_to_end * end_to_end, axis=-1))

    bead_rank = jnp.arange(beads, dtype=positions.dtype)[None, :]
    phase = (
        jnp.pi
        * jnp.arange(1, plan.maximum_modes + 1, dtype=positions.dtype)[:, None, None]
        * (bead_rank[None, ...] + 0.5)
        / chain_counts[None, :, None]
    )
    basis = jnp.cos(phase) * weights[None, ...]
    modes = contract("mcn,tcnk->tmck", basis, chain_positions)
    modes = modes / chain_counts[None, None, :, None]

    monomer_values = []
    center_values = []
    internal_values = []
    end_values = []
    mode_values = []
    counts = []
    active_beads = jnp.sum(weights)
    for lag in lags_host.tolist():
        origins = frames - int(lag)
        displacement = chain_positions[lag:] - chain_positions[:origins]
        center_displacement = center_of_mass[lag:] - center_of_mass[:origins]
        internal_displacement = relative[lag:] - relative[:origins]
        mode_displacement = modes[lag:] - modes[:origins]
        monomer_values.append(
            jnp.sum(jnp.sum(displacement * displacement, axis=-1) * weights[None, ...])
            / (origins * active_beads)
        )
        center_values.append(
            jnp.mean(jnp.sum(center_displacement * center_displacement, axis=-1))
        )
        internal_values.append(
            jnp.sum(
                jnp.sum(internal_displacement * internal_displacement, axis=-1)
                * weights[None, ...]
            )
            / (origins * active_beads)
        )
        end_values.append(
            jnp.mean(jnp.sum(end_to_end[lag:] * end_to_end[:origins], axis=-1))
            / jnp.maximum(end_to_end_norm, jnp.finfo(positions.dtype).tiny)
        )
        mode_values.append(
            jnp.mean(jnp.sum(mode_displacement * mode_displacement, axis=-1), axis=(0, 2))
        )
        counts.append(origins)

    lags = jnp.asarray(lags_host)
    monomer = jnp.stack(monomer_values)
    center = jnp.stack(center_values)
    internal = jnp.stack(internal_values)
    end_correlation = jnp.stack(end_values)
    mode_msd = jnp.stack(mode_values)
    origin_counts = jnp.asarray(counts, dtype=jnp.int32)
    finite = (
        jnp.all(jnp.isfinite(monomer))
        & jnp.all(jnp.isfinite(center))
        & jnp.all(jnp.isfinite(internal))
        & jnp.all(jnp.isfinite(end_correlation))
        & jnp.all(jnp.isfinite(mode_msd))
    )
    successful = finite & jnp.all(origin_counts >= plan.minimum_origins)
    return ReptationObservableResult(
        lags,
        lags.astype(positions.dtype) * float(time_step),
        monomer,
        center,
        internal,
        end_correlation,
        mode_msd,
        origin_counts,
        finite,
        successful,
        plan.plan_id,
    )


__all__ = [
    "ReptationObservablePlan",
    "ReptationObservableResult",
    "reptation_observables",
]
